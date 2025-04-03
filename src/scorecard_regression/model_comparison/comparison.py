"""
Two-Model Comparison for OAF Loan Repayment Rate Prediction

This module implements and compares two models:
1. Model 1: Using only application-time features
2. Model 2: Using application-time features + September payment data

This allows for fair evaluation of model performance in different
business contexts, with clear separation of features available at
different time points.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple, Optional, Union, Any

# Import our metrics and analysis functions
from ..profitability.metrics import calculate_business_metrics
from ..profitability.threshold_analysis import analyze_multiple_thresholds, analyze_cutoff_tradeoffs
from ..utils import calculate_metrics

# Define feature groups
APPLICATION_FEATURES = [
    # Basic loan details
    'region', 'duka_name', 'nominal_contract_value',
    'deposit_amount', 'area', 'sales_territory',
    
    # Derived features available at application time
    'deposit_ratio', 'contract_start_day', 'contract_day_name',
    'month', 'contract_month', 'contract_quarter', 'is_weekend',
    
    # Historical metrics (available at application time)
    'historical_cum_loans_region', 'historical_cum_deposit_region',
    'historical_cum_value_region', 'historical_cum_customers_region',
    'historical_cum_loans_sales_territory', 'historical_cum_deposit_sales_territory',
    'historical_cum_value_sales_territory', 'historical_cum_customers_sales_territory',
    'historical_cum_loans_area', 'historical_cum_deposit_area',
    'historical_cum_value_area', 'historical_cum_customers_area',
    
    # Relative position metrics
    'deposit_ratio_rank_region', 'contract_value_rank_region',
    'historical_loans_rank_region', 'historical_value_rank_region', 'historical_deposit_rank_region',
    'deposit_ratio_rank_sales_territory', 'contract_value_rank_sales_territory',
    'historical_loans_rank_sales_territory', 'historical_value_rank_sales_territory', 
    'historical_deposit_rank_sales_territory',
    'deposit_ratio_rank_area', 'contract_value_rank_area',
    'historical_loans_rank_area', 'historical_value_rank_area', 'historical_deposit_rank_area',
    
    # Infrastructure metrics
    'distinct_dukas_region', 'distinct_customers_region',
    'distinct_dukas_sales_territory', 'distinct_customers_sales_territory',
    'distinct_dukas_area', 'distinct_customers_area',
    
    # Geographic features
    'distance_to_centroid', 'distance_to_nairobi',
    'distance_to_mombasa', 'distance_to_kisumu'
]

SEPTEMBER_PAYMENT_FEATURES = [
    # Payment data from September
    'cumulative_amount_paid_start',
    'sept_23_repayment_rate',
    
    # Time-based features
    'days_diff_contract_start_to_sept_23',
    'month_diff_contract_start_to_sept_23',
    'months_since_start',
    'days_since_start',
    
    # Derived payment features
    'post_payment_velocity',
    'post_days_to_sept',
    'post_payment_ratio',
    'post_deposit_to_paid_ratio',
    'post_sept_repayment_rate'
]

# Constants for business calculations
BUSINESS_PARAMS = {
    'margin': 0.16,  # 16% profit margin
    'default_loss_rate': 1.0  # 100% loss on defaulted amount
}

def load_and_prepare_data(
    data_path: str,
    test_size: float = 0.3,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Load and prepare data for model training and evaluation.
    
    Args:
        data_path: Path to the features CSV file
        test_size: Fraction of data to use for testing
        random_state: Random seed for reproducibility
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Define target variable (November repayment rate)
    target_col = 'nov_23_repayment_rate'
    
    # Drop rows with missing target
    if df[target_col].isna().any():
        print(f"Dropping {df[target_col].isna().sum()} rows with missing target values")
        df = df.dropna(subset=[target_col])
    
    # Split data into training and testing sets
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state
    )
    
    print(f"Data split: {len(train_df)} training samples, {len(test_df)} testing samples")
    
    # Extract target variable
    y_train = train_df[target_col]
    y_test = test_df[target_col]
    
    return train_df, test_df, y_train, y_test

def filter_features(
    df: pd.DataFrame,
    feature_list: List[str],
    handle_categorical: bool = True,
    handle_missing: str = 'mean',
    encoder_dict: Optional[Dict] = None
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict]]:
    """
    Filter DataFrame to include only specified features and handle categorical variables.
    
    Args:
        df: Input DataFrame
        feature_list: List of feature names to include
        handle_categorical: Whether to encode categorical variables
        handle_missing: Strategy for handling missing values ('mean', 'median', 'most_frequent')
    
    Returns:
        Filtered DataFrame with encoded categorical features
    """
    # Find which features are actually available in the dataframe
    available_features = [feat for feat in feature_list if feat in df.columns]
    
    # Warn about missing features
    missing_features = set(feature_list) - set(available_features)
    if missing_features:
        print(f"Warning: {len(missing_features)} features not found in data: {missing_features}")
    
    # Get filtered DataFrame
    filtered_df = df[available_features].copy()
    
    # First handle missing values in numeric columns
    numeric_cols = filtered_df.select_dtypes(include=['int64', 'float64']).columns
    if len(numeric_cols) > 0:
        from sklearn.impute import SimpleImputer
        numeric_imputer = SimpleImputer(strategy=handle_missing)
        filtered_df[numeric_cols] = numeric_imputer.fit_transform(filtered_df[numeric_cols])
    
    if handle_categorical:
        # Detect non-numeric columns
        non_numeric_cols = []
        for col in filtered_df.columns:
            if not pd.api.types.is_numeric_dtype(filtered_df[col]) or pd.api.types.is_object_dtype(filtered_df[col]):
                unique_count = filtered_df[col].nunique()
                if unique_count <= 30:  # Limit cardinality
                    non_numeric_cols.append(col)
                else:
                    print(f"Warning: Column '{col}' has {unique_count} unique values, too many for encoding. Column will be dropped.")
                    filtered_df = filtered_df.drop(columns=[col])
        
        if non_numeric_cols:
            print(f"\nEncoding {len(non_numeric_cols)} non-numeric features:")
            for col in non_numeric_cols:
                print(f"  - {col} ({filtered_df[col].nunique()} unique values)")
            
            # Handle missing values in categorical columns
            cat_imputer = SimpleImputer(strategy='most_frequent')
            filtered_df[non_numeric_cols] = cat_imputer.fit_transform(filtered_df[non_numeric_cols])
            
    # Handle categorical encoding
    if non_numeric_cols:
        # Create and fit/use OneHotEncoder
        from sklearn.preprocessing import OneHotEncoder
        
        if encoder_dict is None:
            # Training mode - create new encoder
            encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            encoded_data = encoder.fit_transform(filtered_df[non_numeric_cols])
            
            # Store encoder and feature names
            encoder_dict = {
                'encoder': encoder,
                'columns': non_numeric_cols,
                'feature_names': []
            }
            
            # Get feature names
            for i, col in enumerate(non_numeric_cols):
                for val in encoder.categories_[i]:
                    encoder_dict['feature_names'].append(f"{col}_{val}")
        else:
            # Test mode - use existing encoder
            encoder = encoder_dict['encoder']
            # Ensure all columns from training are present
            missing_cols = set(encoder_dict['columns']) - set(filtered_df.columns)
            if missing_cols:
                for col in missing_cols:
                    filtered_df[col] = None  # Add missing columns with NaN values
            
            # Reorder columns to match training order
            filtered_df_ordered = filtered_df[encoder_dict['columns']]
            encoded_data = encoder.transform(filtered_df_ordered)
        
        # Create DataFrame with encoded features
        encoded_df = pd.DataFrame(
            encoded_data,
            columns=encoder_dict['feature_names'],
            index=filtered_df.index
        )
            
        # Drop original non-numeric columns and add encoded ones
        filtered_df = filtered_df.drop(columns=non_numeric_cols)
        filtered_df = pd.concat([filtered_df, encoded_df], axis=1)
        
        print(f"Added {len(encoder_dict['feature_names'])} encoded features")
    
    # Verify all columns are numeric
    non_numeric = [col for col in filtered_df.columns if not pd.api.types.is_numeric_dtype(filtered_df[col])]
    if non_numeric:
        print(f"Warning: Dropping {len(non_numeric)} remaining non-numeric columns: {non_numeric}")
        filtered_df = filtered_df.drop(columns=non_numeric)
    
    # Final check for any remaining missing values
    if filtered_df.isna().any().any():
        print("Warning: Some missing values remain after processing. Using mean imputation.")
        final_imputer = SimpleImputer(strategy='mean')
        filtered_df = pd.DataFrame(
            final_imputer.fit_transform(filtered_df),
            columns=filtered_df.columns,
            index=filtered_df.index
        )
    
    # Return DataFrame and encoder dict if in training mode
    if encoder_dict is None:
        # Create encoder dict if we have categorical columns
        if 'non_numeric_cols' in locals() and non_numeric_cols:
            return filtered_df, encoder_dict
        else:
            # No categorical columns to encode
            return filtered_df, None
    else:
        # Return just the DataFrame in test mode
        return filtered_df, None

def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_params: Optional[Dict] = None
) -> GradientBoostingRegressor:
    """
    Train a Gradient Boosting regression model.
    
    Args:
        X_train: Training features
        y_train: Training target
        model_params: Model hyperparameters
    
    Returns:
        Trained model
    """
    # Default model parameters
    if model_params is None:
        model_params = {
            'n_estimators': 100,
            'max_depth': 5,
            'learning_rate': 0.1,
            'min_samples_split': 5,
            'min_samples_leaf': 5,
            'random_state': 42
        }
    
    # Initialize and train model
    model = GradientBoostingRegressor(**model_params)
    model.fit(X_train, y_train)
    
    return model

def evaluate_model(
    model: GradientBoostingRegressor,
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str
) -> Dict:
    """
    Evaluate model performance on data.
    
    Args:
        model: Trained model
        X: Feature data
        y: Target data
        model_name: Name of the model for reporting
    
    Returns:
        Dictionary of evaluation metrics
    """
    # Make predictions
    predictions = model.predict(X)
    
    # Calculate metrics
    metrics = calculate_metrics(y, predictions, include_advanced=True)
    
    # Print results
    print(f"\n{model_name} Performance:")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"MAE: {metrics['mae']:.4f}")
    print(f"R²: {metrics['r2']:.4f}")
    
    # Return predictions and metrics
    return {
        'predictions': predictions,
        'metrics': metrics
    }

def analyze_model_profit(
    y_true: pd.Series,
    y_pred: np.ndarray,
    loan_values: pd.Series,
    thresholds: List[float],
    model_name: str,
    output_dir: str
) -> Dict:
    """
    Analyze model profitability across different thresholds.
    
    Args:
        y_true: True repayment rates
        y_pred: Predicted repayment rates
        loan_values: Loan amounts
        thresholds: List of thresholds to evaluate
        model_name: Name of the model for reporting
        output_dir: Directory to save results
    
    Returns:
        Dictionary with profit analysis results
    """
    # Make sure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Analyze multiple thresholds
    metrics_df = analyze_multiple_thresholds(
        y_true, y_pred, loan_values, thresholds,
        margin=BUSINESS_PARAMS['margin'],
        default_loss_rate=BUSINESS_PARAMS['default_loss_rate'],
        output_path=os.path.join(output_dir, f"{model_name}_threshold_metrics.csv")
    )
    
    # Analyze cutoff tradeoffs
    tradeoffs = analyze_cutoff_tradeoffs(
        y_true, y_pred, loan_values, thresholds,
        business_params=BUSINESS_PARAMS,
        output_path=os.path.join(output_dir, f"{model_name}_tradeoffs.json")
    )
    
    # Print optimal thresholds
    print(f"\n{model_name} Optimal Thresholds:")
    print(f"Profit focused: {tradeoffs['optimal_thresholds']['profit']['threshold']:.2f}")
    print(f"ROI focused: {tradeoffs['optimal_thresholds']['roi']['threshold']:.2f}")
    print(f"Balanced approach: {tradeoffs['recommendations']['balanced']:.2f}")
    
    return {
        'metrics_df': metrics_df,
        'tradeoffs': tradeoffs
    }

def compare_models(model1_results: Dict, model2_results: Dict, output_dir: str) -> Dict:
    """
    Compare profitability of two models.
    
    Args:
        model1_results: Results from Model 1
        model2_results: Results from Model 2
        output_dir: Directory to save comparison results
    
    Returns:
        Dictionary with comparison results
    """
    # Make sure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Get performance metrics
    model1_metrics = model1_results['metrics']
    model2_metrics = model2_results['metrics']
    
    # Get threshold analysis dataframes
    model1_df = model1_results['profit_analysis']['metrics_df']
    model2_df = model2_results['profit_analysis']['metrics_df']
    
    # Calculate improvement percentages
    performance_improvement = {
        'rmse': (model1_metrics['rmse'] - model2_metrics['rmse']) / model1_metrics['rmse'] * 100,
        'mae': (model1_metrics['mae'] - model2_metrics['mae']) / model1_metrics['mae'] * 100,
        'r2': (model2_metrics['r2'] - model1_metrics['r2']) / max(0.001, model1_metrics['r2']) * 100,
    }
    
    # Find optimal thresholds for each model
    model1_opt = model1_results['profit_analysis']['tradeoffs']['optimal_thresholds']
    model2_opt = model2_results['profit_analysis']['tradeoffs']['optimal_thresholds']
    
    # Calculate profit improvements at optimal thresholds
    profit_improvement = (model2_opt['profit']['value'] - model1_opt['profit']['value']) / model1_opt['profit']['value'] * 100
    roi_improvement = (model2_opt['roi']['value'] - model1_opt['roi']['value']) / max(0.001, model1_opt['roi']['value']) * 100
    
    # Create comparison summary
    comparison = {
        'performance_improvement': performance_improvement,
        'profit_improvement': profit_improvement,
        'roi_improvement': roi_improvement,
        'model1_optimal': model1_opt,
        'model2_optimal': model2_opt
    }
    
    # Save to JSON
    import json
    with open(os.path.join(output_dir, "model_comparison.json"), 'w') as f:
        json.dump(comparison, f, indent=2)
    
    # Create comparison plots
    create_comparison_plots(model1_df, model2_df, output_dir)
    
    # Print key results
    print("\nModel Comparison:")
    print(f"RMSE improvement: {performance_improvement['rmse']:.2f}%")
    print(f"R² improvement: {performance_improvement['r2']:.2f}%")
    print(f"Profit improvement: {profit_improvement:.2f}%")
    print(f"ROI improvement: {roi_improvement:.2f}%")
    
    return comparison

def create_comparison_plots(model1_df: pd.DataFrame, model2_df: pd.DataFrame, output_dir: str) -> None:
    """
    Create comparison plots between two models.
    
    Args:
        model1_df: Threshold metrics for Model 1
        model2_df: Threshold metrics for Model 2
        output_dir: Directory to save plots
    """
    # Ensure path exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Merge dataframes
    model1_df['model'] = 'Application-Time Features'
    model2_df['model'] = 'With September Payment Data'
    combined_df = pd.concat([model1_df, model2_df])
    
    # Plot 1: Profit by threshold
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=combined_df, x='threshold', y='actual_profit', hue='model', marker='o')
    plt.title('Profit by Threshold')
    plt.xlabel('Threshold')
    plt.ylabel('Profit')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'profit_comparison.png'), dpi=300)
    plt.close()
    
    # Plot 2: ROI by threshold
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=combined_df, x='threshold', y='roi', hue='model', marker='o')
    plt.title('ROI by Threshold')
    plt.xlabel('Threshold')
    plt.ylabel('ROI')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'roi_comparison.png'), dpi=300)
    plt.close()
    
    # Plot 3: Approval rate by threshold
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=combined_df, x='threshold', y='approval_rate', hue='model', marker='o')
    plt.title('Approval Rate by Threshold')
    plt.xlabel('Threshold')
    plt.ylabel('Approval Rate')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'approval_rate_comparison.png'), dpi=300)
    plt.close()
    
    # Plot 4: F1 Score by threshold
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=combined_df, x='threshold', y='f1_score', hue='model', marker='o')
    plt.title('F1 Score by Threshold')
    plt.xlabel('Threshold')
    plt.ylabel('F1 Score')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'f1_comparison.png'), dpi=300)
    plt.close()
    
    # Plot 5: Profit-ROI tradeoff (scatter plot)
    plt.figure(figsize=(10, 6))
    for model in ['Application-Time Features', 'With September Payment Data']:
        model_data = combined_df[combined_df['model'] == model]
        plt.scatter(model_data['roi'], model_data['actual_profit'], label=model, alpha=0.8)
        
        # Annotate threshold values
        for i, row in model_data.iterrows():
            plt.annotate(
                f"{row['threshold']:.2f}", 
                (row['roi'], row['actual_profit']),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=8
            )
    
    plt.title('Profit vs ROI Tradeoff')
    plt.xlabel('ROI')
    plt.ylabel('Profit')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'profit_roi_tradeoff.png'), dpi=300)
    plt.close()

def save_predictions_for_holdout(
    holdout_df: pd.DataFrame, 
    model1: GradientBoostingRegressor,
    model2: GradientBoostingRegressor,
    model1_features: List[str],
    model2_features: List[str],
    output_path: str
) -> None:
    """
    Generate predictions for holdout data and save them.
    
    Args:
        holdout_df: Holdout dataset
        model1: Trained Model 1 (application-time features)
        model2: Trained Model 2 (with September payment data)
        model1_features: Features for Model 1
        model2_features: Features for Model 2
        output_path: Path to save predictions
    """
    # Ensure holdout data has all necessary features
    X1_holdout = filter_features(holdout_df, model1_features)
    X2_holdout = filter_features(holdout_df, model2_features)
    
    # Generate predictions
    model1_pred = model1.predict(X1_holdout)
    model2_pred = model2.predict(X2_holdout)
    
    # Extract loan value
    loan_values = holdout_df['nominal_contract_value']
    
    # Calculate predicted repayment amounts
    model1_amount = model1_pred * loan_values
    model2_amount = model2_pred * loan_values
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'client_id': holdout_df['client_id'],
        'nominal_contract_value': loan_values,
        'model1_predicted_rate': model1_pred,
        'model2_predicted_rate': model2_pred,
        'model1_predicted_amount': model1_amount,
        'model2_predicted_amount': model2_amount,
        'september_repayment_rate': holdout_df['sept_23_repayment_rate'],
        'september_amount_paid': holdout_df['cumulative_amount_paid_start']
    })
    
    # Add actual November data if available
    if 'nov_23_repayment_rate' in holdout_df.columns:
        results_df['actual_november_rate'] = holdout_df['nov_23_repayment_rate']
    
    if 'cumulative_amount_paid' in holdout_df.columns:
        results_df['actual_november_amount'] = holdout_df['cumulative_amount_paid']
    
    # Save results
    results_df.to_csv(output_path, index=False)
    print(f"Holdout predictions saved to {output_path}")

def feature_importance_comparison(
    model1: GradientBoostingRegressor,
    model2: GradientBoostingRegressor,
    model1_features: List[str],
    model2_features: List[str],
    output_dir: str
) -> None:
    """
    Compare feature importance between the two models.
    
    Args:
        model1: Trained Model 1
        model2: Trained Model 2
        model1_features: Features used in Model 1
        model2_features: Features used in Model 2
        output_dir: Directory to save comparison
    """
    # Get feature importances
    model1_imp = pd.DataFrame({
        'feature': model1_features,
        'importance': model1.feature_importances_,
        'model': 'Application-Time Features'
    }).sort_values('importance', ascending=False)
    
    model2_imp = pd.DataFrame({
        'feature': model2_features,
        'importance': model2.feature_importances_,
        'model': 'With September Payment Data'
    }).sort_values('importance', ascending=False)
    
    # Save to CSV
    model1_imp.to_csv(os.path.join(output_dir, "model1_feature_importance.csv"), index=False)
    model2_imp.to_csv(os.path.join(output_dir, "model2_feature_importance.csv"), index=False)
    
    # Create comparative plot for top features
    plt.figure(figsize=(12, 10))
    
    # Get top 15 features from each model
    top_features_model1 = model1_imp.head(15)
    top_features_model2 = model2_imp.head(15)
    
    # Plot top features for Model 1
    plt.subplot(2, 1, 1)
    sns.barplot(x='importance', y='feature', data=top_features_model1, color='skyblue')
    plt.title('Top 15 Features - Model 1 (Application-Time Features)')
    plt.tight_layout()
    
    # Plot top features for Model 2
    plt.subplot(2, 1, 2)
    sns.barplot(x='importance', y='feature', data=top_features_model2, color='salmon')
    plt.title('Top 15 Features - Model 2 (With September Payment Data)')
    plt.tight_layout()
    
    # Save plot
    plt.savefig(os.path.join(output_dir, "feature_importance_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print top 5 features for each model
    print("\nTop 5 Features - Model 1 (Application-Time Features):")
    for _, row in top_features_model1.head(5).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")
    
    print("\nTop 5 Features - Model 2 (With September Payment Data):")
    for _, row in top_features_model2.head(5).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")
