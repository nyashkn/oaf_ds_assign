"""
Variable selection functions for regression-based loan repayment rate prediction.

This module provides functions for feature selection, multicollinearity checking,
and data partitioning for regression modeling of loan repayment rates.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Union
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

from .constants import EXCLUDE_VARS, VARIABLE_SELECTION


def exclude_leakage_variables(
    df: pd.DataFrame, 
    target_var: str, 
    additional_exclusions: Optional[List[str]] = None,
    ignore_warnings: bool = False,
    output_path: Optional[str] = None
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Exclude variables that might leak information about the target.
    
    Args:
        df: DataFrame with features
        target_var: Target variable name (will be retained)
        additional_exclusions: Additional variables to exclude
        ignore_warnings: If True, suppress warnings about missing columns
        output_path: Path to save the filtered DataFrame
        
    Returns:
        Tuple containing (filtered DataFrame, list of excluded variables)
    """
    # Start with standard leakage variables
    exclude_vars = EXCLUDE_VARS.copy()
    
    # Add additional exclusions if provided
    if additional_exclusions:
        exclude_vars.extend(additional_exclusions)
    
    # Get columns that actually exist in the dataframe
    existing_cols = set(df.columns)
    exclude_vars = [col for col in exclude_vars if col in existing_cols or not ignore_warnings]
    
    # Get variables that will be excluded
    excluded = [col for col in exclude_vars if col in existing_cols]
    
    # Print warning for non-existent columns
    if not ignore_warnings:
        missing = [col for col in exclude_vars if col not in existing_cols]
        if missing:
            print(f"Warning: The following exclusion variables don't exist in the dataframe: {missing}")
    
    # Get columns to keep (all except excluded, but always keep target)
    keep_cols = [col for col in df.columns if col not in excluded or col == target_var]
    
    # Create filtered dataframe
    filtered_df = df[keep_cols]
    
    print("\n=== Variable Filtering ===")
    print(f"Excluded {len(excluded)} variables as potential leakage: {', '.join(excluded)}")
    print(f"Retained {len(keep_cols)} variables")
    
    # Save results if output_path provided
    if output_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save DataFrame
        filtered_df.to_csv(output_path, index=False)
        
        # Also save exclusion info
        exclusion_info = {
            "excluded_variables": excluded,
            "missing_variables": missing if 'missing' in locals() else [],
            "retained_variables": keep_cols
        }
        
        exclusion_info_path = os.path.join(os.path.dirname(output_path), "exclusion_info.json")
        with open(exclusion_info_path, 'w') as f:
            import json
            json.dump(exclusion_info, f, indent=2)
        
        print(f"Filtered DataFrame saved to {output_path}")
        print(f"Exclusion information saved to {exclusion_info_path}")
    
    return filtered_df, excluded


def partition_data(
    df: pd.DataFrame,
    target_var: str,
    train_ratio: float = 0.7,
    validation_ratio: float = 0.15,
    include_validation: bool = False,
    stratify: bool = False,
    random_state: int = 42,
    output_dir: Optional[str] = None
) -> Dict[str, pd.DataFrame]:
    """
    Split data into training, validation, and testing sets.
    
    Args:
        df: DataFrame to split
        target_var: Target variable name 
        train_ratio: Ratio of training data
        validation_ratio: Ratio of validation data (if include_validation=True)
        include_validation: Whether to include a validation set
        stratify: Whether to use stratified sampling based on binned target
        random_state: Random seed for reproducibility
        output_dir: Directory to save train, validation, and test sets
        
    Returns:
        Dictionary with train, validation (if requested), and test DataFrames
    """
    print("\n=== Data Partitioning ===")
    
    from sklearn.model_selection import train_test_split
    
    # Initialize result dictionary
    result = {}
    
    # Create stratification target if needed
    if stratify:
        # Bin the continuous target for stratification
        bins = np.linspace(df[target_var].min(), df[target_var].max(), num=5)
        strat_target = pd.cut(df[target_var], bins=bins, labels=False)
        print("Using stratified sampling with binned target variable")
    else:
        strat_target = None
    
    # Split into train and remaining data
    train_df, remaining_df = train_test_split(
        df, 
        train_size=train_ratio,
        stratify=strat_target, 
        random_state=random_state
    )
    
    result['train'] = train_df
    
    # If validation set is needed, split the remaining data
    if include_validation:
        # Calculate test_ratio from remaining data
        test_ratio = 1.0 - (train_ratio + validation_ratio)
        # Adjust for the new total
        validation_ratio_adjusted = validation_ratio / (1 - train_ratio)
        
        # Create stratification for remaining data if needed
        if stratify:
            remaining_strat = pd.cut(remaining_df[target_var], bins=bins, labels=False)
        else:
            remaining_strat = None
        
        validation_df, test_df = train_test_split(
            remaining_df,
            train_size=validation_ratio_adjusted,
            stratify=remaining_strat,
            random_state=random_state
        )
        
        result['validation'] = validation_df
        result['test'] = test_df
        
        print(f"Training set: {train_df.shape[0]} rows, {train_df.shape[1]} columns")
        print(f"Validation set: {validation_df.shape[0]} rows, {validation_df.shape[1]} columns")
        print(f"Testing set: {test_df.shape[0]} rows, {test_df.shape[1]} columns")
        
        # Calculate ratios for verification
        actual_train_ratio = len(train_df) / len(df)
        actual_validation_ratio = len(validation_df) / len(df)
        actual_test_ratio = len(test_df) / len(df)
        
        print(f"Actual ratios - Train: {actual_train_ratio:.2f}, "
              f"Validation: {actual_validation_ratio:.2f}, "
              f"Test: {actual_test_ratio:.2f}")
    else:
        # Just split into train and test
        result['test'] = remaining_df
        
        print(f"Training set: {train_df.shape[0]} rows, {train_df.shape[1]} columns")
        print(f"Testing set: {remaining_df.shape[0]} rows, {remaining_df.shape[1]} columns")
        
        # Calculate ratios for verification
        actual_train_ratio = len(train_df) / len(df)
        actual_test_ratio = len(remaining_df) / len(df)
        
        print(f"Actual ratios - Train: {actual_train_ratio:.2f}, "
              f"Test: {actual_test_ratio:.2f}")
    
    # Check target distribution in the sets
    print("\nTarget distribution:")
    for key, dataset in result.items():
        print(f"{key.capitalize()}: mean={dataset[target_var].mean():.4f}, "
              f"min={dataset[target_var].min():.4f}, "
              f"max={dataset[target_var].max():.4f}, "
              f"std={dataset[target_var].std():.4f}")
    
    # Save results if output_dir provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save DataFrames
        for key, dataset in result.items():
            path = os.path.join(output_dir, f"{key}.csv")
            dataset.to_csv(path, index=False)
            print(f"{key.capitalize()} data saved to {path}")
        
        # Save summary information
        summary = {
            "dataset_sizes": {k: v.shape for k, v in result.items()},
            "target_distribution": {
                k: {
                    "mean": float(v[target_var].mean()),
                    "min": float(v[target_var].min()),
                    "max": float(v[target_var].max()),
                    "std": float(v[target_var].std())
                } for k, v in result.items()
            },
            "train_ratio": train_ratio,
            "validation_ratio": validation_ratio if include_validation else None,
            "random_state": random_state,
            "stratified": stratify
        }
        
        summary_path = os.path.join(output_dir, "partition_summary.json")
        with open(summary_path, 'w') as f:
            import json
            json.dump(summary, f, indent=2)
        
        print(f"Summary information saved to {summary_path}")
    
    return result


def check_multicollinearity(
    X: pd.DataFrame,
    threshold: float = VARIABLE_SELECTION['vif_threshold'],
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Check for multicollinearity between features using VIF (Variance Inflation Factor).
    
    Args:
        X: Feature DataFrame
        threshold: Threshold for VIF (10 is commonly used)
        output_path: Path to save VIF results
        
    Returns:
        Dictionary with VIF results and high correlation features
    """
    print("\n=== Multicollinearity Check ===")
    
    # Drop non-numeric columns
    X_numeric = X.select_dtypes(include=['number'])
    
    # For VIF calculation, we need to handle missing values by imputation
    X_numeric_filled = X_numeric.fillna(X_numeric.mean())
    
    # Add a constant column for statsmodels
    X_with_const = sm.add_constant(X_numeric_filled)
    
    # Use a try-except block to handle potential errors in VIF calculation
    try:
        # Calculate VIF for each feature
        vif_data = pd.DataFrame()
        vif_data['feature'] = X_numeric_filled.columns
        
        # Calculate VIF values safely
        vif_values = []
        for i in range(X_with_const.shape[1] - 1):  # Skip the constant term (index 0)
            vif_values.append(variance_inflation_factor(X_with_const.values, i + 1))
        
        # Handle any infinity values that might occur due to perfect collinearity
        vif_values = [float('inf') if np.isinf(v) else v for v in vif_values]
        vif_values = [1000.0 if v > 1000 else v for v in vif_values]  # Cap extremely high values
        
        vif_data['VIF'] = vif_values
        
        # Sort by VIF
        vif_data = vif_data.sort_values('VIF', ascending=False).reset_index(drop=True)
        
    except Exception as e:
        print(f"WARNING: VIF calculation encountered an error: {str(e)}")
        print("Proceeding with correlation analysis only...")
        # Create a placeholder DataFrame with default values
        vif_data = pd.DataFrame({'feature': X_numeric_filled.columns, 'VIF': [1.0] * len(X_numeric_filled.columns)})
    
    # Sort by VIF
    vif_data = vif_data.sort_values('VIF', ascending=False).reset_index(drop=True)
    
    # Identify high VIF features
    high_vif_features = vif_data[vif_data['VIF'] > threshold]
    
    # Calculate correlation matrix
    corr_matrix = X_numeric_filled.corr().abs()
    
    # Extract highly correlated pairs
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            col1 = corr_matrix.columns[i]
            col2 = corr_matrix.columns[j]
            corr = corr_matrix.iloc[i, j]
            if corr >= VARIABLE_SELECTION['correlation_threshold']:
                high_corr_pairs.append({
                    'feature1': col1,
                    'feature2': col2,
                    'correlation': float(corr)
                })
    
    # Sort by correlation
    high_corr_pairs = sorted(high_corr_pairs, key=lambda x: x['correlation'], reverse=True)
    
    # Print VIF results
    print("\nVariance Inflation Factor (VIF):")
    print(f"Features with VIF > {threshold} (potential multicollinearity issue):")
    if len(high_vif_features) > 0:
        for _, row in high_vif_features.iterrows():
            print(f"  {row['feature']}: VIF = {row['VIF']:.2f}")
    else:
        print("  None")
    
    # Print correlation results
    print("\nHighly correlated feature pairs:")
    print(f"Features with correlation >= {VARIABLE_SELECTION['correlation_threshold']}:")
    if len(high_corr_pairs) > 0:
        for pair in high_corr_pairs[:10]:  # Show top 10
            print(f"  {pair['feature1']} - {pair['feature2']}: {pair['correlation']:.4f}")
        if len(high_corr_pairs) > 10:
            print(f"  ... and {len(high_corr_pairs) - 10} more pairs")
    else:
        print("  None")
    
    # Create result dictionary
    result = {
        'vif_data': vif_data.to_dict('records'),
        'high_vif_features': high_vif_features.to_dict('records'),
        'high_corr_pairs': high_corr_pairs,
        'threshold': threshold,
        'correlation_threshold': VARIABLE_SELECTION['correlation_threshold']
    }
    
    # Save results if output_path provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            import json
            json.dump(result, f, indent=2)
        
        # Create and save correlation heatmap
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        heatmap = sns.heatmap(
            corr_matrix, 
            mask=mask, 
            vmin=0, 
            vmax=1, 
            annot=False, 
            cmap='coolwarm', 
            linewidths=.5
        )
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        
        heatmap_path = os.path.join(os.path.dirname(output_path), 'correlation_heatmap.png')
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Multicollinearity results saved to {output_path}")
        print(f"Correlation heatmap saved to {heatmap_path}")
    
    return result


def select_variables_correlation(
    X: pd.DataFrame,
    target_var: pd.Series,
    correlation_measure: str = 'pearson',
    output_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Select variables based on correlation with the target.
    
    Args:
        X: Feature DataFrame
        target_var: Target variable Series
        correlation_measure: Correlation measure to use ('pearson', 'spearman', 'kendall')
        output_path: Path to save correlation results
        
    Returns:
        DataFrame with feature correlations
    """
    print(f"\n=== Variable Selection by {correlation_measure.capitalize()} Correlation ===")
    
    # Drop non-numeric columns
    X_numeric = X.select_dtypes(include=['number'])
    
    # Add target to calculate correlations
    data_with_target = X_numeric.copy()
    data_with_target['target'] = target_var
    
    # Calculate correlations
    if correlation_measure == 'pearson':
        correlations = data_with_target.corr(method='pearson')['target']
    elif correlation_measure == 'spearman':
        correlations = data_with_target.corr(method='spearman')['target']
    elif correlation_measure == 'kendall':
        correlations = data_with_target.corr(method='kendall')['target']
    else:
        raise ValueError(f"Unsupported correlation measure: {correlation_measure}")
    
    # Remove target from correlations
    correlations = correlations.drop('target')
    
    # Create a DataFrame with absolute correlations
    corr_df = pd.DataFrame({
        'feature': correlations.index,
        'correlation': correlations.values,
        'abs_correlation': np.abs(correlations.values)
    }).sort_values('abs_correlation', ascending=False).reset_index(drop=True)
    
    # Print top correlations
    print("\nTop 10 features by absolute correlation with target:")
    for _, row in corr_df.head(10).iterrows():
        print(f"  {row['feature']}: {row['correlation']:.4f}")
    
    # Save results if output_path provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save DataFrame
        corr_df.to_csv(output_path, index=False)
        
        # Create and save correlation bar plot
        plt.figure(figsize=(12, 8))
        top_features = corr_df.head(20)['feature'].values
        top_corrs = corr_df.head(20)['correlation'].values
        
        plt.barh(
            y=top_features,
            width=top_corrs,
            color=[plt.cm.RdYlGn(0.8 * (x + 1) / 2) for x in top_corrs]
        )
        plt.xlabel('Correlation with Target')
        plt.title(f'Top 20 Features by {correlation_measure.capitalize()} Correlation with Target')
        plt.tight_layout()
        
        plot_path = os.path.join(os.path.dirname(output_path), f'{correlation_measure}_correlation_plot.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Correlation results saved to {output_path}")
        print(f"Correlation plot saved to {plot_path}")
    
    return corr_df


def select_variables_importance(
    X: pd.DataFrame,
    target_var: pd.Series,
    model_type: str = 'rf',
    n_estimators: int = 100,
    max_depth: int = 5,
    random_state: int = 42,
    output_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Select variables based on feature importance from a tree-based model.
    
    Args:
        X: Feature DataFrame
        target_var: Target variable Series
        model_type: Type of model to use ('rf' for Random Forest, 'gb' for Gradient Boosting)
        n_estimators: Number of estimators for the model
        max_depth: Maximum depth of the trees
        random_state: Random seed for reproducibility
        output_path: Path to save importance results
        
    Returns:
        DataFrame with feature importances
    """
    print(f"\n=== Variable Selection by {'Random Forest' if model_type == 'rf' else 'Gradient Boosting'} Importance ===")
    
    # Drop non-numeric columns and handle missing values
    X_numeric = X.select_dtypes(include=['number'])
    X_filled = X_numeric.fillna(X_numeric.mean())
    
    # Create and fit model
    if model_type == 'rf':
        model = RandomForestRegressor(
            n_estimators=n_estimators, 
            max_depth=max_depth, 
            random_state=random_state,
            n_jobs=-1
        )
    elif model_type == 'gb':
        model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    model.fit(X_filled, target_var)
    
    # Get feature importances
    importances = model.feature_importances_
    
    # Create a DataFrame with feature importances
    imp_df = pd.DataFrame({
        'feature': X_filled.columns,
        'importance': importances
    }).sort_values('importance', ascending=False).reset_index(drop=True)
    
    # Print top importances
    print("\nTop 10 features by importance:")
    for _, row in imp_df.head(10).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")
    
    # Save results if output_path provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save DataFrame
        imp_df.to_csv(output_path, index=False)
        
        # Create and save importance bar plot
        plt.figure(figsize=(12, 8))
        top_features = imp_df.head(20)['feature'].values
        top_imps = imp_df.head(20)['importance'].values
        
        plt.barh(y=top_features, width=top_imps, color='darkgreen')
        plt.xlabel('Feature Importance')
        plt.title(f"Top 20 Features by {'Random Forest' if model_type == 'rf' else 'Gradient Boosting'} Importance")
        plt.tight_layout()
        
        plot_path = os.path.join(os.path.dirname(output_path), f'{model_type}_importance_plot.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Importance results saved to {output_path}")
        print(f"Importance plot saved to {plot_path}")
    
    return imp_df


def select_significant_variables(
    X: pd.DataFrame,
    y: pd.Series,
    correlation_threshold: float = 0.1,
    importance_threshold: float = 0.01,
    p_value_threshold: float = 0.05,
    model_type: str = 'rf',
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Perform comprehensive variable selection using multiple methods and return significant variables.
    
    Args:
        X: Feature DataFrame
        y: Target variable
        correlation_threshold: Minimum absolute correlation to retain a feature
        importance_threshold: Minimum feature importance to retain a feature
        p_value_threshold: Maximum p-value for statistical significance
        model_type: Type of model for importance calculation ('rf' or 'gb')
        output_dir: Directory to save selection results
        
    Returns:
        Dictionary with selected variables and selection metrics
    """
    print("\n=== Comprehensive Variable Selection ===")
    
    # Create output directory if provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Correlation with target
    corr_df = select_variables_correlation(
        X, y, 
        correlation_measure='pearson',
        output_path=os.path.join(output_dir, 'correlation_results.csv') if output_dir else None
    )
    
    # Step 2: Feature importance
    imp_df = select_variables_importance(
        X, y,
        model_type=model_type,
        output_path=os.path.join(output_dir, 'importance_results.csv') if output_dir else None
    )
    
    # Step 3: Statistical significance (using f_regression)
    X_numeric = X.select_dtypes(include=['number'])
    X_filled = X_numeric.fillna(X_numeric.mean())
    
    f_stats, p_values = f_regression(X_filled, y)
    stat_df = pd.DataFrame({
        'feature': X_filled.columns,
        'f_statistic': f_stats,
        'p_value': p_values
    }).sort_values('p_value').reset_index(drop=True)
    
    if output_dir:
        stat_df.to_csv(os.path.join(output_dir, 'statistical_significance.csv'), index=False)
    
    # Step 4: Combine all criteria
    # 4.1: Identify significant features by correlation
    sig_by_corr = set(corr_df[corr_df['abs_correlation'] >= correlation_threshold]['feature'])
    
    # 4.2: Identify significant features by importance
    sig_by_imp = set(imp_df[imp_df['importance'] >= importance_threshold]['feature'])
    
    # 4.3: Identify significant features by p-value
    sig_by_pval = set(stat_df[stat_df['p_value'] <= p_value_threshold]['feature'])
    
    # 4.4: Get the intersection of all methods
    selected_by_all = sig_by_corr.intersection(sig_by_imp).intersection(sig_by_pval)
    
    # 4.5: Get features selected by at least two methods
    selected_by_two = (sig_by_corr.intersection(sig_by_imp)
                       .union(sig_by_corr.intersection(sig_by_pval))
                       .union(sig_by_imp.intersection(sig_by_pval)))
    
    # 4.6: Get features selected by at least one method
    selected_by_one = sig_by_corr.union(sig_by_imp).union(sig_by_pval)
    
    # Create a DataFrame with combined results
    all_features = list(selected_by_one)
    combined_df = pd.DataFrame({
        'feature': all_features,
        'selected_by_correlation': [feature in sig_by_corr for feature in all_features],
        'selected_by_importance': [feature in sig_by_imp for feature in all_features],
        'selected_by_pvalue': [feature in sig_by_pval for feature in all_features],
        'selected_by_all_methods': [feature in selected_by_all for feature in all_features],
        'selected_by_two_methods': [feature in selected_by_two for feature in all_features],
        'num_methods': [
            sum([
                feature in sig_by_corr,
                feature in sig_by_imp,
                feature in sig_by_pval
            ]) for feature in all_features
        ]
    })
    
    # Add additional metrics
    for feature in combined_df['feature']:
        corr_row = corr_df[corr_df['feature'] == feature]
        imp_row = imp_df[imp_df['feature'] == feature]
        stat_row = stat_df[stat_df['feature'] == feature]
        
        if not corr_row.empty:
            combined_df.loc[combined_df['feature'] == feature, 'correlation'] = corr_row['correlation'].values[0]
            combined_df.loc[combined_df['feature'] == feature, 'abs_correlation'] = corr_row['abs_correlation'].values[0]
        
        if not imp_row.empty:
            combined_df.loc[combined_df['feature'] == feature, 'importance'] = imp_row['importance'].values[0]
        
        if not stat_row.empty:
            combined_df.loc[combined_df['feature'] == feature, 'p_value'] = stat_row['p_value'].values[0]
            combined_df.loc[combined_df['feature'] == feature, 'f_statistic'] = stat_row['f_statistic'].values[0]
    
    # Sort by number of methods, then by importance
    combined_df = combined_df.sort_values(['num_methods', 'importance'], ascending=[False, False]).reset_index(drop=True)
    
    # Fill NaN values with 0 for numeric columns
    combined_df = combined_df.fillna(0)
    
    # Prepare final selected variables (e.g., by at least two methods)
    final_selected = list(selected_by_two)
    
    print("\nVariable Selection Summary:")
    print(f"  Total features: {len(X.columns)}")
    print(f"  Features with significant correlation: {len(sig_by_corr)}")
    print(f"  Features with significant importance: {len(sig_by_imp)}")
    print(f"  Features with significant p-value: {len(sig_by_pval)}")
    print(f"  Features selected by all three methods: {len(selected_by_all)}")
    print(f"  Features selected by at least two methods: {len(selected_by_two)}")
    
    print("\nTop 10 selected features:")
    for feature in combined_df.head(10)['feature']:
        row = combined_df[combined_df['feature'] == feature].iloc[0]
        print(f"  {feature}: selected by {int(row['num_methods'])} methods, "
              f"importance: {row.get('importance', 0):.4f}, "
              f"correlation: {row.get('correlation', 0):.4f}, "
              f"p-value: {row.get('p_value', 1):.4f}")
    
    # Save combined results if output_dir provided
    if output_dir:
        combined_df.to_csv(os.path.join(output_dir, 'combined_selection_results.csv'), index=False)
        
        # Save list of selected features
        with open(os.path.join(output_dir, 'selected_features.json'), 'w') as f:
            import json
            json.dump({
                'selected_by_all_methods': list(selected_by_all),
                'selected_by_two_methods': list(selected_by_two),
                'selected_by_one_method': list(selected_by_one),
                'thresholds': {
                    'correlation_threshold': correlation_threshold,
                    'importance_threshold': importance_threshold,
                    'p_value_threshold': p_value_threshold
                }
            }, f, indent=2)
        
        print(f"\nVariable selection results saved to {output_dir}")
    
    # Create result dictionary
    result = {
        'selected_features': final_selected,
        'selected_by_all': list(selected_by_all),
        'selected_by_two': list(selected_by_two),
        'selected_by_one': list(selected_by_one),
        'metrics': {
            'correlation_threshold': correlation_threshold,
            'importance_threshold': importance_threshold,
            'p_value_threshold': p_value_threshold
        },
        'correlation_results': corr_df.to_dict('records'),
        'importance_results': imp_df.to_dict('records'),
        'statistical_results': stat_df.to_dict('records'),
        'combined_results': combined_df.to_dict('records')
    }
    
    return result
