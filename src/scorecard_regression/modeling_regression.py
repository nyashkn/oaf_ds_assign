"""
Regression modeling functions for loan repayment rate prediction.

This module provides functions for training, evaluating, and comparing regression
models for predicting loan repayment rates, as well as comparing regression and
classification approaches.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from typing import Dict, List, Optional, Tuple, Any, Union
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, KFold
import joblib

from .constants import DEFAULT_REGRESSION_PARAMS

def train_regression_model(
    X: pd.DataFrame,
    y: pd.Series,
    model_type: str = 'histgb',
    model_params: Optional[Dict[str, Any]] = None,
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Train a regression model for loan repayment rate prediction.
    
    Args:
        X: Feature DataFrame
        y: Target Series (repayment rates)
        model_type: Type of regression model ('linear', 'ridge', 'lasso', 'elasticnet',
                    'histgb', 'xgboost', 'lightgbm', 'catboost', 'rf')
        model_params: Dictionary of model parameters
        output_dir: Directory to save model and results
        
    Returns:
        Dictionary with model and training results
    """
    print(f"\n=== Training {model_type.upper()} Regression Model ===")
    
    # Get model parameters (default or custom)
    params = model_params or DEFAULT_REGRESSION_PARAMS.get(model_type, {})
    
    # Initialize model based on type
    if model_type == 'linear':
        model = LinearRegression(**params)
    elif model_type == 'ridge':
        model = Ridge(**params)
    elif model_type == 'lasso':
        model = Lasso(**params)
    elif model_type == 'elasticnet':
        model = ElasticNet(**params)
    elif model_type == 'histgb':
        model = GradientBoostingRegressor(**params)
    elif model_type == 'rf':
        model = RandomForestRegressor(**params)
    elif model_type == 'xgboost':
        try:
            from xgboost import XGBRegressor
            model = XGBRegressor(**params)
        except ImportError:
            print("XGBoost not installed. Please install with: pip install xgboost")
            raise
    elif model_type == 'lightgbm':
        try:
            from lightgbm import LGBMRegressor
            model = LGBMRegressor(**params)
        except ImportError:
            print("LightGBM not installed. Please install with: pip install lightgbm")
            raise
    elif model_type == 'catboost':
        try:
            from catboost import CatBoostRegressor
            model = CatBoostRegressor(**params)
        except ImportError:
            print("CatBoost not installed. Please install with: pip install catboost")
            raise
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Train the model
    model.fit(X, y)
    
    # Get training predictions
    train_preds = model.predict(X)
    
    # Calculate training metrics
    train_metrics = {
        'rmse': float(np.sqrt(mean_squared_error(y, train_preds))),
        'mae': float(mean_absolute_error(y, train_preds)),
        'r2': float(r2_score(y, train_preds)),
        'mean_abs_pct_error': float(np.mean(np.abs((y - train_preds) / y))) * 100
    }
    
    # Print training metrics
    print("Training Metrics:")
    print(f"  RMSE: {train_metrics['rmse']:.4f}")
    print(f"  MAE: {train_metrics['mae']:.4f}")
    print(f"  R²: {train_metrics['r2']:.4f}")
    print(f"  Mean Absolute Percentage Error: {train_metrics['mean_abs_pct_error']:.2f}%")
    
    # Extract feature importance if available
    feature_importance = None
    if hasattr(model, 'feature_importances_'):
        feature_importance = {
            feature: float(importance)
            for feature, importance in zip(X.columns, model.feature_importances_)
        }
        
        # Print top 10 feature importances
        print("\nTop 10 feature importances:")
        for feature, importance in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {feature}: {importance:.4f}")
    
    # Save model and results if output_dir provided
    if output_dir:
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(output_dir, f"{model_type}_regression_model.pkl")
        joblib.dump(model, model_path)
        
        # Save training metrics
        metrics_path = os.path.join(output_dir, "training_metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(train_metrics, f, indent=2)
        
        # Save feature importance if available
        if feature_importance:
            importance_path = os.path.join(output_dir, "feature_importance.json")
            with open(importance_path, 'w') as f:
                json.dump(feature_importance, f, indent=2)
            
            # Create and save feature importance plot
            plt.figure(figsize=(12, 8))
            
            # Sort feature importance
            sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            features = [x[0] for x in sorted_importance[:20]]  # Top 20 features
            importance = [x[1] for x in sorted_importance[:20]]
            
            # Create bar plot
            plt.barh(range(len(features)), importance, align='center')
            plt.yticks(range(len(features)), features)
            plt.xlabel('Importance')
            plt.title(f'Top 20 Feature Importances - {model_type.upper()}')
            plt.tight_layout()
            
            # Save plot
            plot_path = os.path.join(output_dir, "feature_importance_plot.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"\nModel saved to {model_path}")
        print(f"Training metrics saved to {metrics_path}")
        if feature_importance:
            print(f"Feature importance saved to {importance_path}")
            print(f"Feature importance plot saved to {plot_path}")
    
    # Return results
    return {
        'model': model,
        'model_type': model_type,
        'train_metrics': train_metrics,
        'feature_importance': feature_importance,
        'train_predictions': train_preds
    }


def evaluate_regression_model(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    output_dir: Optional[str] = None,
    threshold_cutoffs: Optional[List[float]] = None,
    return_plots: bool = False
) -> Dict[str, Any]:
    """
    Evaluate a trained regression model on test data.
    
    Args:
        model: Trained regression model
        X: Test feature DataFrame
        y: Test target Series
        output_dir: Directory to save evaluation results
        threshold_cutoffs: Optional list of threshold cutoffs for binary metrics
        return_plots: Whether to return plot objects
        
    Returns:
        Dictionary with evaluation metrics and results
    """
    print("\n=== Evaluating Regression Model ===")
    
    # Get predictions
    predictions = model.predict(X)
    
    # Calculate regression metrics
    metrics = {
        'rmse': float(np.sqrt(mean_squared_error(y, predictions))),
        'mae': float(mean_absolute_error(y, predictions)),
        'r2': float(r2_score(y, predictions)),
        'mean_abs_pct_error': float(np.mean(np.abs((y - predictions) / np.maximum(y, 1e-10)))) * 100
    }
    
    # Calculate additional metrics
    pct_diff = (predictions - y) / np.maximum(y, 1e-10)
    
    metrics.update({
        'mean_pct_diff': float(np.mean(pct_diff)) * 100,
        'median_pct_diff': float(np.median(pct_diff)) * 100,
        'std_pct_diff': float(np.std(pct_diff)) * 100,
        'over_prediction_rate': float(np.mean(predictions > y)),
        'under_prediction_rate': float(np.mean(predictions < y))
    })
    
    # Calculate classification metrics at different thresholds
    if threshold_cutoffs is None:
        threshold_cutoffs = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    
    threshold_metrics = {}
    for threshold in threshold_cutoffs:
        # Create binary labels
        y_true_bin = (y >= threshold).astype(int)
        y_pred_bin = (predictions >= threshold).astype(int)
        
        # Calculate classification metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        threshold_metrics[threshold] = {
            'accuracy': float(accuracy_score(y_true_bin, y_pred_bin)),
            'precision': float(precision_score(y_true_bin, y_pred_bin, zero_division=0)),
            'recall': float(recall_score(y_true_bin, y_pred_bin, zero_division=0)),
            'f1_score': float(f1_score(y_true_bin, y_pred_bin, zero_division=0))
        }
    
    # Print metrics
    print("Test Metrics:")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  MAE: {metrics['mae']:.4f}")
    print(f"  R²: {metrics['r2']:.4f}")
    print(f"  Mean Absolute Percentage Error: {metrics['mean_abs_pct_error']:.2f}%")
    print(f"  Over-prediction Rate: {metrics['over_prediction_rate']:.2%}")
    print(f"  Under-prediction Rate: {metrics['under_prediction_rate']:.2%}")
    
    print("\nClassification Metrics at Different Thresholds:")
    for threshold, metrics_dict in threshold_metrics.items():
        print(f"  Threshold = {threshold}:")
        print(f"    Accuracy: {metrics_dict['accuracy']:.4f}")
        print(f"    Precision: {metrics_dict['precision']:.4f}")
        print(f"    Recall: {metrics_dict['recall']:.4f}")
        print(f"    F1 Score: {metrics_dict['f1_score']:.4f}")
    
    # Create plots
    plots = {}
    
    # Actual vs Predicted plot
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    
    # Scatter plot
    ax1.scatter(y, predictions, alpha=0.5)
    
    # Perfect prediction line
    min_val = min(min(y), min(predictions))
    max_val = max(max(y), max(predictions))
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    # Add regression line
    from scipy import stats
    slope, intercept, r_value, p_value, std_err = stats.linregress(y, predictions)
    ax1.plot(y, intercept + slope * y, 'g-', label=f'Regression Line (r²={r_value**2:.3f})')
    
    ax1.set_xlabel('Actual Repayment Rate')
    ax1.set_ylabel('Predicted Repayment Rate')
    ax1.set_title('Actual vs Predicted Repayment Rate')
    ax1.grid(True)
    ax1.legend()
    
    plots['actual_vs_predicted'] = fig1
    
    # Prediction error histogram
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    
    # Calculate errors
    errors = predictions - y
    
    # Create histogram
    ax2.hist(errors, bins=30, edgecolor='black')
    ax2.axvline(x=0, color='r', linestyle='--')
    ax2.set_xlabel('Prediction Error')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Prediction Error Distribution')
    ax2.grid(True)
    
    # Add mean and std annotations
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    ax2.annotate(f'Mean Error: {mean_error:.3f}\nStd Dev: {std_error:.3f}',
                xy=(0.05, 0.95), xycoords='axes fraction',
                ha='left', va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plots['error_distribution'] = fig2
    
    # Save results if output_dir provided
    if output_dir:
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save metrics
        metrics_path = os.path.join(output_dir, "test_metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump({
                'regression_metrics': metrics,
                'threshold_metrics': threshold_metrics
            }, f, indent=2)
        
        # Save predictions
        preds_path = os.path.join(output_dir, "predictions.csv")
        pd.DataFrame({
            'actual': y,
            'predicted': predictions,
            'error': predictions - y,
            'pct_error': pct_diff * 100
        }).to_csv(preds_path, index=False)
        
        # Save plots
        for name, fig in plots.items():
            plot_path = os.path.join(output_dir, f"{name}.png")
            fig.savefig(plot_path, dpi=300, bbox_inches='tight')
        
        print(f"\nEvaluation metrics saved to {metrics_path}")
        print(f"Predictions saved to {preds_path}")
        print(f"Plots saved to {output_dir}")
    
    # Close plots if not returning them
    if not return_plots:
        for fig in plots.values():
            plt.close(fig)
    
    # Return results
    return {
        'test_metrics': metrics,
        'threshold_metrics': threshold_metrics,
        'predictions': predictions,
        'plots': plots if return_plots else None
    }


def cross_validate_regression(
    X: pd.DataFrame,
    y: pd.Series,
    model_type: str = 'histgb',
    model_params: Optional[Dict[str, Any]] = None,
    n_folds: int = 5,
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Perform cross-validation for regression model.
    
    Args:
        X: Feature DataFrame
        y: Target Series
        model_type: Type of regression model
        model_params: Dictionary of model parameters
        n_folds: Number of cross-validation folds
        output_dir: Directory to save cross-validation results
        
    Returns:
        Dictionary with cross-validation results
    """
    print(f"\n=== Cross-Validation ({n_folds} Folds) ===")
    
    # Get model parameters (default or custom)
    params = model_params or DEFAULT_REGRESSION_PARAMS.get(model_type, {})
    
    # Initialize model based on type
    if model_type == 'linear':
        model = LinearRegression(**params)
    elif model_type == 'ridge':
        model = Ridge(**params)
    elif model_type == 'lasso':
        model = Lasso(**params)
    elif model_type == 'elasticnet':
        model = ElasticNet(**params)
    elif model_type == 'histgb':
        model = GradientBoostingRegressor(**params)
    elif model_type == 'rf':
        model = RandomForestRegressor(**params)
    elif model_type == 'xgboost':
        from xgboost import XGBRegressor
        model = XGBRegressor(**params)
    elif model_type == 'lightgbm':
        from lightgbm import LGBMRegressor
        model = LGBMRegressor(**params)
    elif model_type == 'catboost':
        from catboost import CatBoostRegressor
        model = CatBoostRegressor(**params)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Set up KFold cross-validation
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # Cross-validate for different metrics
    cv_rmse = -cross_val_score(model, X, y, cv=kf, scoring='neg_root_mean_squared_error')
    cv_mae = -cross_val_score(model, X, y, cv=kf, scoring='neg_mean_absolute_error')
    cv_r2 = cross_val_score(model, X, y, cv=kf, scoring='r2')
    
    # Calculate average metrics
    avg_rmse = cv_rmse.mean()
    avg_mae = cv_mae.mean()
    avg_r2 = cv_r2.mean()
    
    # Calculate standard deviations
    std_rmse = cv_rmse.std()
    std_mae = cv_mae.std()
    std_r2 = cv_r2.std()
    
    # Print cross-validation results
    print("Cross-Validation Results:")
    print(f"  RMSE: {avg_rmse:.4f} ± {std_rmse:.4f}")
    print(f"  MAE: {avg_mae:.4f} ± {std_mae:.4f}")
    print(f"  R²: {avg_r2:.4f} ± {std_r2:.4f}")
    
    # Create fold-by-fold results
    fold_results = []
    for i, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Train model on this fold
        model.fit(X_train, y_train)
        
        # Predict on test fold
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        fold_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        fold_mae = mean_absolute_error(y_test, y_pred)
        fold_r2 = r2_score(y_test, y_pred)
        
        # Add to fold results
        fold_results.append({
            'fold': i + 1,
            'rmse': float(fold_rmse),
            'mae': float(fold_mae),
            'r2': float(fold_r2),
            'train_size': len(X_train),
            'test_size': len(X_test)
        })
    
    # Create final cross-validation results dictionary
    cv_results = {
        'model_type': model_type,
        'n_folds': n_folds,
        'avg_metrics': {
            'rmse': float(avg_rmse),
            'mae': float(avg_mae),
            'r2': float(avg_r2)
        },
        'std_metrics': {
            'rmse': float(std_rmse),
            'mae': float(std_mae),
            'r2': float(std_r2)
        },
        'fold_results': fold_results
    }
    
    # Save results if output_dir provided
    if output_dir:
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save cross-validation results
        results_path = os.path.join(output_dir, f"{model_type}_cv_results.json")
        with open(results_path, 'w') as f:
            json.dump(cv_results, f, indent=2)
        
        # Create and save cross-validation plot
        plt.figure(figsize=(12, 8))
        
        # Set up data for plot
        fold_nums = [r['fold'] for r in fold_results]
        rmse_vals = [r['rmse'] for r in fold_results]
        mae_vals = [r['mae'] for r in fold_results]
        r2_vals = [r['r2'] for r in fold_results]
        
        # Create subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
        
        # Plot RMSE
        ax1.bar(fold_nums, rmse_vals, color='skyblue')
        ax1.axhline(y=avg_rmse, color='r', linestyle='--', label=f'Mean: {avg_rmse:.3f}')
        ax1.set_ylabel('RMSE')
        ax1.set_title('RMSE by Fold')
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.legend()
        
        # Plot MAE
        ax2.bar(fold_nums, mae_vals, color='lightgreen')
        ax2.axhline(y=avg_mae, color='r', linestyle='--', label=f'Mean: {avg_mae:.3f}')
        ax2.set_ylabel('MAE')
        ax2.set_title('MAE by Fold')
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.legend()
        
        # Plot R²
        ax3.bar(fold_nums, r2_vals, color='salmon')
        ax3.axhline(y=avg_r2, color='r', linestyle='--', label=f'Mean: {avg_r2:.3f}')
        ax3.set_xlabel('Fold')
        ax3.set_ylabel('R²')
        ax3.set_title('R² by Fold')
        ax3.grid(True, linestyle='--', alpha=0.7)
        ax3.legend()
        
        # Set xticks to be integers
        ax3.set_xticks(fold_nums)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(output_dir, f"{model_type}_cv_plot.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\nCross-validation results saved to {results_path}")
        print(f"Cross-validation plot saved to {plot_path}")
    
    return cv_results


def compare_regression_vs_classification(
    y_true: np.ndarray,
    regression_predictions: np.ndarray,
    classification_probas: np.ndarray,
    loan_values: np.ndarray,
    thresholds: Optional[List[float]] = None,
    margin: float = 0.16,
    default_loss_rate: float = 1.0,
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Compare regression and classification approaches for loan repayment prediction.
    
    Args:
        y_true: True repayment rates
        regression_predictions: Predicted repayment rates from regression model
        classification_probas: Class probabilities from classification model
        loan_values: Loan amounts for profitability calculations
        thresholds: List of thresholds to evaluate (default is [0.70, 0.75, 0.80, 0.85, 0.90, 0.95])
        margin: Gross margin as a decimal (e.g., 0.16 = 16%)
        default_loss_rate: Loss rate on defaulted loans (1.0 = 100%)
        output_dir: Directory to save comparison results
        
    Returns:
        Dictionary with comparison results
    """
    print("\n=== Comparing Regression vs Classification ===")
    
    # Use default thresholds if none provided
    if thresholds is None:
        thresholds = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    
    # Import profitability functions here to avoid circular imports
    from .profitability.metrics import calculate_business_metrics
    
    # Compare at each threshold
    comparison_results = []
    
    for threshold in thresholds:
        # Calculate metrics for regression approach
        reg_metrics = calculate_business_metrics(
            y_true,
            regression_predictions,
            loan_values,
            threshold=threshold,
            margin=margin,
            default_loss_rate=default_loss_rate
        )
        
        # Calculate metrics for classification approach
        clf_metrics = calculate_business_metrics(
            y_true,
            classification_probas,
            loan_values,
            threshold=threshold,
            margin=margin,
            default_loss_rate=default_loss_rate
        )
        
        # Extract key metrics for comparison
        reg_profit = reg_metrics['profit_metrics']['actual_profit']
        clf_profit = clf_metrics['profit_metrics']['actual_profit']
        
        reg_roi = reg_metrics['profit_metrics']['roi']
        clf_roi = clf_metrics['profit_metrics']['roi']
        
        reg_approval_rate = reg_metrics['loan_metrics']['n_loans']['approval_rate']
        clf_approval_rate = clf_metrics['loan_metrics']['n_loans']['approval_rate']
        
        # Calculate difference
        profit_diff = reg_profit - clf_profit
        roi_diff = reg_roi - clf_roi
        approval_rate_diff = reg_approval_rate - clf_approval_rate
        
        # Calculate percentage difference
        profit_pct_diff = profit_diff / max(abs(clf_profit), 1e-10) * 100
        roi_pct_diff = roi_diff / max(abs(clf_roi), 1e-10) * 100
        approval_rate_pct_diff = approval_rate_diff / max(clf_approval_rate, 1e-10) * 100
        
        # Store comparison for this threshold
        comparison_results.append({
            'threshold': threshold,
            'regression': {
                'profit': float(reg_profit),
                'roi': float(reg_roi),
                'approval_rate': float(reg_approval_rate)
            },
            'classification': {
                'profit': float(clf_profit),
                'roi': float(clf_roi),
                'approval_rate': float(clf_approval_rate)
            },
            'difference': {
                'profit': float(profit_diff),
                'roi': float(roi_diff),
                'approval_rate': float(approval_rate_diff)
            },
            'percentage_difference': {
                'profit': float(profit_pct_diff),
                'roi': float(roi_pct_diff),
                'approval_rate': float(approval_rate_pct_diff)
            }
        })
    
    # Find the best threshold for each approach based on profit
    best_reg_threshold = max(comparison_results, key=lambda x: x['regression']['profit'])['threshold']
    best_clf_threshold = max(comparison_results, key=lambda x: x['classification']['profit'])['threshold']
    
    # Extract best metrics
    best_reg_result = next(result for result in comparison_results if result['threshold'] == best_reg_threshold)
    best_clf_result = next(result for result in comparison_results if result['threshold'] == best_clf_threshold)
    
    # Calculate performance difference at optimal thresholds
    best_profit_diff = best_reg_result['regression']['profit'] - best_clf_result['classification']['profit']
    best_roi_diff = best_reg_result['regression']['roi'] - best_clf_result['classification']['roi']
    
    best_profit_pct_diff = best_profit_diff / max(abs(best_clf_result['classification']['profit']), 1e-10) * 100
    best_roi_pct_diff = best_roi_diff / max(abs(best_clf_result['classification']['roi']), 1e-10) * 100
    
    # Print comparison results
    print("Comparison at Common Thresholds:")
    for result in comparison_results:
        print(f"  Threshold = {result['threshold']:.2f}:")
        print(f"    Regression Profit: {result['regression']['profit']:.2f}, "
              f"ROI: {result['regression']['roi']:.2%}, "
              f"Approval Rate: {result['regression']['approval_rate']:.2%}")
        print(f"    Classification Profit: {result['classification']['profit']:.2f}, "
              f"ROI: {result['classification']['roi']:.2%}, "
              f"Approval Rate: {result['classification']['approval_rate']:.2%}")
        print(f"    Profit Difference: {result['difference']['profit']:.2f} "
              f"({result['percentage_difference']['profit']:.1f}%)")
        print(f"    ROI Difference: {result['difference']['roi']:.4f} "
              f"({result['percentage_difference']['roi']:.1f}%)")
    
    print("\nOptimal Thresholds:")
    print(f"  Regression Best Threshold: {best_reg_threshold:.2f} "
          f"(Profit: {best_reg_result['regression']['profit']:.2f}, "
          f"ROI: {best_reg_result['regression']['roi']:.2%})")
    print(f"  Classification Best Threshold: {best_clf_threshold:.2f} "
          f"(Profit: {best_clf_result['classification']['profit']:.2f}, "
          f"ROI: {best_clf_result['classification']['roi']:.2%})")
    print(f"  Performance Difference at Optimal Thresholds:")
    print(f"    Profit: {best_profit_diff:.2f} ({best_profit_pct_diff:.1f}%)")
    print(f"    ROI: {best_roi_diff:.4f} ({best_roi_pct_diff:.1f}%)")
    
    # Create result dictionary
    result = {
        'thresholds': thresholds,
        'comparison_by_threshold': comparison_results,
        'optimal_thresholds': {
            'regression': float(best_reg_threshold),
            'classification': float(best_clf_threshold)
        },
        'performance_difference': {
            'profit': {
                'absolute': float(best_profit_diff),
                'relative': float(best_profit_pct_diff)
            },
            'roi': {
                'absolute': float(best_roi_diff),
                'relative': float(best_roi_pct_diff)
            }
        }
    }
    
    # Save results if output_dir provided
    if output_dir:
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save comparison results
        results_path = os.path.join(output_dir, "regression_vs_classification.json")
        with open(results_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        # Create and save comparison plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        
        # Extract data for plotting
        thresholds = [r['threshold'] for r in comparison_results]
        reg_profits = [r['regression']['profit'] for r in comparison_results]
        clf_profits = [r['classification']['profit'] for r in comparison_results]
        reg_rois = [r['regression']['roi'] * 100 for r in comparison_results]
        clf_rois = [r['classification']['roi'] * 100 for r in comparison_results]
        
        # Plot profits
        ax1.plot(thresholds, reg_profits, 'o-', color='#1A9641', label='Regression')
        ax1.plot(thresholds, clf_profits, 's-', color='#D73027', label='Classification')
        
        # Highlight optimal thresholds
        ax1.axvline(x=best_reg_threshold, color='#1A9641', linestyle='--', alpha=0.7,
                   label=f'Regression Best ({best_reg_threshold:.2f})')
        ax1.axvline(x=best_clf_threshold, color='#D73027', linestyle='--', alpha=0.7,
                   label=f'Classification Best ({best_clf_threshold:.2f})')
        
        ax1.set_xlabel('Threshold')
        ax1.set_ylabel('Profit')
        ax1.set_title('Profit Comparison')
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.legend()
        
        # Plot ROIs
        ax2.plot(thresholds, reg_rois, 'o-', color='#1A9641', label='Regression')
        ax2.plot(thresholds, clf_rois, 's-', color='#D73027', label='Classification')
        
        # Highlight optimal thresholds
        ax2.axvline(x=best_reg_threshold, color='#1A9641', linestyle='--', alpha=0.7)
        ax2.axvline(x=best_clf_threshold, color='#D73027', linestyle='--', alpha=0.7)
        
        ax2.set_xlabel('Threshold')
        ax2.set_ylabel('ROI (%)')
        ax2.set_title('ROI Comparison')
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.legend()
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(output_dir, "regression_vs_classification_plot.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\nComparison results saved to {results_path}")
        print(f"Comparison plot saved to {plot_path}")
    
    return result
