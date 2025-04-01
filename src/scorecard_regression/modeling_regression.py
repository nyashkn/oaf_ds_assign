"""
Regression modeling functions for loan repayment prediction.

This module replaces the classification approach in the original scorecard package
with regression models to predict repayment rates directly.
"""

import pandas as pd
import numpy as np
import os
import json
import pickle
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Any, Union

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
import warnings

# Import optional model packages with fallbacks
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    warnings.warn("XGBoost not available. XGBRegressor will not be available.")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    warnings.warn("LightGBM not available. LGBMRegressor will not be available.")

# Defer CatBoost import to runtime to avoid compatibility issues
CATBOOST_AVAILABLE = False

from .constants import DEFAULT_REGRESSION_PARAMS

def select_regression_model(
    model_type: str = 'histgb',
    model_params: Optional[Dict] = None,
    random_state: int = 42
) -> Any:
    """
    Select a regression model based on specified type and parameters.
    
    Args:
        model_type: Type of regression model
        model_params: Additional parameters for the model
        random_state: Random seed for reproducibility
        
    Returns:
        Configured regression model
    """
    # Get default parameters for the selected model type
    default_params = DEFAULT_REGRESSION_PARAMS.get(model_type, {})
    
    # Merge with provided parameters, if any
    params = default_params.copy()
    if model_params:
        params.update(model_params)
    
    # Select model based on type
    if model_type == 'histgb':
        # Add random_state to models that support it
        if 'random_state' not in params:
            params['random_state'] = random_state
        model = HistGradientBoostingRegressor(**params)
    elif model_type == 'linear':
        # LinearRegression doesn't use random_state
        model = LinearRegression(**params)
    elif model_type == 'ridge':
        # Ridge only supports random_state with solver='sag'
        if 'solver' in params and params['solver'] == 'sag':
            if 'random_state' not in params:
                params['random_state'] = random_state
        elif 'random_state' in params:
            # Remove random_state if solver is not 'sag'
            del params['random_state']
        model = Ridge(**params)
    elif model_type == 'lasso':
        # Lasso doesn't support random_state
        if 'random_state' in params:
            del params['random_state']
        model = Lasso(**params)
    elif model_type == 'elasticnet':
        # ElasticNet doesn't support random_state
        if 'random_state' in params:
            del params['random_state']
        model = ElasticNet(**params)
    elif model_type == 'randomforest':
        if 'random_state' not in params:
            params['random_state'] = random_state
        model = RandomForestRegressor(**params)
    elif model_type == 'xgboost' and XGBOOST_AVAILABLE:
        model = xgb.XGBRegressor(**params)
    elif model_type == 'lightgbm' and LIGHTGBM_AVAILABLE:
        model = lgb.LGBMRegressor(**params)
    elif model_type == 'catboost' and CATBOOST_AVAILABLE:
        # Import CatBoost only when needed
        try:
            import catboost as cb
            model = cb.CatBoostRegressor(**params)
        except ImportError:
            print("CatBoost not available. Using HistGradientBoostingRegressor instead.")
            model = HistGradientBoostingRegressor(**DEFAULT_REGRESSION_PARAMS['histgb'])
    else:
        available_models = [
            'histgb', 'linear', 'ridge', 'lasso', 'elasticnet', 'randomforest'
        ]
        if XGBOOST_AVAILABLE:
            available_models.append('xgboost')
        if LIGHTGBM_AVAILABLE:
            available_models.append('lightgbm')
        if CATBOOST_AVAILABLE:
            available_models.append('catboost')
            
        print(f"Warning: Unknown model type '{model_type}'. Using HistGradientBoostingRegressor.")
        print(f"Available models: {', '.join(available_models)}")
        model = HistGradientBoostingRegressor(**DEFAULT_REGRESSION_PARAMS['histgb'])
    
    return model

def develop_regression_model(
    train_scaled: pd.DataFrame,
    test_scaled: pd.DataFrame,
    target_var: str,
    output_dir: Optional[str] = None,
    model_type: str = 'histgb',
    model_params: Optional[Dict] = None,
    cv_folds: int = 5,
    perform_cv: bool = True,
    random_state: int = 42
) -> Dict:
    """
    Develop a regression model using the specified model type.
    
    Args:
        train_scaled: Scaled training data
        test_scaled: Scaled testing data
        target_var: Target variable name
        output_dir: Directory to save model results
        model_type: Type of regression model
        model_params: Additional parameters for the model
        cv_folds: Number of cross-validation folds
        perform_cv: Whether to perform cross-validation
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary with model and results
    """
    print("\n=== Regression Model Development ===")
    print(f"Using model: {model_type}")
    
    # Prepare data for modeling
    X_train = train_scaled.drop(columns=[target_var])
    y_train = train_scaled[target_var]
    X_test = test_scaled.drop(columns=[target_var])
    y_test = test_scaled[target_var]
    
    # Select model with specified configuration
    model = select_regression_model(model_type, model_params, random_state)
    
    # Cross-validation if requested
    if perform_cv:
        print("\nPerforming cross-validation...")
        cv_scores = cross_val_score(
            model, X_train, y_train, 
            cv=cv_folds, 
            scoring='neg_mean_squared_error'
        )
        
        cv_rmse_scores = np.sqrt(-cv_scores)
        print(f"Cross-validation RMSE: {cv_rmse_scores.mean():.4f} (±{cv_rmse_scores.std():.4f})")
    
    # Fit model on full training set
    print("\nTraining final model...")
    model.fit(X_train, y_train)
    
    # Make predictions
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    # Calculate metrics
    train_metrics = calculate_regression_metrics(y_train, train_pred)
    test_metrics = calculate_regression_metrics(y_test, test_pred)
    
    # Print key metrics
    print("\nTraining set metrics:")
    print(f"  RMSE: {train_metrics['rmse']:.4f}")
    print(f"  MAE: {train_metrics['mae']:.4f}")
    print(f"  R²: {train_metrics['r2']:.4f}")
    
    print("\nTesting set metrics:")
    print(f"  RMSE: {test_metrics['rmse']:.4f}")
    print(f"  MAE: {test_metrics['mae']:.4f}")
    print(f"  R²: {test_metrics['r2']:.4f}")
    
    # Get feature importances if available
    feature_importances = extract_feature_importances(model, X_train.columns)
    
    if feature_importances is not None:
        print("\nTop 10 features by importance:")
        for feature, importance in sorted(
            feature_importances.items(), 
            key=lambda x: abs(x[1]), 
            reverse=True
        )[:10]:
            print(f"  {feature}: {importance:.4f}")
    
    # Create results dictionary
    results = {
        'model': model,
        'model_type': model_type,
        'model_params': model_params,
        'target_var': target_var,
        'feature_names': list(X_train.columns),
        'feature_importances': feature_importances,
        'predictions': {
            'train': train_pred,
            'test': test_pred
        },
        'metrics': {
            'train': train_metrics,
            'test': test_metrics
        }
    }
    
    if perform_cv:
        results['cv_scores'] = {
            'rmse': cv_rmse_scores,
            'mean_rmse': cv_rmse_scores.mean(),
            'std_rmse': cv_rmse_scores.std()
        }
    
    # Save results if output_dir provided
    if output_dir:
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(output_dir, "regression_model.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Save predictions
        train_pred_df = pd.DataFrame({
            'actual': y_train,
            'predicted': train_pred
        })
        test_pred_df = pd.DataFrame({
            'actual': y_test,
            'predicted': test_pred
        })
        
        train_pred_path = os.path.join(output_dir, "train_predictions.csv")
        test_pred_path = os.path.join(output_dir, "test_predictions.csv")
        
        train_pred_df.to_csv(train_pred_path, index=False)
        test_pred_df.to_csv(test_pred_path, index=False)
        
        # Save metrics
        metrics = {
            'train': train_metrics,
            'test': test_metrics
        }
        if perform_cv:
            metrics['cv'] = {
                'mean_rmse': float(cv_rmse_scores.mean()),
                'std_rmse': float(cv_rmse_scores.std()),
                'all_rmse': [float(score) for score in cv_rmse_scores]
            }
        
        metrics_path = os.path.join(output_dir, "regression_metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Save feature importances if available
        if feature_importances is not None:
            importances_df = pd.DataFrame({
                'feature': list(feature_importances.keys()),
                'importance': list(feature_importances.values())
            }).sort_values('importance', ascending=False)
            
            importances_path = os.path.join(output_dir, "feature_importances.csv")
            importances_df.to_csv(importances_path, index=False)
        
        # Create and save actual vs predicted plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Training set
        ax1.scatter(y_train, train_pred, alpha=0.5)
        ax1.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--')
        ax1.set_xlabel('Actual')
        ax1.set_ylabel('Predicted')
        ax1.set_title('Training Set: Actual vs Predicted')
        
        # Testing set
        ax2.scatter(y_test, test_pred, alpha=0.5)
        ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        ax2.set_xlabel('Actual')
        ax2.set_ylabel('Predicted')
        ax2.set_title('Testing Set: Actual vs Predicted')
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, "actual_vs_predicted.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()
        
        print(f"\nModel saved to {model_path}")
        print(f"Predictions saved to {os.path.dirname(train_pred_path)}")
        print(f"Metrics saved to {metrics_path}")
        print(f"Plot saved to {plot_path}")
        if feature_importances is not None:
            print(f"Feature importances saved to {importances_path}")
    
    return results

def calculate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate regression performance metrics.
    
    Args:
        y_true: Actual target values
        y_pred: Predicted target values
        
    Returns:
        Dictionary with calculated metrics
    """
    metrics = {
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred),
        'mse': mean_squared_error(y_true, y_pred),
        'ev': explained_variance_score(y_true, y_pred)
    }
    
    # Calculate MAPE only if no true values are zero
    if np.all(y_true != 0):
        metrics['mape'] = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    else:
        metrics['mape'] = np.nan
        
    return metrics

def extract_feature_importances(model, feature_names) -> Optional[Dict[str, float]]:
    """
    Extract feature importances from the model if available.
    
    Args:
        model: Trained model
        feature_names: Feature names
        
    Returns:
        Dictionary with feature importances or None if not available
    """
    # For models with .feature_importances_ attribute
    if hasattr(model, 'feature_importances_'):
        return dict(zip(feature_names, model.feature_importances_))
    
    # For linear models with .coef_ attribute
    elif hasattr(model, 'coef_'):
        if model.coef_.ndim > 1:
            return dict(zip(feature_names, model.coef_[0]))
        else:
            return dict(zip(feature_names, model.coef_))
    
    # For XGBoost
    elif hasattr(model, 'get_booster') and hasattr(model.get_booster(), 'get_score'):
        try:
            importance_dict = model.get_booster().get_score(importance_type='gain')
            # XGBoost feature names might be different, so map them to original names
            return {feature_names[int(key.replace('f', ''))]: value 
                    for key, value in importance_dict.items()}
        except:
            return None
    
    # For LightGBM
    elif hasattr(model, 'booster_') and hasattr(model.booster_, 'feature_importance'):
        try:
            return dict(zip(feature_names, model.booster_.feature_importance(importance_type='gain')))
        except:
            return None
    
    return None

def predict_repayment_rate(
    model,
    df: pd.DataFrame,
    scaling_info: Dict,
    target_var: Optional[str] = None
) -> pd.DataFrame:
    """
    Predict repayment rate using the trained regression model.
    
    Args:
        model: Trained regression model
        df: DataFrame with features
        scaling_info: Scaling information dictionary
        target_var: Target variable name (if included in df)
        
    Returns:
        DataFrame with predictions
    """
    # Extract components from scaling_info
    scaling_pipeline = scaling_info['pipeline']
    feature_names = scaling_info['feature_names']
    target_column = scaling_info['target_var']
    
    # If target_var is not provided, use the one from scaling_info
    if target_var is None:
        target_var = target_column
    
    # Check if target column is in the dataframe
    has_target = target_var in df.columns
    
    # Apply scaling transformation
    if has_target:
        X = df.drop(columns=[target_var])
        y_true = df[target_var]
    else:
        X = df
    
    # Transform features using the scaling pipeline
    X_scaled = pd.DataFrame(
        scaling_pipeline.transform(X),
        index=X.index,
        columns=feature_names
    )
    
    # Make predictions
    y_pred = model.predict(X_scaled)
    
    # Create results dataframe
    if has_target:
        results = pd.DataFrame({
            'actual': y_true,
            'predicted': y_pred
        })
        
        # Calculate error metrics
        metrics = calculate_regression_metrics(y_true, y_pred)
        print("\nPrediction metrics:")
        print(f"  RMSE: {metrics['rmse']:.4f}")
        print(f"  MAE: {metrics['mae']:.4f}")
        print(f"  R²: {metrics['r2']:.4f}")
    else:
        results = pd.DataFrame({
            'predicted': y_pred
        })
    
    return results
