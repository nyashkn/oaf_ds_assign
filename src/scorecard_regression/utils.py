"""
Utility functions for regression-based loan repayment rate prediction.

This module provides helper functions for common operations like calculating metrics,
formatting predictions, creating directories, etc.
"""

import os
import datetime
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Any, Union
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error, explained_variance_score,
    max_error, median_absolute_error
)

def create_version_path(base_path: str = "data/processed/regression_modelling") -> str:
    """
    Create a new version directory for the current modeling iteration.
    
    Args:
        base_path: Base directory for regression modeling outputs
        
    Returns:
        Path to the new version directory
    """
    # Ensure base path exists
    os.makedirs(base_path, exist_ok=True)
    
    # Find highest existing version
    existing_versions = [d for d in os.listdir(base_path) 
                        if os.path.isdir(os.path.join(base_path, d)) and d.startswith('v')]
    
    if not existing_versions:
        new_version = "v0"
    else:
        # Extract version numbers and find highest
        version_numbers = [int(v[1:]) for v in existing_versions if v[1:].isdigit()]
        if not version_numbers:
            new_version = "v0"
        else:
            new_version = f"v{max(version_numbers) + 1}"
    
    # Create new version directory
    version_path = os.path.join(base_path, new_version)
    os.makedirs(version_path, exist_ok=True)
    
    # Create subdirectories for organization
    subdirs = ['data', 'models', 'plots', 'metrics', 'reports']
    for subdir in subdirs:
        os.makedirs(os.path.join(version_path, subdir), exist_ok=True)
    
    print(f"Created version directory: {version_path}")
    return version_path

def calculate_metrics(
    y_true: Union[np.ndarray, pd.Series], 
    y_pred: np.ndarray,
    include_advanced: bool = False
) -> Dict[str, float]:
    """
    Calculate various regression metrics for model evaluation.
    
    Args:
        y_true: True target values
        y_pred: Predicted target values
        include_advanced: Whether to include additional metrics
        
    Returns:
        Dictionary of metric names and values
    """
    # Convert inputs to numpy arrays
    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)
    
    # Calculate basic metrics
    metrics = {
        'r2': float(r2_score(y_true_np, y_pred_np)),
        'rmse': float(np.sqrt(mean_squared_error(y_true_np, y_pred_np))),
        'mae': float(mean_absolute_error(y_true_np, y_pred_np)),
        'explained_variance': float(explained_variance_score(y_true_np, y_pred_np))
    }
    
    # Calculate advanced metrics if requested
    if include_advanced:
        try:
            # MAPE can fail if y_true contains zeros
            metrics['mape'] = float(mean_absolute_percentage_error(y_true_np, y_pred_np))
        except:
            metrics['mape'] = float('nan')
            
        metrics['max_error'] = float(max_error(y_true_np, y_pred_np))
        metrics['median_ae'] = float(median_absolute_error(y_true_np, y_pred_np))
        
        # Calculate mean bias error (MBE)
        metrics['mbe'] = float(np.mean(y_pred_np - y_true_np))
        
        # Calculate coefficient of variation of RMSE
        if np.mean(y_true_np) != 0:
            metrics['cv_rmse'] = float(metrics['rmse'] / np.mean(y_true_np))
        else:
            metrics['cv_rmse'] = float('nan')
    
    return metrics

def format_predictions(
    X: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    id_col: Optional[str] = None
) -> pd.DataFrame:
    """
    Format predictions into a DataFrame with features and errors.
    
    Args:
        X: Feature DataFrame
        y_true: True target values
        y_pred: Predicted target values
        id_col: Name of ID column if present
        
    Returns:
        DataFrame with predictions and errors
    """
    # Start with features
    result = X.copy()
    
    # Add actual and predicted values
    result['actual'] = y_true
    result['predicted'] = y_pred
    
    # Calculate errors
    result['error'] = result['predicted'] - result['actual']
    result['abs_error'] = np.abs(result['error'])
    
    # Calculate percentage error (handle zeros gracefully)
    with np.errstate(divide='ignore', invalid='ignore'):
        pct_error = np.where(
            np.abs(y_true) > 1e-10,  # avoid division by zero
            100.0 * (y_pred - y_true) / y_true,
            np.nan
        )
    result['pct_error'] = pct_error
    
    # Move ID column to front if provided
    if id_col and id_col in result.columns:
        cols = [id_col] + [c for c in result.columns if c != id_col]
        result = result[cols]
    
    return result

def prepare_model_summary(
    model: Any,
    model_type: str,
    metrics: Dict[str, Any],
    feature_importances: Optional[Dict[str, float]] = None,
    model_params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Prepare a comprehensive model summary dictionary.
    
    Args:
        model: Trained model
        model_type: Type of model
        metrics: Dictionary of evaluation metrics
        feature_importances: Dictionary of feature importances
        model_params: Dictionary of model parameters
        
    Returns:
        Dictionary with model summary information
    """
    # Create summary dictionary
    summary = {
        'model_type': model_type,
        'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'metrics': metrics
    }
    
    # Add feature importances if available
    if feature_importances:
        summary['feature_importances'] = feature_importances
    
    # Add model parameters if available
    if model_params:
        summary['model_params'] = {k: str(v) if not isinstance(v, (int, float, bool, str)) else v 
                                 for k, v in model_params.items()}
    elif hasattr(model, 'get_params'):
        # Extract from model if not provided
        try:
            params = model.get_params()
            summary['model_params'] = {k: str(v) if not isinstance(v, (int, float, bool, str)) else v 
                                     for k, v in params.items()}
        except:
            pass
    
    return summary

def save_model(
    model: Any,
    scaler: Any,
    model_path: str,
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """
    Save model, scaler, and metadata to disk.
    
    Args:
        model: Trained model
        scaler: Feature scaler (or None)
        model_path: Path to save the model
        metadata: Additional metadata to save
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Create model package
    model_package = {
        'model': model,
        'scaler': scaler,
        'metadata': metadata or {}
    }
    
    # Add timestamp to metadata
    model_package['metadata']['saved_at'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Save using joblib for efficient storage of scikit-learn models
    import joblib
    joblib.dump(model_package, model_path)
    
    print(f"Model saved to {model_path}")
    
    # Save metadata separately as JSON
    if metadata:
        metadata_path = model_path.replace('.pkl', '_metadata.json')
        with open(metadata_path, 'w') as f:
            # Convert metadata to JSON-serializable format
            serializable_metadata = {}
            for k, v in metadata.items():
                if isinstance(v, (dict, list)):
                    serializable_metadata[k] = v
                elif isinstance(v, (int, float, bool, str, type(None))):
                    serializable_metadata[k] = v
                else:
                    serializable_metadata[k] = str(v)
            
            json.dump(serializable_metadata, f, indent=2)
        
        print(f"Model metadata saved to {metadata_path}")

def load_model(model_path: str) -> Dict[str, Any]:
    """
    Load model, scaler, and metadata from disk.
    
    Args:
        model_path: Path to the saved model
        
    Returns:
        Dictionary with model, scaler, and metadata
    """
    # Load using joblib
    import joblib
    model_package = joblib.load(model_path)
    
    print(f"Model loaded from {model_path}")
    return model_package

def create_threshold_prediction(
    y_pred_continuous: np.ndarray,
    threshold: float = 0.8
) -> np.ndarray:
    """
    Convert continuous predictions to binary predictions using a threshold.
    
    Args:
        y_pred_continuous: Continuous predictions
        threshold: Threshold for binary classification
        
    Returns:
        Binary predictions
    """
    return (y_pred_continuous >= threshold).astype(int)

def normalize_feature_importances(importances: Dict[str, float]) -> Dict[str, float]:
    """
    Normalize feature importances to sum to 1.0.
    
    Args:
        importances: Dictionary of feature importances
        
    Returns:
        Dictionary with normalized importances
    """
    total = sum(importances.values())
    if total > 0:
        return {feature: importance/total for feature, importance in importances.items()}
    return importances
