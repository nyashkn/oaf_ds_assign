"""
Utility functions for regression-based loan repayment prediction.
"""

import os
import json
from typing import Dict, List, Optional, Any

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
    
    print(f"Created version directory: {version_path}")
    return version_path

def save_model_info(model: Any, model_type: str, model_params: Dict, 
                   feature_names: List[str], output_path: str) -> None:
    """
    Save model information to a JSON file.
    
    Args:
        model: Trained model
        model_type: Type of model
        model_params: Model parameters
        feature_names: Names of features used in the model
        output_path: Path to save model information
    """
    model_info = {
        'model_type': model_type,
        'model_params': model_params,
        'feature_count': len(feature_names),
        'features': feature_names
    }
    
    # Add model-specific information if available
    if hasattr(model, 'coef_'):
        try:
            # For linear models
            coefficients = model.coef_.tolist() if hasattr(model.coef_, 'tolist') else float(model.coef_)
            intercept = model.intercept_.tolist() if hasattr(model.intercept_, 'tolist') else float(model.intercept_)
            
            model_info['coefficients'] = dict(zip(feature_names, coefficients)) if isinstance(coefficients, list) else coefficients
            model_info['intercept'] = intercept
        except (TypeError, ValueError) as e:
            print(f"Warning: Could not convert coefficients to JSON serializable format: {e}")
    
    if hasattr(model, 'feature_importances_'):
        try:
            # For tree-based models
            importances = model.feature_importances_.tolist() if hasattr(model.feature_importances_, 'tolist') else list(model.feature_importances_)
            model_info['feature_importances'] = dict(zip(feature_names, importances))
        except (TypeError, ValueError) as e:
            print(f"Warning: Could not convert feature importances to JSON serializable format: {e}")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save model info
    with open(output_path, 'w') as f:
        json.dump(model_info, f, indent=2)
