"""
Feature scaling and transformation functions for regression modeling.

This module replaces the WOE binning approach from the original scorecard package
with scalers and encoders more suitable for regression models.
"""

import pandas as pd
import numpy as np
import os
import json
import pickle
from typing import Dict, List, Optional, Tuple, Any, Union
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from .constants import SCALING_METHODS

def detect_variable_types(df: pd.DataFrame, target_var: str) -> Dict[str, List[str]]:
    """
    Detect variable types for appropriate scaling/encoding.
    
    Args:
        df: DataFrame with features and target
        target_var: Target variable name
        
    Returns:
        Dictionary with variable types (numerical, categorical, date)
    """
    # Initialize variable type lists
    var_types = {
        'numerical': [],
        'categorical': [],
        'date': [],
        'exclude': [target_var]  # Exclude target variable from transformation
    }
    
    # Examine each column
    for col in df.columns:
        if col == target_var:
            continue
            
        # Detect date columns - they are usually handled separately
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            var_types['date'].append(col)
        # Numeric columns
        elif pd.api.types.is_numeric_dtype(df[col]):
            var_types['numerical'].append(col)
        # Consider categorical
        elif pd.api.types.is_categorical_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]):
            unique_count = df[col].nunique()
            # Only treat as categorical if limited unique values
            if unique_count <= 30:  # Limit cardinality to avoid too many one-hot columns
                var_types['categorical'].append(col)
            else:
                print(f"Warning: Column '{col}' has {unique_count} unique values, too many for categorical encoding. Excluding.")
                var_types['exclude'].append(col)
        else:
            # Exclude anything we don't know how to handle
            var_types['exclude'].append(col)
    
    # Print summary
    print("\n=== Variable Type Detection ===")
    print(f"Numerical variables: {len(var_types['numerical'])}")
    print(f"Categorical variables: {len(var_types['categorical'])}")
    print(f"Date variables: {len(var_types['date'])}")
    print(f"Excluded variables: {len(var_types['exclude'])}")
    
    return var_types

def create_scaling_pipeline(
    var_types: Dict[str, List[str]],
    scaling_method: str = 'robust',
    handle_missing: str = 'mean'
) -> ColumnTransformer:
    """
    Create a pipeline for scaling numerical features and encoding categorical features.
    
    Args:
        var_types: Dictionary with variable types
        scaling_method: Method for scaling numerical variables
        handle_missing: Strategy for handling missing values
        
    Returns:
        ColumnTransformer pipeline
    """
    # Set up missing value handling strategies
    if handle_missing not in ['mean', 'median', 'most_frequent', 'constant']:
        handle_missing = 'mean'
    
    # Get the scaler based on method
    if scaling_method not in SCALING_METHODS:
        print(f"Warning: Unknown scaling method '{scaling_method}'. Using RobustScaler.")
        scaling_method = 'robust'
    
    scaler_name = SCALING_METHODS[scaling_method]
    if scaler_name == 'RobustScaler':
        scaler = RobustScaler()
    elif scaler_name == 'StandardScaler':
        scaler = StandardScaler()
    elif scaler_name == 'MinMaxScaler':
        scaler = MinMaxScaler()
    elif scaler_name == 'MaxAbsScaler':
        scaler = MaxAbsScaler()
    elif scaler_name is None:
        scaler = None
    else:
        print(f"Warning: Unrecognized scaler '{scaler_name}'. Using RobustScaler.")
        scaler = RobustScaler()
    
    # Create transformers
    transformers = []
    
    # Numerical pipeline with imputation and scaling
    if var_types['numerical']:
        if scaler:
            num_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy=handle_missing)),
                ('scaler', scaler)
            ])
        else:
            num_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy=handle_missing))
            ])
        
        transformers.append(
            ('num', num_pipeline, var_types['numerical'])
        )
    
    # Categorical pipeline with imputation and one-hot encoding
    if var_types['categorical']:
        cat_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
        ])
        
        transformers.append(
            ('cat', cat_pipeline, var_types['categorical'])
        )
    
    # Create the column transformer
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder='drop'  # Drop columns not specified in transformers
    )
    
    return preprocessor

def scale_features(
    df: pd.DataFrame,
    target_var: str,
    output_dir: Optional[str] = None,
    scaling_method: str = 'robust',
    handle_missing: str = 'mean'
) -> Dict:
    """
    Scale features for regression modeling.
    
    Args:
        df: DataFrame with features and target
        target_var: Target variable name
        output_dir: Directory to save scaling results
        scaling_method: Method for scaling numerical variables
        handle_missing: Strategy for handling missing values
        
    Returns:
        Dictionary with scaling information
    """
    print("\n=== Feature Scaling ===")
    
    # Detect variable types
    var_types = detect_variable_types(df, target_var)
    
    # Create scaling pipeline
    scaling_pipeline = create_scaling_pipeline(
        var_types, 
        scaling_method=scaling_method,
        handle_missing=handle_missing
    )
    
    # Fit the pipeline
    X = df.drop(columns=[target_var])
    scaling_pipeline.fit(X)
    
    # Get transformed feature names
    feature_names = get_feature_names(scaling_pipeline, X.columns)
    
    print(f"\nScaling complete: {len(feature_names)} features after transformation")
    
    # Create scaling info dict
    scaling_info = {
        'pipeline': scaling_pipeline,
        'target_var': target_var,
        'var_types': var_types,
        'feature_names': feature_names,
        'scaling_method': scaling_method,
        'handle_missing': handle_missing
    }
    
    # Save results if output_dir provided
    if output_dir:
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save pipeline
        pipeline_path = os.path.join(output_dir, "scaling_pipeline.pkl")
        with open(pipeline_path, 'wb') as f:
            pickle.dump(scaling_pipeline, f)
        
        # Save var_types and other info (excluding the pipeline object)
        info = {k: v for k, v in scaling_info.items() if k != 'pipeline'}
        info_path = os.path.join(output_dir, "scaling_info.json")
        with open(info_path, 'w') as f:
            # Convert any non-serializable items (like numpy arrays) to lists
            json_info = {}
            for k, v in info.items():
                if isinstance(v, dict):
                    json_info[k] = {
                        k2: (v2.tolist() if hasattr(v2, 'tolist') else v2)
                        for k2, v2 in v.items()
                    }
                elif hasattr(v, 'tolist'):
                    json_info[k] = v.tolist()
                else:
                    json_info[k] = v
            
            json.dump(json_info, f, indent=2)
        
        print(f"\nScaling pipeline saved to {pipeline_path}")
        print(f"Scaling info saved to {info_path}")
    
    return scaling_info

def get_feature_names(ct, original_features):
    """
    Get feature names from a ColumnTransformer, handling sklearn 1.0+ compatibility.
    
    Args:
        ct: Fitted ColumnTransformer
        original_features: Original feature names
        
    Returns:
        List of transformed feature names
    """
    # Handle sklearn 1.0+ with get_feature_names_out
    if hasattr(ct, 'get_feature_names_out'):
        return ct.get_feature_names_out()
    
    # Pre-sklearn 1.0 approach
    feature_names = []
    
    # Handle column transformer
    if hasattr(ct, 'transformers_'):
        for name, trans, cols in ct.transformers_:
            if name == 'remainder':
                if cols == 'drop':
                    continue
                elif cols == 'passthrough':
                    feature_names.extend(original_features.tolist())
                else:
                    feature_names.extend([original_features[i] for i in cols])
            else:
                # Handle nested pipelines
                if hasattr(trans, 'steps'):
                    # Get the last step of the pipeline
                    last_step = trans.steps[-1][1]
                    
                    # Handle different transformer types
                    if hasattr(last_step, 'get_feature_names'):
                        transformer_feature_names = last_step.get_feature_names()
                    elif hasattr(last_step, 'classes_'):
                        transformer_feature_names = last_step.classes_
                    else:
                        transformer_feature_names = cols
                        
                    # Prefix feature names with transformer name
                    if isinstance(cols, list):
                        transformer_feature_names = [f"{name}_{fn}" for fn in transformer_feature_names]
                    else:
                        transformer_feature_names = [f"{name}_{cols}_{fn}" for fn in transformer_feature_names]
                    
                    feature_names.extend(transformer_feature_names)
                else:
                    # Handle simple transformers
                    if hasattr(trans, 'get_feature_names'):
                        transformer_feature_names = trans.get_feature_names()
                    else:
                        transformer_feature_names = cols
                        
                    feature_names.extend([f"{name}_{fn}" for fn in transformer_feature_names])
    
    return feature_names

def apply_scaling_transformation(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    scaling_info: Dict,
    output_dir: Optional[str] = None
) -> Dict[str, pd.DataFrame]:
    """
    Apply scaling transformation to training and testing data.
    
    Args:
        train_df: Training DataFrame
        test_df: Testing DataFrame
        scaling_info: Scaling information dictionary
        output_dir: Directory to save transformed data
        
    Returns:
        Dictionary with transformed train and test DataFrames
    """
    print("\n=== Applying Scaling Transformation ===")
    
    # Extract components from scaling_info
    scaling_pipeline = scaling_info['pipeline']
    target_var = scaling_info['target_var']
    feature_names = scaling_info['feature_names']
    
    # Apply transformation
    X_train = train_df.drop(columns=[target_var])
    y_train = train_df[target_var]
    X_test = test_df.drop(columns=[target_var])
    y_test = test_df[target_var]
    
    X_train_scaled = pd.DataFrame(
        scaling_pipeline.transform(X_train),
        index=X_train.index,
        columns=feature_names
    )
    
    X_test_scaled = pd.DataFrame(
        scaling_pipeline.transform(X_test),
        index=X_test.index,
        columns=feature_names
    )
    
    # Add target variables back
    train_scaled = X_train_scaled.copy()
    train_scaled[target_var] = y_train
    test_scaled = X_test_scaled.copy()
    test_scaled[target_var] = y_test
    
    print(f"Applied scaling to {len(feature_names)} features")
    print(f"Training set: {train_scaled.shape[0]} rows, {train_scaled.shape[1]} columns")
    print(f"Testing set: {test_scaled.shape[0]} rows, {test_scaled.shape[1]} columns")
    
    # Save results if output_dir provided
    if output_dir:
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save transformed DataFrames
        train_scaled_path = os.path.join(output_dir, "train_scaled.csv")
        test_scaled_path = os.path.join(output_dir, "test_scaled.csv")
        
        train_scaled.to_csv(train_scaled_path, index=False)
        test_scaled.to_csv(test_scaled_path, index=False)
        
        print(f"\nScaled training data saved to {train_scaled_path}")
        print(f"Scaled testing data saved to {test_scaled_path}")
    
    return {"train_scaled": train_scaled, "test_scaled": test_scaled}
