"""
Data inspection and cleaning functions for regression-based loan repayment prediction.

This module provides functions to analyze data quality, handle missing values, 
and deal with date-like columns.
"""

import pandas as pd
import numpy as np
import os
import re
import json
from typing import Dict, List, Optional, Any, Tuple

from .constants import DATE_PATTERNS

def inspect_data(
    df: pd.DataFrame, 
    target_var: Optional[str] = None,
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Inspect dataframe for potential issues and data quality problems.
    
    Args:
        df: DataFrame to inspect
        target_var: Target variable name (if available)
        output_path: Path to save inspection results
        
    Returns:
        Dictionary with inspection results
    """
    print("\n=== Data Inspection ===")
    
    # Basic dataframe info
    result = {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": {col: str(df[col].dtype) for col in df.columns},
        "missing_values": {col: int(df[col].isna().sum()) for col in df.columns},
        "missing_pct": {col: float(df[col].isna().mean()) for col in df.columns},
        "unique_values": {col: int(df[col].nunique()) for col in df.columns}
    }
    
    # Check for date-like strings
    date_like_columns = []
    sample_values = {}
    
    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]):
            # Sample non-null values
            sample = df[col].dropna().sample(min(5, df[col].count())).tolist()
            sample_values[col] = sample
            
            # Check for date pattern matches
            date_matches = False
            for val in sample:
                if isinstance(val, str):
                    for pattern in DATE_PATTERNS:
                        if re.match(pattern, val):
                            date_like_columns.append(col)
                            date_matches = True
                            break
                    if date_matches:
                        break
    
    result["date_like_columns"] = date_like_columns
    result["sample_values"] = sample_values
    
    # Target variable statistics if provided
    if target_var and target_var in df.columns:
        if pd.api.types.is_numeric_dtype(df[target_var]):
            result["target_stats"] = {
                "mean": float(df[target_var].mean()),
                "median": float(df[target_var].median()),
                "min": float(df[target_var].min()),
                "max": float(df[target_var].max()),
                "std": float(df[target_var].std()),
                "percentiles": {p: float(df[target_var].quantile(p/100)) for p in range(0, 101, 5)}
            }
        else:
            result["target_value_counts"] = df[target_var].value_counts().to_dict()
    
    # Print summary
    print(f"DataFrame shape: {result['shape']}")
    print(f"Missing values summary: {sum(result['missing_values'].values())} total missing values")
    
    if date_like_columns:
        print("\nWARNING: Potential date-formatted strings found in columns:")
        for col in date_like_columns:
            print(f"  - {col}: sample values = {sample_values[col][:2]}")
    
    # Save results if output path provided
    if output_path:
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Convert to serializable types for JSON
        serializable_result = {
            "shape": result["shape"],
            "columns": result["columns"],
            "dtypes": result["dtypes"],
            "missing_values": result["missing_values"],
            "missing_pct": result["missing_pct"],
            "unique_values": result["unique_values"],
            "date_like_columns": result["date_like_columns"]
        }
        
        if "target_stats" in result:
            serializable_result["target_stats"] = result["target_stats"]
        if "target_value_counts" in result:
            serializable_result["target_value_counts"] = {str(k): v for k, v in result["target_value_counts"].items()}
        
        # Convert sample values to strings for JSON serialization
        serializable_result["sample_values"] = {
            col: [str(v) if not isinstance(v, (int, float, bool, type(None))) else v 
                 for v in vals]
            for col, vals in result["sample_values"].items()
        }
        
        with open(output_path, 'w') as f:
            json.dump(serializable_result, f, indent=2)
        
        print(f"Inspection results saved to {output_path}")
    
    return result

def handle_date_like_columns(
    df: pd.DataFrame,
    date_like_columns: List[str],
    method: str = 'exclude',
    output_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Handle columns containing date-like strings.
    
    Args:
        df: DataFrame to process
        date_like_columns: List of columns with date-like strings
        method: How to handle these columns ('exclude', 'parse', 'categorical')
        output_path: Path to save handling results
        
    Returns:
        Processed DataFrame
    """
    if not date_like_columns:
        return df
    
    result_df = df.copy()
    handling_results = {}
    
    for col in date_like_columns:
        if method == 'exclude':
            if col in result_df.columns:
                result_df = result_df.drop(columns=[col])
                handling_results[col] = "excluded"
        
        elif method == 'categorical':
            if col in result_df.columns and pd.api.types.is_object_dtype(result_df[col]):
                # Convert to categorical and then to codes
                result_df[col] = result_df[col].astype('category').cat.codes
                handling_results[col] = "converted to categorical codes"
                
        elif method == 'parse':
            if col in result_df.columns:
                try:
                    # Try to parse as datetime and extract useful features
                    date_series = pd.to_datetime(result_df[col], errors='coerce')
                    
                    # Extract year, month, quarter features
                    result_df[f"{col}_year"] = date_series.dt.year
                    result_df[f"{col}_month"] = date_series.dt.month
                    result_df[f"{col}_quarter"] = date_series.dt.quarter
                    
                    # Extract day of week and week of year if it contains that level of detail
                    if not (date_series.dt.day == 1).all():
                        result_df[f"{col}_day"] = date_series.dt.day
                        result_df[f"{col}_dayofweek"] = date_series.dt.dayofweek
                        result_df[f"{col}_week"] = date_series.dt.isocalendar().week
                    
                    # Drop the original column
                    result_df = result_df.drop(columns=[col])
                    handling_results[col] = "parsed to year/month/etc components"
                except Exception as e:
                    # If parsing fails, exclude the column
                    result_df = result_df.drop(columns=[col])
                    handling_results[col] = f"excluded (failed to parse: {str(e)})"
    
    print("\n=== Date-like Column Handling ===")
    for col, action in handling_results.items():
        print(f"Column '{col}': {action}")
    
    # Save results if output path provided
    if output_path:
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(handling_results, f, indent=2)
        
        print(f"Date column handling results saved to {output_path}")
    
    return result_df

def handle_missing_values(
    df: pd.DataFrame,
    method: str = 'mean',
    region_col: Optional[str] = None,
    output_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Handle missing values in the DataFrame.
    
    Args:
        df: DataFrame to process
        method: How to handle missing values ('mean', 'median', 'mode', 'drop', 'region_mean')
        region_col: Column name containing region information (for region_mean method)
        output_path: Path to save handling results
        
    Returns:
        Processed DataFrame
    """
    if df.isna().sum().sum() == 0:
        print("No missing values to handle")
        return df
    
    result_df = df.copy()
    handling_results = {}
    
    for col in result_df.columns:
        if result_df[col].isna().any():
            missing_count = result_df[col].isna().sum()
            missing_pct = missing_count / len(result_df)
            
            # Special handling for distance variables using region means
            if method == 'region_mean' and region_col and region_col in result_df.columns:
                # Calculate mean by region
                region_means = result_df.groupby(region_col)[col].mean()
                
                # Apply region-specific means
                for region in result_df[region_col].unique():
                    mask = (result_df[region_col] == region) & result_df[col].isna()
                    if mask.any():
                        if pd.notna(region_means.get(region)):
                            result_df.loc[mask, col] = region_means[region]
                        else:
                            # If region mean is also NA, use overall mean
                            result_df.loc[mask, col] = result_df[col].mean()
                
                action = f"filled {missing_count} values ({missing_pct:.1%}) with means by {region_col}"
                
            else:
                if method == 'drop':
                    # Only drop if less than 30% missing to avoid losing too much data
                    if missing_pct < 0.3:
                        result_df = result_df.dropna(subset=[col])
                        action = f"dropped {missing_count} rows ({missing_pct:.1%})"
                    else:
                        # Too many missing values to drop, fall back to mean
                        result_df[col] = result_df[col].fillna(result_df[col].mean())
                        action = f"filled {missing_count} values ({missing_pct:.1%}) with mean (too many to drop)"
                
                elif method == 'median':
                    # For numeric columns use median, otherwise use mode
                    if pd.api.types.is_numeric_dtype(result_df[col]):
                        result_df[col] = result_df[col].fillna(result_df[col].median())
                        action = f"filled {missing_count} values ({missing_pct:.1%}) with median"
                    else:
                        result_df[col] = result_df[col].fillna(result_df[col].mode()[0])
                        action = f"filled {missing_count} values ({missing_pct:.1%}) with mode"
                        
                elif method == 'mode':
                    # Always use mode
                    result_df[col] = result_df[col].fillna(result_df[col].mode()[0])
                    action = f"filled {missing_count} values ({missing_pct:.1%}) with mode"
                    
                else:  # Default to mean
                    if pd.api.types.is_numeric_dtype(result_df[col]):
                        result_df[col] = result_df[col].fillna(result_df[col].mean())
                        action = f"filled {missing_count} values ({missing_pct:.1%}) with mean"
                    else:
                        result_df[col] = result_df[col].fillna(result_df[col].mode()[0])
                        action = f"filled {missing_count} values ({missing_pct:.1%}) with mode"
            
            handling_results[col] = action
    
    print("\n=== Missing Value Handling ===")
    for col, action in handling_results.items():
        print(f"Column '{col}': {action}")
    
    # Save results if output path provided
    if output_path:
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(handling_results, f, indent=2)
        
        print(f"Missing value handling results saved to {output_path}")
    
    return result_df
