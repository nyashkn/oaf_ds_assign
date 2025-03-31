"""
Data inspection and cleaning functions for scorecard modeling.
"""

import pandas as pd
import numpy as np
import os
import re
import json
from typing import Dict, List, Optional, Any

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
                "std": float(df[target_var].std())
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
        method: How to handle these columns ('exclude', 'convert_to_categorical', or 'parse_date')
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
        
        elif method == 'convert_to_categorical':
            if col in result_df.columns and pd.api.types.is_object_dtype(result_df[col]):
                # Convert to categorical and then to codes
                result_df[col] = result_df[col].astype('category').cat.codes
                handling_results[col] = "converted to categorical codes"
                
        elif method == 'parse_date':
            if col in result_df.columns:
                try:
                    # Try to parse as datetime and extract useful features
                    result_df[f"{col}_year"] = pd.to_datetime(result_df[col], errors='coerce').dt.year
                    result_df[f"{col}_month"] = pd.to_datetime(result_df[col], errors='coerce').dt.month
                    result_df = result_df.drop(columns=[col])
                    handling_results[col] = "parsed to year/month components"
                except:
                    # If parsing fails, exclude the column
                    result_df = result_df.drop(columns=[col])
                    handling_results[col] = "excluded (failed to parse)"
    
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
