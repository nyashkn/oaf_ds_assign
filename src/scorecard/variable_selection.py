"""
Variable selection and data partitioning functions for scorecard modeling.
"""

import pandas as pd
import os
import json
import scorecardpy as sc
from typing import Dict, List, Optional, Tuple, Any, Set

from .constants import EXCLUDE_VARS

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
    
    # Save results if output path provided
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
            json.dump(exclusion_info, f, indent=2)
        
        print(f"Filtered DataFrame saved to {output_path}")
        print(f"Exclusion information saved to {exclusion_info_path}")
    
    return filtered_df, excluded

def partition_data(
    df: pd.DataFrame,
    target_var: str,
    train_ratio: float = 0.7,
    random_state: int = 42,
    output_dir: Optional[str] = None
) -> Dict[str, pd.DataFrame]:
    """
    Split data into training and testing sets.
    
    Args:
        df: DataFrame to split
        target_var: Target variable name for stratified split
        train_ratio: Ratio of training data
        random_state: Random seed for reproducibility
        output_dir: Directory to save train and test sets
        
    Returns:
        Dictionary with train and test DataFrames
    """
    print("\n=== Data Partitioning ===")
    
    # Use scorecardpy's split_df for consistent API
    split_result = sc.split_df(df, y=target_var, ratio=train_ratio, seed=random_state)
    
    train_df = split_result['train']
    test_df = split_result['test']
    
    print(f"Training set: {train_df.shape[0]} rows, {train_df.shape[1]} columns")
    print(f"Testing set: {test_df.shape[0]} rows, {test_df.shape[1]} columns")
    
    # Check target distribution in train and test
    train_target_dist = train_df[target_var].value_counts(normalize=True)
    test_target_dist = test_df[target_var].value_counts(normalize=True)
    
    print("\nTarget distribution:")
    print(f"Training: {train_target_dist.to_dict()}")
    print(f"Testing: {test_target_dist.to_dict()}")
    
    # Save results if output_dir provided
    if output_dir:
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save DataFrames
        train_path = os.path.join(output_dir, "train.csv")
        test_path = os.path.join(output_dir, "test.csv")
        
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        
        # Save summary information
        summary = {
            "train_shape": train_df.shape,
            "test_shape": test_df.shape,
            "train_target_distribution": train_target_dist.to_dict(),
            "test_target_distribution": test_target_dist.to_dict(),
            "train_ratio": train_ratio,
            "random_state": random_state
        }
        
        summary_path = os.path.join(output_dir, "partition_summary.json")
        with open(summary_path, 'w') as f:
            json.dump({k: str(v) if isinstance(v, pd.Series) else v for k, v in summary.items()}, f, indent=2)
        
        print(f"\nTrain data saved to {train_path}")
        print(f"Test data saved to {test_path}")
        print(f"Summary information saved to {summary_path}")
    
    return {"train": train_df, "test": test_df}

def select_variables(
    df: pd.DataFrame,
    target_var: str,
    iv_threshold: float = 0.02,
    output_dir: Optional[str] = None
) -> Tuple[pd.DataFrame, Dict]:
    """
    Select variables based on information value.
    
    Args:
        df: DataFrame with features and target
        target_var: Target variable name
        iv_threshold: Minimum IV threshold for variable selection
        output_dir: Directory to save variable selection results
        
    Returns:
        Tuple of (filtered DataFrame, selection results)
    """
    print("\n=== Variable Selection ===")
    print(f"Selecting variables with Information Value >= {iv_threshold}")
    
    # Apply variable filter using scorecardpy
    selected_df = sc.var_filter(df, y=target_var, iv_limit=iv_threshold)
    
    # Calculate information values for all variables
    iv_df = sc.iv(df, y=target_var, order=True)
    
    # Get the list of selected variables
    selected_vars = [col for col in selected_df.columns if col != target_var]
    
    # Identify variables that were filtered out
    filtered_out = [var for var in iv_df['variable'] if var not in selected_vars and var != target_var]
    
    print(f"Selected {len(selected_vars)} variables with IV >= {iv_threshold}")
    print(f"Filtered out {len(filtered_out)} variables with IV < {iv_threshold}")
    
    # Create detailed results
    selection_results = {
        "selected_variables": selected_vars,
        "filtered_out_variables": filtered_out,
        "iv_threshold": iv_threshold,
        "iv_values": iv_df.to_dict(orient='records')
    }
    
    # Sort and display top variables by IV
    top_iv = iv_df.head(10)
    print("\nTop variables by Information Value:")
    for _, row in top_iv.iterrows():
        print(f"  {row['variable']}: {row['info_value']:.4f}")
    
    # Save results if output_dir provided
    if output_dir:
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save filtered DataFrame
        selected_path = os.path.join(output_dir, "selected_features.csv")
        selected_df.to_csv(selected_path, index=False)
        
        # Save IV values
        iv_path = os.path.join(output_dir, "information_values.csv")
        iv_df.to_csv(iv_path, index=False)
        
        # Save selection summary
        summary_path = os.path.join(output_dir, "selection_summary.json")
        
        # Convert to serializable format
        serializable_results = {
            "selected_variables": selection_results["selected_variables"],
            "filtered_out_variables": selection_results["filtered_out_variables"],
            "iv_threshold": selection_results["iv_threshold"]
        }
        
        with open(summary_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"\nSelected features saved to {selected_path}")
        print(f"Information values saved to {iv_path}")
        print(f"Selection summary saved to {summary_path}")
    
    return selected_df, selection_results
