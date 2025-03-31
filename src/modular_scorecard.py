"""
OAF Loan Performance Scorecard Modeling - Modular Approach

This module provides a modular approach to credit scorecard development, 
with each step saved separately to facilitate iterative modeling.
It handles:
- Data inspection and cleaning
- Target variable creation with customizable cutoffs
- Data partitioning (train/test split)
- Feature selection based on Information Value
- WOE binning and transformation
- Scorecard development
- Model evaluation

Each step saves its outputs to a structured directory, allowing 
inspection and iteration at any point in the process.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
import datetime
import json
from pathlib import Path
from typing import List, Dict, Tuple, Union, Optional, Any, Set
import scorecardpy as sc
from sklearn.linear_model import LogisticRegression

# Define variables that might leak information
EXCLUDE_VARS = [
    'client_id',
    'cumulative_amount_paid',
    'nominal_contract_value',
    'contract_start_date',
    # These variables directly relate to or derive from the target
    'sept_23_repayment_rate', 
    'nov_23_repayment_rate',
    'months_since_start',
    'days_since_start'
]

# Date patterns to identify date-formatted columns
DATE_PATTERNS = [
    r'^\d{4}-\d{2}$',       # YYYY-MM
    r'^\d{4}-\d{2}-\d{2}$',  # YYYY-MM-DD
    r'^\d{2}/\d{2}/\d{4}$',  # MM/DD/YYYY
    r'^\d{2}/\d{2}/\d{2}$',  # MM/DD/YY
    r'^\d{2}-\d{2}-\d{4}$',  # DD-MM-YYYY
    r'^\d{4}$'               # YYYY
]

def create_version_path(base_path: str = "data/processed/scorecard_modelling") -> str:
    """
    Create a new version directory for the current modeling iteration.
    
    Args:
        base_path: Base directory for scorecard modeling outputs
        
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

def create_binary_target(
    df: pd.DataFrame, 
    target_var: str = 'repayment_rate',
    cutoff: float = 0.8, 
    target_name: str = 'good_loan',
    output_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Create a binary target variable based on repayment rate.
    
    Args:
        df: DataFrame with features
        target_var: Column to use for creating binary target
        cutoff: Threshold for good/bad classification (default 0.8)
        target_name: Name for the new binary target variable
        output_path: Path to save the results
        
    Returns:
        DataFrame with added binary target variable
    """
    # Add binary target
    result_df = df.copy()
    result_df[target_name] = (result_df[target_var] >= cutoff).astype(int)
    
    # Print distribution
    good_count = result_df[target_name].sum()
    bad_count = len(result_df) - good_count
    good_pct = good_count / len(result_df) * 100
    
    print("\n=== Target Creation ===")
    print(f"Target Distribution (cutoff = {cutoff}):")
    print(f"  Good loans: {good_count:,} ({good_pct:.1f}%)")
    print(f"  Bad loans:  {bad_count:,} ({100-good_pct:.1f}%)")
    
    # Save results if output path provided
    if output_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save DataFrame
        result_df.to_csv(output_path, index=False)
        print(f"DataFrame with binary target saved to {output_path}")
    
    return result_df

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

def perform_woe_binning(
    df: pd.DataFrame,
    target_var: str,
    output_dir: Optional[str] = None,
    check_cate_num: bool = False  # Set to False to avoid interactive prompts
) -> Dict:
    """
    Perform WOE binning on all variables.
    
    Args:
        df: DataFrame with features and target
        target_var: Target variable name
        output_dir: Directory to save binning results
        check_cate_num: Whether to check categorical variables unique value count
        
    Returns:
        Dictionary with binning information
    """
    print("\n=== WOE Binning ===")
    
    # Check for date-like columns that might cause issues
    date_like_check = {}
    for col in df.columns:
        if col != target_var and pd.api.types.is_object_dtype(df[col]):
            sample = df[col].dropna().sample(min(5, df[col].count())).tolist()
            
            for val in sample:
                if isinstance(val, str):
                    for pattern in DATE_PATTERNS:
                        if re.match(pattern, val):
                            date_like_check[col] = sample
                            print(f"WARNING: Column '{col}' contains date-like strings: {sample[:3]}")
                            print(f"  This may cause issues in WOE binning.")
                            break
    
    # Perform WOE binning
    try:
        bins = sc.woebin(df, y=target_var, check_cate_num=check_cate_num)
        
        # Count bins for each variable
        bin_counts = {var: len(bin_df) for var, bin_df in bins.items()}
        
        print(f"WOE binning completed for {len(bins)} variables")
        print("\nBin counts for each variable:")
        for var, count in bin_counts.items():
            print(f"  {var}: {count} bins")
        
        # Save results if output_dir provided
        if output_dir:
            # Create directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Save bins as CSV (combine all variables)
            bins_df = pd.concat(bins.values())
            bins_path = os.path.join(output_dir, "woe_bins.csv")
            bins_df.to_csv(bins_path, index=False)
            
            # Save binning summary
            summary = {
                "variables": list(bins.keys()),
                "bin_counts": bin_counts,
                "date_like_warnings": date_like_check
            }
            
            summary_path = os.path.join(output_dir, "binning_summary.json")
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            
            print(f"\nWOE bins saved to {bins_path}")
            print(f"Binning summary saved to {summary_path}")
        
        return bins
        
    except Exception as e:
        print(f"ERROR in WOE binning: {str(e)}")
        # Try to identify the problematic variable
        for col in df.columns:
            if col != target_var:
                try:
                    print(f"Testing binning for '{col}'...")
                    sc.woebin(df[[col, target_var]], y=target_var, check_cate_num=False)
                    print(f"  Binning successful for '{col}'")
                except Exception as var_e:
                    print(f"  ERROR: Variable '{col}' failed with: {str(var_e)}")
                    print(f"  Sample values: {df[col].dropna().sample(5).tolist()}")
        
        raise e

def apply_woe_transformation(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    bins: Dict,
    target_var: str,
    output_dir: Optional[str] = None
) -> Dict[str, pd.DataFrame]:
    """
    Apply WOE transformation to training and testing data.
    
    Args:
        train_df: Training DataFrame
        test_df: Testing DataFrame
        bins: WOE binning information
        target_var: Target variable name
        output_dir: Directory to save transformed data
        
    Returns:
        Dictionary with transformed train and test DataFrames
    """
    print("\n=== WOE Transformation ===")
    
    # Apply WOE transformation
    train_woe = sc.woebin_ply(train_df, bins)
    test_woe = sc.woebin_ply(test_df, bins)
    
    print(f"WOE transformation applied to {train_woe.shape[1]-1} variables")
    print(f"Training set: {train_woe.shape[0]} rows, {train_woe.shape[1]} columns")
    print(f"Testing set: {test_woe.shape[0]} rows, {test_woe.shape[1]} columns")
    
    # Check for any variables with issues
    for col in train_woe.columns:
        if col != target_var and train_woe[col].isna().any():
            print(f"WARNING: Column '{col}' has {train_woe[col].isna().sum()} missing values after WOE transformation")
    
    # Save results if output_dir provided
    if output_dir:
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save transformed DataFrames
        train_woe_path = os.path.join(output_dir, "train_woe.csv")
        test_woe_path = os.path.join(output_dir, "test_woe.csv")
        
        train_woe.to_csv(train_woe_path, index=False)
        test_woe.to_csv(test_woe_path, index=False)
        
        print(f"\nWOE-transformed training data saved to {train_woe_path}")
        print(f"WOE-transformed testing data saved to {test_woe_path}")
    
    return {"train_woe": train_woe, "test_woe": test_woe}

def develop_scorecard_model(
    train_woe: pd.DataFrame,
    test_woe: pd.DataFrame,
    bins: Dict,
    target_var: str,
    output_dir: Optional[str] = None,
    penalty: str = 'l1',
    C: float = 0.9,
    solver: str = 'saga',
    random_state: int = 42
) -> Dict:
    """
    Develop a scorecard model using logistic regression.
    
    Args:
        train_woe: WOE-transformed training data
        test_woe: WOE-transformed testing data
        bins: WOE binning information
        target_var: Target variable name
        output_dir: Directory to save model results
        penalty: Regularization penalty type
        C: Inverse of regularization strength
        solver: Algorithm for optimization problem
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary with scorecard model and results
    """
    print("\n=== Scorecard Development ===")
    
    # Prepare data for modeling
    X_train = train_woe.drop(columns=[target_var])
    y_train = train_woe[target_var]
    X_test = test_woe.drop(columns=[target_var])
    y_test = test_woe[target_var]
    
    # Fit logistic regression model
    lr = LogisticRegression(penalty=penalty, C=C, solver=solver, random_state=random_state)
    lr.fit(X_train, y_train)
    
    # Get model coefficients
    coef_df = pd.DataFrame({
        'variable': X_train.columns,
        'coefficient': lr.coef_[0]
    }).sort_values('coefficient', ascending=False)
    
    print("\nModel coefficients (top 10):")
    for _, row in coef_df.head(10).iterrows():
        print(f"  {row['variable']}: {row['coefficient']:.4f}")
    
    # Calculate predicted probabilities
    train_pred = lr.predict_proba(X_train)[:, 1]
    test_pred = lr.predict_proba(X_test)[:, 1]
    
    # Create scorecard
    card = sc.scorecard(bins, lr, X_train.columns)
    
    # Apply scorecard to get scores
    train_score = sc.scorecard_ply(train_woe, card, print_step=0)
    test_score = sc.scorecard_ply(test_woe, card, print_step=0)
    
    print(f"\nScorecard created with {len(card)} components")
    print(f"Score ranges: Train [{train_score['score'].min()}, {train_score['score'].max()}], "
          f"Test [{test_score['score'].min()}, {test_score['score'].max()}]")
    
    # Save results if output_dir provided
    if output_dir:
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save scorecard as CSV (combine all components)
        card_df = pd.concat(card.values())
        card_path = os.path.join(output_dir, "scorecard.csv")
        card_df.to_csv(card_path, index=False)
        
        # Save scores
        train_score_path = os.path.join(output_dir, "train_scores.csv")
        test_score_path = os.path.join(output_dir, "test_scores.csv")
        
        train_score.to_csv(train_score_path, index=False)
        test_score.to_csv(test_score_path, index=False)
        
        # Save model coefficients
        coef_path = os.path.join(output_dir, "model_coefficients.csv")
        coef_df.to_csv(coef_path, index=False)
        
        print(f"\nScorecard saved to {card_path}")
        print(f"Training scores saved to {train_score_path}")
        print(f"Testing scores saved to {test_score_path}")
        print(f"Model coefficients saved to {coef_path}")
    
    # Results dictionary
    return {
        'card': card,
        'scorecard_df': pd.concat(card.values()),
        'model': lr,
        'coefficients': coef_df,
        'predictions': {
            'train': train_pred,
            'test': test_pred
        },
        'scores': {
            'train': train_score,
            'test': test_score
        }
    }

def evaluate_model_performance(
    train_actual: pd.Series,
    test_actual: pd.Series,
    train_pred: np.ndarray,
    test_pred: np.ndarray,
    output_dir: Optional[str] = None
) -> Dict:
    """
    Evaluate model performance using various metrics.
    
    Args:
        train_actual: Actual target values for training set
        test_actual: Actual target values for testing set
        train_pred: Predicted probabilities for training set
        test_pred: Predicted probabilities for testing set
        output_dir: Directory to save evaluation results
        
    Returns:
        Dictionary with performance metrics
    """
    print("\n=== Model Performance Evaluation ===")
    
    # Calculate performance metrics using scorecardpy
    train_perf = sc.perf_eva(train_actual, train_pred, title="Train", show_plot=False)
    test_perf = sc.perf_eva(test_actual, test_pred, title="Test", show_plot=False)
    
    # Print key metrics
    print("\nTraining Performance:")
    print(f"  KS: {train_perf['KS']:.4f}")
    print(f"  AUC: {train_perf['AUC']:.4f}")
    print(f"  Gini: {train_perf['Gini']:.4f}")
    
    print("\nTesting Performance:")
    print(f"  KS: {test_perf['KS']:.4f}")
    print(f"  AUC: {test_perf['AUC']:.4f}")
    print(f"  Gini: {test_perf['Gini']:.4f}")
    
    # Calculate PSI
    psi_result = sc.perf_psi(
        score={'train': pd.DataFrame({'score': train_pred}), 
               'test': pd.DataFrame({'score': test_pred})},
        label={'train': train_actual, 'test': test_actual}
    )
    
    print(f"\nPopulation Stability Index (PSI): {psi_result['psi']['PSI'].values[0]:.4f}")
    
    # Save results if output_dir provided
    if output_dir:
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save performance metrics
        metrics = {
            'train': {
                'KS': float(train_perf['KS']),
                'AUC': float(train_perf['AUC']),
                'Gini': float(train_perf['Gini'])
            },
            'test': {
                'KS': float(test_perf['KS']),
                'AUC': float(test_perf['AUC']),
                'Gini': float(test_perf['Gini'])
            },
            'psi': float(psi_result['psi']['PSI'].values[0])
        }
        
        metrics_path = os.path.join(output_dir, "performance_metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Save plots
        if 'pic' in train_perf:
            train_plot_path = os.path.join(output_dir, "train_performance.png")
            train_perf['pic'].savefig(train_plot_path)
            plt.close(train_perf['pic'])
        
        if 'pic' in test_perf:
            test_plot_path = os.path.join(output_dir, "test_performance.png")
            test_perf['pic'].savefig(test_plot_path)
            plt.close(test_perf['pic'])
        
        if 'pic' in psi_result:
            psi_plot_path = os.path.join(output_dir, "psi_plot.png")
            list(psi_result['pic'].values())[0].savefig(psi_plot_path)
            plt.close(list(psi_result['pic'].values())[0])
        
        print(f"\nPerformance metrics saved to {metrics_path}")
        print("Performance plots saved to output directory")
    
    # Return performance metrics
    return {
        'train_perf': train_perf,
        'test_perf': test_perf,
        'psi': psi_result
    }

def run_modular_scorecard(
    features_path: str,
    target_var: str,
    output_base_path: str = "data/processed/scorecard_modelling",
    cutoff: Optional[float] = None,
    sample_size: Optional[int] = None,
    exclude_vars: Optional[List[str]] = None,
    handle_date_columns: str = 'exclude',  # 'exclude', 'convert_to_categorical', or 'parse_date'
    iv_threshold: float = 0.02,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Run the full modular scorecard modeling workflow.
    
    Args:
        features_path: Path to features CSV file
        target_var: Name of target variable
        output_base_path: Base path for output files
        cutoff: Cutoff for good/bad classification (if None, determined automatically)
        sample_size: Optional sample size for testing
        exclude_vars: Additional variables to exclude from modeling
        handle_date_columns: How to handle date-like columns 
        iv_threshold: Minimum IV for variable selection
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary with paths to all outputs
    """
    print(f"[INFO] Starting modular scorecard modeling workflow...")
    print(f"Target variable: {target_var}")
    print(f"Features path: {features_path}")
    
    # Create version directory
    version_path = create_version_path(output_base_path)
    
    # Step 1: Load and inspect data
    print(f"\n[STEP 1] Loading and inspecting data...")
    df = pd.read_csv(features_path)
    
    if sample_size is not None and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=random_state)
        print(f"Using sample of {sample_size} loans for testing")
    
    # Inspect data
    inspection_path = os.path.join(version_path, "1_data_inspection.json")
    inspection_results = inspect_data(df, target_var, inspection_path)
    
    # Step 2: Handle date-like columns
    print(f"\n[STEP 2] Handling date-like columns...")
    date_handling_path = os.path.join(version_path, "2_date_handling.json")
    df = handle_date_like_columns(
        df, 
        inspection_results.get("date_like_columns", []),
        method=handle_date_columns,
        output_path=date_handling_path
    )
    
    # Step 3: Create binary target
    print(f"\n[STEP 3] Creating binary target...")
    binary_target_path = os.path.join(version_path, "3_binary_target.csv")
    if cutoff is None:
        # Find optimal cutoff based on data
        print("Determining optimal cutoff...")
        cutoff_result = find_optimal_cutoff(df, target_var=target_var)
        cutoff = cutoff_result['optimal_cutoff']
        print(f"Optimal cutoff determined: {cutoff:.2f}")
        
        # Save cutoff info
        cutoff_info_path = os.path.join(version_path, "3_cutoff_info.json")
        with open(cutoff_info_path, 'w') as f:
            json.dump({
                'optimal_cutoff': float(cutoff),
                'cutoff_stats': cutoff_result['cutoff_stats'].to_dict(orient='records')
            }, f, indent=2)
    
    df_with_target = create_binary_target(
        df, 
        target_var=target_var, 
        cutoff=cutoff,
        output_path=binary_target_path
    )
    
    # Step 4: Exclude leakage variables
    print(f"\n[STEP 4] Excluding leakage variables...")
    filtered_path = os.path.join(version_path, "4_filtered_data.csv")
    filtered_df, excluded = exclude_leakage_variables(
        df_with_target, 
        'good_loan', 
        additional_exclusions=exclude_vars,
        output_path=filtered_path
    )
    
    # Step 5: Partition data
    print(f"\n[STEP 5] Partitioning data...")
    partition_dir = os.path.join(version_path, "5_partition")
    partitioned_data = partition_data(
        filtered_df,
        'good_loan',
        train_ratio=0.7,
        random_state=random_state,
        output_dir=partition_dir
    )
    
    # Step 6: Variable selection based on IV
    print(f"\n[STEP 6] Selecting variables...")
    selection_dir = os.path.join(version_path, "6_variable_selection")
    selected_df, selection_results = select_variables(
        partitioned_data['train'],
        'good_loan',
        iv_threshold=iv_threshold,
        output_dir=selection_dir
    )
    
    # Adjust test set to have the same variables as selected_df
    selected_vars = [col for col in selected_df.columns if col != 'good_loan']
    test_selected = partitioned_data['test'][selected_vars + ['good_loan']]
    
    # Step 7: WOE binning
    print(f"\n[STEP 7] Performing WOE binning...")
    binning_dir = os.path.join(version_path, "7_woe_binning")
    try:
        bins = perform_woe_binning(
            selected_df,
            'good_loan',
            output_dir=binning_dir,
            check_cate_num=False
        )
        
        # Step 8: WOE transformation
        print(f"\n[STEP 8] Applying WOE transformation...")
        transformation_dir = os.path.join(version_path, "8_woe_transformation")
        woe_data = apply_woe_transformation(
            selected_df,
            test_selected,
            bins,
            'good_loan',
            output_dir=transformation_dir
        )
        
        # Step 9: Develop scorecard
        print(f"\n[STEP 9] Developing scorecard model...")
        scorecard_dir = os.path.join(version_path, "9_scorecard")
        scorecard_results = develop_scorecard_model(
            woe_data['train_woe'],
            woe_data['test_woe'],
            bins,
            'good_loan',
            output_dir=scorecard_dir,
            random_state=random_state
        )
        
        # Step 10: Evaluate model performance
        print(f"\n[STEP 10] Evaluating model performance...")
        evaluation_dir = os.path.join(version_path, "10_performance_evaluation")
        performance_results = evaluate_model_performance(
            woe_data['train_woe']['good_loan'],
            woe_data['test_woe']['good_loan'],
            scorecard_results['predictions']['train'],
            scorecard_results['predictions']['test'],
            output_dir=evaluation_dir
        )
        
        # Create summary report
        summary = {
            'version_path': version_path,
            'date': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'target_variable': target_var,
            'cutoff': float(cutoff),
            'sample_size': len(df) if sample_size is None else sample_size,
            'excluded_variables': excluded,
            'selected_variables': selected_vars,
            'performance': {
                'train_ks': float(performance_results['train_perf']['KS']),
                'test_ks': float(performance_results['test_perf']['KS']),
                'train_auc': float(performance_results['train_perf']['AUC']),
                'test_auc': float(performance_results['test_perf']['AUC']),
                'psi': float(performance_results['psi']['psi']['PSI'].values[0])
            }
        }
        
        summary_path = os.path.join(version_path, "summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n[SUCCESS] Scorecard modeling complete!")
        print(f"Results saved to {version_path}")
        print(f"Summary report saved to {summary_path}")
        
        # Return paths to outputs
        return {
            'version_path': version_path,
            'summary': summary,
            'data_inspection': inspection_path,
            'binary_target': binary_target_path,
            'filtered_data': filtered_path,
            'partition': partition_dir,
            'variable_selection': selection_dir,
            'woe_binning': binning_dir,
            'woe_transformation': transformation_dir,
            'scorecard': scorecard_dir,
            'performance_evaluation': evaluation_dir
        }
        
    except Exception as e:
        error_log_path = os.path.join(version_path, "error_log.txt")
        with open(error_log_path, 'w') as f:
            f.write(f"Error occurred during modeling: {str(e)}\n")
            import traceback
            f.write(traceback.format_exc())
        
        print(f"\n[ERROR] An error occurred during modeling: {str(e)}")
        print(f"Error details saved to {error_log_path}")
        raise e

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='OAF Loan Scorecard Modeling - Modular Approach')
    parser.add_argument('--features', type=str, default="data/processed/all_features.csv",
                       help='Path to features CSV file')
    parser.add_argument('--target', type=str, default='sept_23_repayment_rate',
                       help='Target variable name')
    parser.add_argument('--cutoff', type=float, default=None,
                       help='Cutoff for good/bad classification')
    parser.add_argument('--sample', type=int, default=None,
                       help='Sample size for testing')
    parser.add_argument('--output', type=str, default="data/processed/scorecard_modelling",
                       help='Base path for output files')
    parser.add_argument('--handle-dates', type=str, choices=['exclude', 'convert_to_categorical', 'parse_date'],
                       default='exclude', help='How to handle date-like columns')
    parser.add_argument('--iv-threshold', type=float, default=0.02,
                       help='Minimum IV for variable selection')
    
    args = parser.parse_args()
    
    # Run the modeling workflow
    run_modular_scorecard(
        features_path=args.features,
        target_var=args.target,
        output_base_path=args.output,
        cutoff=args.cutoff,
        sample_size=args.sample,
        handle_date_columns=args.handle_dates,
        iv_threshold=args.iv_threshold
    )
