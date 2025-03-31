"""
WOE binning and transformation functions for scorecard modeling.
"""

import pandas as pd
import os
import re
import json
import scorecardpy as sc
from typing import Dict, List, Optional, Tuple, Any
import matplotlib.pyplot as plt

from .constants import DATE_PATTERNS

def perform_woe_binning(
    df: pd.DataFrame,
    target_var: str,
    output_dir: Optional[str] = None,
    check_cate_num: bool = False,  # Set to False to avoid interactive prompts
    save_plots: bool = True,
    handle_missing: str = 'mean',  # 'mean', 'median', 'mode', 'drop', or 'region_mean'
    region_col: Optional[str] = None  # Column name containing region information
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
    
    # Handle missing values
    df_clean = df.copy()
    missing_handling = {}
    
    for col in df_clean.columns:
        if col != target_var and df_clean[col].isna().any():
            missing_count = df_clean[col].isna().sum()
            missing_pct = missing_count / len(df_clean)
            
            # Special handling for distance variables using region means
            if handle_missing == 'region_mean' and region_col and region_col in df_clean.columns and 'distance' in col.lower():
                region_means = df_clean.groupby(region_col)[col].mean()
                for region in df_clean[region_col].unique():
                    mask = (df_clean[region_col] == region) & df_clean[col].isna()
                    if mask.any():
                        df_clean.loc[mask, col] = region_means[region]
                method = 'region mean'
                missing_handling[col] = f"filled {missing_count} values ({missing_pct:.1%}) with {method} by region"
            else:
                if handle_missing == 'drop' and missing_pct < 0.3:  # Only drop if less than 30% missing
                    df_clean = df_clean.dropna(subset=[col])
                    missing_handling[col] = f"dropped {missing_count} rows ({missing_pct:.1%})"
                else:
                    if pd.api.types.is_numeric_dtype(df_clean[col]):
                        if handle_missing == 'median':
                            fill_value = df_clean[col].median()
                            method = 'median'
                        elif handle_missing == 'mode':
                            fill_value = df_clean[col].mode()[0]
                            method = 'mode'
                        else:  # default to mean
                            fill_value = df_clean[col].mean()
                            method = 'mean'
                    else:
                        fill_value = df_clean[col].mode()[0]
                        method = 'mode'
                    
                    df_clean[col] = df_clean[col].fillna(fill_value)
                    missing_handling[col] = f"filled {missing_count} values ({missing_pct:.1%}) with {method}"
    
    # Perform WOE binning
    try:
        bins = sc.woebin(df_clean, y=target_var, check_cate_num=check_cate_num)
        
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
                "date_like_warnings": date_like_check,
                "missing_value_handling": missing_handling
            }
            
            summary_path = os.path.join(output_dir, "binning_summary.json")
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            
            # Save WOE plots if requested
            if save_plots:
                plots_dir = os.path.join(output_dir, "woe_plots")
                os.makedirs(plots_dir, exist_ok=True)
                
                for var in bins.keys():
                    plt.figure(figsize=(10, 6))
                    sc.woebin_plot(bins[var])
                    plt.title(f"WOE Binning Plot - {var}")
                    plt.tight_layout()
                    plt.savefig(os.path.join(plots_dir, f"{var}_woe_plot.png"))
                    plt.close()
            
            print(f"\nWOE bins saved to {bins_path}")
            print(f"Binning summary saved to {summary_path}")
            if save_plots:
                print(f"WOE plots saved to {plots_dir}")
        
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
