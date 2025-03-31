"""
OAF Loan Performance Scorecard Modeling

This module provides functions for developing a credit scorecard
based on the features generated in the feature engineering step.
It handles:
- Target variable creation with customizable cutoffs
- Excluding leakage variables
- Feature selection
- WOE binning and transformation
- Scorecard development using logistic regression
- Model evaluation
- Feature inspection and data type validation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
from pathlib import Path
from typing import List, Dict, Tuple, Union, Optional, Any, Set
import scorecardpy as sc
from sklearn.linear_model import LogisticRegression

# Define variables that might leak information
EXLUDE_VARS = [
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

def create_binary_target(
    df: pd.DataFrame, 
    target_var: str = 'repayment_rate',
    cutoff: float = 0.8, 
    target_name: str = 'good_loan'
) -> pd.DataFrame:
    """
    Create a binary target variable based on repayment rate.
    
    Args:
        df: DataFrame with features
        target_var: Column to use for creating binary target
        cutoff: Threshold for good/bad classification (default 0.8)
        target_name: Name for the new binary target variable
        
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
    
    print(f"Target Distribution (cutoff = {cutoff}):")
    print(f"  Good loans: {good_count:,} ({good_pct:.1f}%)")
    print(f"  Bad loans:  {bad_count:,} ({100-good_pct:.1f}%)")
    
    return result_df

def exclude_leakage_variables(
    df: pd.DataFrame, 
    target_var: str, 
    additional_exclusions: Optional[List[str]] = None,
    ignore_warnings: bool = False
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Exclude variables that might leak information about the target.
    
    Args:
        df: DataFrame with features
        target_var: Target variable name (will be retained)
        additional_exclusions: Additional variables to exclude
        ignore_warnings: If True, suppress warnings about missing columns
        
    Returns:
        Tuple containing (filtered DataFrame, list of excluded variables)
    """
    # Start with standard leakage variables
    exclude_vars = EXLUDE_VARS.copy()
    
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
    
    # Return filtered df and list of excluded variables
    return df[keep_cols], excluded

def find_optimal_cutoff(
    df: pd.DataFrame, 
    target_var: str = 'repayment_rate',
    min_cutoff: float = 0.6,
    max_cutoff: float = 0.95,
    step: float = 0.05,
    min_bad_rate: float = 0.1,
    max_bad_rate: float = 0.3,
    plot: bool = True
) -> Dict[str, Union[float, pd.DataFrame]]:
    """
    Find optimal cutoff for creating binary target variable.
    
    Args:
        df: DataFrame with repayment rate
        target_var: Column name for repayment rate
        min_cutoff: Minimum cutoff to consider
        max_cutoff: Maximum cutoff to consider
        step: Step size for cutoff values
        min_bad_rate: Minimum acceptable bad rate
        max_bad_rate: Maximum acceptable bad rate
        plot: Whether to show distribution plot
        
    Returns:
        Dictionary with optimal cutoff and distribution statistics
    """
    # Check if target_var exists
    if target_var not in df.columns:
        raise ValueError(f"Target variable '{target_var}' not found in dataframe")
    
    # Create range of cutoffs to test
    cutoffs = np.arange(min_cutoff, max_cutoff + step, step)
    
    # Calculate statistics for each cutoff
    results = []
    for cutoff in cutoffs:
        bad_count = (df[target_var] < cutoff).sum()
        total_count = len(df)
        bad_rate = bad_count / total_count
        
        results.append({
            'cutoff': cutoff,
            'bad_count': bad_count,
            'good_count': total_count - bad_count,
            'total_count': total_count,
            'bad_rate': bad_rate,
            'good_rate': 1 - bad_rate
        })
    
    # Convert to dataframe
    stats_df = pd.DataFrame(results)
    
    # Find cutoffs within desired bad rate range
    valid_df = stats_df[(stats_df['bad_rate'] >= min_bad_rate) & 
                         (stats_df['bad_rate'] <= max_bad_rate)]
    
    # Select optimal cutoff (middle of valid range or closest to 20% bad rate if available)
    if len(valid_df) > 0:
        # Find cutoff closest to 20% bad rate (common industry standard)
        optimal_cutoff = valid_df.iloc[(valid_df['bad_rate'] - 0.2).abs().argsort()[0]]['cutoff']
    else:
        # If no cutoffs in valid range, pick closest to min_bad_rate
        optimal_cutoff = stats_df.iloc[(stats_df['bad_rate'] - min_bad_rate).abs().argsort()[0]]['cutoff']
    
    # Plot distribution if requested
    if plot:
        plt.figure(figsize=(10, 6))
        
        # Plot bad rate by cutoff
        plt.plot(stats_df['cutoff'], stats_df['bad_rate'] * 100, 'o-', color='#D55E00', label='Bad Rate (%)')
        
        # Highlight valid region
        plt.axhspan(min_bad_rate * 100, max_bad_rate * 100, alpha=0.2, color='green', 
                   label=f'Target Bad Rate ({min_bad_rate*100:.0f}%-{max_bad_rate*100:.0f}%)')
        
        # Highlight optimal cutoff
        opt_bad_rate = stats_df[stats_df['cutoff'] == optimal_cutoff]['bad_rate'].values[0] * 100
        plt.axvline(optimal_cutoff, color='red', linestyle='--', 
                   label=f'Optimal Cutoff = {optimal_cutoff:.2f} (Bad Rate = {opt_bad_rate:.1f}%)')
        
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xlabel('Repayment Rate Cutoff')
        plt.ylabel('Bad Rate (%)')
        plt.title('Bad Rate by Repayment Rate Cutoff')
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    return {
        'optimal_cutoff': optimal_cutoff,
        'cutoff_stats': stats_df,
        'optimal_stats': stats_df[stats_df['cutoff'] == optimal_cutoff].iloc[0].to_dict()
    }

def analyze_repayment_distribution(
    df: pd.DataFrame, 
    target_var: str = 'repayment_rate'
) -> Dict[str, Any]:
    """
    Analyze the distribution of repayment rates to help determine a suitable cutoff.
    
    Args:
        df: DataFrame with loan data
        target_var: Name of repayment rate column
        
    Returns:
        Dictionary with distribution statistics
    """
    if target_var not in df.columns:
        raise ValueError(f"Target variable '{target_var}' not found in dataframe")
    
    # Calculate basic statistics
    stats = {
        'mean': df[target_var].mean(),
        'median': df[target_var].median(),
        'min': df[target_var].min(),
        'max': df[target_var].max(),
        'std': df[target_var].std(),
        'percentiles': {p: df[target_var].quantile(p/100) for p in range(0, 101, 5)}
    }
    
    # Calculate some common cutoffs
    common_cutoffs = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.98]
    cutoff_stats = {}
    
    for cutoff in common_cutoffs:
        bad_loans = (df[target_var] < cutoff).sum()
        good_loans = (df[target_var] >= cutoff).sum()
        bad_rate = bad_loans / len(df)
        
        cutoff_stats[cutoff] = {
            'bad_count': bad_loans,
            'good_count': good_loans,
            'bad_rate': bad_rate,
            'good_rate': 1 - bad_rate
        }
    
    stats['cutoffs'] = cutoff_stats
    
    # Plot distribution
    plt.figure(figsize=(12, 6))
    
    # Main distribution plot
    plt.subplot(1, 2, 1)
    sns.histplot(df[target_var], bins=50, kde=True)
    
    # Add vertical lines for common cutoffs
    for cutoff in common_cutoffs:
        plt.axvline(cutoff, linestyle='--', alpha=0.7, 
                   label=f'{cutoff:.0%} cutoff ({cutoff_stats[cutoff]["bad_rate"]:.1%} bad)')
    
    plt.title(f'Distribution of {target_var}')
    plt.xlabel(target_var)
    plt.ylabel('Count')
    plt.legend()
    
    # CDF plot
    plt.subplot(1, 2, 2)
    plt.hist(df[target_var], bins=50, density=True, cumulative=True, 
             histtype='step', alpha=0.8, color='blue')
    
    # Add horizontal line at common bad rates
    common_bad_rates = [0.1, 0.2, 0.3]
    for bad_rate in common_bad_rates:
        plt.axhline(bad_rate, linestyle=':', color='red', alpha=0.7)
        for cutoff in common_cutoffs:
            if abs(cutoff_stats[cutoff]['bad_rate'] - bad_rate) < 0.02:
                plt.text(cutoff, bad_rate, f'{cutoff:.0%}', 
                        ha='center', va='bottom', bbox=dict(facecolor='white', alpha=0.8))
    
    plt.title('Cumulative Distribution')
    plt.xlabel(target_var)
    plt.ylabel('Cumulative Probability')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()
    
    return stats

def develop_scorecard(
    df: pd.DataFrame,
    target_var: str,
    exclude_vars: Optional[List[str]] = None,
    iv_threshold: float = 0.02,
    woe_bins: Optional[Dict] = None,
    sample_rate: float = 0.8,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Develop a full credit scorecard using the scorecardpy package.
    
    Args:
        df: DataFrame with features and binary target
        target_var: Name of binary target variable (0=bad, 1=good)
        exclude_vars: Variables to exclude from modeling
        iv_threshold: Minimum IV threshold for variable selection
        woe_bins: Pre-defined WOE bins (if None, will be calculated)
        sample_rate: Train/test split ratio
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary with scorecard model and performance metrics
    """
    # Handle exclude_vars
    if exclude_vars is None:
        exclude_vars = []
    
    # Filter variables
    filtered_df, _ = exclude_leakage_variables(
        df, target_var, additional_exclusions=exclude_vars, ignore_warnings=True
    )
    
    # Split train/test
    train_df, test_df = sc.split_df(filtered_df, y=target_var, ratio=sample_rate, seed=random_state).values()
    
    # Filter variables based on IV
    selected_df = sc.var_filter(filtered_df, y=target_var, iv_limit=iv_threshold)
    
    # Get the list of variables after filtering
    feature_vars = [col for col in selected_df.columns if col != target_var]
    
    # Split the selected variables dataframe
    train_selected = train_df[feature_vars + [target_var]]
    test_selected = test_df[feature_vars + [target_var]]
    
    # Perform WOE binning - disable the categorical check to avoid interactive prompt
    if woe_bins is None:
        bins = sc.woebin(selected_df, y=target_var, check_cate_num=False)
    else:
        bins = woe_bins
    
    # Apply WOE transformation
    train_woe = sc.woebin_ply(train_selected, bins)
    test_woe = sc.woebin_ply(test_selected, bins)
    
    # Prepare for modeling
    y_train = train_woe[target_var]
    X_train = train_woe.drop(columns=[target_var])
    
    y_test = test_woe[target_var]
    X_test = test_woe.drop(columns=[target_var])
    
    # Logistic regression
    lr = LogisticRegression(penalty='l1', solver='saga', C=0.9, random_state=random_state)
    lr.fit(X_train, y_train)
    
    # Predictions
    train_pred = lr.predict_proba(X_train)[:, 1]
    test_pred = lr.predict_proba(X_test)[:, 1]
    
    # Performance evaluation
    train_perf = sc.perf_eva(y_train, train_pred, title="Train")
    test_perf = sc.perf_eva(y_test, test_pred, title="Test")
    
    # Create scorecard
    card = sc.scorecard(bins, lr, X_train.columns)
    
    # Apply scorecard
    train_score = sc.scorecard_ply(train_df, card, print_step=0)
    test_score = sc.scorecard_ply(test_df, card, print_step=0)
    
    # PSI between train and test
    psi_result = sc.perf_psi(
        score={'train': train_score, 'test': test_score},
        label={'train': y_train, 'test': y_test}
    )
    
    # Return results
    return {
        'scorecard': card,
        'bins': bins,
        'model': lr,
        'feature_vars': feature_vars,
        'train_perf': train_perf,
        'test_perf': test_perf,
        'psi': psi_result,
        'datasets': {
            'train': train_df,
            'test': test_df,
            'train_woe': train_woe,
            'test_woe': test_woe
        },
        'scores': {
            'train': train_score,
            'test': test_score
        }
    }

def main(
    features_path: str = "data/processed/all_features.csv",
    target_var: str = 'repayment_rate',
    cutoff: Optional[float] = None,
    sample_size: Optional[int] = None,
    exclude_vars: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Run the full scorecard modeling workflow.
    
    Args:
        features_path: Path to features CSV file
        target_var: Target variable name
        cutoff: Cutoff for good/bad classification (if None, will be determined)
        sample_size: Optional sample size for testing
        exclude_vars: Additional variables to exclude from modeling
        
    Returns:
        Dictionary with scorecard model and results
    """
    # Load features
    print(f"Loading features from {features_path}...")
    df = pd.read_csv(features_path)
    
    if sample_size is not None and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=42)
        print(f"Using sample of {sample_size} loans for testing")
    
    # Analyze repayment distribution
    print("\nAnalyzing repayment rate distribution...")
    analyze_repayment_distribution(df, target_var=target_var)
    
    # Determine optimal cutoff if not provided
    if cutoff is None:
        print("\nFinding optimal cutoff for binary classification...")
        cutoff_result = find_optimal_cutoff(df, target_var=target_var)
        cutoff = cutoff_result['optimal_cutoff']
        print(f"Optimal cutoff determined: {cutoff:.2f}")
    
    # Create binary target
    print("\nCreating binary target variable...")
    df_with_target = create_binary_target(df, target_var=target_var, cutoff=cutoff)
    
    # Exclude leakage variables
    print("\nExcluding potential leakage variables...")
    filtered_df, excluded = exclude_leakage_variables(
        df_with_target, 'good_loan', additional_exclusions=exclude_vars
    )
    print(f"Excluded {len(excluded)} variables")
    
    # Develop scorecard
    print("\nDeveloping scorecard model...")
    results = develop_scorecard(filtered_df, 'good_loan')
    
    print("\nScorecard modeling complete!")
    return results

if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description='OAF Loan Scorecard Modeling')
    parser.add_argument('--features', type=str, default="data/processed/all_features.csv",
                       help='Path to features CSV file')
    parser.add_argument('--target', type=str, default='repayment_rate',
                       help='Target variable name')
    parser.add_argument('--cutoff', type=float, default=None,
                       help='Cutoff for good/bad classification')
    parser.add_argument('--sample', type=int, default=None,
                       help='Sample size for testing')
    
    args = parser.parse_args()
    
    # Run main function
    main(
        features_path=args.features,
        target_var=args.target,
        cutoff=args.cutoff,
        sample_size=args.sample
    )
