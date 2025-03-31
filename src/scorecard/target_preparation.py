"""
Target variable preparation functions for scorecard modeling.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from typing import Dict, List, Optional, Union, Any

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

def find_optimal_cutoff(
    df: pd.DataFrame, 
    target_var: str = 'repayment_rate',
    min_cutoff: float = 0.6,
    max_cutoff: float = 0.95,
    step: float = 0.05,
    min_bad_rate: float = 0.1,
    max_bad_rate: float = 0.3,
    plot: bool = True,
    output_path: Optional[str] = None
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
        
        # Save plot to file
        if output_path:
            plot_dir = os.path.dirname(output_path)
            plot_path = os.path.join(plot_dir, "cutoff_plot.png")
            plt.savefig(plot_path)
            print(f"Cutoff plot saved to {plot_path}")
        
        plt.close()
    
    return {
        'optimal_cutoff': optimal_cutoff,
        'cutoff_stats': stats_df,
        'optimal_stats': stats_df[stats_df['cutoff'] == optimal_cutoff].iloc[0].to_dict()
    }

def analyze_repayment_distribution(
    df: pd.DataFrame, 
    target_var: str = 'repayment_rate',
    output_path: Optional[str] = None
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
    import seaborn as sns
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
    
    # Save plot to file
    if output_path:
        plot_dir = os.path.dirname(output_path)
        plot_path = os.path.join(plot_dir, "repayment_distribution.png")
        plt.savefig(plot_path)
        print(f"Repayment distribution plot saved to {plot_path}")
    
    plt.close()
    return stats
