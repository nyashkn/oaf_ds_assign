"""
Data inspection and cleaning functions for regression modeling.

This module reuses functionality from the original scorecard package.
"""

# Import directly from the original scorecard package
from src.scorecard.data_inspection import inspect_data, handle_date_like_columns

# The imports above expose the exact same functionality as the original
# We're simply re-exporting them for consistent module structure

# Additional regression-specific inspection functions can be added here
def identify_outliers(df, target_var, threshold=3.0):
    """
    Identify outliers in the target variable using Z-score method.
    
    Args:
        df: DataFrame with features and target
        target_var: Target variable name
        threshold: Z-score threshold for outlier detection (default 3.0)
        
    Returns:
        DataFrame with outliers identified and z-scores
    """
    from scipy import stats
    import pandas as pd
    import numpy as np
    
    # Create copy of dataframe
    result_df = df.copy()
    
    # Calculate z-scores for target variable
    z_scores = np.abs(stats.zscore(result_df[target_var].dropna()))
    
    # Create mask for the full dataframe (handling NAs)
    z_score_mask = pd.Series(index=result_df.index, dtype=float)
    non_na_indices = result_df[~result_df[target_var].isna()].index
    z_score_mask.loc[non_na_indices] = z_scores
    z_score_mask.fillna(0, inplace=True)
    
    # Add z-score column
    result_df[f'{target_var}_zscore'] = z_score_mask
    
    # Identify outliers
    result_df[f'{target_var}_is_outlier'] = z_score_mask > threshold
    
    # Print summary
    outlier_count = result_df[f'{target_var}_is_outlier'].sum()
    print(f"Identified {outlier_count} outliers in {target_var} ({outlier_count/len(df)*100:.1f}%)")
    
    return result_df

def check_target_distribution(df, target_var):
    """
    Check the distribution of the target variable for regression.
    
    Args:
        df: DataFrame with features and target
        target_var: Target variable name
        
    Returns:
        Dictionary with distribution statistics
    """
    import pandas as pd
    import numpy as np
    
    # Basic statistics
    stats = {
        'count': len(df),
        'missing': df[target_var].isna().sum(),
        'missing_pct': df[target_var].isna().mean() * 100,
        'mean': df[target_var].mean(),
        'median': df[target_var].median(),
        'std': df[target_var].std(),
        'min': df[target_var].min(),
        'max': df[target_var].max(),
        'range': df[target_var].max() - df[target_var].min(),
        'skew': df[target_var].skew(),
        'kurtosis': df[target_var].kurtosis()
    }
    
    # Percentiles
    for p in [1, 5, 10, 25, 75, 90, 95, 99]:
        stats[f'p{p}'] = np.percentile(df[target_var].dropna(), p)
    
    # Print summary
    print(f"\n=== Target Variable Distribution: {target_var} ===")
    print(f"Count:    {stats['count']:,} observations ({stats['missing']:,} missing, {stats['missing_pct']:.1f}%)")
    print(f"Range:    {stats['min']:.2f} to {stats['max']:.2f}")
    print(f"Central:  Mean = {stats['mean']:.4f}, Median = {stats['median']:.4f}")
    print(f"Spread:   Std Dev = {stats['std']:.4f}, Range = {stats['range']:.4f}")
    print(f"Shape:    Skewness = {stats['skew']:.4f}, Kurtosis = {stats['kurtosis']:.4f}")
    
    return stats
