"""
OAF Loan Performance Analysis - Modular Functions

This module contains functions for analyzing loan performance data,
designed to be imported and used sequentially in a Marimo notebook.
Each function takes input data and returns transformed data or visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

# OAF brand colors
OAF_GREEN = '#1B5E20'  # Primary green
OAF_BROWN = '#8B4513'  # Accent color
OAF_GRAY = '#58595B'   # Dark gray
OAF_LIGHT_GREEN = '#81C784'  # Light green
OAF_BLUE = '#1976D2'   # Blue for accent

# Set style for static plots
plt.style.use('seaborn-v0_8')
sns.set_palette([OAF_GREEN, OAF_BLUE, OAF_BROWN, OAF_LIGHT_GREEN])

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load raw data from CSV file.
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        Loaded DataFrame
    """
    return pd.read_csv(filepath)

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess loan data by calculating derived metrics.
    
    Args:
        df: Raw loan data
        
    Returns:
        Preprocessed DataFrame with additional metrics
    """
    # Create a copy to avoid modifying the original
    df = df.copy()
    
    # Convert date columns
    df['contract_start_date'] = pd.to_datetime(df['contract_start_date'])
    
    # Calculate derived metrics
    df['sept_23_repayment_rate'] = df['cumulative_amount_paid_start'] / df['nominal_contract_value']
    df['deposit_ratio'] = df['deposit_amount'] / df['nominal_contract_value']
    df['nov_23_repayment_rate'] = df['cumulative_amount_paid'] / df['nominal_contract_value'] 
    
    # Calculate time-based features
    df['month'] = df['contract_start_date'].dt.to_period('M')
    df['months_since_start'] = (df['contract_start_date'] - df['contract_start_date'].min()).dt.days / 30
    df['days_since_start'] = (df['contract_start_date'] - df['contract_start_date'].min()).dt.days

    df['days_diff_contract_start_to_sept_23'] = (pd.to_datetime('2023-09-01') - df['contract_start_date']).dt.days
    df['days_diff_contract_start_to_nov_23'] = (pd.to_datetime('2023-11-01') - df['contract_start_date']).dt.days
    df['month_diff_contract_start_to_sept_23'] = (pd.to_datetime('2023-09-01') - df['contract_start_date']).dt.days / 30
    df['month_diff_contract_start_to_nov_23'] = (pd.to_datetime('2023-11-01') - df['contract_start_date']).dt.days / 30
    
    df['diff_nov_23_to_sept_23_repayment_rate'] = df['nov_23_repayment_rate'] - df['sept_23_repayment_rate']
    
    df['contract_start_day'] = df['contract_start_date'].dt.day
    
    return df

def get_summary_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate key summary statistics from the loan data.
    
    Args:
        df: Preprocessed loan data
        
    Returns:
        Dictionary of summary statistics
    """
    stats = {
        'loan_count': len(df),
        'avg_loan_value': df['nominal_contract_value'].mean(),
        'median_loan_value': df['nominal_contract_value'].median(),
        'avg_repayment_rate': df['repayment_rate'].mean(),
        'median_repayment_rate': df['repayment_rate'].median(),
        'avg_deposit_ratio': df['deposit_ratio'].mean(),
        'median_deposit_ratio': df['deposit_ratio'].median(),
        'target_achievement_rate': (df['repayment_rate'] >= 0.98).mean(),
        'loan_type_counts': df['Loan_Type'].value_counts().to_dict(),
        'region_counts': df['region'].value_counts().to_dict()
    }
    
    return stats

def plot_loan_portfolio_composition(df: pd.DataFrame) -> plt.Figure:
    """
    Create pie chart showing loan portfolio composition.
    
    Args:
        df: Preprocessed loan data
        
    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    loan_type_counts = df['Loan_Type'].value_counts()
    ax.pie(loan_type_counts.values, 
           labels=loan_type_counts.index,
           autopct='%1.1f%%', 
           colors=sns.color_palette([OAF_GREEN, OAF_BLUE, OAF_BROWN]),
           startangle=90)
    
    ax.set_title('Loan Portfolio Composition', fontsize=14, fontweight='bold')
    
    return fig

def plot_repayment_distribution(df: pd.DataFrame) -> plt.Figure:
    """
    Create histogram of repayment rate distribution.
    
    Args:
        df: Preprocessed loan data
        
    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.histplot(data=df, x='repayment_rate', bins=30, ax=ax)
    ax.axvline(x=0.98, color='red', linestyle='--', label='98% Target')
    
    # Add summary statistics
    stats_text = (
        f"Mean: {df['repayment_rate'].mean():.1%}\n"
        f"Median: {df['repayment_rate'].median():.1%}\n"
        f"≥ 98% Target: {(df['repayment_rate'] >= 0.98).mean():.1%}\n"
        f"Sample Size: {len(df):,}"
    )
    
    ax.text(0.02, 0.98, stats_text,
            transform=ax.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_title('Repayment Rate Distribution', fontsize=14, fontweight='bold')
    ax.set_xlabel('Repayment Rate')
    ax.set_ylabel('Count')
    ax.legend()
    
    plt.tight_layout()
    return fig

def plot_regional_performance(df: pd.DataFrame) -> plt.Figure:
    """
    Create horizontal bar chart of repayment rates by region.
    
    Args:
        df: Preprocessed loan data
        
    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate regional performance with confidence intervals
    regional_stats = df.groupby('region')['repayment_rate'].agg(['mean', 'std', 'count']).reset_index()
    regional_stats['ci'] = 1.96 * regional_stats['std'] / np.sqrt(regional_stats['count'])
    regional_stats = regional_stats.sort_values('mean', ascending=True)
    
    # Plot horizontal bars with error bars
    bars = ax.barh(regional_stats['region'], 
                  regional_stats['mean'],
                  xerr=regional_stats['ci'],
                  color=OAF_GREEN,
                  capsize=5)
    
    # Add percentage labels
    for bar in bars:
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2,
                f'{width:.1%}', ha='left', va='center')
    
    ax.set_title('Regional Repayment Performance', fontsize=14, fontweight='bold')
    ax.set_xlabel('Average Repayment Rate (with 95% CI)')
    
    plt.tight_layout()
    return fig

def plot_loan_value_repayment(df: pd.DataFrame) -> plt.Figure:
    """
    Create hexbin plot of loan value vs repayment rate.
    
    Args:
        df: Preprocessed loan data
        
    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    h = ax.hexbin(df['nominal_contract_value'], df['repayment_rate'],
                 gridsize=30, cmap='YlOrRd')
    plt.colorbar(h, label='Count')
    
    ax.axhline(y=0.98, color='red', linestyle='--', label='98% Target')
    
    # Add summary statistics
    stats_text = (
        f"Mean Loan: {df['nominal_contract_value'].mean():,.0f} KES\n"
        f"Median Loan: {df['nominal_contract_value'].median():,.0f} KES\n"
        f"Mean Repayment: {df['repayment_rate'].mean():.1%}\n"
        f"Loans Meeting Target: {(df['repayment_rate'] >= 0.98).mean():.1%}"
    )
    
    ax.text(0.02, 0.98, stats_text,
           transform=ax.transAxes,
           verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_xlabel('Loan Value (KES)')
    ax.set_ylabel('Repayment Rate')
    ax.set_title('Loan Value vs Repayment Rate Distribution', fontsize=14, fontweight='bold')
    ax.legend()
    
    plt.tight_layout()
    return fig

def analyze_deposit_ratio(df: pd.DataFrame, n_bins: int = 5) -> Tuple[plt.Figure, pd.DataFrame]:
    """
    Analyze and visualize the relationship between deposit ratio and repayment rate.
    
    Args:
        df: Preprocessed loan data
        n_bins: Number of deposit ratio bins to create
        
    Returns:
        Tuple containing the Matplotlib figure and binned statistics DataFrame
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create bins and calculate statistics
    deposit_bins = pd.qcut(df['deposit_ratio'], q=n_bins)
    deposit_stats = df.groupby(deposit_bins, observed=True)['repayment_rate'].agg(['mean', 'std', 'count'])
    deposit_stats['ci'] = 1.96 * deposit_stats['std'] / np.sqrt(deposit_stats['count'])
    
    # Plot bars with error bars
    ax.bar(range(len(deposit_stats)), deposit_stats['mean'],
           yerr=deposit_stats['ci'],
           color=OAF_GREEN,
           capsize=5)
    
    # Set x-tick labels
    ax.set_xticks(range(len(deposit_stats)))
    ax.set_xticklabels([f'{b.left:.0%}-{b.right:.0%}' for b in deposit_stats.index],
                       rotation=45)
    
    # Add count labels
    for i, (_, row) in enumerate(deposit_stats.iterrows()):
        ax.text(i, row['mean'],
               f'n={int(row["count"]):,}',
               ha='center', va='bottom')
    
    ax.set_xlabel('Deposit Ratio Range')
    ax.set_ylabel('Average Repayment Rate')
    ax.set_title('Repayment Performance by Deposit Ratio Quintiles\nwith 95% Confidence Intervals',
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig, deposit_stats

def analyze_temporal_trends(df: pd.DataFrame, n_bins: int = 12) -> Tuple[plt.Figure, pd.DataFrame]:
    """
    Analyze and visualize temporal trends in repayment rates.
    
    Args:
        df: Preprocessed loan data
        n_bins: Number of time bins to create
        
    Returns:
        Tuple containing the Matplotlib figure and temporal statistics DataFrame
    """
    # Create time bins
    df = df.copy()  # Avoid modifying the original
    df['time_bin'] = pd.qcut(df['months_since_start'], q=n_bins, labels=[f'Period {i+1}' for i in range(n_bins)])
    
    # Calculate statistics by time bin
    time_stats = df.groupby('time_bin', observed=True).agg({
        'repayment_rate': ['mean', 'std', 'count'],
        'nominal_contract_value': 'sum'
    }).round(4)
    
    # Create figure with dual axes
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()
    
    # Plot on primary axis: repayment rate with confidence intervals
    x = range(len(time_stats))
    means = time_stats[('repayment_rate', 'mean')]
    stds = time_stats[('repayment_rate', 'std')]
    counts = time_stats[('repayment_rate', 'count')]
    ci = 1.96 * stds / np.sqrt(counts)
    
    ax1.plot(x, means, 'o-', color=OAF_GREEN, label='Repayment Rate')
    ax1.fill_between(x, means - ci, means + ci, color=OAF_GREEN, alpha=0.2)
    
    # Plot on secondary axis: number of loans
    ax2.bar(x, counts, alpha=0.2, color=OAF_BLUE, label='Number of Loans')
    
    # Customize axes
    ax1.set_xlabel('Time Period')
    ax1.set_ylabel('Average Repayment Rate')
    ax2.set_ylabel('Number of Loans')
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(time_stats.index.get_level_values(0), rotation=45)
    
    # Add legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.title('Repayment Rate Trends Over Time\nwith 95% Confidence Intervals',
             fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig, time_stats

def compare_loan_types(df: pd.DataFrame) -> Tuple[plt.Figure, pd.DataFrame]:
    """
    Compare performance metrics between different loan types.
    
    Args:
        df: Preprocessed loan data
        
    Returns:
        Tuple containing the Matplotlib figure and loan type comparison DataFrame
    """
    # Calculate statistics by loan type
    loan_stats = df.groupby('Loan_Type').agg({
        'repayment_rate': ['mean', 'std', 'count'],
        'nominal_contract_value': ['mean', 'median'],
        'deposit_ratio': ['mean', 'median']
    }).round(4)
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot repayment rate
    loan_types = loan_stats.index
    
    # 1. Repayment rate
    means = loan_stats[('repayment_rate', 'mean')]
    stds = loan_stats[('repayment_rate', 'std')]
    counts = loan_stats[('repayment_rate', 'count')]
    ci = 1.96 * stds / np.sqrt(counts)
    
    bars1 = axes[0].bar(loan_types, means, yerr=ci, capsize=5, color=OAF_GREEN)
    axes[0].set_title('Repayment Rate')
    axes[0].set_ylabel('Average Repayment Rate')
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1%}',
                    ha='center', va='bottom')
    
    # 2. Loan value
    bars2 = axes[1].bar(loan_types, loan_stats[('nominal_contract_value', 'mean')], color=OAF_BLUE)
    axes[1].set_title('Average Loan Value')
    axes[1].set_ylabel('KES')
    
    # Add value labels
    for bar in bars2:
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:,.0f}',
                    ha='center', va='bottom')
    
    # 3. Deposit ratio
    bars3 = axes[2].bar(loan_types, loan_stats[('deposit_ratio', 'mean')], color=OAF_BROWN)
    axes[2].set_title('Average Deposit Ratio')
    axes[2].set_ylabel('Ratio')
    
    # Add value labels
    for bar in bars3:
        height = bar.get_height()
        axes[2].text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1%}',
                    ha='center', va='bottom')
    
    plt.suptitle('Performance Comparison by Loan Type', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    return fig, loan_stats

def analyze_geographic_patterns(df: pd.DataFrame) -> Tuple[plt.Figure, pd.DataFrame]:
    """
    Analyze and visualize geographic patterns in loan performance.
    
    Args:
        df: Preprocessed loan data
        
    Returns:
        Tuple containing the Matplotlib figure and geographic statistics DataFrame
    """
    # Calculate statistics by region and area
    region_stats = df.groupby(['region', 'area']).agg({
        'repayment_rate': ['mean', 'count'],
        'client_id': 'nunique'
    }).round(4)
    
    # Reshape for visualization
    region_area_pivot = df.groupby(['region', 'area'])['repayment_rate'].mean().unstack()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot heatmap
    sns.heatmap(region_area_pivot, cmap='YlGnBu', annot=True, fmt='.1%', ax=ax)
    
    ax.set_title('Repayment Rate by Region and Area', fontsize=14, fontweight='bold')
    ax.set_ylabel('Region')
    ax.set_xlabel('Area')
    
    plt.tight_layout()
    return fig, region_stats

def filter_data(df: pd.DataFrame, 
                min_loan_value: Optional[float] = None,
                max_loan_value: Optional[float] = None,
                regions: Optional[list] = None,
                loan_types: Optional[list] = None,
                min_repayment: Optional[float] = None,
                max_repayment: Optional[float] = None) -> pd.DataFrame:
    """
    Filter the dataframe based on specified criteria.
    
    Args:
        df: Preprocessed loan data
        min_loan_value: Minimum loan value to include
        max_loan_value: Maximum loan value to include
        regions: List of regions to include
        loan_types: List of loan types to include
        min_repayment: Minimum repayment rate to include
        max_repayment: Maximum repayment rate to include
        
    Returns:
        Filtered DataFrame
    """
    filtered_df = df.copy()
    
    if min_loan_value is not None:
        filtered_df = filtered_df[filtered_df['nominal_contract_value'] >= min_loan_value]
        
    if max_loan_value is not None:
        filtered_df = filtered_df[filtered_df['nominal_contract_value'] <= max_loan_value]
        
    if regions is not None:
        filtered_df = filtered_df[filtered_df['region'].isin(regions)]
        
    if loan_types is not None:
        filtered_df = filtered_df[filtered_df['Loan_Type'].isin(loan_types)]
        
    if min_repayment is not None:
        filtered_df = filtered_df[filtered_df['repayment_rate'] >= min_repayment]
        
    if max_repayment is not None:
        filtered_df = filtered_df[filtered_df['repayment_rate'] <= max_repayment]
    
    return filtered_df

def create_executive_dashboard(df: pd.DataFrame) -> plt.Figure:
    """
    Create a four-panel executive dashboard summarizing key metrics.
    
    Args:
        df: Preprocessed loan data
        
    Returns:
        Matplotlib figure with four subplots
    """
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle('One Acre Fund: Loan Performance Analysis\nExecutive Summary', 
                fontsize=16, fontweight='bold')
    
    # 1. Key Metrics Overview (Top Left)
    ax1 = plt.subplot(2, 2, 1)
    metrics = {
        'Average Loan Value': f"KES {df['nominal_contract_value'].mean():,.0f}",
        'Median Deposit Ratio': f"{df['deposit_ratio'].median():.1%}",
        'Average Repayment Rate': f"{df['repayment_rate'].mean():.1%}",
        'Total Active Loans': f"{len(df):,}",
        'Target Achievement': f"{(df['repayment_rate'] >= 0.98).mean():.1%}"
    }
    
    # Create text box with metrics
    metrics_text = '\n'.join([f"{k}: {v}" for k, v in metrics.items()])
    ax1.text(0.5, 0.5, metrics_text, 
            ha='center', va='center',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor=OAF_GREEN),
            fontsize=12)
    ax1.axis('off')
    ax1.set_title('Key Performance Indicators', fontsize=12, fontweight='bold')
    
    # 2. Loan Type Distribution (Top Right)
    ax2 = plt.subplot(2, 2, 2)
    loan_type_counts = df['Loan_Type'].value_counts()
    ax2.pie(loan_type_counts.values, labels=loan_type_counts.index,
            autopct='%1.1f%%', colors=sns.color_palette([OAF_GREEN, OAF_BLUE, OAF_BROWN]))
    ax2.set_title('Loan Portfolio Composition', fontsize=12, fontweight='bold')
    
    # 3. Repayment Rate Distribution (Bottom Left)
    ax3 = plt.subplot(2, 2, 3)
    sns.histplot(data=df, x='repayment_rate', bins=30, ax=ax3)
    ax3.axvline(x=0.98, color='red', linestyle='--', label='98% Target')
    ax3.set_title('Repayment Rate Distribution', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Repayment Rate')
    ax3.set_ylabel('Count')
    ax3.legend()
    
    # 4. Regional Performance (Bottom Right)
    ax4 = plt.subplot(2, 2, 4)
    regional_performance = df.groupby('region')['repayment_rate'].agg(['mean', 'count']).reset_index()
    regional_performance = regional_performance.sort_values('mean', ascending=True)
    
    bars = ax4.barh(regional_performance['region'], 
                    regional_performance['mean'],
                    color=OAF_GREEN)
    
    # Add percentage labels
    for bar in bars:
        width = bar.get_width()
        ax4.text(width, bar.get_y() + bar.get_height()/2,
                f'{width:.1%}', ha='left', va='center')
    
    ax4.set_title('Regional Repayment Performance', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Average Repayment Rate')
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    return fig

# ---- Advanced Analysis Functions ----

def segment_clients(df: pd.DataFrame) -> Tuple[plt.Figure, pd.DataFrame]:
    """
    Segment clients based on repayment behavior and loan characteristics.
    
    Args:
        df: Preprocessed loan data
        
    Returns:
        Tuple containing the Matplotlib figure and segment statistics DataFrame
    """
    # Create client segments
    df = df.copy()
    
    # Determine segments
    df.loc[df['repayment_rate'] >= 0.98, 'segment'] = 'Target Achievers'
    df.loc[(df['repayment_rate'] < 0.98) & (df['repayment_rate'] >= 0.75), 'segment'] = 'Strong Performers'
    df.loc[(df['repayment_rate'] < 0.75) & (df['repayment_rate'] >= 0.50), 'segment'] = 'Moderate Performers'
    df.loc[(df['repayment_rate'] < 0.50) & (df['repayment_rate'] >= 0.25), 'segment'] = 'Underperformers'
    df.loc[df['repayment_rate'] < 0.25, 'segment'] = 'At Risk'
    
    # Calculate segment statistics
    segment_stats = df.groupby('segment').agg({
        'client_id': 'count',
        'nominal_contract_value': ['mean', 'sum'],
        'deposit_ratio': 'mean',
        'repayment_rate': ['mean', 'median']
    })
    
    # Calculate percentages for visualization
    segment_counts = df['segment'].value_counts()
    segment_pcts = segment_counts / segment_counts.sum()
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Pie chart of segment distribution
    ax1.pie(segment_pcts, labels=segment_pcts.index, autopct='%1.1f%%', 
           colors=sns.color_palette("YlOrRd_r", n_colors=len(segment_pcts)))
    ax1.set_title('Client Segment Distribution', fontsize=14, fontweight='bold')
    
    # Bar chart of average metrics by segment
    segments = segment_stats.index
    x = np.arange(len(segments))
    width = 0.35
    
    ax2.bar(x - width/2, segment_stats[('repayment_rate', 'mean')], width, label='Repayment Rate')
    ax2.bar(x + width/2, segment_stats[('deposit_ratio', 'mean')], width, label='Deposit Ratio')
    
    ax2.set_xticks(x)
    ax2.set_xticklabels(segments, rotation=45)
    ax2.set_ylabel('Ratio')
    ax2.set_title('Key Metrics by Client Segment', fontsize=14, fontweight='bold')
    ax2.legend()
    
    plt.tight_layout()
    return fig, segment_stats
