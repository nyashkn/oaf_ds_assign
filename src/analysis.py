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
    # Contract start day, Day name
    df['contract_day_name'] = df['contract_start_date'].dt.day_name()
    
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

def plot_repayment_distribution_comparison(df: pd.DataFrame) -> plt.Figure:
    """
    Create histogram comparing September and November repayment rate distributions.
    
    Args:
        df: Preprocessed loan data
        
    Returns:
        Matplotlib figure object
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # September distribution
    sns.histplot(data=df, x='sept_23_repayment_rate', bins=30, ax=ax1, color=OAF_BLUE)
    ax1.axvline(x=0.98, color='red', linestyle='--', label='98% Target')
    
    # Add September summary statistics
    sept_stats_text = (
        f"Mean: {df['sept_23_repayment_rate'].mean():.1%}\n"
        f"Median: {df['sept_23_repayment_rate'].median():.1%}\n"
        f"≥ 98% Target: {(df['sept_23_repayment_rate'] >= 0.98).mean():.1%}\n"
        f"Sample Size: {len(df):,}"
    )
    
    ax1.text(0.02, 0.98, sept_stats_text,
            transform=ax1.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax1.set_title('September 2023 Repayment Rate Distribution', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Repayment Rate')
    ax1.set_ylabel('Count')
    ax1.legend()
    
    # November distribution
    sns.histplot(data=df, x='nov_23_repayment_rate', bins=30, ax=ax2, color=OAF_GREEN)
    ax2.axvline(x=0.98, color='red', linestyle='--', label='98% Target')
    
    # Add November summary statistics
    nov_stats_text = (
        f"Mean: {df['nov_23_repayment_rate'].mean():.1%}\n"
        f"Median: {df['nov_23_repayment_rate'].median():.1%}\n"
        f"≥ 98% Target: {(df['nov_23_repayment_rate'] >= 0.98).mean():.1%}\n"
        f"Sample Size: {len(df):,}"
    )
    
    ax2.text(0.02, 0.98, nov_stats_text,
            transform=ax2.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax2.set_title('November 2023 Repayment Rate Distribution', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Repayment Rate')
    ax2.set_ylabel('Count')
    ax2.legend()
    
    plt.tight_layout()
    plt.suptitle('Comparison of Repayment Rate Distributions', fontsize=14, fontweight='bold', y=1.05)
    return fig


def plot_repayment_curve_with_cure_rates(df: pd.DataFrame) -> plt.Figure:
    """
    Create a dual-panel visualization showing:
    1. Left panel: Repayment progression over time with vertical lines for September and November
    2. Right panel: Bar chart of cure rates by contract start day
    
    Args:
        df: Preprocessed loan data
        
    Returns:
        Matplotlib figure object
    """
    fig = plt.figure(figsize=(16, 7))
    
    # Create a 1x2 grid of subplots with different widths (left panel wider)
    gs = fig.add_gridspec(1, 2, width_ratios=[2, 1])
    ax1 = fig.add_subplot(gs[0, 0])  # Left panel
    ax2 = fig.add_subplot(gs[0, 1])  # Right panel
    
    # ---- LEFT PANEL: Repayment progression curves ----
    
    # Calculate median days for September and November from contract start
    sept_days = df['days_diff_contract_start_to_sept_23'].median()
    nov_days = df['days_diff_contract_start_to_nov_23'].median()
    
    # Create a DataFrame with overall repayment progress data
    overall_progress = pd.DataFrame({
        'Time Point': ['Contract Start', 'September 2023', 'November 2023'],
        'Days Since Start': [0, sept_days, nov_days],
        'Average Repayment Rate': [0, df['sept_23_repayment_rate'].mean(), 
                                  df['nov_23_repayment_rate'].mean()]
    })
    
    # Plot overall repayment curve
    ax1.plot(overall_progress['Days Since Start'], overall_progress['Average Repayment Rate'], 
            'o-', color=OAF_GREEN, linewidth=3, markersize=8, label='All Loans', zorder=10)
    
    # Calculate cure rates by day
    day_cure_rates = {}
    colors = plt.cm.viridis(np.linspace(0, 1, 31))  # Color spectrum for days 1-31
    
    # Plot individual lines for each contract start day
    for i, day in enumerate(sorted(df['contract_start_day'].unique())):
        # if day > 30:  # Skip outliers
        #     continue
            
        day_df = df[df['contract_start_day'] == day]
        # if len(day_df) < 50:  # Skip days with too few loans for statistical significance
        #     continue
        
        # Calculate day's repayment progression
        day_data = pd.DataFrame({
            'Time Point': ['Contract Start', 'September 2023', 'November 2023'],
            'Days Since Start': [0, day_df['days_diff_contract_start_to_sept_23'].median(), 
                                day_df['days_diff_contract_start_to_nov_23'].median()],
            'Average Repayment Rate': [0, day_df['sept_23_repayment_rate'].mean(), 
                                     day_df['nov_23_repayment_rate'].mean()]
        })
        
        # Calculate cure rate for this day
        day_cure_rate = day_data['Average Repayment Rate'].iloc[2] - day_data['Average Repayment Rate'].iloc[1]
        day_cure_rates[day] = day_cure_rate
        
        # Plot this day's curve
        ax1.plot(day_data['Days Since Start'], day_data['Average Repayment Rate'],
               '--', linewidth=1, alpha=0.7, color=colors[i], label=f'Day {day}')
    
    # Calculate overall cure rate
    overall_cure_rate = overall_progress['Average Repayment Rate'].iloc[2] - overall_progress['Average Repayment Rate'].iloc[1]
    
    # Annotate overall cure rate
    cure_x = (overall_progress['Days Since Start'].iloc[1] + overall_progress['Days Since Start'].iloc[2]) / 2
    cure_y = (overall_progress['Average Repayment Rate'].iloc[1] + overall_progress['Average Repayment Rate'].iloc[2]) / 2
    
    ax1.annotate(f"Cure rate: {overall_cure_rate:.1%}", 
               xy=(cure_x, cure_y),
               xytext=(0, 20), textcoords='offset points',
               arrowprops=dict(arrowstyle='->'),
               bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.5),
               fontweight='bold', zorder=15)
    
    # Add vertical lines for September and November
    ax1.axvline(x=sept_days, color='darkred', linestyle='--', alpha=0.7, label='September 2023')
    ax1.axvline(x=nov_days, color='darkblue', linestyle='--', alpha=0.7, label='November 2023')
    
    # Add curly brace to indicate estimated period
    mid_early_period = sept_days / 2
    ax1.annotate('', xy=(5, 0.15), xytext=(sept_days-5, 0.15),
                arrowprops=dict(arrowstyle='|-|, widthA=0.5, widthB=0.5',
                                connectionstyle='arc3, rad=0.5', color='black'))
    ax1.text(mid_early_period, 0.1, 'Estimated curve\n(no observations)', 
            ha='center', va='center', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Add key performance targets
    ax1.axhline(y=0.98, color='red', linestyle='--', label='98% Target')
    
    # Format the left panel
    ax1.set_xlabel('Days Since Contract Start')
    ax1.set_ylabel('Repayment Rate')
    ax1.set_title('Loan Repayment Progression Over Time', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Add stats on cured contracts
    cured_contracts = (df['nov_23_repayment_rate'] >= 0.98) & (df['sept_23_repayment_rate'] < 0.98)
    pct_cured = cured_contracts.mean()
    
    stats_text = (
        f"Contracts below 98% in September: {(df['sept_23_repayment_rate'] < 0.98).mean():.1%}\n"
        f"Contracts below 98% in November: {(df['nov_23_repayment_rate'] < 0.98).mean():.1%}\n"
        f"Contracts that cured (reached 98%): {pct_cured:.1%}"
    )
    
    ax1.text(0.02, 0.02, stats_text,
           transform=ax1.transAxes,
           verticalalignment='bottom',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Custom legend with fewer entries to avoid overcrowding
    handles, labels = ax1.get_legend_handles_labels()
    # Keep 'All Loans', '98% Target', vertical lines, and some representative days
    important_indices = [i for i, label in enumerate(labels) 
                       if label in ['All Loans', '98% Target', 'September 2023', 'November 2023']]
    selected_days = [1, 10, 20, 30]  # Representative days
    day_indices = [i for i, label in enumerate(labels) 
                 if label.startswith('Day') and int(label.split(' ')[1]) in selected_days]
    ax1.legend([handles[i] for i in important_indices + day_indices],
              [labels[i] for i in important_indices + day_indices],
              loc='upper left', fontsize=8)
    
    # ---- RIGHT PANEL: Cure rate by contract start day bar chart ----
    
    # Calculate cure rate stats
    cure_rates = list(day_cure_rates.values())
    min_cure_rate = min(cure_rates)
    max_cure_rate = max(cure_rates)
    median_cure_rate = np.median(cure_rates)
    
    # Sort days by cure rate to see patterns
    days = list(day_cure_rates.keys())
    cure_rates_sorted = [day_cure_rates[day] for day in days]
    
    # Create bar chart
    bars = ax2.bar(days, cure_rates_sorted, color=OAF_BLUE, alpha=0.7)
    
    # Highlight bars with highest and lowest cure rates
    min_idx = cure_rates_sorted.index(min_cure_rate)
    max_idx = cure_rates_sorted.index(max_cure_rate)
    bars[min_idx].set_color('red')
    bars[min_idx].set_alpha(1.0)
    bars[max_idx].set_color('green')
    bars[max_idx].set_alpha(1.0)
    
    # Add horizontal line for median and overall cure rate
    ax2.axhline(y=median_cure_rate, color='black', linestyle='--', label=f'Median: {median_cure_rate:.1%}')
    ax2.axhline(y=overall_cure_rate, color=OAF_GREEN, linestyle='-', label=f'Overall: {overall_cure_rate:.1%}')
    
    # Format the right panel
    ax2.set_xlabel('Contract Start Day')
    ax2.set_ylabel('Cure Rate (Nov - Sept)')
    ax2.set_title('Cure Rate by Contract Start Day', fontsize=14, fontweight='bold')
    ax2.set_xticks(range(0, 31, 5))  # Show every 5th day on x-axis
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add cure rate stats
    cure_stats_text = (
        f"Cure Rate Statistics:\n"
        f"Min: {min_cure_rate:.1%} (Day {days[min_idx]})\n"
        f"Median: {median_cure_rate:.1%}\n"
        f"Max: {max_cure_rate:.1%} (Day {days[max_idx]})\n"
        f"Overall: {overall_cure_rate:.1%}"
    )
    
    ax2.text(0.98, 0.02, cure_stats_text,
           transform=ax2.transAxes,
           verticalalignment='bottom',
           horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax2.legend()
    
    # Adjust layout and spacing
    plt.tight_layout()
    
    return fig

def plot_repayment_curve_by_weekday(df: pd.DataFrame) -> plt.Figure:
    """
    Create a dual-panel visualization showing:
    1. Left panel: Repayment progression over time with curves for each day of week
    2. Right panel: Bar chart of cure rates by day of week
    
    Args:
        df: Preprocessed loan data with day_name column
        
    Returns:
        Matplotlib figure object
    """
    fig = plt.figure(figsize=(16, 7))
    
    # Create a 1x2 grid of subplots with different widths (left panel wider)
    gs = fig.add_gridspec(1, 2, width_ratios=[2, 1])
    ax1 = fig.add_subplot(gs[0, 0])  # Left panel
    ax2 = fig.add_subplot(gs[0, 1])  # Right panel
    
    # ---- LEFT PANEL: Repayment progression curves by day of week ----
    
    # Calculate median days for September and November from contract start
    sept_days = df['days_diff_contract_start_to_sept_23'].median()
    nov_days = df['days_diff_contract_start_to_nov_23'].median()
    
    # Create a DataFrame with overall repayment progress data
    overall_progress = pd.DataFrame({
        'Time Point': ['Contract Start', 'September 2023', 'November 2023'],
        'Days Since Start': [0, sept_days, nov_days],
        'Average Repayment Rate': [0, df['sept_23_repayment_rate'].mean(), 
                                  df['nov_23_repayment_rate'].mean()]
    })
    
    # Plot overall repayment curve
    ax1.plot(overall_progress['Days Since Start'], overall_progress['Average Repayment Rate'], 
            'o-', color=OAF_GREEN, linewidth=3, markersize=8, label='All Loans', zorder=10)
    
    # Calculate cure rates by day of week
    weekday_cure_rates = {}
    
    # Define standard weekday order and colors
    weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekday_colors = {
        'Monday': '#3366CC',     # Blue
        'Tuesday': '#DC3912',    # Red
        'Wednesday': '#FF9900',  # Orange
        'Thursday': '#109618',   # Green
        'Friday': '#990099',     # Purple
        'Saturday': '#0099C6',   # Teal
        'Sunday': '#DD4477'      # Pink
    }
    
    # Plot separate curves for each day of the week
    for weekday in weekdays:
        weekday_df = df[df['contract_day_name'] == weekday]
        if len(weekday_df) < 50:  # Skip days with too few loans
            continue
        
        # Calculate weekday's repayment progression
        weekday_data = pd.DataFrame({
            'Time Point': ['Contract Start', 'September 2023', 'November 2023'],
            'Days Since Start': [0, weekday_df['days_diff_contract_start_to_sept_23'].median(), 
                                weekday_df['days_diff_contract_start_to_nov_23'].median()],
            'Average Repayment Rate': [0, weekday_df['sept_23_repayment_rate'].mean(), 
                                     weekday_df['nov_23_repayment_rate'].mean()]
        })
        
        # Calculate cure rate for this weekday
        weekday_cure_rate = weekday_data['Average Repayment Rate'].iloc[2] - weekday_data['Average Repayment Rate'].iloc[1]
        weekday_cure_rates[weekday] = weekday_cure_rate
        
        # Plot this weekday's curve
        ax1.plot(weekday_data['Days Since Start'], weekday_data['Average Repayment Rate'],
               'o-', linewidth=2, alpha=0.8, color=weekday_colors[weekday], label=weekday)
    
    # Calculate overall cure rate
    overall_cure_rate = overall_progress['Average Repayment Rate'].iloc[2] - overall_progress['Average Repayment Rate'].iloc[1]
    
    # Annotate overall cure rate
    cure_x = (overall_progress['Days Since Start'].iloc[1] + overall_progress['Days Since Start'].iloc[2]) / 2
    cure_y = (overall_progress['Average Repayment Rate'].iloc[1] + overall_progress['Average Repayment Rate'].iloc[2]) / 2
    
    ax1.annotate(f"Overall Cure rate: {overall_cure_rate:.1%}", 
               xy=(cure_x, cure_y),
               xytext=(0, 20), textcoords='offset points',
               arrowprops=dict(arrowstyle='->'),
               bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.5),
               fontweight='bold', zorder=15)
    
    # Add vertical lines for September and November
    ax1.axvline(x=sept_days, color='darkred', linestyle='--', alpha=0.7, label='September 2023')
    ax1.axvline(x=nov_days, color='darkblue', linestyle='--', alpha=0.7, label='November 2023')
    
    # Add curly brace to indicate estimated period
    mid_early_period = sept_days / 2
    ax1.annotate('', xy=(5, 0.15), xytext=(sept_days-5, 0.15),
                arrowprops=dict(arrowstyle='|-|, widthA=0.5, widthB=0.5',
                                connectionstyle='arc3, rad=0.5', color='black'))
    ax1.text(mid_early_period, 0.1, 'Estimated curve\n(no observations)', 
            ha='center', va='center', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Add key performance targets
    ax1.axhline(y=0.98, color='red', linestyle='--', label='98% Target')
    
    # Format the left panel
    ax1.set_xlabel('Days Since Contract Start')
    ax1.set_ylabel('Repayment Rate')
    ax1.set_title('Loan Repayment Progression by Day of Week', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Add stats on cured contracts
    cured_contracts = (df['nov_23_repayment_rate'] >= 0.98) & (df['sept_23_repayment_rate'] < 0.98)
    pct_cured = cured_contracts.mean()
    
    stats_text = (
        f"Contracts below 98% in September: {(df['sept_23_repayment_rate'] < 0.98).mean():.1%}\n"
        f"Contracts below 98% in November: {(df['nov_23_repayment_rate'] < 0.98).mean():.1%}\n"
        f"Contracts that cured (reached 98%): {pct_cured:.1%}"
    )
    
    ax1.text(0.02, 0.02, stats_text,
           transform=ax1.transAxes,
           verticalalignment='bottom',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Legend with reasonable size
    ax1.legend(loc='upper left', fontsize=8)
    
    # ---- RIGHT PANEL: Cure rate by day of week bar chart ----
    
    # Calculate cure rate stats
    cure_rates = list(weekday_cure_rates.values())
    if cure_rates:  # Check if we have any valid cure rates
        min_cure_rate = min(cure_rates)
        max_cure_rate = max(cure_rates)
        median_cure_rate = np.median(cure_rates)
        
        # Get weekdays in proper order with their cure rates
        ordered_weekdays = [day for day in weekdays if day in weekday_cure_rates]
        ordered_rates = [weekday_cure_rates[day] for day in ordered_weekdays]
        
        # Create bar chart with consistent colors matching the line plot
        bars = ax2.bar(ordered_weekdays, ordered_rates, 
                     color=[weekday_colors[day] for day in ordered_weekdays])
        
        # Add horizontal line for median and overall cure rate
        ax2.axhline(y=median_cure_rate, color='black', linestyle='--', label=f'Median: {median_cure_rate:.1%}')
        ax2.axhline(y=overall_cure_rate, color=OAF_GREEN, linestyle='-', label=f'Overall: {overall_cure_rate:.1%}')
        
        # Format the right panel
        ax2.set_xlabel('Contract Start Day of Week')
        ax2.set_ylabel('Cure Rate (Nov - Sept)')
        ax2.set_title('Cure Rate by Day of Week', fontsize=14, fontweight='bold')
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Find days with min and max cure rates
        min_weekday = ordered_weekdays[ordered_rates.index(min_cure_rate)]
        max_weekday = ordered_weekdays[ordered_rates.index(max_cure_rate)]
        
        # Add cure rate stats
        cure_stats_text = (
            f"Cure Rate Statistics:\n"
            f"Min: {min_cure_rate:.1%} ({min_weekday})\n"
            f"Median: {median_cure_rate:.1%}\n"
            f"Max: {max_cure_rate:.1%} ({max_weekday})\n"
            f"Overall: {overall_cure_rate:.1%}"
        )
        
        ax2.text(0.98, 0.02, cure_stats_text,
               transform=ax2.transAxes,
               verticalalignment='bottom',
               horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax2.legend()
    
    # Adjust layout and spacing
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

# ---- Analysis Functions ----

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
