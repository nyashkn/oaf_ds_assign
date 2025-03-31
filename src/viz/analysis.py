"""
OAF Loan Performance Visualizations

This module contains visualization functions for loan performance analysis.
Each function takes preprocessed data and returns Matplotlib figures.
"""

import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, Any, List
import pandas as pd
import numpy as np

# OAF brand colors
OAF_GREEN = '#1B5E20'  # Primary green
OAF_BROWN = '#8B4513'  # Accent color
OAF_GRAY = '#58595B'   # Dark gray
OAF_LIGHT_GREEN = '#81C784'  # Light green
OAF_BLUE = '#1976D2'   # Blue for accent

# Set style for static plots
plt.style.use('seaborn-v0_8')
sns.set_palette([OAF_GREEN, OAF_BLUE, OAF_BROWN, OAF_LIGHT_GREEN])

def plot_loan_portfolio_composition(loan_type_counts: pd.Series) -> plt.Figure:
    """
    Create pie chart showing loan portfolio composition.
    
    Args:
        loan_type_counts: Series with loan type counts
        
    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.pie(loan_type_counts.values, 
           labels=loan_type_counts.index,
           autopct='%1.1f%%', 
           colors=sns.color_palette([OAF_GREEN, OAF_BLUE, OAF_BROWN]),
           startangle=90)
    
    ax.set_title('Loan Portfolio Composition', fontsize=14, fontweight='bold')
    
    return fig

def plot_repayment_distribution(df: pd.DataFrame, stats: Dict[str, float]) -> plt.Figure:
    """
    Create histogram of repayment rate distribution.
    
    Args:
        df: Preprocessed loan data
        stats: Dictionary with summary statistics
        
    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.histplot(data=df, x='repayment_rate', bins=30, ax=ax)
    ax.axvline(x=0.98, color='red', linestyle='--', label='98% Target')
    
    # Add summary statistics
    stats_text = (
        f"Mean: {stats['mean_repayment']:.1%}\n"
        f"Median: {stats['median_repayment']:.1%}\n"
        f"≥ 98% Target: {stats['target_achieved']:.1%}\n"
        f"Sample Size: {stats['count']:,}"
    )
    
    ax.text(0.02, 0.98, stats_text,
            transform=ax.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_title('Repayment Rate Distribution', fontsize=14, fontweight='bold')
    ax.set_xlabel('Repayment Rate')
    ax.set_ylabel('Count')
    ax.legend()
    
    return fig

def plot_repayment_curve_with_cure_rates(df: pd.DataFrame, 
                                       overall_progress: pd.DataFrame,
                                       day_cure_rates: Dict[int, float],
                                       stats: Dict[str, float]) -> plt.Figure:
    """
    Create a dual-panel visualization showing repayment progression and cure rates.
    
    Args:
        df: Preprocessed loan data
        overall_progress: DataFrame with overall repayment progress
        day_cure_rates: Dictionary mapping days to cure rates
        stats: Dictionary with summary statistics
        
    Returns:
        Matplotlib figure object
    """
    fig = plt.figure(figsize=(16, 7))
    
    # Create a 1x2 grid of subplots with different widths
    gs = fig.add_gridspec(1, 2, width_ratios=[2, 1])
    ax1 = fig.add_subplot(gs[0, 0])  # Left panel
    ax2 = fig.add_subplot(gs[0, 1])  # Right panel
    
    # ---- LEFT PANEL: Repayment progression curves ----
    sept_days = df['days_diff_contract_start_to_sept_23'].median()
    nov_days = df['days_diff_contract_start_to_nov_23'].median()
    
    # Plot overall repayment curve
    ax1.plot(overall_progress['Days Since Start'], overall_progress['Average Repayment Rate'], 
            'o-', color=OAF_GREEN, linewidth=3, markersize=8, label='All Loans', zorder=10)
    
    # Plot individual lines for each contract start day
    colors = plt.cm.viridis(np.linspace(0, 1, 31))
    days = sorted(day_cure_rates.keys())
    
    for i, day in enumerate(days):
        day_df = df[df['contract_start_day'] == day]
        day_data = pd.DataFrame({
            'Days Since Start': [0, 
                               day_df['days_diff_contract_start_to_sept_23'].median(), 
                               day_df['days_diff_contract_start_to_nov_23'].median()],
            'Average Repayment Rate': [0, 
                                     day_df['sept_23_repayment_rate'].mean(), 
                                     day_df['nov_23_repayment_rate'].mean()]
        })
        ax1.plot(day_data['Days Since Start'], day_data['Average Repayment Rate'],
               '--', linewidth=1, alpha=0.7, color=colors[i], label=f'Day {day}')
    
    # Add vertical lines and annotations
    ax1.axvline(x=sept_days, color='darkred', linestyle='--', alpha=0.7, label='September 2023')
    ax1.axvline(x=nov_days, color='darkblue', linestyle='--', alpha=0.7, label='November 2023')
    ax1.axhline(y=0.98, color='red', linestyle='--', label='98% Target')
    
    # Add stats text
    stats_text = (
        f"Contracts below 98% in September: {stats['sept_below_target']:.1%}\n"
        f"Contracts below 98% in November: {stats['nov_below_target']:.1%}\n"
        f"Contracts that cured (reached 98%): {stats['pct_cured']:.1%}"
    )
    
    ax1.text(0.02, 0.02, stats_text,
           transform=ax1.transAxes,
           verticalalignment='bottom',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Format left panel
    ax1.set_xlabel('Days Since Contract Start')
    ax1.set_ylabel('Repayment Rate')
    ax1.set_title('Loan Repayment Progression Over Time', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Custom legend
    handles, labels = ax1.get_legend_handles_labels()
    important_indices = [i for i, label in enumerate(labels) 
                       if label in ['All Loans', '98% Target', 'September 2023', 'November 2023']]
    selected_days = [1, 10, 20, 30]
    day_indices = [i for i, label in enumerate(labels) 
                 if label.startswith('Day') and int(label.split(' ')[1]) in selected_days]
    ax1.legend([handles[i] for i in important_indices + day_indices],
              [labels[i] for i in important_indices + day_indices],
              loc='upper left', fontsize=8)
    
    # ---- RIGHT PANEL: Cure rate by contract start day bar chart ----
    days = list(day_cure_rates.keys())
    cure_rates = list(day_cure_rates.values())
    
    bars = ax2.bar(days, cure_rates, color=OAF_BLUE, alpha=0.7)
    
    # Highlight min/max
    min_idx = cure_rates.index(min(cure_rates))
    max_idx = cure_rates.index(max(cure_rates))
    bars[min_idx].set_color('red')
    bars[min_idx].set_alpha(1.0)
    bars[max_idx].set_color('green')
    bars[max_idx].set_alpha(1.0)
    
    # Add horizontal lines for median and overall
    ax2.axhline(y=np.median(cure_rates), color='black', linestyle='--', 
                label=f"Median: {np.median(cure_rates):.1%}")
    ax2.axhline(y=stats['overall_cure_rate'], color=OAF_GREEN, linestyle='-', 
                label=f"Overall: {stats['overall_cure_rate']:.1%}")
    
    # Format right panel
    ax2.set_xlabel('Contract Start Day')
    ax2.set_ylabel('Cure Rate (Nov - Sept)')
    ax2.set_title('Cure Rate by Contract Start Day', fontsize=14, fontweight='bold')
    ax2.set_xticks(range(0, 31, 5))
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.legend()
    
    # Add cure rate stats
    cure_stats_text = (
        f"Cure Rate Statistics:\n"
        f"Min: {min(cure_rates):.1%} (Day {days[min_idx]})\n"
        f"Median: {np.median(cure_rates):.1%}\n"
        f"Max: {max(cure_rates):.1%} (Day {days[max_idx]})\n"
        f"Overall: {stats['overall_cure_rate']:.1%}"
    )
    
    ax2.text(0.98, 0.02, cure_stats_text,
           transform=ax2.transAxes,
           verticalalignment='bottom',
           horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    return fig

def plot_repayment_curve_by_weekday(df: pd.DataFrame,
                                   overall_progress: pd.DataFrame,
                                   weekday_cure_rates: Dict[str, float],
                                   weekday_colors: Dict[str, str],
                                   stats: Dict[str, float]) -> plt.Figure:
    """
    Create a dual-panel visualization showing repayment progression by weekday.
    
    Args:
        df: Preprocessed loan data
        overall_progress: DataFrame with overall repayment progress
        weekday_cure_rates: Dictionary mapping weekdays to cure rates
        weekday_colors: Dictionary mapping weekdays to colors
        stats: Dictionary with summary statistics
        
    Returns:
        Matplotlib figure object
    """
    fig = plt.figure(figsize=(16, 7))
    
    # Create a 1x2 grid of subplots with different widths
    gs = fig.add_gridspec(1, 2, width_ratios=[2, 1])
    ax1 = fig.add_subplot(gs[0, 0])  # Left panel
    ax2 = fig.add_subplot(gs[0, 1])  # Right panel
    
    # ---- LEFT PANEL: Repayment progression curves by weekday ----
    sept_days = df['days_diff_contract_start_to_sept_23'].median()
    nov_days = df['days_diff_contract_start_to_nov_23'].median()
    
    # Plot overall repayment curve
    ax1.plot(overall_progress['Days Since Start'], overall_progress['Average Repayment Rate'], 
            'o-', color=OAF_GREEN, linewidth=3, markersize=8, label='All Loans', zorder=10)
    
    # Plot separate curves for each weekday
    for weekday, color in weekday_colors.items():
        weekday_df = df[df['contract_day_name'] == weekday]
        if len(weekday_df) < 50:  # Skip days with too few loans
            continue
            
        weekday_data = pd.DataFrame({
            'Days Since Start': [0, 
                               weekday_df['days_diff_contract_start_to_sept_23'].median(), 
                               weekday_df['days_diff_contract_start_to_nov_23'].median()],
            'Average Repayment Rate': [0, 
                                     weekday_df['sept_23_repayment_rate'].mean(), 
                                     weekday_df['nov_23_repayment_rate'].mean()]
        })
        
        ax1.plot(weekday_data['Days Since Start'], weekday_data['Average Repayment Rate'],
               'o-', linewidth=2, alpha=0.8, color=color, label=weekday)
    
    # Add vertical lines and annotations
    ax1.axvline(x=sept_days, color='darkred', linestyle='--', alpha=0.7, label='September 2023')
    ax1.axvline(x=nov_days, color='darkblue', linestyle='--', alpha=0.7, label='November 2023')
    ax1.axhline(y=0.98, color='red', linestyle='--', label='98% Target')
    
    # Add stats text
    stats_text = (
        f"Contracts below 98% in September: {stats['sept_below_target']:.1%}\n"
        f"Contracts below 98% in November: {stats['nov_below_target']:.1%}\n"
        f"Contracts that cured (reached 98%): {stats['pct_cured']:.1%}"
    )
    
    ax1.text(0.02, 0.02, stats_text,
           transform=ax1.transAxes,
           verticalalignment='bottom',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Format left panel
    ax1.set_xlabel('Days Since Contract Start')
    ax1.set_ylabel('Repayment Rate')
    ax1.set_title('Loan Repayment Progression by Day of Week', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left', fontsize=8)
    
    # ---- RIGHT PANEL: Cure rate by weekday bar chart ----
    weekdays = list(weekday_cure_rates.keys())
    cure_rates = list(weekday_cure_rates.values())
    
    bars = ax2.bar(weekdays, cure_rates, 
                  color=[weekday_colors[day] for day in weekdays])
    
    # Add horizontal lines for median and overall
    ax2.axhline(y=np.median(cure_rates), color='black', linestyle='--', 
                label=f"Median: {np.median(cure_rates):.1%}")
    ax2.axhline(y=stats['overall_cure_rate'], color=OAF_GREEN, linestyle='-', 
                label=f"Overall: {stats['overall_cure_rate']:.1%}")
    
    # Format right panel
    ax2.set_xlabel('Contract Start Day of Week')
    ax2.set_ylabel('Cure Rate (Nov - Sept)')
    ax2.set_title('Cure Rate by Day of Week', fontsize=14, fontweight='bold')
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Find min/max weekdays
    min_idx = cure_rates.index(min(cure_rates))
    max_idx = cure_rates.index(max(cure_rates))
    min_weekday = weekdays[min_idx]
    max_weekday = weekdays[max_idx]
    
    # Add cure rate stats
    cure_stats_text = (
        f"Cure Rate Statistics:\n"
        f"Min: {min(cure_rates):.1%} ({min_weekday})\n"
        f"Median: {np.median(cure_rates):.1%}\n"
        f"Max: {max(cure_rates):.1%} ({max_weekday})\n"
        f"Overall: {stats['overall_cure_rate']:.1%}"
    )
    
    ax2.text(0.98, 0.02, cure_stats_text,
           transform=ax2.transAxes,
           verticalalignment='bottom',
           horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax2.legend()
    plt.tight_layout()
    return fig
