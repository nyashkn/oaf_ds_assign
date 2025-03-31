"""
OAF Loan Performance Analysis - Modular Functions

This module contains functions for analyzing loan performance data.
Each function takes input data and returns processed data or statistics.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple

# Constants
KES_TO_USD = 1/130  # Conversion rate: 130 KES = 1 USD

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
    # Calculate portfolio totals
    total_portfolio_kes = df['nominal_contract_value'].sum()
    total_portfolio_usd = total_portfolio_kes * KES_TO_USD
    
    stats = {
        'loan_count': len(df),
        'total_portfolio_kes': total_portfolio_kes,
        'total_portfolio_usd': total_portfolio_usd,
        'avg_loan_value': df['nominal_contract_value'].mean(),
        'median_loan_value': df['nominal_contract_value'].median(),
        'avg_loan_value_usd': df['nominal_contract_value'].mean() * KES_TO_USD,
        'median_loan_value_usd': df['nominal_contract_value'].median() * KES_TO_USD,
        'avg_repayment_rate': df['nov_23_repayment_rate'].mean(),
        'median_repayment_rate': df['nov_23_repayment_rate'].median(),
        'avg_deposit_ratio': df['deposit_ratio'].mean(),
        'median_deposit_ratio': df['deposit_ratio'].median(),
        'target_achievement_rate': (df['nov_23_repayment_rate'] >= 0.98).mean(),
        'loan_type_counts': df['Loan_Type'].value_counts().to_dict(),
        'region_counts': df['region'].value_counts().to_dict()
    }
    
    return stats

def analyze_repayment_curves(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[int, float], Dict[str, float]]:
    """
    Analyze repayment curves and calculate cure rates by contract start day.
    
    Args:
        df: Preprocessed loan data
        
    Returns:
        Tuple containing:
        - DataFrame with overall progress data
        - Dictionary mapping days to cure rates
        - Dictionary with summary statistics
    """
    # Calculate overall progress
    sept_days = df['days_diff_contract_start_to_sept_23'].median()
    nov_days = df['days_diff_contract_start_to_nov_23'].median()
    
    overall_progress = pd.DataFrame({
        'Time Point': ['Contract Start', 'September 2023', 'November 2023'],
        'Days Since Start': [0, sept_days, nov_days],
        'Average Repayment Rate': [0, df['sept_23_repayment_rate'].mean(), 
                                  df['nov_23_repayment_rate'].mean()]
    })
    
    # Calculate cure rates by day
    day_cure_rates = {}
    for day in sorted(df['contract_start_day'].unique()):
        day_df = df[df['contract_start_day'] == day]
        cure_rate = day_df['nov_23_repayment_rate'].mean() - day_df['sept_23_repayment_rate'].mean()
        day_cure_rates[day] = cure_rate
    
    # Calculate overall cure rate and statistics
    stats = {
        'sept_below_target': (df['sept_23_repayment_rate'] < 0.98).mean(),
        'nov_below_target': (df['nov_23_repayment_rate'] < 0.98).mean(),
        'pct_cured': ((df['nov_23_repayment_rate'] >= 0.98) & (df['sept_23_repayment_rate'] < 0.98)).mean(),
        'overall_cure_rate': overall_progress['Average Repayment Rate'].iloc[2] - overall_progress['Average Repayment Rate'].iloc[1]
    }
    
    return overall_progress, day_cure_rates, stats

def analyze_repayment_by_weekday(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float], Dict[str, str], Dict[str, float]]:
    """
    Analyze repayment curves and calculate cure rates by weekday.
    
    Args:
        df: Preprocessed loan data
        
    Returns:
        Tuple containing:
        - DataFrame with overall progress data
        - Dictionary mapping weekdays to cure rates
        - Dictionary mapping weekdays to colors
        - Dictionary with summary statistics
    """
    # Calculate overall progress
    sept_days = df['days_diff_contract_start_to_sept_23'].median()
    nov_days = df['days_diff_contract_start_to_nov_23'].median()
    
    overall_progress = pd.DataFrame({
        'Time Point': ['Contract Start', 'September 2023', 'November 2023'],
        'Days Since Start': [0, sept_days, nov_days],
        'Average Repayment Rate': [0, df['sept_23_repayment_rate'].mean(), 
                                  df['nov_23_repayment_rate'].mean()]
    })
    
    # Define weekday colors
    weekday_colors = {
        'Monday': '#3366CC',     # Blue
        'Tuesday': '#DC3912',    # Red
        'Wednesday': '#FF9900',  # Orange
        'Thursday': '#109618',   # Green
        'Friday': '#990099',     # Purple
        'Saturday': '#0099C6',   # Teal
        'Sunday': '#DD4477'      # Pink
    }
    
    # Calculate cure rates by weekday
    weekday_cure_rates = {}
    for weekday in sorted(df['contract_day_name'].unique()):
        weekday_df = df[df['contract_day_name'] == weekday]
        if len(weekday_df) >= 50:  # Only include days with enough loans
            cure_rate = weekday_df['nov_23_repayment_rate'].mean() - weekday_df['sept_23_repayment_rate'].mean()
            weekday_cure_rates[weekday] = cure_rate
    
    # Calculate overall cure rate and statistics
    stats = {
        'sept_below_target': (df['sept_23_repayment_rate'] < 0.98).mean(),
        'nov_below_target': (df['nov_23_repayment_rate'] < 0.98).mean(),
        'pct_cured': ((df['nov_23_repayment_rate'] >= 0.98) & (df['sept_23_repayment_rate'] < 0.98)).mean(),
        'overall_cure_rate': overall_progress['Average Repayment Rate'].iloc[2] - overall_progress['Average Repayment Rate'].iloc[1]
    }
    
    return overall_progress, weekday_cure_rates, weekday_colors, stats
