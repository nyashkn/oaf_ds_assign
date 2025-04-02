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
    Only calculates metrics that can be derived from available columns.
    
    Args:
        df: Raw loan data
        
    Returns:
        Preprocessed DataFrame with additional metrics
    """
    # Create a copy to avoid modifying the original
    df = df.copy()
    
    # Convert date columns
    df['contract_start_date'] = pd.to_datetime(df['contract_start_date'])
    
    # Calculate derived metrics available for all datasets
    df['sept_23_repayment_rate'] = df['cumulative_amount_paid_start'] / df['nominal_contract_value']
    df['deposit_ratio'] = df['deposit_amount'] / df['nominal_contract_value']
    
    # Calculate time-based features
    df['month'] = df['contract_start_date'].dt.to_period('M')
    df['months_since_start'] = (df['contract_start_date'] - df['contract_start_date'].min()).dt.days / 30
    df['days_since_start'] = (df['contract_start_date'] - df['contract_start_date'].min()).dt.days

    df['days_diff_contract_start_to_sept_23'] = (pd.to_datetime('2023-09-01') - df['contract_start_date']).dt.days
    df['days_diff_contract_start_to_nov_23'] = (pd.to_datetime('2023-11-01') - df['contract_start_date']).dt.days
    df['month_diff_contract_start_to_sept_23'] = (pd.to_datetime('2023-09-01') - df['contract_start_date']).dt.days / 30
    df['month_diff_contract_start_to_nov_23'] = (pd.to_datetime('2023-11-01') - df['contract_start_date']).dt.days / 30
    
    df['contract_start_day'] = df['contract_start_date'].dt.day
    df['contract_day_name'] = df['contract_start_date'].dt.day_name()
    
    return df

def prepare_training_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare training dataset with all necessary metrics.
    Assumes cumulative_amount_paid (November data) is available.
    
    Args:
        df: Raw training data that should contain cumulative_amount_paid
        
    Returns:
        Processed training data with November-specific metrics
    """
    # First apply common preprocessing
    df = preprocess_data(df)
    
    # Verify that we have November data
    if 'cumulative_amount_paid' not in df.columns:
        raise ValueError("Training data must contain 'cumulative_amount_paid' column")
    
    # Calculate November-specific metrics
    df['nov_23_repayment_rate'] = df['cumulative_amount_paid'] / df['nominal_contract_value']
    df['diff_nov_23_to_sept_23_repayment_rate'] = df['nov_23_repayment_rate'] - df['sept_23_repayment_rate']
    
    return df

def prepare_holdout_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare holdout dataset for prediction purposes.
    Does NOT add any placeholder November data to maintain clean separation of concerns.
    
    Args:
        df: Raw holdout data (without November data)
        
    Returns:
        Processed holdout data ready for feature engineering
    """
    # Apply common preprocessing
    df = preprocess_data(df)
    
    print("Preparing holdout dataset (no November data available)")
    
    # Verify that this is indeed holdout data
    if 'cumulative_amount_paid' in df.columns:
        print("Warning: Holdout data unexpectedly contains 'cumulative_amount_paid' column")
    
    return df

def get_summary_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate key summary statistics from the loan data.
    Works with both training and holdout data.
    
    Args:
        df: Preprocessed loan data
        
    Returns:
        Dictionary of summary statistics
    """
    # Calculate portfolio totals
    total_portfolio_kes = df['nominal_contract_value'].sum()
    total_portfolio_usd = total_portfolio_kes * KES_TO_USD
    
    # Basic statistics available for all datasets
    stats = {
        'loan_count': len(df),
        'total_portfolio_kes': total_portfolio_kes,
        'total_portfolio_usd': total_portfolio_usd,
        'avg_loan_value': df['nominal_contract_value'].mean(),
        'median_loan_value': df['nominal_contract_value'].median(),
        'avg_loan_value_usd': df['nominal_contract_value'].mean() * KES_TO_USD,
        'median_loan_value_usd': df['nominal_contract_value'].median() * KES_TO_USD,
        'avg_sept_repayment_rate': df['sept_23_repayment_rate'].mean(),
        'median_sept_repayment_rate': df['sept_23_repayment_rate'].median(),
        'avg_deposit_ratio': df['deposit_ratio'].mean(),
        'median_deposit_ratio': df['deposit_ratio'].median(),
        'loan_type_counts': df['Loan_Type'].value_counts().to_dict(),
        'region_counts': df['region'].value_counts().to_dict()
    }
    
    # Add November metrics if available
    if 'nov_23_repayment_rate' in df.columns:
        stats.update({
            'avg_nov_repayment_rate': df['nov_23_repayment_rate'].mean(),
            'median_nov_repayment_rate': df['nov_23_repayment_rate'].median(),
            'target_achievement_rate': (df['nov_23_repayment_rate'] >= 0.98).mean(),
        })
        
        if 'diff_nov_23_to_sept_23_repayment_rate' in df.columns:
            stats['avg_cure_rate'] = df['diff_nov_23_to_sept_23_repayment_rate'].mean()
    
    return stats

def analyze_repayment_curves(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[int, float], Dict[str, float]]:
    """
    Analyze repayment curves and calculate cure rates by contract start day.
    Works with both training and holdout data.
    
    Args:
        df: Preprocessed loan data
        
    Returns:
        Tuple containing:
        - DataFrame with overall progress data
        - Dictionary mapping days to cure rates (empty for holdout data)
        - Dictionary with summary statistics
    """
    # Check if we have November data
    has_nov_data = 'nov_23_repayment_rate' in df.columns
    
    # Calculate overall progress
    sept_days = df['days_diff_contract_start_to_sept_23'].median()
    nov_days = df['days_diff_contract_start_to_nov_23'].median()
    
    # For holdout data without November data
    if not has_nov_data:
        overall_progress = pd.DataFrame({
            'Time Point': ['Contract Start', 'September 2023'],
            'Days Since Start': [0, sept_days],
            'Average Repayment Rate': [0, df['sept_23_repayment_rate'].mean()]
        })
        
        # Cannot calculate cure rates without November data
        day_cure_rates = {}
        
        # Only September statistics
        stats = {
            'sept_below_target': (df['sept_23_repayment_rate'] < 0.98).mean(),
            'sept_above_target': (df['sept_23_repayment_rate'] >= 0.98).mean(),
        }
        
        return overall_progress, day_cure_rates, stats
    
    # For training data with November data
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
    Works with both training and holdout data.
    
    Args:
        df: Preprocessed loan data
        
    Returns:
        Tuple containing:
        - DataFrame with overall progress data
        - Dictionary mapping weekdays to cure rates (empty for holdout data)
        - Dictionary mapping weekdays to colors
        - Dictionary with summary statistics
    """
    # Check if we have November data
    has_nov_data = 'nov_23_repayment_rate' in df.columns
    
    # Calculate overall progress
    sept_days = df['days_diff_contract_start_to_sept_23'].median()
    nov_days = df['days_diff_contract_start_to_nov_23'].median()
    
    # Define weekday colors (same for both scenarios)
    weekday_colors = {
        'Monday': '#3366CC',     # Blue
        'Tuesday': '#DC3912',    # Red
        'Wednesday': '#FF9900',  # Orange
        'Thursday': '#109618',   # Green
        'Friday': '#990099',     # Purple
        'Saturday': '#0099C6',   # Teal
        'Sunday': '#DD4477'      # Pink
    }
    
    # For holdout data without November data
    if not has_nov_data:
        overall_progress = pd.DataFrame({
            'Time Point': ['Contract Start', 'September 2023'],
            'Days Since Start': [0, sept_days],
            'Average Repayment Rate': [0, df['sept_23_repayment_rate'].mean()]
        })
        
        # Cannot calculate cure rates without November data
        weekday_cure_rates = {}
        
        # Calculate September repayment rates by weekday
        weekday_sept_rates = {}
        for weekday in sorted(df['contract_day_name'].unique()):
            weekday_df = df[df['contract_day_name'] == weekday]
            if len(weekday_df) >= 50:  # Only include days with enough loans
                weekday_sept_rates[weekday] = weekday_df['sept_23_repayment_rate'].mean()
        
        # Only September statistics
        stats = {
            'sept_below_target': (df['sept_23_repayment_rate'] < 0.98).mean(),
            'sept_above_target': (df['sept_23_repayment_rate'] >= 0.98).mean(),
            'weekday_sept_rates': weekday_sept_rates
        }
        
        return overall_progress, weekday_cure_rates, weekday_colors, stats
    
    # For training data with November data
    overall_progress = pd.DataFrame({
        'Time Point': ['Contract Start', 'September 2023', 'November 2023'],
        'Days Since Start': [0, sept_days, nov_days],
        'Average Repayment Rate': [0, df['sept_23_repayment_rate'].mean(), 
                                  df['nov_23_repayment_rate'].mean()]
    })
    
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
