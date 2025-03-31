"""
OAF Loan Performance Model Development

This module contains functions for feature engineering and model development.
Ensures all features are calculated using only data available at loan application time.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple

def calculate_historical_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate historical metrics using only data available before each loan's application date.
    
    Args:
        df: Preprocessed loan data
        
    Returns:
        DataFrame with added historical metrics
    """
    # Sort by date for historical calculations
    df = df.sort_values('contract_start_date').copy()
    
    # Calculate historical metrics for each level
    for level in ['region', 'sales_territory', 'area']:
        # For each loan, calculate metrics using only previous loans
        df[f'historical_cum_loans_{level}'] = df.groupby(level).cumcount()
        
        # Create temporary series for calculations
        temp_deposits = df.groupby(level)['deposit_amount'].cumsum()
        temp_value = df.groupby(level)['nominal_contract_value'].cumsum()
        
        # Shift to get values before current loan
        df[f'historical_cum_deposit_{level}'] = temp_deposits.shift(1).fillna(0)
        df[f'historical_cum_value_{level}'] = temp_value.shift(1).fillna(0)
        
        # Count distinct customers before current loan
        df[f'historical_cum_customers_{level}'] = (
            df.groupby(level)
            .apply(lambda x: x.assign(
                count=x.groupby('client_id').cumcount().shift(1).fillna(-1)
            ).apply(lambda row: len(x[
                (x['contract_start_date'] < row['contract_start_date']) & 
                (x['client_id'] != row['client_id'])
            ]['client_id'].unique()), axis=1))
            .reset_index(level=0, drop=True)
        )

def calculate_relative_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate relative position metrics using only historical data.
    
    Args:
        df: Preprocessed loan data
        
    Returns:
        DataFrame with added relative position metrics
    """
    df = df.copy()
    
    # Calculate deposit ratio (available at application time)
    df['deposit_ratio'] = df['deposit_amount'] / df['nominal_contract_value']
    
    # Calculate relative position metrics for each level using only previous loans
    for level in ['region', 'sales_territory', 'area']:
        # Create temporary dataframe for historical rankings
        temp_df = df.sort_values('contract_start_date').copy()
        
        # For each loan, rank against only previous loans
        df[f'deposit_ratio_rank_{level}'] = temp_df.groupby(level).apply(
            lambda x: x['deposit_ratio'].expanding().rank(pct=True)
        ).reset_index(level=0, drop=True)
        
        df[f'contract_value_rank_{level}'] = temp_df.groupby(level).apply(
            lambda x: x['nominal_contract_value'].expanding().rank(pct=True)
        ).reset_index(level=0, drop=True)
        
        # Historical metrics ranks
        for metric in ['loans', 'value', 'deposit']:
            df[f'historical_{metric}_rank_{level}'] = temp_df.groupby(level).apply(
                lambda x: x[f'historical_cum_{metric}_{level}'].expanding().rank(pct=True)
            ).reset_index(level=0, drop=True)
    
    return df

def calculate_infrastructure_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate infrastructure metrics using only data available before each loan.
    
    Args:
        df: Preprocessed loan data
        
    Returns:
        DataFrame with added infrastructure metrics
    """
    df = df.copy()
    
    # Calculate distinct counts for each level using only previous data
    for level in ['region', 'sales_territory', 'area']:
        # For each loan, count distinct entities up to its date
        df[f'distinct_dukas_{level}'] = (
            df.groupby(level)
            .apply(lambda x: x.assign(
                count=x.groupby('duka_name').cumcount()
            ).apply(lambda row: len(x[
                x['contract_start_date'] < row['contract_start_date']
            ]['duka_name'].unique()), axis=1))
            .reset_index(level=0, drop=True)
        )
        
        df[f'distinct_customers_{level}'] = (
            df.groupby(level)
            .apply(lambda x: x.assign(
                count=x.groupby('client_id').cumcount()
            ).apply(lambda row: len(x[
                x['contract_start_date'] < row['contract_start_date']
            ]['client_id'].unique()), axis=1))
            .reset_index(level=0, drop=True)
        )
    
    return df

def calculate_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate temporal features from contract start date.
    These are legitimate features as they're available at application time.
    
    Args:
        df: Preprocessed loan data
        
    Returns:
        DataFrame with added temporal features
    """
    df = df.copy()
    
    # Extract day of week features
    df['contract_day_name'] = df['contract_start_date'].dt.day_name()
    df['contract_start_day'] = df['contract_start_date'].dt.day
    
    # Create day of week encoding
    df['is_weekend'] = df['contract_day_name'].isin(['Saturday', 'Sunday']).astype(int)
    
    return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine all feature engineering steps.
    All features are calculated using only data available at loan application time.
    
    Args:
        df: Preprocessed loan data
        
    Returns:
        DataFrame with all engineered features
    """
    # Add historical cumulative metrics
    df = calculate_historical_metrics(df)
    
    # Add relative position metrics
    df = calculate_relative_metrics(df)
    
    # Add infrastructure metrics
    df = calculate_infrastructure_metrics(df)
    
    # Add temporal features
    df = calculate_temporal_features(df)
    
    return df
