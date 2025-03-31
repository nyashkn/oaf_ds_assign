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
    Calculate historical metrics using explicit filtering for each loan.
    This ensures strict temporal integrity by only considering loans that
    existed before the current loan's application date.
    
    Args:
        df: Preprocessed loan data
        
    Returns:
        DataFrame with added historical metrics
    """
    # Sort by date for historical calculations
    df = df.sort_values('contract_start_date').copy()
    
    print("Calculating historical metrics (direct filtering method)")
    print("This may take a few minutes for large datasets...")
    
    # Calculate historical metrics for each level
    for level in ['region', 'sales_territory', 'area']:
        print(f"  Processing {level} level...")
        
        # Explicitly count previous loans for each row
        df[f'historical_cum_loans_{level}'] = df.apply(
            lambda row: df[
                (df[level] == row[level]) & 
                (df['contract_start_date'] < row['contract_start_date'])
            ].shape[0], 
            axis=1
        )
        
        # Explicitly sum deposits for previous loans
        df[f'historical_cum_deposit_{level}'] = df.apply(
            lambda row: df[
                (df[level] == row[level]) & 
                (df['contract_start_date'] < row['contract_start_date'])
            ]['deposit_amount'].sum(), 
            axis=1
        )
        
        # Explicitly sum contract values for previous loans
        df[f'historical_cum_value_{level}'] = df.apply(
            lambda row: df[
                (df[level] == row[level]) & 
                (df['contract_start_date'] < row['contract_start_date'])
            ]['nominal_contract_value'].sum(), 
            axis=1
        )
        
        # Count distinct customers before current loan
        df[f'historical_cum_customers_{level}'] = df.apply(
            lambda row: df[
                (df[level] == row[level]) & 
                (df['contract_start_date'] < row['contract_start_date'])
            ]['client_id'].nunique(), 
            axis=1
        )
    
    print("Historical metrics calculation complete")
    return df

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
    
    print("Calculating relative position metrics...")
    
    # Calculate relative position metrics for each level
    for level in ['region', 'sales_territory', 'area']:
        print(f"  Processing {level} level...")
        
        # Deposit ratio rank
        df[f'deposit_ratio_rank_{level}'] = df.apply(
            lambda row: len(df[
                (df[level] == row[level]) & 
                (df['contract_start_date'] < row['contract_start_date']) &
                (df['deposit_ratio'] < row['deposit_ratio'])
            ]) / max(1, df[
                (df[level] == row[level]) & 
                (df['contract_start_date'] < row['contract_start_date'])
            ].shape[0]),
            axis=1
        )
        
        # Contract value rank
        df[f'contract_value_rank_{level}'] = df.apply(
            lambda row: len(df[
                (df[level] == row[level]) & 
                (df['contract_start_date'] < row['contract_start_date']) &
                (df['nominal_contract_value'] < row['nominal_contract_value'])
            ]) / max(1, df[
                (df[level] == row[level]) & 
                (df['contract_start_date'] < row['contract_start_date'])
            ].shape[0]),
            axis=1
        )
        
        # Historical metrics ranks
        for metric in ['loans', 'value', 'deposit']:
            metric_col = f'historical_cum_{metric}_{level}'
            
            # Skip if the column doesn't exist yet
            if metric_col not in df.columns:
                continue
                
            # Calculate percentile rank
            df[f'historical_{metric}_rank_{level}'] = df.apply(
                lambda row: len(df[
                    (df[level] == row[level]) & 
                    (df['contract_start_date'] < row['contract_start_date']) &
                    (df[metric_col] < row[metric_col])
                ]) / max(1, df[
                    (df[level] == row[level]) & 
                    (df['contract_start_date'] < row['contract_start_date'])
                ].shape[0]),
                axis=1
            )
    
    print("Relative position metrics calculation complete")
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
    
    print("Calculating infrastructure metrics...")
    
    # Calculate distinct counts for each level using only previous data
    for level in ['region', 'sales_territory', 'area']:
        print(f"  Processing {level} level...")
        
        # Count distinct dukas before each loan
        df[f'distinct_dukas_{level}'] = df.apply(
            lambda row: df[
                (df[level] == row[level]) & 
                (df['contract_start_date'] < row['contract_start_date'])
            ]['duka_name'].nunique(),
            axis=1
        )
        
        # Count distinct customers before each loan
        df[f'distinct_customers_{level}'] = df.apply(
            lambda row: df[
                (df[level] == row[level]) & 
                (df['contract_start_date'] < row['contract_start_date'])
            ]['client_id'].nunique(),
            axis=1
        )
    
    print("Infrastructure metrics calculation complete")
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
    
    print("Calculating temporal features...")
    
    # Extract day of week features
    df['contract_day_name'] = df['contract_start_date'].dt.day_name()
    df['contract_start_day'] = df['contract_start_date'].dt.day
    
    # Create day of week encoding
    df['is_weekend'] = df['contract_day_name'].isin(['Saturday', 'Sunday']).astype(int)
    
    # Month features
    df['contract_month'] = df['contract_start_date'].dt.month
    df['contract_quarter'] = df['contract_start_date'].dt.quarter
    
    print("Temporal features calculation complete")
    return df

def engineer_features(df: pd.DataFrame, sample_size: Optional[int] = None) -> pd.DataFrame:
    """
    Combine all feature engineering steps.
    All features are calculated using only data available at loan application time.
    
    Args:
        df: Preprocessed loan data
        sample_size: Optional sample size for testing (limits rows processed)
        
    Returns:
        DataFrame with all engineered features
    """
    # Take a sample if requested (for testing)
    if sample_size is not None:
        df = df.head(sample_size).copy()
        print(f"Using sample of {sample_size} loans")
    else:
        df = df.copy()
        print(f"Processing all {len(df)} loans")
    
    print("\nStarting feature engineering...")
    
    # Add historical cumulative metrics
    df = calculate_historical_metrics(df)
    
    # Add relative position metrics
    df = calculate_relative_metrics(df)
    
    # Add infrastructure metrics
    df = calculate_infrastructure_metrics(df)
    
    # Add temporal features
    df = calculate_temporal_features(df)
    
    print(f"\nFeature engineering complete! Generated {len(df.columns)} features.")
    return df

def create_sample_dataset(df: pd.DataFrame, save_path: str = "../data/processed/feature_sample.csv") -> None:
    """
    Create a small sample dataset with engineered features for testing.
    
    Args:
        df: Original preprocessed data
        save_path: Path to save the sample dataset
    """
    print("Creating sample dataset with features...")
    sample = df.sample(n=100, random_state=42)
    sample_with_features = engineer_features(sample)
    sample_with_features.to_csv(save_path, index=False)
    print(f"Sample dataset saved to {save_path}")
