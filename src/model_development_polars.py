"""
OAF Loan Performance Model Development (Polars Implementation)

This module contains functions for feature engineering and model development,
implemented using Polars for high-performance data processing.
All features ensure strict temporal integrity, only using data available
at loan application time.
"""

import pandas as pd
import polars as pl
import numpy as np
from typing import Dict, Any, Optional, Tuple
import time

def prepare_for_polars(df_pandas: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare pandas DataFrame for conversion to Polars by handling unsupported types
    
    Args:
        df_pandas: Original pandas DataFrame
        
    Returns:
        Modified pandas DataFrame ready for Polars conversion
    """
    df = df_pandas.copy()
    
    # Handle period dtype (month column)
    if 'month' in df.columns and hasattr(df['month'], 'dt'):
        # Convert period to string
        df['month'] = df['month'].astype(str)
    
    # Handle any other problematic columns
    for col in df.columns:
        # Check for period dtype
        if hasattr(df[col], 'dt') and hasattr(df[col].dt, 'to_timestamp'):
            df[col] = df[col].dt.to_timestamp()
        
        # Check for other extension types
        if pd.api.types.is_extension_array_dtype(df[col].dtype):
            # Convert to standard numpy type if possible
            try:
                df[col] = df[col].astype('float64')
            except:
                try:
                    df[col] = df[col].astype('int64')
                except:
                    try:
                        df[col] = df[col].astype(str)
                    except:
                        print(f"Warning: Couldn't convert column {col} to Polars-compatible type")
    
    return df

def calculate_historical_metrics(df: pl.DataFrame) -> pl.DataFrame:
    """
    Calculate historical metrics using Polars expressions.
    This ensures strict temporal integrity by only considering loans that
    existed before the current loan's application date.
    
    Args:
        df: Preprocessed loan data in Polars DataFrame format
        
    Returns:
        DataFrame with added historical metrics
    """
    print("Calculating historical metrics (Polars implementation)")
    
    # Sort by date for historical calculations
    df = df.sort("contract_start_date")
    
    # Calculate historical metrics for each level - loop through each level
    for level in ['region', 'sales_territory', 'area']:
        print(f"  Processing {level} level...")
        
        # Process each row individually using apply on the DataFrame
        # We need to do this row by row to maintain the temporal integrity
        
        results = []
        for row in df.iter_rows(named=True):
            current_level = row[level]
            current_date = row["contract_start_date"]
            
            # First, filter for non-null values in the level column
            filtered_df = df.filter(pl.col(level).is_not_null())
            
            # Now filter for same level and earlier dates
            if current_level is not None:
                historical_loans = filtered_df.filter(
                    (pl.col(level) == current_level) & 
                    (pl.col("contract_start_date") < current_date)
                )
            else:
                # If current_level is None, we want loans with null values in this column
                historical_loans = filtered_df.filter(
                    pl.col(level).is_null() & 
                    (pl.col("contract_start_date") < current_date)
                )
            
            # Calculate metrics
            loan_count = len(historical_loans)
            deposit_sum = historical_loans.select(pl.sum("deposit_amount")).item() if loan_count > 0 else 0
            contract_value_sum = historical_loans.select(pl.sum("nominal_contract_value")).item() if loan_count > 0 else 0
            unique_customers = historical_loans.select(pl.col("client_id").n_unique()).item() if loan_count > 0 else 0
            
            # Store the results for this row
            results.append({
                f"historical_cum_loans_{level}": loan_count,
                f"historical_cum_deposit_{level}": float(deposit_sum),
                f"historical_cum_value_{level}": float(contract_value_sum),
                f"historical_cum_customers_{level}": unique_customers
            })
        
        # Add computed columns to dataframe
        for col_name in results[0].keys():
            df = df.with_columns([
                # Use strict=False to allow mixed types
                pl.Series(name=col_name, values=[r[col_name] for r in results], strict=False)
            ])
    
    print("Historical metrics calculation complete")
    return df

def calculate_relative_metrics(df: pl.DataFrame) -> pl.DataFrame:
    """
    Calculate relative position metrics using Polars expressions.
    
    Args:
        df: Preprocessed loan data
        
    Returns:
        DataFrame with added relative position metrics
    """
    print("Calculating relative position metrics...")
    
    # Calculate deposit ratio (available at application time)
    df = df.with_columns([
        (pl.col("deposit_amount") / pl.col("nominal_contract_value")).alias("deposit_ratio")
    ])
    
    # Calculate relative position metrics for each level
    for level in ['region', 'sales_territory', 'area']:
        print(f"  Processing {level} level...")
        
        # Process row by row to maintain temporal integrity
        results_deposit_ratio = []
        results_contract_value = []
        
        for row in df.iter_rows(named=True):
            current_level = row[level]
            current_date = row["contract_start_date"]
            current_deposit_ratio = row["deposit_ratio"]
            current_contract_value = row["nominal_contract_value"]
            
            # First, filter for non-null values in the level column
            filtered_df = df.filter(pl.col(level).is_not_null())
            
            # Now filter for same level and earlier dates
            if current_level is not None:
                historical_loans = filtered_df.filter(
                    (pl.col(level) == current_level) & 
                    (pl.col("contract_start_date") < current_date)
                )
            else:
                # If current_level is None, we want loans with null values in this column
                historical_loans = filtered_df.filter(
                    pl.col(level).is_null() & 
                    (pl.col("contract_start_date") < current_date)
                )
            
            if len(historical_loans) == 0:
                # No historical loans, so rank is 0
                deposit_ratio_rank = 0.0
                contract_value_rank = 0.0
            else:
                # Calculate ranks
                deposit_ratio_rank = float(len(historical_loans.filter(
                    pl.col("deposit_ratio") < current_deposit_ratio
                )) / len(historical_loans))
                
                contract_value_rank = float(len(historical_loans.filter(
                    pl.col("nominal_contract_value") < current_contract_value
                )) / len(historical_loans))
            
            # Store results
            results_deposit_ratio.append(deposit_ratio_rank)
            results_contract_value.append(contract_value_rank)
        
        # Add columns to dataframe
        df = df.with_columns([
            pl.Series(name=f"deposit_ratio_rank_{level}", values=results_deposit_ratio, strict=False),
            pl.Series(name=f"contract_value_rank_{level}", values=results_contract_value, strict=False)
        ])
        
        # Historical metrics ranks
        for metric in ['loans', 'value', 'deposit']:
            metric_col = f"historical_cum_{metric}_{level}"
            
            # Skip if the column doesn't exist yet
            if metric_col not in df.columns:
                continue
                
            # Calculate ranks
            results_metric_rank = []
            
            for row in df.iter_rows(named=True):
                current_level = row[level]
                current_date = row["contract_start_date"]
                current_metric_value = row[metric_col]
                
                # First, filter for non-null values in the level column
                filtered_df = df.filter(pl.col(level).is_not_null())
                
                # Now filter for same level and earlier dates
                if current_level is not None:
                    historical_loans = filtered_df.filter(
                        (pl.col(level) == current_level) & 
                        (pl.col("contract_start_date") < current_date)
                    )
                else:
                    # If current_level is None, we want loans with null values in this column
                    historical_loans = filtered_df.filter(
                        pl.col(level).is_null() & 
                        (pl.col("contract_start_date") < current_date)
                    )
                
                if len(historical_loans) == 0:
                    metric_rank = 0.0
                else:
                    metric_rank = float(len(historical_loans.filter(
                        pl.col(metric_col) < current_metric_value
                    )) / len(historical_loans))
                
                results_metric_rank.append(metric_rank)
            
            df = df.with_columns([
                pl.Series(name=f"historical_{metric}_rank_{level}", values=results_metric_rank, strict=False)
            ])
    
    print("Relative position metrics calculation complete")
    return df

def calculate_infrastructure_metrics(df: pl.DataFrame) -> pl.DataFrame:
    """
    Calculate infrastructure metrics using Polars expressions.
    
    Args:
        df: Preprocessed loan data
        
    Returns:
        DataFrame with added infrastructure metrics
    """
    print("Calculating infrastructure metrics...")
    
    # Calculate distinct counts for each level
    for level in ['region', 'sales_territory', 'area']:
        print(f"  Processing {level} level...")
        
        # Process row by row
        results_dukas = []
        results_customers = []
        
        for row in df.iter_rows(named=True):
            current_level = row[level]
            current_date = row["contract_start_date"]
            
            # First, filter for non-null values in the level column
            filtered_df = df.filter(pl.col(level).is_not_null())
            
            # Now filter for same level and earlier dates
            if current_level is not None:
                historical_loans = filtered_df.filter(
                    (pl.col(level) == current_level) & 
                    (pl.col("contract_start_date") < current_date)
                )
            else:
                # If current_level is None, we want loans with null values in this column
                historical_loans = filtered_df.filter(
                    pl.col(level).is_null() & 
                    (pl.col("contract_start_date") < current_date)
                )
            
            # Calculate distinct counts
            if len(historical_loans) == 0:
                distinct_dukas = 0
                distinct_customers = 0
            else:
                distinct_dukas = historical_loans.select(pl.col("duka_name").n_unique()).item()
                distinct_customers = historical_loans.select(pl.col("client_id").n_unique()).item()
            
            # Store results
            results_dukas.append(distinct_dukas)
            results_customers.append(distinct_customers)
        
        # Add columns to dataframe
        df = df.with_columns([
            pl.Series(name=f"distinct_dukas_{level}", values=results_dukas, strict=False),
            pl.Series(name=f"distinct_customers_{level}", values=results_customers, strict=False)
        ])
    
    print("Infrastructure metrics calculation complete")
    return df

def calculate_temporal_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Calculate temporal features from contract start date using Polars expressions.
    
    Args:
        df: Preprocessed loan data
        
    Returns:
        DataFrame with added temporal features
    """
    print("Calculating temporal features...")
    
    # Day name mapping
    day_name_map = {
        0: "Monday",
        1: "Tuesday",
        2: "Wednesday",
        3: "Thursday",
        4: "Friday",
        5: "Saturday",
        6: "Sunday",
        7: "Sunday"  # Adding 7 as an additional mapping for Sunday (some systems use 0-6, others 1-7)
    }
    
    # Extract day of week features
    df = df.with_columns([
        pl.col("contract_start_date").dt.day().alias("contract_start_day"),
        pl.col("contract_start_date").dt.month().alias("contract_month"),
        pl.col("contract_start_date").dt.quarter().alias("contract_quarter"),
        pl.col("contract_start_date").dt.weekday().alias("weekday_num")
    ])
    
    # Map weekday numbers to day names - with safety check
    weekday_nums = df.select(pl.col("weekday_num")).to_series().to_list()
    
    # Safely map weekday numbers to day names
    day_names = []
    for num in weekday_nums:
        if num in day_name_map:
            day_names.append(day_name_map[num])
        else:
            # Default to "Unknown" for any unexpected weekday numbers
            print(f"Warning: Unexpected weekday number {num}, using 'Unknown'")
            day_names.append("Unknown")
    
    # Determine weekend days (both 0/6 and 1/7 systems)
    is_weekend = [1 if (num == 5 or num == 6 or num == 7) else 0 for num in weekday_nums]
    
    # Add day names and weekend flag
    df = df.with_columns([
        pl.Series(name="contract_day_name", values=day_names, strict=False),
        pl.Series(name="is_weekend", values=is_weekend, strict=False)
    ])
    
    # Drop temporary column
    df = df.drop("weekday_num")
    
    print("Temporal features calculation complete")
    return df

def calculate_post_sept_metrics(df: pl.DataFrame, is_holdout: bool = False) -> pl.DataFrame:
    """
    Calculate metrics using post-application September data.
    These features are prefixed with 'post_' to clearly indicate they are
    post-application features that weren't available at loan decision time
    and should only be used for analysis, not in production models.
    
    Args:
        df: Preprocessed loan data
        is_holdout: Whether this is the holdout dataset (affects feature calculation)
        
    Returns:
        DataFrame with added post-application metrics
    """
    if is_holdout:
        print("Calculating post-September metrics (holdout dataset)...")
    else:
        print("Calculating post-September metrics (training dataset)...")
    
    # Skip if we don't have the required columns
    if "cumulative_amount_paid_start" not in df.columns or "nominal_contract_value" not in df.columns:
        print("  Warning: Required columns for post-September metrics not found, skipping")
        return df
    
    # Add September repayment rate
    df = df.with_columns([
        (pl.col("cumulative_amount_paid_start") / pl.col("nominal_contract_value")).alias("post_sept_repayment_rate")
    ])
    
    # Calculate days since contract start to September 1, 2023
    if "contract_start_date" in df.columns:
        # Use string parsing approach which is compatible with all Polars versions
        try:
            # Create reference date directly in the calculation
            sept_1_2023_str = "2023-09-01"
            
            # Calculate days between dates - using different methods based on Polars version
            # Use a simpler approach with datetime conversion
            # Convert contract start date to Unix timestamp (seconds)
            df = df.with_columns([
                pl.col("contract_start_date").dt.timestamp().alias("contract_timestamp")
            ])
            
            # Hard-code the September 1, 2023 timestamp (Unix seconds)
            sept_timestamp = 1693526400  # Sept 1, 2023 00:00:00 UTC timestamp
            
            # Calculate days difference by timestamp subtraction and conversion
            df = df.with_columns([
                ((sept_timestamp - pl.col("contract_timestamp")) / 86400).alias("post_days_to_sept")
            ])
            
            # Drop temporary column
            df = df.drop("contract_timestamp")
        except Exception as e:
            # If all else fails, use a simple fallback method
            print(f"  Warning: Date calculation error - {str(e)}")
            print("  Using fallback date calculation method")
            
            # Use a very basic estimate - add a constant value
            df = df.with_columns([
                pl.lit(180).alias("post_days_to_sept")  # 6 months as a rough estimate
            ])
    
        # Calculate payment velocity (amount paid per day)
        # Need to handle the case where days could be zero or negative (shouldn't happen but for safety)
        df = df.with_columns([
            (pl.col("cumulative_amount_paid_start") / 
             pl.when(pl.col("post_days_to_sept") > 0)
             .then(pl.col("post_days_to_sept"))
             .otherwise(1))
             .alias("post_payment_velocity")
        ])
    
    # Calculate payment ratio (how much of the contract value was paid by September)
    df = df.with_columns([
        (pl.col("cumulative_amount_paid_start") / pl.col("nominal_contract_value")).alias("post_payment_ratio")
    ])
    
    # Calculate the proportion of deposit to total paid by September
    # Handle the case where amount paid could be zero
    df = df.with_columns([
        (pl.col("deposit_amount") / 
         pl.when(pl.col("cumulative_amount_paid_start") > 0)
         .then(pl.col("cumulative_amount_paid_start"))
         .otherwise(1))
         .alias("post_deposit_to_paid_ratio")
    ])
    
    # If we also have November data (cumulative_amount_paid), calculate cure rate
    if "cumulative_amount_paid" in df.columns:
        df = df.with_columns([
            (pl.col("cumulative_amount_paid") / pl.col("nominal_contract_value")).alias("post_nov_repayment_rate"),
            ((pl.col("cumulative_amount_paid") - pl.col("cumulative_amount_paid_start")) / 
             pl.col("nominal_contract_value")).alias("post_sept_to_nov_increase"),
        ])
        
        # Calculate cure rate (increase in repayment rate from Sept to Nov)
        df = df.with_columns([
            (pl.col("post_nov_repayment_rate") - pl.col("post_sept_repayment_rate")).alias("post_cure_rate")
        ])
    
    print("Post-September metrics calculation complete")
    return df

def engineer_features(df_pandas: pd.DataFrame, sample_size: Optional[int] = None, is_holdout: bool = False) -> pd.DataFrame:
    """
    Combine all feature engineering steps using Polars for high performance.
    Features are divided into:
    1. Application-time features (available at loan decision time)
    2. Post-application features (prefixed with 'post_', for analysis only)
    
    Args:
        df_pandas: Preprocessed loan data (pandas DataFrame)
        sample_size: Optional sample size for testing (limits rows processed)
        is_holdout: Whether this is the holdout dataset (affects feature calculation)
        
    Returns:
        pandas DataFrame with all engineered features
    """
    start_time = time.time()
    
    # Take a sample if requested (for testing)
    if sample_size is not None and sample_size < len(df_pandas):
        df_pandas = df_pandas.head(sample_size).copy()
        print(f"Using sample of {sample_size} loans")
    else:
        df_pandas = df_pandas.copy()
        print(f"Processing all {len(df_pandas)} loans")
    
    # Log whether we're processing holdout data
    if is_holdout:
        print("Processing HOLDOUT dataset (no November data available)")
    else:
        print("Processing TRAINING dataset (with November data)")
    
    # Prepare data for Polars conversion
    df_pandas = prepare_for_polars(df_pandas)
    
    # Convert to Polars DataFrame
    print("Converting pandas DataFrame to Polars...")
    df = pl.from_pandas(df_pandas)
    
    print(f"\nStarting feature engineering with Polars...")
    
    # Add historical cumulative metrics
    df = calculate_historical_metrics(df)
    
    # Add relative position metrics
    df = calculate_relative_metrics(df)
    
    # Add infrastructure metrics
    df = calculate_infrastructure_metrics(df)
    
    # Add temporal features
    df = calculate_temporal_features(df)
    
    # Add post-September metrics (for insights, not for production models)
    # Pass the is_holdout flag to the function
    df = calculate_post_sept_metrics(df, is_holdout=is_holdout)
    
    # Convert back to pandas
    print("Converting results back to pandas DataFrame...")
    result_df = df.to_pandas()
    
    end_time = time.time()
    print(f"\nFeature engineering complete! Generated {len(result_df.columns)} features in {end_time - start_time:.2f} seconds.")
    
    # Add a column to identify which features are post-application (for documentation)
    post_features = [col for col in result_df.columns if col.startswith('post_')]
    if post_features:
        print(f"\nGenerated {len(post_features)} post-application features (prefixed with 'post_'):")
        for feature in post_features[:5]:
            print(f"  - {feature}")
        if len(post_features) > 5:
            print(f"  - ... and {len(post_features) - 5} more")
        
        print("\nWARNING: Post-application features are only available for analysis:")
        print("  - These features use data collected AFTER loan application")
        print("  - They should NOT be used in production models")
        print("  - Only use them for insights and understanding repayment patterns")
    
    return result_df

def create_sample_dataset(df: pd.DataFrame, save_path: str = "data/processed/feature_sample.csv") -> None:
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
