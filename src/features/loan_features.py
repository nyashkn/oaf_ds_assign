# src/features/loan_features.py

import pandas as pd
import polars as pl
import numpy as np
from typing import Dict, Any, Optional, List, Union

from .feature_registry import feature, feature_group, feature_registry,target

# Basic helper function (not a feature itself)
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

# Features

@feature(
    description="Ratio of deposit amount to nominal contract value",
    entity_id="loan_id",
    time_reference="contract_start_date"
)
def deposit_ratio(df: pl.DataFrame) -> pl.DataFrame:
    """Calculate the deposit ratio for each loan"""
    return df.with_columns([
        (pl.col("deposit_amount") / pl.col("nominal_contract_value")).alias("deposit_ratio")
    ])

@feature(
    description="Historical cumulative loan count for region",
    entity_id="region",
    time_reference="contract_start_date",
    relative_time=True
)
def historical_cum_loans_region(df: pl.DataFrame) -> pl.DataFrame:
    """
    Calculate the historical cumulative number of loans for each region.
    This ensures strict temporal integrity by only considering loans that
    existed before the current loan's application date.
    """
    results = []
    
    for row in df.iter_rows(named=True):
        current_region = row["region"]
        current_date = row["contract_start_date"]
        
        # First, filter for non-null values in the region column
        filtered_df = df.filter(pl.col("region").is_not_null())
        
        # Now filter for same region and earlier dates
        if current_region is not None:
            historical_loans = filtered_df.filter(
                (pl.col("region") == current_region) & 
                (pl.col("contract_start_date") < current_date)
            )
        else:
            # If current_region is None, we want loans with null values in this column
            historical_loans = filtered_df.filter(
                pl.col("region").is_null() & 
                (pl.col("contract_start_date") < current_date)
            )
        
        # Calculate metric
        loan_count = len(historical_loans)
        
        # Store the result for this row
        results.append(loan_count)
    
    # Add computed column to dataframe
    return df.with_columns([
        pl.Series(name="historical_cum_loans_region", values=results, strict=False)
    ])

@feature(
    description="Historical cumulative deposit sum for region",
    entity_id="region",
    time_reference="contract_start_date",
    relative_time=True,
    dependencies=["historical_cum_loans_region"]
)
def historical_cum_deposit_region(df: pl.DataFrame) -> pl.DataFrame:
    """
    Calculate the historical cumulative deposit amount for each region.
    """
    results = []
    
    for row in df.iter_rows(named=True):
        current_region = row["region"]
        current_date = row["contract_start_date"]
        
        # First, filter for non-null values in the region column
        filtered_df = df.filter(pl.col("region").is_not_null())
        
        # Now filter for same region and earlier dates
        if current_region is not None:
            historical_loans = filtered_df.filter(
                (pl.col("region") == current_region) & 
                (pl.col("contract_start_date") < current_date)
            )
        else:
            # If current_region is None, we want loans with null values in this column
            historical_loans = filtered_df.filter(
                pl.col("region").is_null() & 
                (pl.col("contract_start_date") < current_date)
            )
        
        # Calculate metric
        deposit_sum = historical_loans.select(pl.sum("deposit_amount")).item() if len(historical_loans) > 0 else 0
        
        # Store the result for this row
        results.append(float(deposit_sum))
    
    # Add computed column to dataframe
    return df.with_columns([
        pl.Series(name="historical_cum_deposit_region", values=results, strict=False)
    ])

@feature(
    description="Historical cumulative contract value sum for region",
    entity_id="region",
    time_reference="contract_start_date",
    relative_time=True,
    dependencies=["historical_cum_loans_region"]
)
def historical_cum_value_region(df: pl.DataFrame) -> pl.DataFrame:
    """
    Calculate the historical cumulative contract value for each region.
    """
    results = []
    
    for row in df.iter_rows(named=True):
        current_region = row["region"]
        current_date = row["contract_start_date"]
        
        # First, filter for non-null values in the region column
        filtered_df = df.filter(pl.col("region").is_not_null())
        
        # Now filter for same region and earlier dates
        if current_region is not None:
            historical_loans = filtered_df.filter(
                (pl.col("region") == current_region) & 
                (pl.col("contract_start_date") < current_date)
            )
        else:
            # If current_region is None, we want loans with null values in this column
            historical_loans = filtered_df.filter(
                pl.col("region").is_null() & 
                (pl.col("contract_start_date") < current_date)
            )
        
        # Calculate metric
        value_sum = historical_loans.select(pl.sum("nominal_contract_value")).item() if len(historical_loans) > 0 else 0
        
        # Store the result for this row
        results.append(float(value_sum))
    
    # Add computed column to dataframe
    return df.with_columns([
        pl.Series(name="historical_cum_value_region", values=results, strict=False)
    ])

@feature(
    description="Historical unique customers count for region",
    entity_id="region",
    time_reference="contract_start_date",
    relative_time=True,
    dependencies=["historical_cum_loans_region"]
)
def historical_cum_customers_region(df: pl.DataFrame) -> pl.DataFrame:
    """
    Calculate the historical unique customer count for each region.
    """
    results = []
    
    for row in df.iter_rows(named=True):
        current_region = row["region"]
        current_date = row["contract_start_date"]
        
        # First, filter for non-null values in the region column
        filtered_df = df.filter(pl.col("region").is_not_null())
        
        # Now filter for same region and earlier dates
        if current_region is not None:
            historical_loans = filtered_df.filter(
                (pl.col("region") == current_region) & 
                (pl.col("contract_start_date") < current_date)
            )
        else:
            # If current_region is None, we want loans with null values in this column
            historical_loans = filtered_df.filter(
                pl.col("region").is_null() & 
                (pl.col("contract_start_date") < current_date)
            )
        
        # Calculate metric
        unique_customers = historical_loans.select(pl.col("client_id").n_unique()).item() if len(historical_loans) > 0 else 0
        
        # Store the result for this row
        results.append(unique_customers)
    
    # Add computed column to dataframe
    return df.with_columns([
        pl.Series(name="historical_cum_customers_region", values=results, strict=False)
    ])

@feature(
    description="Deposit ratio rank within region",
    entity_id="loan_id",
    time_reference="contract_start_date",
    relative_time=True,
    dependencies=["deposit_ratio"]
)
def deposit_ratio_rank_region(df: pl.DataFrame) -> pl.DataFrame:
    """
    Calculate the percentile rank of deposit ratio within the region, 
    based on historical loans.
    """
    results = []
    
    for row in df.iter_rows(named=True):
        current_region = row["region"]
        current_date = row["contract_start_date"]
        current_deposit_ratio = row["deposit_ratio"]
        
        # First, filter for non-null values in the region column
        filtered_df = df.filter(pl.col("region").is_not_null())
        
        # Now filter for same region and earlier dates
        if current_region is not None:
            historical_loans = filtered_df.filter(
                (pl.col("region") == current_region) & 
                (pl.col("contract_start_date") < current_date)
            )
        else:
            # If current_region is None, we want loans with null values in this column
            historical_loans = filtered_df.filter(
                pl.col("region").is_null() & 
                (pl.col("contract_start_date") < current_date)
            )
        
        if len(historical_loans) == 0:
            # No historical loans, so rank is 0
            deposit_ratio_rank = 0.0
        else:
            # Calculate rank
            deposit_ratio_rank = float(len(historical_loans.filter(
                pl.col("deposit_ratio") < current_deposit_ratio
            )) / len(historical_loans))
        
        # Store result
        results.append(deposit_ratio_rank)
    
    # Add column to dataframe
    return df.with_columns([
        pl.Series(name="deposit_ratio_rank_region", values=results, strict=False)
    ])

@feature(
    description="Contract value rank within region",
    entity_id="loan_id",
    time_reference="contract_start_date",
    relative_time=True
)
def contract_value_rank_region(df: pl.DataFrame) -> pl.DataFrame:
    """
    Calculate the percentile rank of contract value within the region,
    based on historical loans.
    """
    results = []
    
    for row in df.iter_rows(named=True):
        current_region = row["region"]
        current_date = row["contract_start_date"]
        current_contract_value = row["nominal_contract_value"]
        
        # First, filter for non-null values in the region column
        filtered_df = df.filter(pl.col("region").is_not_null())
        
        # Now filter for same region and earlier dates
        if current_region is not None:

            historical_loans = filtered_df.filter(
                (pl.col("region") == current_region) & 
                (pl.col("contract_start_date") < current_date)
            )
        else:
            # If current_region is None, we want loans with null values in this column
            historical_loans = filtered_df.filter(
                pl.col("region").is_null() & 
                (pl.col("contract_start_date") < current_date)
            )
        
        if len(historical_loans) == 0:
            # No historical loans, so rank is 0
            contract_value_rank = 0.0
        else:
            # Calculate rank
            contract_value_rank = float(len(historical_loans.filter(
                pl.col("nominal_contract_value") < current_contract_value
            )) / len(historical_loans))
        
        # Store result
        results.append(contract_value_rank)
    
    # Add column to dataframe
    return df.with_columns([
        pl.Series(name="contract_value_rank_region", values=results, strict=False)
    ])

@feature(
    description="Number of distinct dukas in the region",
    entity_id="region",
    time_reference="contract_start_date",
    relative_time=True
)
def distinct_dukas_region(df: pl.DataFrame) -> pl.DataFrame:
    """
    Calculate the number of distinct dukas in the region,
    based on historical loans.
    """
    results = []
    
    for row in df.iter_rows(named=True):
        current_region = row["region"]
        current_date = row["contract_start_date"]
        
        # First, filter for non-null values in the region column
        filtered_df = df.filter(pl.col("region").is_not_null())
        
        # Now filter for same region and earlier dates
        if current_region is not None:
            historical_loans = filtered_df.filter(
                (pl.col("region") == current_region) & 
                (pl.col("contract_start_date") < current_date)
            )
        else:
            # If current_region is None, we want loans with null values in this column
            historical_loans = filtered_df.filter(
                pl.col("region").is_null() & 
                (pl.col("contract_start_date") < current_date)
            )
        
        # Calculate distinct counts
        if len(historical_loans) == 0:
            distinct_dukas = 0
        else:
            distinct_dukas = historical_loans.select(pl.col("duka_name").n_unique()).item()
        
        # Store result
        results.append(distinct_dukas)
    
    # Add column to dataframe
    return df.with_columns([
        pl.Series(name="distinct_dukas_region", values=results, strict=False)
    ])

@feature(
    description="Number of distinct customers in the region",
    entity_id="region",
    time_reference="contract_start_date",
    relative_time=True
)
def distinct_customers_region(df: pl.DataFrame) -> pl.DataFrame:
    """
    Calculate the number of distinct customers in the region,
    based on historical loans.
    """
    results = []
    
    for row in df.iter_rows(named=True):
        current_region = row["region"]
        current_date = row["contract_start_date"]
        
        # First, filter for non-null values in the region column
        filtered_df = df.filter(pl.col("region").is_not_null())
        
        # Now filter for same region and earlier dates
        if current_region is not None:
            historical_loans = filtered_df.filter(
                (pl.col("region") == current_region) & 
                (pl.col("contract_start_date") < current_date)
            )
        else:
            # If current_region is None, we want loans with null values in this column
            historical_loans = filtered_df.filter(
                pl.col("region").is_null() & 
                (pl.col("contract_start_date") < current_date)
            )
        
        # Calculate distinct counts
        if len(historical_loans) == 0:
            distinct_customers = 0
        else:
            distinct_customers = historical_loans.select(pl.col("client_id").n_unique()).item()
        
        # Store result
        results.append(distinct_customers)
    
    # Add column to dataframe
    return df.with_columns([
        pl.Series(name="distinct_customers_region", values=results, strict=False)
    ])

@feature(
    description="Day of month for contract start date",
    entity_id="loan_id",
    time_reference="contract_start_date"
)
def contract_start_day(df: pl.DataFrame) -> pl.DataFrame:
    """Extract the day of month from contract start date"""
    return df.with_columns([
        pl.col("contract_start_date").dt.day().alias("contract_start_day")
    ])

@feature(
    description="Month of contract start date",
    entity_id="loan_id",
    time_reference="contract_start_date"
)
def contract_month(df: pl.DataFrame) -> pl.DataFrame:
    """Extract the month from contract start date"""
    return df.with_columns([
        pl.col("contract_start_date").dt.month().alias("contract_month")
    ])

@feature(
    description="Quarter of contract start date",
    entity_id="loan_id",
    time_reference="contract_start_date"
)
def contract_quarter(df: pl.DataFrame) -> pl.DataFrame:
    """Extract the quarter from contract start date"""
    return df.with_columns([
        pl.col("contract_start_date").dt.quarter().alias("contract_quarter")
    ])

@feature(
    description="Name of weekday for contract start date",
    entity_id="loan_id",
    time_reference="contract_start_date"
)
def contract_day_name(df: pl.DataFrame) -> pl.DataFrame:
    """Extract the day name from contract start date"""
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
    
    # Get weekday numbers first
    df_with_weekday = df.with_columns([
        pl.col("contract_start_date").dt.weekday().alias("weekday_num")
    ])
    
    # Map weekday numbers to day names
    weekday_nums = df_with_weekday.select(pl.col("weekday_num")).to_series().to_list()
    
    # Safely map weekday numbers to day names
    day_names = []
    for num in weekday_nums:
        if num in day_name_map:
            day_names.append(day_name_map[num])
        else:
            # Default to "Unknown" for any unexpected weekday numbers
            day_names.append("Unknown")
    
    # Add day names and drop temporary column
    result = df_with_weekday.with_columns([
        pl.Series(name="contract_day_name", values=day_names, strict=False)
    ])
    
    return result.drop("weekday_num")

@feature(
    description="Flag indicating if contract start date is on a weekend",
    entity_id="loan_id",
    time_reference="contract_start_date"
)
def is_weekend(df: pl.DataFrame) -> pl.DataFrame:
    """Determine if contract start date is on a weekend"""
    # Get weekday numbers first
    df_with_weekday = df.with_columns([
        pl.col("contract_start_date").dt.weekday().alias("weekday_num")
    ])
    
    # Map weekday numbers to weekend flag
    weekday_nums = df_with_weekday.select(pl.col("weekday_num")).to_series().to_list()
    
    # Determine weekend days (both 0/6 and 1/7 systems)
    is_weekend_values = [1 if (num == 5 or num == 6 or num == 7) else 0 for num in weekday_nums]
    
    # Add weekend flag and drop temporary column
    result = df_with_weekday.with_columns([
        pl.Series(name="is_weekend", values=is_weekend_values, strict=False)
    ])
    
    return result.drop("weekday_num")


# Feature Groups

@feature_group("historical_metrics", "Historical metrics based on data available at loan application time")
def calculate_historical_metrics(df: pl.DataFrame) -> pl.DataFrame:
    """
    Calculate historical metrics using Polars expressions.
    This ensures strict temporal integrity by only considering loans that
    existed before the current loan's application date.
    """
    print("Calculating historical metrics (Feature Registry implementation)")
    
    # Apply individual historical metrics features
    df = feature_registry.apply_feature(df, "historical_cum_loans_region")
    df = feature_registry.apply_feature(df, "historical_cum_deposit_region")
    df = feature_registry.apply_feature(df, "historical_cum_value_region")
    df = feature_registry.apply_feature(df, "historical_cum_customers_region")
    
    print("Historical metrics calculation complete")
    return df

@feature_group("relative_metrics", "Relative metrics that show ranking within groups")
def calculate_relative_metrics(df: pl.DataFrame) -> pl.DataFrame:
    """
    Calculate relative position metrics.
    """
    print("Calculating relative position metrics...")
    
    # Apply deposit ratio feature
    df = feature_registry.apply_feature(df, "deposit_ratio")
    
    # Apply rank metrics
    df = feature_registry.apply_feature(df, "deposit_ratio_rank_region")
    df = feature_registry.apply_feature(df, "contract_value_rank_region")
    
    print("Relative position metrics calculation complete")
    return df

@feature_group("infrastructure_metrics", "Metrics related to physical infrastructure")
def calculate_infrastructure_metrics(df: pl.DataFrame) -> pl.DataFrame:
    """
    Calculate infrastructure metrics.
    """
    print("Calculating infrastructure metrics...")
    
    # Apply infrastructure metrics
    df = feature_registry.apply_feature(df, "distinct_dukas_region")
    df = feature_registry.apply_feature(df, "distinct_customers_region")
    
    print("Infrastructure metrics calculation complete")
    return df

@feature_group("temporal_features", "Features based on time dimensions")
def calculate_temporal_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Calculate temporal features from contract start date.
    """
    print("Calculating temporal features...")
    
    # Apply temporal features
    df = feature_registry.apply_feature(df, "contract_start_day")
    df = feature_registry.apply_feature(df, "contract_month")
    df = feature_registry.apply_feature(df, "contract_quarter")
    df = feature_registry.apply_feature(df, "contract_day_name")
    df = feature_registry.apply_feature(df, "is_weekend")
    
    print("Temporal features calculation complete")
    return df

# Target variable
@target(
    description="Repayment rate on November 1, 2023",
    time_point="2023-11-01"
)
def tar_sept_23_repayment_rate(df: pl.DataFrame) -> pl.DataFrame:
    """
    Calculate the September 2023 repayment rate.
    This is the target variable for prediction.
    """
    # Note: In a real implementation this would calculate the repayment rate,
    # but for prediction purposes we use the existing column
    
    # Just ensure the column exists
    if "sept_23_repayment_rate" not in df.columns:
        print("Warning: Target column 'sept_23_repayment_rate' not found in DataFrame")
        # Return unchanged DataFrame
        return df
    
    return df

# Add features to appropriate groups
feature_registry.add_to_group("historical_metrics", "historical_cum_loans_region")
feature_registry.add_to_group("historical_metrics", "historical_cum_deposit_region")
feature_registry.add_to_group("historical_metrics", "historical_cum_value_region")
feature_registry.add_to_group("historical_metrics", "historical_cum_customers_region")

feature_registry.add_to_group("relative_metrics", "deposit_ratio")
feature_registry.add_to_group("relative_metrics", "deposit_ratio_rank_region")
feature_registry.add_to_group("relative_metrics", "contract_value_rank_region")

feature_registry.add_to_group("infrastructure_metrics", "distinct_dukas_region")
feature_registry.add_to_group("infrastructure_metrics", "distinct_customers_region")

feature_registry.add_to_group("temporal_features", "contract_start_day")
feature_registry.add_to_group("temporal_features", "contract_month")
feature_registry.add_to_group("temporal_features", "contract_quarter")
feature_registry.add_to_group("temporal_features", "contract_day_name")
feature_registry.add_to_group("temporal_features", "is_weekend")

# Combined feature engineering function
def engineer_features(df_pandas: pd.DataFrame, sample_size: Optional[int] = None) -> pd.DataFrame:
    """
    Combine all feature engineering steps using the feature registry.
    All features are calculated using only data available at loan application time.
    
    Args:
        df_pandas: Preprocessed loan data (pandas DataFrame)
        sample_size: Optional sample size for testing (limits rows processed)
        
    Returns:
        pandas DataFrame with all engineered features
    """
    import time
    start_time = time.time()
    
    # Take a sample if requested (for testing)
    if sample_size is not None and sample_size < len(df_pandas):
        df_pandas = df_pandas.head(sample_size).copy()
        print(f"Using sample of {sample_size} loans")
    else:
        df_pandas = df_pandas.copy()
        print(f"Processing all {len(df_pandas)} loans")
    
    # Prepare data for Polars conversion
    df_pandas = prepare_for_polars(df_pandas)
    
    # Convert to Polars DataFrame
    print("Converting pandas DataFrame to Polars...")
    df = pl.from_pandas(df_pandas)
    
    print(f"\nStarting feature engineering with Feature Registry...")
    
    # Apply feature groups
    df = feature_registry.apply_feature_group(df, "historical_metrics")
    df = feature_registry.apply_feature_group(df, "relative_metrics")
    df = feature_registry.apply_feature_group(df, "infrastructure_metrics")
    df = feature_registry.apply_feature_group(df, "temporal_features")
    
    # Convert back to pandas
    print("Converting results back to pandas DataFrame...")
    result_df = df.to_pandas()
    
    end_time = time.time()
    print(f"\nFeature engineering complete! Generated {len(result_df.columns)} features in {end_time - start_time:.2f} seconds.")
    
    return result_df