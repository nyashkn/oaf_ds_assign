"""
OAF Loan Performance Model Development

This module contains functions for feature engineering and model development,
including location-based features using geocoding.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import os
import time

def calculate_regional_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate regional features including cumulative metrics and distinct counts.
    
    Args:
        df: Preprocessed loan data
        
    Returns:
        DataFrame with added regional features:
        - Cumulative loan count per region
        - Cumulative deposit amount per region
        - Cumulative distinct customers per region
        - Cumulative nominal contract value per region
        - Distinct dukas count per region
        - Distinct sales territories per region
        - Distinct areas per region
    """
    # Sort by contract start date to calculate cumulative metrics
    df = df.sort_values('contract_start_date').copy()
    
    # Calculate cumulative metrics by region
    df['cum_loans_in_region'] = df.groupby('region').cumcount() + 1
    df['cum_deposit_amount_in_region'] = df.groupby('region')['deposit_amount'].cumsum()
    df['cum_contract_value_in_region'] = df.groupby('region')['nominal_contract_value'].cumsum()
    
    # Calculate cumulative distinct customers
    df['cum_distinct_customers_in_region'] = df.groupby('region').apply(
        lambda x: x.groupby('client_id').ngroup() + 1
    ).reset_index(level=0, drop=True)
    
    # Calculate distinct counts
    region_stats = df.groupby('region').agg({
        'duka_name': 'nunique',  # Changed from duka_id
        'sales_territory_name': 'nunique',  # Changed from sales_territory
        'area_name': 'nunique'  # Changed from area
    }).reset_index()
    
    region_stats.columns = ['region', 'distinct_dukas', 'distinct_territories', 'distinct_areas']
    
    # Merge distinct counts back to main DataFrame
    df = df.merge(region_stats, on='region', how='left')
    
    # Calculate percentile ranks for cumulative metrics
    df['cum_loans_rank_in_region'] = df.groupby('region')['cum_loans_in_region'].rank(pct=True)
    df['cum_value_rank_in_region'] = df.groupby('region')['cum_contract_value_in_region'].rank(pct=True)
    df['cum_deposit_rank_in_region'] = df.groupby('region')['cum_deposit_amount_in_region'].rank(pct=True)
    
    return df

def get_unique_dukas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract unique duka locations with their regions.
    
    Args:
        df: Preprocessed loan data
        
    Returns:
        DataFrame with unique duka_name and region combinations
    """
    duka_df = df[['duka_name', 'region']].drop_duplicates().reset_index(drop=True)
    return duka_df

def geocode_dukas(duka_df: pd.DataFrame, 
                  use_cached: bool = True,
                  cache_file: str = '../data/processed/duka_locations.csv') -> pd.DataFrame:
    """
    Add latitude and longitude to duka locations using OpenStreetMap's Nominatim service.
    Can use cached results from previous geocoding to avoid repeated API calls.
    
    Args:
        duka_df: DataFrame with duka_name and region columns
        use_cached: Whether to try loading cached results first
        cache_file: Path to cache file
        
    Returns:
        DataFrame with added latitude and longitude columns
    """
    # Check for cached results
    cache_path = Path(cache_file)
    if use_cached and cache_path.exists():
        try:
            cached_locations = pd.read_csv(cache_path)
            print(f"Loaded {len(cached_locations)} cached duka locations")
            
            # Merge cached locations with input dataframe
            result_df = duka_df.merge(
                cached_locations[['duka_name', 'region', 'latitude', 'longitude']],
                on=['duka_name', 'region'],
                how='left'
            )
            
            # Only geocode locations not in cache
            missing_locations = result_df[result_df['latitude'].isna()]
            if len(missing_locations) == 0:
                print("All locations found in cache")
                return result_df
            
            print(f"Geocoding {len(missing_locations)} new locations")
            duka_df = missing_locations
        except Exception as e:
            print(f"Error loading cache, will perform fresh geocoding: {e}")
            result_df = duka_df.copy()
            result_df['latitude'] = None
            result_df['longitude'] = None
    else:
        result_df = duka_df.copy()
        result_df['latitude'] = None
        result_df['longitude'] = None
    
    # Geocode locations
    from geopy.geocoders import Nominatim
    from geopy.extra.rate_limiter import RateLimiter
    
    # Initialize geocoder with custom user agent
    geolocator = Nominatim(user_agent="oaf_analysis")
    # Rate limit to respect API limits
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
    
    for idx, row in duka_df.iterrows():
        # Skip if we already have coordinates
        if pd.notnull(result_df.at[idx, 'latitude']):
            continue
            
        # Combine duka name and region for better accuracy
        search_query = f"{row['duka_name']}, {row['region']}, Kenya"
        try:
            location = geocode(search_query)
            if location:
                result_df.at[idx, 'latitude'] = location.latitude
                result_df.at[idx, 'longitude'] = location.longitude
            time.sleep(1)  # Additional delay to be safe
        except Exception as e:
            print(f"Error geocoding {search_query}: {e}")
    
    # Update cache with new results
    if cache_path.parent.exists():
        result_df.to_csv(cache_path, index=False)
        print(f"Updated cache with {len(result_df)} duka locations")
    
    return result_df

def calculate_region_centroids(duka_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the centroid (average lat/long) for each region.
    
    Args:
        duka_df: DataFrame with duka locations including latitude and longitude
        
    Returns:
        DataFrame with region centroids
    """
    region_centroids = duka_df.groupby('region').agg({
        'latitude': 'mean',
        'longitude': 'mean'
    }).reset_index()
    
    return region_centroids

def calculate_distances(duka_df: pd.DataFrame, region_centroids: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate distances from each duka to region centroid and major cities.
    
    Args:
        duka_df: DataFrame with duka locations including latitude and longitude
        region_centroids: DataFrame with region centroids
        
    Returns:
        DataFrame with added distance columns
    """
    from geopy.distance import geodesic
    
    result_df = duka_df.copy()
    
    # Define major cities
    major_cities = {
        'Nairobi': (-1.2921, 36.8219),
        'Mombasa': (-4.0435, 39.6682),
        'Kisumu': (-0.1022, 34.7617)
    }
    
    # Calculate distance to region centroid
    result_df['distance_to_centroid'] = result_df.apply(
        lambda row: geodesic(
            (row['latitude'], row['longitude']),
            (region_centroids[region_centroids['region'] == row['region']]['latitude'].iloc[0],
             region_centroids[region_centroids['region'] == row['region']]['longitude'].iloc[0])
        ).kilometers if pd.notnull(row['latitude']) else None,
        axis=1
    )
    
    # Calculate distances to major cities
    for city, coords in major_cities.items():
        result_df[f'distance_to_{city.lower()}'] = result_df.apply(
            lambda row: geodesic(
                (row['latitude'], row['longitude']),
                coords
            ).kilometers if pd.notnull(row['latitude']) else None,
            axis=1
        )
    
    return result_df

def add_location_features(df: pd.DataFrame, duka_distances: pd.DataFrame) -> pd.DataFrame:
    """
    Add location-based features to the main dataframe.
    
    Args:
        df: Main loan data DataFrame
        duka_distances: DataFrame with duka distance calculations
        
    Returns:
        DataFrame with added location-based features
    """
    # Merge distance features
    result_df = df.merge(
        duka_distances[[
            'duka_name',
            'distance_to_centroid',
            'distance_to_nairobi',
            'distance_to_mombasa',
            'distance_to_kisumu'
        ]],
        on='duka_name',
        how='left'
    )
    
    return result_df

def process_duka_locations(df: pd.DataFrame, 
                         use_cached: bool = True,
                         cache_file: str = '../data/processed/duka_locations.csv') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Process duka locations to add geographic features.
    This function combines all the location analysis steps.
    
    Args:
        df: Preprocessed loan data
        use_cached: Whether to try loading cached geocoding results
        cache_file: Path to cache file
        
    Returns:
        Tuple containing:
        - Original DataFrame with added location features
        - DataFrame with duka location details
    """
    # Get unique dukas
    duka_df = get_unique_dukas(df)
    
    # Geocode dukas (using cache if available)
    duka_locations = geocode_dukas(duka_df, use_cached=use_cached, cache_file=cache_file)
    
    # Calculate region centroids
    region_centroids = calculate_region_centroids(duka_locations)
    
    # Calculate distances
    duka_distances = calculate_distances(duka_locations, region_centroids)
    
    # Add features to main dataframe
    result_df = add_location_features(df, duka_distances)
    
    # Add regional features
    result_df = calculate_regional_features(result_df)
    
    return result_df, duka_distances
