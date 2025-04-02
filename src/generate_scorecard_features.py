"""
OAF Loan Performance Feature Generation

This script generates all features needed for loan performance prediction,
including historical metrics, relative position metrics, infrastructure metrics,
temporal features, and geographic features.

Run with the --full flag to process the entire dataset:
python src/generate_scorecard_features.py --full

Or with --sample flag to process a smaller sample for testing:
python src/generate_scorecard_features.py --sample 100
"""

import pandas as pd
import time
import os
import argparse
from pathlib import Path

# Ensure paths work regardless of where script is run from
current_dir = Path(os.getcwd())
if current_dir.name == 'src':
    project_dir = current_dir.parent
else:
    project_dir = current_dir

# Import local modules
import sys
sys.path.append(str(project_dir))
from src import analysis
from src import geo_features
import src.model_development_polars as mdp

def main(sample_size=None, full=False, holdout=False):
    """Main function to generate scorecard features"""
    print("One Acre Fund: Loan Performance Feature Generation")
    print("-" * 70)
    
    # Load and preprocess data
    print("\n1. Loading and preprocessing data...")
    if holdout:
        file_path = "data/raw/holdout_loan_processed.csv"
        print("   Processing HOLDOUT dataset")
        raw_data = analysis.load_data(file_path)
        df = analysis.prepare_holdout_data(raw_data)
    else:
        file_path = "data/raw/training_loan_processed.csv"
        print("   Processing TRAINING dataset")
        raw_data = analysis.load_data(file_path)
        df = analysis.prepare_training_data(raw_data)
        
    print(f"   Loaded {len(df)} loans")
    
    # Take a sample if requested (for testing)
    if not full and sample_size is not None:
        if sample_size < len(df):
            df = df.sample(n=sample_size, random_state=42)
            print(f"   Created sample of {sample_size} loans for testing")
    
    # Generate features in stages
    print("\n2. Generating features...")
    start_time = time.time()
    
    # Apply feature engineering using Polars for better performance
    print("   Applying feature engineering...")
    df_features = mdp.engineer_features(df, is_holdout=holdout)
    
    # Add geographic features
    print("   Adding geographic features...")
    try:
        # Load cached duka locations if available
        cache_file = "data/processed/duka_locations.csv"
        
        # First check if we have cached duka locations
        try:
            cached_locations = pd.read_csv(cache_file)
            print(f"   Loaded {len(cached_locations)} cached duka locations")
            
            # Impute missing coordinates 
            cached_locations = geo_features.impute_missing_coordinates(cached_locations)
            
            # Calculate region centroids
            region_centroids = geo_features.calculate_region_centroids(cached_locations)
            
            # Calculate distances
            duka_distances = geo_features.calculate_distances(cached_locations, region_centroids)
            
            # Add geographic features to the main dataframe
            df_features = df_features.merge(
                duka_distances[[
                    'duka_name',
                    'distance_to_centroid',
                    'distance_to_nairobi',
                    'distance_to_mombasa',
                    'distance_to_kisumu',
                    'coords_imputed'
                ]],
                on='duka_name',
                how='left'
            )
            print("   Geographic features added successfully")
        except FileNotFoundError:
            print(f"   No cached duka locations found at {cache_file}")
            print("   To add geographic features, first run geocoding to generate the cache file:")
            print("   python src/generate_scorecard_features.py --geocode")
            print("   This requires internet access and can take several hours for all dukas")
            print("   You can still use the other features for modeling")
    except Exception as e:
        print(f"   Warning: Could not add geographic features: {e}")
        print("   Continuing with other features only")
    
    end_time = time.time()
    print(f"\nFeature generation complete! Generated {len(df_features.columns)} features in {(end_time - start_time)/60:.2f} minutes.")

    # Display feature groups
    print("\n3. Feature groups in generated dataset:")
    
    feature_groups = {
        "Historical Metrics": [col for col in df_features.columns if 'historical_cum' in col],
        "Relative Position": [col for col in df_features.columns if 'rank' in col],
        "Infrastructure": [col for col in df_features.columns if 'distinct' in col],
        "Temporal": [col for col in df_features.columns 
                   if any(x in col for x in ['day', 'month', 'quarter', 'weekend'])],
        "Geographic": [col for col in df_features.columns 
                      if any(x in col for x in ['distance_to', 'coords_imputed'])],
        "Post-Application": [col for col in df_features.columns if col.startswith('post_')]
    }
    
    for group, columns in feature_groups.items():
        if not columns:  # Skip if no columns in this group
            continue
        print(f"\n   {group} ({len(columns)} features):")
        for col in columns[:5]:  # Show first 5 of each group
            print(f"     - {col}")
        if len(columns) > 5:
            print(f"     - ... and {len(columns) - 5} more")
    
    # Save dataset
    if holdout:
        output_path = "data/processed/holdout_all_features.csv"
        print("   Saving HOLDOUT features")
    elif full:
        output_path = "data/processed/all_features.csv"
    else:
        output_path = "data/processed/feature_sample.csv"
    
    df_features.to_csv(output_path, index=False)
    print(f"\n4. Features saved to {output_path}")
    
    # If it's a sample, explain how to run for full dataset
    if not full:
        print("\n5. To process the full dataset, run:")
        print("   python src/generate_scorecard_features.py --full")
    
    return df_features

def geocode_dukas(force_refresh=False):
    """
    Run geocoding for all dukas to create the location cache.
    This uses OpenStreetMap's Nominatim service to geocode duka locations.
    
    By default, it will:
    - Use existing cached results for dukas already geocoded
    - Only geocode dukas that don't already have coordinates
    - Create a new cached locations file if one doesn't exist
    
    Args:
        force_refresh: Whether to geocode all dukas again, even those already cached
    """
    print("One Acre Fund: Geocoding Duka Locations")
    print("-" * 70)
    print("Warning: This process requires internet access and may take several hours")
    print("         It will generate cached location data for use in feature generation")
    
    # Load data
    print("\n1. Loading data...")
    raw_data = analysis.load_data("data/raw/training_loan_processed.csv")
    # Use common preprocessing since we only need basic fields for geocoding
    df = analysis.preprocess_data(raw_data)
    
    # Get unique dukas
    duka_df = geo_features.get_unique_dukas(df)
    print(f"   Found {len(duka_df)} unique dukas to geocode")
    
    # Geocode dukas
    print("\n2. Geocoding dukas...")
    cache_file = "data/processed/duka_locations.csv"
    
    # Check if cache exists and determine what to process
    if os.path.exists(cache_file) and not force_refresh:
        cached_locations = pd.read_csv(cache_file)
        print(f"   Loaded {len(cached_locations)} cached duka locations")
        
        # Merge with the dukas we need to geocode
        merged_dukas = duka_df.merge(
            cached_locations[['duka_name', 'region', 'latitude', 'longitude']],
            on=['duka_name', 'region'],
            how='left'
        )
        
        # Count how many dukas already have coordinates
        cached_count = merged_dukas['latitude'].notna().sum()
        missing_count = merged_dukas['latitude'].isna().sum()
        
        print(f"   Already geocoded: {cached_count} dukas")
        print(f"   Need to geocode: {missing_count} dukas")
        
        if missing_count == 0:
            print("   All dukas already geocoded! No further geocoding needed.")
            # Still need to ensure we have all the needed columns
            if 'coords_imputed' not in cached_locations.columns:
                cached_locations = geo_features.impute_missing_coordinates(cached_locations)
                cached_locations.to_csv(cache_file, index=False)
                print("   Updated cache file with imputed coordinates")
            return
    else:
        if force_refresh:
            print("   Force refresh requested - geocoding all dukas")
        else:
            print("   No cache found - geocoding all dukas")
        missing_count = len(duka_df)
    
    print(f"   Will geocode {missing_count} dukas (this will take approximately {missing_count * 2 / 60:.1f} minutes)")
    
    # Perform the geocoding
    duka_locations = geo_features.geocode_dukas(
        duka_df, 
        use_cached=not force_refresh,
        cache_file=cache_file
    )
    
    # Impute missing coordinates
    print("\n3. Imputing missing coordinates...")
    duka_locations = geo_features.impute_missing_coordinates(duka_locations)
    
    # Save results
    duka_locations.to_csv(cache_file, index=False)
    print(f"\n4. Duka locations saved to {cache_file}")
    print(f"   Geocoded: {duka_locations['latitude'].notna().sum()} dukas")
    print(f"   Missing/Imputed: {duka_locations['coords_imputed'].sum()} dukas")
    
    # Verification
    if force_refresh:
        print("\nGeocoding complete with force refresh - all dukas were processed")
    else:
        print("\nGeocoding complete - only processed dukas not in the cache")
    print("You can now run the feature generation with geographic features included")

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='OAF Loan Feature Generation')
    parser.add_argument('--full', action='store_true', help='Process full dataset')
    parser.add_argument('--sample', type=int, help='Sample size to process')
    parser.add_argument('--geocode', action='store_true', help='Run geocoding process')
    parser.add_argument('--force-refresh', action='store_true', help='Force refresh geocoding cache')
    parser.add_argument('--holdout', action='store_true', help='Process holdout dataset instead of training')
    args = parser.parse_args()

    # Check if geocoding is requested
    if args.geocode:
        geocode_dukas(force_refresh=args.force_refresh)
    else:
        # Run feature generation
        main(sample_size=args.sample, full=args.full, holdout=args.holdout)
