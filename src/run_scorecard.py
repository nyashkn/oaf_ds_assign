#!/usr/bin/env python
"""
Command-line script to run the modular scorecard modeling workflow.

Example usage:
    python src/run_scorecard.py --features data/processed/all_features.csv --target sept_23_repayment_rate --sample 1000

This script uses the modular scorecard package to develop a credit scorecard
with better error handling and diagnostic capabilities.
"""

import argparse
import sys
# Use relative import since this is inside the src directory
from .scorecard.main import run_modular_scorecard

def main():
    """Run the modular scorecard modeling workflow from command line arguments."""
    parser = argparse.ArgumentParser(description='OAF Loan Scorecard Modeling - Modular Approach')
    
    # Required parameters
    parser.add_argument('--features', type=str, required=True,
                       help='Path to features CSV file')
    parser.add_argument('--target', type=str, required=True,
                       help='Target variable name')
    
    # Optional parameters with defaults
    parser.add_argument('--cutoff', type=float, default=None,
                       help='Cutoff for good/bad classification (if omitted, determined automatically)')
    parser.add_argument('--sample', type=int, default=None,
                       help='Sample size for testing (if omitted, uses full dataset)')
    parser.add_argument('--output', type=str, default="data/processed/scorecard_modelling",
                       help='Base path for output files')
    parser.add_argument('--handle-dates', type=str, 
                       choices=['exclude', 'convert_to_categorical', 'parse_date'],
                       default='exclude', 
                       help='How to handle date-like columns')
    parser.add_argument('--iv-threshold', type=float, default=0.02,
                       help='Minimum IV for variable selection')
    parser.add_argument('--exclude-vars', type=str, default=None,
                       help='Comma-separated list of additional variables to exclude')
    
    args = parser.parse_args()
    
    # Process exclude-vars if provided
    exclude_vars = None
    if args.exclude_vars:
        exclude_vars = [var.strip() for var in args.exclude_vars.split(',')]
    
    try:
        # Print configuration
        print("\n=== Scorecard Modeling Configuration ===")
        print(f"Features file: {args.features}")
        print(f"Target variable: {args.target}")
        print(f"Cutoff: {'Auto-determined' if args.cutoff is None else args.cutoff}")
        print(f"Sample size: {'Full dataset' if args.sample is None else args.sample}")
        print(f"Output directory: {args.output}")
        print(f"Date handling: {args.handle_dates}")
        print(f"IV threshold: {args.iv_threshold}")
        print(f"Additional excluded variables: {exclude_vars}")
        print("=" * 50)
        
        # Run the modeling workflow
        result = run_modular_scorecard(
            features_path=args.features,
            target_var=args.target,
            output_base_path=args.output,
            cutoff=args.cutoff,
            sample_size=args.sample,
            exclude_vars=exclude_vars,
            handle_date_columns=args.handle_dates,
            iv_threshold=args.iv_threshold
        )
        
        # Print final paths
        print("\n=== Scorecard Modeling Results ===")
        print(f"Version path: {result['version_path']}")
        print(f"Summary file: {result['version_path']}/summary.json")
        
        # Print performance metrics
        if 'summary' in result and 'performance' in result['summary']:
            perf = result['summary']['performance']
            print("\nModel Performance:")
            print(f"  Train KS: {perf['train_ks']:.4f}")
            print(f"  Test KS: {perf['test_ks']:.4f}")
            print(f"  Train AUC: {perf['train_auc']:.4f}")
            print(f"  Test AUC: {perf['test_auc']:.4f}")
            print(f"  PSI: {perf['psi']:.4f}")
        
        return 0
    
    except Exception as e:
        print(f"\n[ERROR] {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
