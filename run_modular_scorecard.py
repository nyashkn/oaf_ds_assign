#!/usr/bin/env python
"""
Standalone script to run the modular scorecard modeling workflow.

This script directly imports all necessary functions from the src/scorecard modules
and provides a command-line interface to run the workflow.

Example usage:
    python run_modular_scorecard.py --features data/processed/all_features.csv --target sept_23_repayment_rate --sample 1000
"""

# Set matplotlib backend to non-interactive
import matplotlib
matplotlib.use('Agg')

import argparse
import sys
import os
import json
import datetime
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Any, Union

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import necessary functions from the scorecard modules
from src.scorecard.utils import create_version_path
from src.scorecard.report_utils import (
    initialize_modeling_report, add_data_inspection_section,
    add_target_preparation_section, add_variable_selection_section,
    add_woe_binning_section, add_model_development_section,
    add_performance_evaluation_section
)
from src.scorecard.data_inspection import inspect_data, handle_date_like_columns
from src.scorecard.target_preparation import create_binary_target, find_optimal_cutoff
from src.scorecard.variable_selection import exclude_leakage_variables, partition_data, select_variables
from src.scorecard.binning import perform_woe_binning, apply_woe_transformation
from src.scorecard.modeling import develop_scorecard_model, evaluate_model_performance

def run_modular_scorecard(
    features_path: str,
    target_var: str,
    output_base_path: str = "data/processed/scorecard_modelling",
    cutoff: float = None,
    sample_size: int = None,
    exclude_vars: list = None,
    handle_date_columns: str = 'exclude',
    iv_threshold: float = 0.02,
    handle_missing: str = 'mean',
    region_col: Optional[str] = None,
    save_plots: bool = True,
    classifier_type: str = 'logistic',
    classifier_params: Optional[dict] = None,
    handle_missing_classifier: bool = False,
    random_state: int = 42
):
    """
    Run the full modular scorecard modeling workflow.
    """
    print(f"[INFO] Starting modular scorecard modeling workflow...")
    print(f"Target variable: {target_var}")
    print(f"Features path: {features_path}")
    print(f"Using classifier: {classifier_type}")
    
    # Create version directory and initialize report
    version_path = create_version_path(output_base_path)
    report = initialize_modeling_report(version_path)
    
    # Step 1: Load and inspect data
    print(f"\n[STEP 1] Loading and inspecting data...")
    df = pd.read_csv(features_path)
    
    if sample_size is not None and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=random_state)
        print(f"Using sample of {sample_size} loans for testing")
    
    # Inspect data
    inspection_path = os.path.join(version_path, "1_data_inspection.json")
    inspection_results = inspect_data(df, target_var, inspection_path)
    add_data_inspection_section(report, inspection_results)
    
    # Step 2: Handle date-like columns
    print(f"\n[STEP 2] Handling date-like columns...")
    date_handling_path = os.path.join(version_path, "2_date_handling.json")
    df = handle_date_like_columns(
        df, 
        inspection_results.get("date_like_columns", []),
        method=handle_date_columns,
        output_path=date_handling_path
    )
    
    # Step 3: Create binary target
    print(f"\n[STEP 3] Creating binary target...")
    binary_target_path = os.path.join(version_path, "3_binary_target.csv")
    if cutoff is None:
        # Find optimal cutoff based on data
        print("Determining optimal cutoff...")
        cutoff_result = find_optimal_cutoff(df, target_var=target_var)
        cutoff = cutoff_result['optimal_cutoff']
        print(f"Optimal cutoff determined: {cutoff:.2f}")
        
        # Save cutoff info
        cutoff_info_path = os.path.join(version_path, "3_cutoff_info.json")
        with open(cutoff_info_path, 'w') as f:
            json.dump({
                'optimal_cutoff': float(cutoff),
                'cutoff_stats': cutoff_result['cutoff_stats'].to_dict(orient='records')
            }, f, indent=2)
    
    df_with_target = create_binary_target(
        df, 
        target_var=target_var, 
        cutoff=cutoff,
        output_path=binary_target_path
    )
    
    # Add target preparation to report
    binary_target_stats = {
        'distribution': {
            'good_count': len(df_with_target[df_with_target['good_loan'] == 1]),
            'bad_count': len(df_with_target[df_with_target['good_loan'] == 0]),
            'good_pct': len(df_with_target[df_with_target['good_loan'] == 1]) / len(df_with_target),
            'bad_pct': len(df_with_target[df_with_target['good_loan'] == 0]) / len(df_with_target)
        }
    }
    add_target_preparation_section(report, {'optimal_cutoff': cutoff}, binary_target_stats)
    
    # Step 4: Exclude leakage variables
    print(f"\n[STEP 4] Excluding leakage variables...")
    filtered_path = os.path.join(version_path, "4_filtered_data.csv")
    filtered_df, excluded = exclude_leakage_variables(
        df_with_target, 
        'good_loan', 
        additional_exclusions=exclude_vars,
        output_path=filtered_path
    )
    
    # Step 5: Partition data
    print(f"\n[STEP 5] Partitioning data...")
    partition_dir = os.path.join(version_path, "5_partition")
    partitioned_data = partition_data(
        filtered_df,
        'good_loan',
        train_ratio=0.7,
        random_state=random_state,
        output_dir=partition_dir
    )
    
    # Step 6: Variable selection based on IV
    print(f"\n[STEP 6] Selecting variables...")
    selection_dir = os.path.join(version_path, "6_variable_selection")
    selected_df, selection_results = select_variables(
        partitioned_data['train'],
        'good_loan',
        iv_threshold=iv_threshold,
        output_dir=selection_dir
    )
    # Add variable selection to report
    selection_results['output_dir'] = selection_dir
    add_variable_selection_section(report, selection_results)
    
    # Adjust test set to have the same variables as selected_df
    selected_vars = [col for col in selected_df.columns if col != 'good_loan']
    test_selected = partitioned_data['test'][selected_vars + ['good_loan']]
    
    # Step 7: WOE binning
    print(f"\n[STEP 7] Performing WOE binning...")
    binning_dir = os.path.join(version_path, "7_woe_binning")
    try:
        bins = perform_woe_binning(
            selected_df,
            'good_loan',
            output_dir=binning_dir,
            check_cate_num=False,
            handle_missing=handle_missing,
            region_col=region_col,
            save_plots=save_plots
        )
        
        # Add WOE binning to report
        binning_summary = {
            'bin_stats': {var: {'n_bins': len(bin_info), 'iv': bin_info['bin_iv'].sum()} 
                         for var, bin_info in bins.items()}
        }
        
        # Pass the directory containing WOE plot images instead of figure objects
        woe_plots_dir = os.path.join(binning_dir, "woe_plots") if save_plots else None
        add_woe_binning_section(report, binning_summary, woe_plots_dir)
        
        # Step 8: WOE transformation
        print(f"\n[STEP 8] Applying WOE transformation...")
        transformation_dir = os.path.join(version_path, "8_woe_transformation")
        woe_data = apply_woe_transformation(
            selected_df,
            test_selected,
            bins,
            'good_loan',
            output_dir=transformation_dir
        )
        
        # Step 9: Develop scorecard
        print(f"\n[STEP 9] Developing scorecard model...")
        scorecard_dir = os.path.join(version_path, "9_scorecard")
        scorecard_results = develop_scorecard_model(
            woe_data['train_woe'],
            woe_data['test_woe'],
            bins,
            'good_loan',
            output_dir=scorecard_dir,
            classifier_type=classifier_type,
            handle_missing=handle_missing_classifier,
            classifier_params=classifier_params,
            random_state=random_state
        )
        
        # Add model development to report
        model_results = {
            'classifier_type': classifier_type,
            'handle_missing': handle_missing_classifier,
            'classifier_params': classifier_params,
            'coefficients': dict(zip(selected_vars, scorecard_results['coefficients']['coefficient']))
        }
        add_model_development_section(report, model_results)
        
        # Step 10: Evaluate model performance
        print(f"\n[STEP 10] Evaluating model performance...")
        evaluation_dir = os.path.join(version_path, "10_performance_evaluation")
        performance_results = evaluate_model_performance(
            woe_data['train_woe']['good_loan'],
            woe_data['test_woe']['good_loan'],
            scorecard_results['predictions']['train'],
            scorecard_results['predictions']['test'],
            output_dir=evaluation_dir
        )
        
        # Add performance evaluation to report
        perf_results = {
            'train_metrics': {
                'KS': float(performance_results['train_perf']['KS']),
                'AUC': float(performance_results['train_perf']['AUC']),
                'Gini': float(performance_results['train_perf']['Gini'])
            },
            'test_metrics': {
                'KS': float(performance_results['test_perf']['KS']),
                'AUC': float(performance_results['test_perf']['AUC']),
                'Gini': float(performance_results['test_perf']['Gini'])
            },
            'psi': float(performance_results['psi']['psi']['PSI'].values[0])
        }
        add_performance_evaluation_section(report, perf_results)
        
        # Save report
        report.save()
        
        # Create summary report
        summary = {
            'version_path': version_path,
            'date': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'target_variable': target_var,
            'cutoff': float(cutoff),
            'sample_size': len(df) if sample_size is None else sample_size,
            'excluded_variables': excluded,
            'selected_variables': selected_vars,
            'classifier': {
                'type': classifier_type,
                'params': classifier_params,
                'handle_missing': handle_missing_classifier
            },
            'performance': perf_results
        }
        
        summary_path = os.path.join(version_path, "summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n[SUCCESS] Scorecard modeling complete!")
        print(f"Results saved to {version_path}")
        print(f"Summary report saved to {summary_path}")
        print(f"PDF report saved to {os.path.join(version_path, 'modeling_report.pdf')}")
        
        # Return paths to outputs
        return {
            'version_path': version_path,
            'summary': summary,
            'data_inspection': inspection_path,
            'binary_target': binary_target_path,
            'filtered_data': filtered_path,
            'partition': partition_dir,
            'variable_selection': selection_dir,
            'woe_binning': binning_dir,
            'woe_transformation': transformation_dir,
            'scorecard': scorecard_dir,
            'performance_evaluation': evaluation_dir,
            'modeling_report': os.path.join(version_path, 'modeling_report.pdf')
        }
        
    except Exception as e:
        error_log_path = os.path.join(version_path, "error_log.txt")
        with open(error_log_path, 'w') as f:
            f.write(f"Error occurred during modeling: {str(e)}\n")
            import traceback
            f.write(traceback.format_exc())
        
        print(f"\n[ERROR] An error occurred during modeling: {str(e)}")
        print(f"Error details saved to {error_log_path}")
        raise e

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
    parser.add_argument('--handle-missing', type=str, 
                       choices=['mean', 'median', 'mode', 'drop', 'region_mean'],
                       default='mean', help='How to handle missing values')
    parser.add_argument('--region-col', type=str, default='sales_territory',
                       help='Column name containing region information for region_mean')
    parser.add_argument('--save-plots', action='store_true',
                       help='Save WOE binning plots')
    parser.add_argument('--classifier', type=str, choices=['logistic', 'histgb'],
                       default='logistic', help='Type of classifier to use')
    parser.add_argument('--classifier-params', type=str, default=None,
                       help='JSON string of additional classifier parameters')
    
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
        print(f"Missing value handling: {args.handle_missing}")
        if args.handle_missing == 'region_mean':
            print(f"Region column: {args.region_col}")
        print(f"Save WOE plots: {args.save_plots}")
        print(f"Classifier: {args.classifier}")
        if args.classifier_params:
            print(f"Classifier parameters: {args.classifier_params}")
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
            iv_threshold=args.iv_threshold,
            handle_missing=args.handle_missing,
            region_col=args.region_col if args.handle_missing == 'region_mean' else None,
            save_plots=args.save_plots,
            classifier_type=args.classifier,
            classifier_params=json.loads(args.classifier_params) if args.classifier_params else None,
            handle_missing_classifier=args.handle_missing == 'region_mean'
        )
        
        # Print final paths
        print("\n=== Scorecard Modeling Results ===")
        print(f"Version path: {result['version_path']}")
        print(f"Summary file: {result['version_path']}/summary.json")
        print(f"PDF report: {result['modeling_report']}")
        
        # Print performance metrics
        if 'summary' in result and 'performance' in result['summary']:
            perf = result['summary']['performance']
            print("\nModel Performance:")
            print(f"  Train KS: {perf['train_metrics']['KS']:.4f}")
            print(f"  Test KS: {perf['test_metrics']['KS']:.4f}")
            print(f"  Train AUC: {perf['train_metrics']['AUC']:.4f}")
            print(f"  Test AUC: {perf['test_metrics']['AUC']:.4f}")
            print(f"  PSI: {perf['psi']:.4f}")
        
        return 0
    
    except Exception as e:
        print(f"\n[ERROR] {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
