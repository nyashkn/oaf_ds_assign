"""
Main orchestration module for scorecard modeling workflow.
"""

import os
import datetime
import json
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Any

from .utils import create_version_path
from .data_inspection import inspect_data, handle_date_like_columns
from .target_preparation import create_binary_target, find_optimal_cutoff
from .variable_selection import exclude_leakage_variables, partition_data, select_variables
from .binning import perform_woe_binning, apply_woe_transformation
from .modeling import develop_scorecard_model, evaluate_model_performance

def run_modular_scorecard(
    features_path: str,
    target_var: str,
    output_base_path: str = "data/processed/scorecard_modelling",
    cutoff: Optional[float] = None,
    sample_size: Optional[int] = None,
    exclude_vars: Optional[List[str]] = None,
    handle_date_columns: str = 'exclude',  # 'exclude', 'convert_to_categorical', or 'parse_date'
    iv_threshold: float = 0.02,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Run the full modular scorecard modeling workflow.
    
    Args:
        features_path: Path to features CSV file
        target_var: Name of target variable
        output_base_path: Base path for output files
        cutoff: Cutoff for good/bad classification (if None, determined automatically)
        sample_size: Optional sample size for testing
        exclude_vars: Additional variables to exclude from modeling
        handle_date_columns: How to handle date-like columns 
        iv_threshold: Minimum IV for variable selection
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary with paths to all outputs
    """
    print(f"[INFO] Starting modular scorecard modeling workflow...")
    print(f"Target variable: {target_var}")
    print(f"Features path: {features_path}")
    
    # Create version directory
    version_path = create_version_path(output_base_path)
    
    # Step 1: Load and inspect data
    print(f"\n[STEP 1] Loading and inspecting data...")
    df = pd.read_csv(features_path)
    
    if sample_size is not None and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=random_state)
        print(f"Using sample of {sample_size} loans for testing")
    
    # Inspect data
    inspection_path = os.path.join(version_path, "1_data_inspection.json")
    inspection_results = inspect_data(df, target_var, inspection_path)
    
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
            check_cate_num=False
        )
        
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
            random_state=random_state
        )
        
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
        
        # Create summary report
        summary = {
            'version_path': version_path,
            'date': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'target_variable': target_var,
            'cutoff': float(cutoff),
            'sample_size': len(df) if sample_size is None else sample_size,
            'excluded_variables': excluded,
            'selected_variables': selected_vars,
            'performance': {
                'train_ks': float(performance_results['train_perf']['KS']),
                'test_ks': float(performance_results['test_perf']['KS']),
                'train_auc': float(performance_results['train_perf']['AUC']),
                'test_auc': float(performance_results['test_perf']['AUC']),
                'psi': float(performance_results['psi']['psi']['PSI'].values[0])
            }
        }
        
        summary_path = os.path.join(version_path, "summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n[SUCCESS] Scorecard modeling complete!")
        print(f"Results saved to {version_path}")
        print(f"Summary report saved to {summary_path}")
        
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
            'performance_evaluation': evaluation_dir
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

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='OAF Loan Scorecard Modeling - Modular Approach')
    parser.add_argument('--features', type=str, default="data/processed/all_features.csv",
                       help='Path to features CSV file')
    parser.add_argument('--target', type=str, default='sept_23_repayment_rate',
                       help='Target variable name')
    parser.add_argument('--cutoff', type=float, default=None,
                       help='Cutoff for good/bad classification')
    parser.add_argument('--sample', type=int, default=None,
                       help='Sample size for testing')
    parser.add_argument('--output', type=str, default="data/processed/scorecard_modelling",
                       help='Base path for output files')
    parser.add_argument('--handle-dates', type=str, choices=['exclude', 'convert_to_categorical', 'parse_date'],
                       default='exclude', help='How to handle date-like columns')
    parser.add_argument('--iv-threshold', type=float, default=0.02,
                       help='Minimum IV for variable selection')
    
    args = parser.parse_args()
    
    # Run the modeling workflow
    run_modular_scorecard(
        features_path=args.features,
        target_var=args.target,
        output_base_path=args.output,
        cutoff=args.cutoff,
        sample_size=args.sample,
        handle_date_columns=args.handle_dates,
        iv_threshold=args.iv_threshold
    )
