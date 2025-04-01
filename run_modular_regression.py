#!/usr/bin/env python
"""
Command-line utility to run regression-based loan repayment prediction
with profitability analysis and optimization.

Example usage:
    python run_modular_regression.py --features data/processed/all_features.csv 
                                    --target sept_23_repayment_rate 
                                    --sample 1000 
                                    --handle-dates exclude 
                                    --handle-missing region_mean 
                                    --region-col sales_territory 
                                    --save-plots 
                                    --optimizer bounded
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import datetime
from sklearn.model_selection import train_test_split
from typing import Dict, List, Optional, Any

from src.scorecard_regression.constants import EXCLUDE_VARS
from src.scorecard_regression.data_inspection import inspect_data, handle_date_like_columns
from src.scorecard_regression.modeling_regression import develop_regression_model as fit_regression_model, predict_repayment_rate
from src.scorecard_regression.evaluation import evaluate_regression_performance
from src.scorecard_regression.utils import create_version_path
from src.scorecard_regression.profitability import (
    analyze_multiple_thresholds,
    find_optimal_threshold,
    advanced_profit_optimization,
    plot_threshold_performance,
    plot_profit_metrics,
    plot_optimization_results,
    plot_pareto_frontier
)

def run_modular_regression(
    features_path: str,
    target_var: str,
    output_base_path: str = "data/processed/regression_modelling",
    sample_size: Optional[int] = None,
    test_size: float = 0.3,
    handle_date_columns: str = 'exclude',  # 'exclude', 'convert_to_categorical', or 'parse_date'
    handle_missing: str = 'mean',  # 'mean', 'median', 'mode', 'drop', or 'region_mean'
    region_col: Optional[str] = None,  # Column name containing region information
    save_plots: bool = False,
    model_type: str = 'linear',  # 'linear', 'ridge', 'lasso', 'elasticnet', 'randomforest', 'gbr', 'histgb'
    model_params: Optional[Dict[str, Any]] = None,
    thresholds: Optional[List[float]] = None,
    optimizer: str = None,  # None, 'bounded', 'brent', 'golden', 'advanced'
    gross_margin: float = 0.3,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Run the full modular regression modeling workflow with profitability analysis.
    
    Args:
        features_path: Path to features CSV file
        target_var: Name of target variable
        output_base_path: Base path for output files
        sample_size: Optional sample size for testing
        test_size: Fraction of data to use for testing
        handle_date_columns: How to handle date-like columns 
        handle_missing: How to handle missing values
        region_col: Column name containing region information
        save_plots: Whether to save plots
        model_type: Type of regression model
        model_params: Additional parameters for the model
        thresholds: List of repayment rate thresholds to evaluate
        optimizer: Optimization method for finding optimal threshold
        gross_margin: Gross margin percentage
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary with paths to all outputs
    """
    print(f"[INFO] Starting modular regression modeling workflow...")
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
    
    # Step 3: Drop excluded columns (including non-numeric ones)
    print(f"\n[STEP 3] Preprocessing data...")
    if target_var not in df.columns:
        raise ValueError(f"Target variable '{target_var}' not found in dataframe")
    
    # Get columns to exclude (from constants.py)
    exclude_cols = [col for col in EXCLUDE_VARS if col in df.columns and col != target_var]
    
    if exclude_cols:
        print(f"Excluding {len(exclude_cols)} columns: {', '.join(exclude_cols[:5])}{'...' if len(exclude_cols) > 5 else ''}")
        df = df.drop(columns=exclude_cols)
    
    # Handle missing values
    # First, check for missing values
    missing_cols = []
    for col in df.columns:
        if col != target_var and df[col].isna().any():
            missing_cols.append((col, df[col].isna().sum()))
    
    if missing_cols:
        print("\nHandling missing values...")
        print(f"Found {len(missing_cols)} columns with missing values:")
        for col, count in sorted(missing_cols, key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {col}: {count} missing values")
        
        if handle_missing == 'drop':
            # Drop rows with any missing values
            original_count = len(df)
            df = df.dropna()
            print(f"Dropped {original_count - len(df)} rows with missing values")
        else:
            # Impute missing values
            for col in df.columns:
                if col != target_var and df[col].isna().any():
                    if handle_missing == 'mean':
                        fill_value = df[col].mean()
                        method = 'mean'
                    elif handle_missing == 'median':
                        fill_value = df[col].median()
                        method = 'median'
                    elif handle_missing == 'mode':
                        fill_value = df[col].mode()[0]
                        method = 'mode'
                    elif handle_missing == 'region_mean' and region_col and region_col in df.columns:
                        # Special case for region-based imputation
                        print(f"  Using region-based imputation for {col}")
                        for region in df[region_col].unique():
                            mask = (df[region_col] == region) & df[col].isna()
                            if mask.any():
                                region_mean = df[df[region_col] == region][col].mean()
                                df.loc[mask, col] = region_mean
                        method = 'region mean'
                        continue
                    else:
                        # Default to mean
                        fill_value = df[col].mean()
                        method = 'mean'
                    
                    # Apply imputation
                    df[col] = df[col].fillna(fill_value)
                    print(f"  Filled {col} missing values with {method}")
    
    # Split into features and target
    features = df.drop(columns=[target_var])
    target = df[target_var]
    
    # Step 4: Split data into training and testing sets
    print(f"\n[STEP 4] Splitting data into training and testing sets...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=test_size, random_state=random_state
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Testing set: {X_test.shape[0]} samples")
    
    # Save train/test split
    split_dir = os.path.join(version_path, "3_train_test_split")
    os.makedirs(split_dir, exist_ok=True)
    
    pd.concat([X_train, y_train], axis=1).to_csv(
        os.path.join(split_dir, "train.csv"), index=False)
    pd.concat([X_test, y_test], axis=1).to_csv(
        os.path.join(split_dir, "test.csv"), index=False)
    
    # Step 4: Train regression model
    print(f"\n[STEP 4] Training regression model ({model_type})...")
    model_dir = os.path.join(version_path, "4_regression_model")
    
    if model_params is None:
        model_params = {}
    
    # Create training and test dataframes with target for the modeling function
    train_df = pd.concat([X_train, pd.Series(y_train, name=target_var)], axis=1)
    test_df = pd.concat([X_test, pd.Series(y_test, name=target_var)], axis=1)
    
    # Call the modeling function
    model_results = fit_regression_model(
        train_df, 
        test_df,
        target_var,
        output_dir=model_dir,
        model_type=model_type,
        model_params=model_params,
        random_state=random_state
    )
    
    # Extract the model from results
    model = model_results['model']
    feature_importance = model_results.get('feature_importances')
    
    # Step 5: Make predictions
    print(f"\n[STEP 5] Making predictions...")
    predictions_dir = os.path.join(version_path, "5_predictions")
    os.makedirs(predictions_dir, exist_ok=True)
    
    # Get predictions directly from model
    y_train_pred = model.predict(X_train)
    train_predictions = pd.DataFrame({
        'actual': y_train,
        'predicted': y_train_pred
    })
    
    # Get predictions directly from model
    y_test_pred = model.predict(X_test)
    test_predictions = pd.DataFrame({
        'actual': y_test,
        'predicted': y_test_pred
    })
    
    # Add loan amount if available
    loan_amount_col = 'nominal_contract_value'
    if loan_amount_col in X_train.columns:
        train_predictions[loan_amount_col] = X_train[loan_amount_col].values
        test_predictions[loan_amount_col] = X_test[loan_amount_col].values
    
    # Save predictions
    train_predictions.to_csv(os.path.join(predictions_dir, "train_predictions.csv"), index=False)
    test_predictions.to_csv(os.path.join(predictions_dir, "test_predictions.csv"), index=False)
    
    # Step 6: Evaluate model
    print(f"\n[STEP 6] Evaluating regression model...")
    evaluation_dir = os.path.join(version_path, "6_model_evaluation")
    
    evaluation_results = evaluate_regression_performance(
        y_train, y_test, y_train_pred, y_test_pred,
        output_dir=evaluation_dir if save_plots else None
    )
    
    # Step 7: Profitability analysis with multiple thresholds
    print(f"\n[STEP 7] Performing profitability analysis...")
    profitability_dir = os.path.join(version_path, "7_profitability_analysis")
    os.makedirs(profitability_dir, exist_ok=True)
    
    # Use test predictions for profitability analysis
    if thresholds is None:
        thresholds = [0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    
    threshold_dir = os.path.join(profitability_dir, "threshold_analysis")
    os.makedirs(threshold_dir, exist_ok=True)
    
    threshold_results = analyze_multiple_thresholds(
        test_predictions,
        thresholds=thresholds,
        predicted_col='predicted',
        actual_col='actual',
        loan_amount_col=loan_amount_col if loan_amount_col in test_predictions.columns else None,
        gross_margin=gross_margin,
        output_dir=threshold_dir
    )
    
    if save_plots:
        # Plot threshold performance
        plot_threshold_performance(
            threshold_results['threshold_df'],
            threshold_results['optimal_threshold'],
            output_path=os.path.join(threshold_dir, "threshold_performance.png")
        )
        
        # Plot profit metrics
        plot_profit_metrics(
            threshold_results['threshold_df'],
            threshold_results['optimal_threshold'],
            output_path=os.path.join(threshold_dir, "profit_metrics.png")
        )
    
    # Step 8: Optimization (if requested)
    if optimizer:
        print(f"\n[STEP 8] Performing threshold optimization ({optimizer})...")
        optimization_dir = os.path.join(profitability_dir, "optimization")
        os.makedirs(optimization_dir, exist_ok=True)
        
        if optimizer == 'advanced':
            # Run multi-objective optimization
            alpha_values = [0.1, 0.25, 0.5, 0.75, 0.9]
            opt_results = advanced_profit_optimization(
                test_predictions,
                predicted_col='predicted',
                actual_col='actual',
                loan_amount_col=loan_amount_col if loan_amount_col in test_predictions.columns else None,
                gross_margin=gross_margin,
                alpha_values=alpha_values,
                output_dir=optimization_dir
            )
            
            if save_plots:
                # Plot Pareto frontier
                plot_pareto_frontier(
                    opt_results['pareto_frontier'],
                    output_path=os.path.join(optimization_dir, "pareto_frontier.png")
                )
                
            # Save results summary
            with open(os.path.join(optimization_dir, "optimization_summary.txt"), 'w') as f:
                f.write(f"Multi-objective Optimization Results\n")
                f.write(f"==================================\n\n")
                
                for alpha, result in opt_results['results'].items():
                    if 'optimal_threshold' in result:
                        f.write(f"Alpha: {result['alpha']:.2f}\n")
                        f.write(f"Optimal threshold: {result['optimal_threshold']:.4f}\n")
                        f.write(f"Total profit: {result['total_profit']:.2f}\n")
                        f.write(f"Money left on table: {result['money_left_on_table']:.2f}\n")
                        f.write(f"Approval rate: {result['approval_rate']:.2%}\n\n")
        else:
            # Run single-objective optimization
            opt_results = find_optimal_threshold(
                test_predictions,
                predicted_col='predicted',
                actual_col='actual',
                loan_amount_col=loan_amount_col if loan_amount_col in test_predictions.columns else None,
                gross_margin=gross_margin,
                metric='total_actual_profit',
                method=optimizer,
                output_dir=optimization_dir
            )
            
            if save_plots:
                # Plot optimization results
                plot_optimization_results(
                    opt_results,
                    output_path=os.path.join(optimization_dir, "optimization_results.png")
                )
                
            # Save optimized result to main directory for easy reference
            with open(os.path.join(profitability_dir, "optimal_threshold.txt"), 'w') as f:
                f.write(f"Optimal threshold: {opt_results['optimal_threshold']:.4f}\n")
                f.write(f"Optimization method: {optimizer}\n")
                f.write(f"Actual repayment rate: {opt_results['loan_metrics']['actual_repayment_rate']:.4f}\n")
                f.write(f"Total profit: {opt_results['profit_metrics']['total_actual_profit']:.2f}\n")
                f.write(f"Approval rate: {opt_results['loan_metrics']['n_approved']/opt_results['loan_metrics']['total_loans']:.2%}\n")
    
    # Create summary report
    summary = {
        'version_path': version_path,
        'date': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'target_variable': target_var,
        'sample_size': len(df) if sample_size is None else sample_size,
        'model_type': model_type,
        'model_params': model_params,
        'performance': {
            'train_r2': float(evaluation_results['train_metrics']['r2']),
            'test_r2': float(evaluation_results['test_metrics']['r2']),
            'train_mae': float(evaluation_results['train_metrics']['mae']),
            'test_mae': float(evaluation_results['test_metrics']['mae']),
            'train_rmse': float(evaluation_results['train_metrics']['rmse']),
            'test_rmse': float(evaluation_results['test_metrics']['rmse'])
        }
    }
    
    # Add profitability metrics
    if threshold_results and 'optimal_threshold' in threshold_results:
        summary['profitability'] = {
            'optimal_threshold_grid': float(threshold_results['optimal_threshold']),
            'approval_rate': float(threshold_results['threshold_df'].loc[
                threshold_results['threshold_df']['threshold'] == threshold_results['optimal_threshold'], 
                'approval_rate'
            ].values[0]),
            'actual_repayment_rate': float(threshold_results['threshold_df'].loc[
                threshold_results['threshold_df']['threshold'] == threshold_results['optimal_threshold'], 
                'actual_repayment_rate'
            ].values[0])
        }
    
    # Add optimization results if available
    if optimizer and 'opt_results' in locals() and 'optimal_threshold' in opt_results:
        if 'profitability' not in summary:
            summary['profitability'] = {}
        
        summary['profitability']['optimal_threshold_' + optimizer] = float(opt_results['optimal_threshold'])
    
    # Save summary report
    summary_path = os.path.join(version_path, "summary.json")
    import json
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n[SUCCESS] Regression modeling with profitability analysis complete!")
    print(f"Results saved to {version_path}")
    print(f"Summary report saved to {summary_path}")
    
    # Return paths to outputs
    return {
        'version_path': version_path,
        'summary': summary,
        'data_inspection': inspection_path,
        'train_test_split': split_dir,
        'regression_model': model_dir,
        'predictions': predictions_dir,
        'model_evaluation': evaluation_dir,
        'profitability_analysis': profitability_dir
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Regression Modeling with Profitability Analysis')
    parser.add_argument('--features', type=str, default="data/processed/all_features.csv",
                       help='Path to features CSV file')
    parser.add_argument('--target', type=str, default='sept_23_repayment_rate',
                       help='Target variable name')
    parser.add_argument('--sample', type=int, default=None,
                       help='Sample size for testing')
    parser.add_argument('--test-size', type=float, default=0.3,
                       help='Fraction of data to use for testing')
    parser.add_argument('--output', type=str, default="data/processed/regression_modelling",
                       help='Base path for output files')
    parser.add_argument('--handle-dates', type=str, choices=['exclude', 'convert_to_categorical', 'parse_date'],
                       default='exclude', help='How to handle date-like columns')
    parser.add_argument('--handle-missing', type=str, choices=['mean', 'median', 'mode', 'drop', 'region_mean'],
                       default='mean', help='How to handle missing values')
    parser.add_argument('--region-col', type=str, default=None,
                       help='Column name containing region information')
    parser.add_argument('--save-plots', action='store_true',
                       help='Whether to save plots')
    parser.add_argument('--model', type=str, choices=['linear', 'ridge', 'lasso', 'elasticnet', 'randomforest', 'gbr', 'histgb'],
                       default='linear', help='Type of regression model')
    parser.add_argument('--model-params', type=str, default=None,
                       help='JSON string with model parameters')
    parser.add_argument('--thresholds', type=str, default=None,
                       help='Comma-separated list of thresholds to evaluate (e.g., "0.7,0.75,0.8,0.85,0.9")')
    parser.add_argument('--optimizer', type=str, choices=['bounded', 'brent', 'golden', 'advanced'],
                       default=None, help='Optimization method for finding optimal threshold')
    parser.add_argument('--gross-margin', type=float, default=0.3,
                       help='Gross margin percentage')
    parser.add_argument('--random-state', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Parse model parameters
    model_params = None
    if args.model_params:
        import json
        try:
            model_params = json.loads(args.model_params)
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON for model parameters: {args.model_params}")
            sys.exit(1)
    
    # Parse thresholds
    thresholds = None
    if args.thresholds:
        try:
            thresholds = [float(t) for t in args.thresholds.split(',')]
        except ValueError:
            print(f"Error: Invalid threshold values: {args.thresholds}")
            sys.exit(1)
    
    # Run the modeling workflow
    run_modular_regression(
        features_path=args.features,
        target_var=args.target,
        output_base_path=args.output,
        sample_size=args.sample,
        test_size=args.test_size,
        handle_date_columns=args.handle_dates,
        handle_missing=args.handle_missing,
        region_col=args.region_col,
        save_plots=args.save_plots,
        model_type=args.model,
        model_params=model_params,
        thresholds=thresholds,
        optimizer=args.optimizer,
        gross_margin=args.gross_margin,
        random_state=args.random_state
    )
