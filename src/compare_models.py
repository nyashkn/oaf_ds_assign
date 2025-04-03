"""
Loan Repayment Model Comparison Script

This script compares two different loan repayment prediction models:
1. Model 1: Using only application-time features 
2. Model 2: Using application-time features + September payment data

The script performs end-to-end model training, evaluation, and comparison,
generating visualizations and reports to help understand the tradeoffs
between these approaches.

Usage:
    python compare_models.py --data-path data/processed/features.csv --output-dir results/model_comparison
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from typing import Dict, List, Tuple

from scorecard_regression.model_comparison import (
    APPLICATION_FEATURES, 
    SEPTEMBER_PAYMENT_FEATURES,
    load_and_prepare_data,
    filter_features,
    train_model,
    evaluate_model,
    analyze_model_profit,
    compare_models,
    create_comparison_plots,
    feature_importance_comparison,
    save_predictions_for_holdout
)

from scorecard_regression.reporting import (
    ModelComparisonReport,
    create_markdown_report
)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Compare loan repayment prediction models')
    
    parser.add_argument('--data-path', type=str, required=True,
                        help='Path to the features CSV file')
    
    parser.add_argument('--output-dir', type=str, default='data/processed/model_comparison',
                        help='Directory to save results and reports')
    
    parser.add_argument('--test-size', type=float, default=0.3,
                        help='Fraction of data to use for testing')
    
    parser.add_argument('--min-threshold', type=float, default=0.5,
                        help='Minimum threshold for repayment rate')
    
    parser.add_argument('--max-threshold', type=float, default=0.95,
                        help='Maximum threshold for repayment rate')
    
    parser.add_argument('--threshold-step', type=float, default=0.02,
                        help='Step size for threshold range')
    
    parser.add_argument('--holdout-path', type=str, default=None,
                        help='Path to holdout dataset for additional predictions (optional)')
    
    return parser.parse_args()

def main():
    """Main function to run the model comparison."""
    # Parse arguments
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate thresholds
    thresholds = np.arange(args.min_threshold, args.max_threshold, args.threshold_step)
    thresholds = [round(t, 2) for t in thresholds]  # Round to avoid floating point issues
    
    print(f"Running model comparison with {len(thresholds)} thresholds from "
          f"{thresholds[0]} to {thresholds[-1]}")
    
    # Load and prepare data
    train_df, test_df, y_train, y_test = load_and_prepare_data(
        args.data_path, 
        test_size=args.test_size
    )
    
    # Extract loan amounts for profit calculations
    loan_amounts = test_df['nominal_contract_value']
    
    # -------------------------------------------------------------------------
    # MODEL 1: Application-time features only
    # -------------------------------------------------------------------------
    print("\n===== MODEL 1: Application-Time Features Only =====")
    
    # Filter to include only application-time features
    X_train_1 = filter_features(train_df, APPLICATION_FEATURES)
    X_test_1 = filter_features(test_df, APPLICATION_FEATURES)
    
    # Train Model 1
    model1 = train_model(X_train_1, y_train)
    
    # Evaluate Model 1
    model1_eval = evaluate_model(model1, X_test_1, y_test, "Model 1 (Application Features)")
    
    # Analyze profit metrics for Model 1
    model1_profit = analyze_model_profit(
        y_test, 
        model1_eval['predictions'], 
        loan_amounts,
        thresholds, 
        "model1",
        args.output_dir
    )
    
    # Combine results for Model 1
    model1_results = {
        'predictions': model1_eval['predictions'],
        'metrics': model1_eval['metrics'],
        'profit_analysis': model1_profit
    }
    
    # -------------------------------------------------------------------------
    # MODEL 2: Application-time features + September payment data
    # -------------------------------------------------------------------------
    print("\n===== MODEL 2: With September Payment Data =====")
    
    # Combine feature lists for Model 2
    model2_features = list(set(APPLICATION_FEATURES + SEPTEMBER_PAYMENT_FEATURES))
    
    # Filter to include both application-time and September payment features
    X_train_2 = filter_features(train_df, model2_features)
    X_test_2 = filter_features(test_df, model2_features)
    
    # Train Model 2
    model2 = train_model(X_train_2, y_train)
    
    # Evaluate Model 2
    model2_eval = evaluate_model(model2, X_test_2, y_test, "Model 2 (With September Data)")
    
    # Analyze profit metrics for Model 2
    model2_profit = analyze_model_profit(
        y_test, 
        model2_eval['predictions'], 
        loan_amounts,
        thresholds, 
        "model2",
        args.output_dir
    )
    
    # Combine results for Model 2
    model2_results = {
        'predictions': model2_eval['predictions'],
        'metrics': model2_eval['metrics'],
        'profit_analysis': model2_profit
    }
    
    # -------------------------------------------------------------------------
    # COMPARE MODELS
    # -------------------------------------------------------------------------
    print("\n===== MODEL COMPARISON =====")
    
    # Compare the models
    comparison = compare_models(model1_results, model2_results, args.output_dir)
    
    # Create feature importance comparison
    feature_importance_comparison(
        model1, model2, 
        list(X_train_1.columns), list(X_train_2.columns),
        args.output_dir
    )
    
    # -------------------------------------------------------------------------
    # GENERATE PREDICTIONS FOR HOLDOUT DATA (if provided)
    # -------------------------------------------------------------------------
    if args.holdout_path and os.path.exists(args.holdout_path):
        print("\n===== GENERATING PREDICTIONS FOR HOLDOUT DATA =====")
        
        # Load holdout data
        holdout_df = pd.read_csv(args.holdout_path)
        
        # Generate and save predictions
        holdout_output_path = os.path.join(args.output_dir, "holdout_predictions.csv")
        save_predictions_for_holdout(
            holdout_df,
            model1, model2,
            list(X_train_1.columns), list(X_train_2.columns),
            holdout_output_path
        )
    
    # -------------------------------------------------------------------------
    # GENERATE REPORTS
    # -------------------------------------------------------------------------
    print("\n===== GENERATING REPORTS =====")
    
    # Create the PDF report
    report_generator = ModelComparisonReport(
        data_dir=args.output_dir,
        output_dir=args.output_dir
    )
    pdf_path = report_generator.generate_report()
    
    # Create the Markdown report
    md_path = create_markdown_report(
        data_dir=args.output_dir,
        output_dir=args.output_dir
    )
    
    print(f"\nReports generated:")
    print(f"  - PDF Report: {pdf_path}")
    print(f"  - Markdown Report: {md_path}")
    
    print("\nModel comparison completed successfully!")

if __name__ == "__main__":
    main()
