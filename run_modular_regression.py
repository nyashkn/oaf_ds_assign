#!/usr/bin/env python3
"""
Modular Regression-based Loan Repayment Rate Prediction Script

This script demonstrates the regression-based approach for predicting loan
repayment rates as an alternative to binary classification. It provides:

1. A complete regression modeling workflow with data preprocessing, feature selection
2. Comparison of regression vs. classification approaches
3. Profitability analysis at different thresholds
4. Comparison of model performance with/without September payment data

The script generates comprehensive reports to help understand the tradeoffs
between different modeling approaches and threshold selection.
"""

import os
import argparse
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

# Import the scorecard regression package
from src.scorecard_regression import (
    # Utils and constants
    create_version_path,
    EXCLUDE_VARS,
    DEFAULT_REGRESSION_PARAMS,
    
    # Data preparation
    exclude_leakage_variables,
    partition_data,
    check_multicollinearity,
    select_significant_variables,
    
    # Modeling
    train_regression_model,
    evaluate_regression_model,
    cross_validate_regression,
    compare_regression_vs_classification,
    
    # Profitability analysis
    analyze_multiple_thresholds,
    analyze_cutoff_tradeoffs,
    calculate_business_metrics
)

# Import model comparison module
from src.scorecard_regression.model_comparison import (
    APPLICATION_FEATURES,
    SEPTEMBER_PAYMENT_FEATURES,
    filter_features,
    feature_importance_comparison,
    evaluate_model,
    analyze_model_profit,
    compare_models
)

# Import reporting module
from src.scorecard_regression.reporting import (
    ModelComparisonReport,
    create_markdown_report
)

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='OAF Loan Repayment Rate Regression Modeling')
    
    # Data parameters
    parser.add_argument('--features', type=str, default="data/processed/all_features.csv",
                       help='Path to features CSV file')
    parser.add_argument('--target', type=str, default='sept_23_repayment_rate',
                       help='Target variable name (continuous repayment rate)')
    parser.add_argument('--id-col', type=str, default='client_id',
                       help='ID column name')
    parser.add_argument('--loan-value-col', type=str, default='nominal_contract_value',
                       help='Column containing loan amounts for profitability calculations')
    
    # Sampling and partitioning
    parser.add_argument('--sample', type=int, default=None,
                       help='Sample size for testing (use None for full dataset)')
    parser.add_argument('--stratify', action='store_true',
                       help='Use stratified sampling for data partitioning')
    parser.add_argument('--validation', action='store_true',
                       help='Include validation set in addition to train/test')
    
    # Feature selection
    parser.add_argument('--correlation-threshold', type=float, default=0.1,
                       help='Minimum correlation to keep a feature')
    parser.add_argument('--importance-threshold', type=float, default=0.01,
                       help='Minimum importance to keep a feature')
    
    # Model parameters
    parser.add_argument('--reg-model', type=str, default='histgb',
                       choices=['linear', 'ridge', 'lasso', 'elasticnet', 'histgb', 'xgboost', 'lightgbm', 'catboost', 'rf'],
                       help='Regression model type')
    parser.add_argument('--clf-model', type=str, default='histgb',
                       choices=['logistic', 'histgb', 'rf'],
                       help='Classification model type for comparison')
    parser.add_argument('--model-params', type=str, default=None,
                       help='Model parameters as JSON string')
    
    # Profitability analysis
    parser.add_argument('--margin', type=float, default=0.16,
                       help='Gross margin as decimal (e.g., 0.16 = 16%%)')
    parser.add_argument('--default-loss-rate', type=float, default=1.0,
                       help='Loss rate on defaulted loans (1.0 = 100%%)')
    parser.add_argument('--thresholds', type=str, default=None,
                       help='Comma-separated list of thresholds to analyze (e.g., "0.7,0.75,0.8,0.85,0.9")')
    
    # Output parameters
    parser.add_argument('--output', type=str, default="data/processed/regression_modelling",
                       help='Base path for output files')
    parser.add_argument('--plots', action='store_true',
                       help='Generate and save plots')
    parser.add_argument('--cross-validate', action='store_true',
                       help='Perform cross-validation')
    parser.add_argument('--folds', type=int, default=5,
                       help='Number of cross-validation folds')
    
    return parser.parse_args()

def main():
    """Run the main regression modeling workflow."""
    args = parse_arguments()
    
    print("=" * 80)
    print("OAF Loan Repayment Rate Regression Modeling")
    print("=" * 80)
    
    # Create version directory
    version_path = create_version_path(args.output)
    print(f"Results will be saved to: {version_path}")
    
    # Parse model parameters if provided
    model_params = None
    if args.model_params:
        try:
            model_params = json.loads(args.model_params)
            print(f"Using custom model parameters: {model_params}")
        except json.JSONDecodeError:
            print(f"Warning: Could not parse model parameters: {args.model_params}")
            print("Using default parameters instead.")
    
    # Parse thresholds if provided
    thresholds = None
    if args.thresholds:
        try:
            thresholds = [float(t) for t in args.thresholds.split(',')]
            print(f"Using custom thresholds: {thresholds}")
        except ValueError:
            print(f"Warning: Could not parse thresholds: {args.thresholds}")
            print("Using default thresholds instead.")
    
    # Step 1: Load and inspect data
    print("\n[STEP 1] Loading and preparing data...")
    try:
        df = pd.read_csv(args.features)
        print(f"Loaded data from {args.features}: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Apply sampling if requested
        if args.sample and args.sample < len(df):
            df = df.sample(n=args.sample, random_state=42)
            print(f"Using sample of {args.sample} loans")
        
        # Make a copy of loan value for later use
        loan_values = df[args.loan_value_col].copy() if args.loan_value_col in df.columns else None
        if loan_values is None:
            print(f"Warning: Loan value column '{args.loan_value_col}' not found. Profitability metrics will use a default value.")
            loan_values = np.ones(len(df))
        
        # Get target variable
        if args.target not in df.columns:
            raise ValueError(f"Target variable '{args.target}' not found in data")
        y = df[args.target].copy()
        print(f"Target variable '{args.target}' statistics:")
        print(f"  Mean: {y.mean():.4f}")
        print(f"  Std: {y.std():.4f}")
        print(f"  Min: {y.min():.4f}")
        print(f"  Max: {y.max():.4f}")
        
        # Save ID column if present
        id_col = df[args.id_col].copy() if args.id_col in df.columns else None
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return
    
    # Step 2: Exclude leakage variables
    print("\n[STEP 2] Excluding leakage variables...")
    filtered_df, excluded = exclude_leakage_variables(
        df, 
        args.target, 
        output_path=os.path.join(version_path, "data", "filtered_data.csv")
    )
    
    # Step 3: Perform multicollinearity check
    print("\n[STEP 3] Checking for multicollinearity...")
    multicollinearity_results = check_multicollinearity(
        filtered_df.drop(columns=[args.target]),
        output_path=os.path.join(version_path, "data", "multicollinearity_results.json")
    )
    
    # Step 4: Partition data
    print("\n[STEP 4] Partitioning data...")
    partitioned_data = partition_data(
        filtered_df,
        args.target,
        include_validation=args.validation,
        stratify=args.stratify,
        output_dir=os.path.join(version_path, "data", "partition")
    )
    
    # Step 5: Variable selection
    print("\n[STEP 5] Selecting significant variables...")
    selection_results = select_significant_variables(
        partitioned_data['train'].drop(columns=[args.target]), 
        partitioned_data['train'][args.target],
        correlation_threshold=args.correlation_threshold,
        importance_threshold=args.importance_threshold,
        output_dir=os.path.join(version_path, "data", "variable_selection")
    )
    
    # Use selected features
    selected_features = selection_results['selected_features']
    print(f"Selected {len(selected_features)} features for modeling")
    
    # Prepare datasets with selected features and handle missing values
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='mean')
    
    X_train = pd.DataFrame(
        imputer.fit_transform(partitioned_data['train'][selected_features]),
        columns=selected_features,
        index=partitioned_data['train'].index
    )
    y_train = partitioned_data['train'][args.target]
    
    X_test = pd.DataFrame(
        imputer.transform(partitioned_data['test'][selected_features]),
        columns=selected_features,
        index=partitioned_data['test'].index
    )
    y_test = partitioned_data['test'][args.target]
    
    # Save test set loan values for later profitability analysis
    test_loan_values = loan_values.loc[partitioned_data['test'].index] if hasattr(loan_values, 'loc') else loan_values[partitioned_data['test'].index]
    
    # Optional: Cross-validation
    if args.cross_validate:
        print("\n[STEP 6] Performing cross-validation...")
        cv_results = cross_validate_regression(
            X_train, 
            y_train,
            model_type=args.reg_model,
            model_params=model_params,
            n_folds=args.folds,
            output_dir=os.path.join(version_path, "models", "cross_validation")
        )
    
    # Step 6/7: Train regression model
    print(f"\n[STEP {'7' if args.cross_validate else '6'}] Training regression model...")
    model_result = train_regression_model(
        X_train,
        y_train,
        model_type=args.reg_model,
        model_params=model_params,
        output_dir=os.path.join(version_path, "models", "regression")
    )
    
    # Step 7/8: Evaluate regression model
    print(f"\n[STEP {'8' if args.cross_validate else '7'}] Evaluating regression model...")
    eval_result = evaluate_regression_model(
        model_result['model'],
        X_test,
        y_test,
        output_dir=os.path.join(version_path, "metrics", "regression")
    )
    
    # Get regression predictions for threshold analysis
    reg_predictions = eval_result['predictions']
    
    # Step 8/9: Train classification model for comparison
    print(f"\n[STEP {'9' if args.cross_validate else '8'}] Training classification model for comparison...")
    # Create binary target for classification
    binary_target_train = (y_train >= 0.8).astype(int)
    binary_target_test = (y_test >= 0.8).astype(int)
    
    # Choose classifier based on args.clf_model
    if args.clf_model == 'logistic':
        clf = LogisticRegression(random_state=42)
    elif args.clf_model == 'histgb':
        clf = GradientBoostingClassifier(random_state=42)
    else:  # rf
        clf = RandomForestClassifier(random_state=42)
    
    # Fit classifier
    clf.fit(X_train, binary_target_train)
    
    # Get probabilities and accuracy
    clf_probas = clf.predict_proba(X_test)[:, 1]
    clf_preds = clf.predict(X_test)
    clf_accuracy = accuracy_score(binary_target_test, clf_preds)
    clf_auc = roc_auc_score(binary_target_test, clf_probas)
    
    print(f"Classification model results:")
    print(f"  Accuracy: {clf_accuracy:.4f}")
    print(f"  AUC: {clf_auc:.4f}")
    
    # Step 9/10: Compare regression and classification approaches
    print(f"\n[STEP {'10' if args.cross_validate else '9'}] Comparing regression and classification approaches...")
    comparison_result = compare_regression_vs_classification(
        y_test,
        reg_predictions,
        clf_probas,
        test_loan_values,
        thresholds=thresholds,
        margin=args.margin,
        default_loss_rate=args.default_loss_rate,
        output_dir=os.path.join(version_path, "metrics", "comparison")
    )
    
    # Step 10/11: Perform detailed threshold analysis for regression model
    print(f"\n[STEP {'11' if args.cross_validate else '10'}] Performing threshold analysis...")
    threshold_results = analyze_cutoff_tradeoffs(
        y_test,
        reg_predictions,
        test_loan_values,
        thresholds=thresholds,
        business_params={
            'margin': args.margin,
            'default_loss_rate': args.default_loss_rate
        },
        output_path=os.path.join(version_path, "metrics", "threshold_analysis.json")
    )
    
    # Extract optimal thresholds for different objectives
    profit_threshold = threshold_results['recommendations']['profit_focused']
    roi_threshold = threshold_results['recommendations']['roi_focused']
    balanced_threshold = threshold_results['recommendations']['balanced']
    
    # Calculate business metrics using optimal thresholds
    profit_metrics = calculate_business_metrics(
        y_test,
        reg_predictions,
        test_loan_values,
        threshold=profit_threshold,
        margin=args.margin,
        default_loss_rate=args.default_loss_rate
    )
    
    roi_metrics = calculate_business_metrics(
        y_test,
        reg_predictions,
        test_loan_values,
        threshold=roi_threshold,
        margin=args.margin,
        default_loss_rate=args.default_loss_rate
    )
    
    balanced_metrics = calculate_business_metrics(
        y_test,
        reg_predictions,
        test_loan_values,
        threshold=balanced_threshold,
        margin=args.margin,
        default_loss_rate=args.default_loss_rate
    )
    
    # Print summary of optimal thresholds
    print("\nOptimal thresholds for regression model:")
    print(f"  Profit-focused: {profit_threshold:.2f}")
    print(f"    Approval rate: {profit_metrics['loan_metrics']['n_loans']['approval_rate']:.1%}")
    print(f"    Profit: {profit_metrics['profit_metrics']['actual_profit']:.2f}")
    print(f"    ROI: {profit_metrics['profit_metrics']['roi']:.1%}")
    
    print(f"\n  ROI-focused: {roi_threshold:.2f}")
    print(f"    Approval rate: {roi_metrics['loan_metrics']['n_loans']['approval_rate']:.1%}")
    print(f"    Profit: {roi_metrics['profit_metrics']['actual_profit']:.2f}")
    print(f"    ROI: {roi_metrics['profit_metrics']['roi']:.1%}")
    
    print(f"\n  Balanced: {balanced_threshold:.2f}")
    print(f"    Approval rate: {balanced_metrics['loan_metrics']['n_loans']['approval_rate']:.1%}")
    print(f"    Profit: {balanced_metrics['profit_metrics']['actual_profit']:.2f}")
    print(f"    ROI: {balanced_metrics['profit_metrics']['roi']:.1%}")
    
    # Save summary
    summary = {
        'version_path': version_path,
        'input_file': args.features,
        'target_variable': args.target,
        'regression_model': args.reg_model,
        'regression_metrics': eval_result['test_metrics'],
        'classification_model': args.clf_model,
        'classification_metrics': {
            'accuracy': float(clf_accuracy),
            'auc': float(clf_auc)
        },
        'optimal_thresholds': {
            'profit_focused': float(profit_threshold),
            'roi_focused': float(roi_threshold),
            'balanced': float(balanced_threshold)
        },
        'profit_metrics': {
            'approval_rate': float(profit_metrics['loan_metrics']['n_loans']['approval_rate']),
            'profit': float(profit_metrics['profit_metrics']['actual_profit']),
            'roi': float(profit_metrics['profit_metrics']['roi'])
        },
        'roi_metrics': {
            'approval_rate': float(roi_metrics['loan_metrics']['n_loans']['approval_rate']),
            'profit': float(roi_metrics['profit_metrics']['actual_profit']),
            'roi': float(roi_metrics['profit_metrics']['roi'])
        },
        'comparison': {
            'regression_vs_classification': {
                'profit_difference': {
                    'absolute': float(comparison_result['performance_difference']['profit']['absolute']),
                    'relative': float(comparison_result['performance_difference']['profit']['relative'])
                },
                'roi_difference': {
                    'absolute': float(comparison_result['performance_difference']['roi']['absolute']),
                    'relative': float(comparison_result['performance_difference']['roi']['relative'])
                }
            }
        }
    }
    
    # Step 11/12: Compare models with/without September payment data
    print(f"\n[STEP {'12' if args.cross_validate else '11'}] Comparing models with/without September payment data...")
    
    # Create directory for model comparison
    model_comparison_dir = os.path.join(version_path, "model_comparison")
    os.makedirs(model_comparison_dir, exist_ok=True)
    
    # Filter features for Model 1 (application-time features only)
    print("\n===== MODEL 1: Application-Time Features Only =====")
    # Filter features for Model 1 (application-time features only)
    X_train_app, encoder_dict1 = filter_features(partitioned_data['train'], APPLICATION_FEATURES)
    X_test_app, _ = filter_features(partitioned_data['test'], APPLICATION_FEATURES, encoder_dict=encoder_dict1)
    
    # Train Model 1
    model1 = train_regression_model(X_train_app, y_train, model_type=args.reg_model, model_params=model_params)['model']
    
    # Evaluate Model 1
    model1_eval = evaluate_model(model1, X_test_app, y_test, "Model 1 (Application Features)")
    
    # Filter features for Model 2 (application + September payment features)
    print("\n===== MODEL 2: With September Payment Data =====")
    model2_features = list(set(APPLICATION_FEATURES + SEPTEMBER_PAYMENT_FEATURES))
    X_train_full, encoder_dict2 = filter_features(partitioned_data['train'], model2_features)
    X_test_full, _ = filter_features(partitioned_data['test'], model2_features, encoder_dict=encoder_dict2)
    
    # Train Model 2
    model2 = train_regression_model(X_train_full, y_train, model_type=args.reg_model, model_params=model_params)['model']
    
    # Evaluate Model 2
    model2_eval = evaluate_model(model2, X_test_full, y_test, "Model 2 (With September Data)")
    
    # Analyze model profits at different thresholds
    from src.scorecard_regression.model_comparison.comparison import analyze_model_profit
    
    # Define thresholds if not provided
    if thresholds is None:
        thresholds = np.arange(0.5, 0.95, 0.02)
        thresholds = [round(t, 2) for t in thresholds]
    
    # Analyze profit metrics for both models
    model1_profit = analyze_model_profit(
        y_test, 
        model1_eval['predictions'], 
        test_loan_values,
        thresholds, 
        "model1",
        model_comparison_dir
    )
    
    model2_profit = analyze_model_profit(
        y_test, 
        model2_eval['predictions'], 
        test_loan_values,
        thresholds, 
        "model2",
        model_comparison_dir
    )
    
    # Combine results
    model1_results = {
        'predictions': model1_eval['predictions'],
        'metrics': model1_eval['metrics'],
        'profit_analysis': model1_profit
    }
    
    model2_results = {
        'predictions': model2_eval['predictions'],
        'metrics': model2_eval['metrics'],
        'profit_analysis': model2_profit
    }
    
    # Compare the models
    from src.scorecard_regression.model_comparison.comparison import compare_models
    comparison = compare_models(model1_results, model2_results, model_comparison_dir)
    
    # Create feature importance comparison
    feature_importance_comparison(
        model1, model2, 
        list(X_train_app.columns), list(X_train_full.columns),
        model_comparison_dir
    )
    
    # Generate comparison reports
    print("\n===== GENERATING MODEL COMPARISON REPORTS =====")
    
    # Create the PDF report
    report_generator = ModelComparisonReport(
        data_dir=model_comparison_dir,
        output_dir=model_comparison_dir
    )
    pdf_path = report_generator.generate_report()
    
    # Create the Markdown report
    md_path = create_markdown_report(
        data_dir=model_comparison_dir,
        output_dir=model_comparison_dir
    )
    
    print(f"\nModel comparison reports generated:")
    print(f"  - PDF Report: {pdf_path}")
    print(f"  - Markdown Report: {md_path}")
    
    # Update summary with model comparison results
    summary['model_comparison'] = {
        'performance_improvement': comparison['performance_improvement'],
        'profit_improvement': comparison['profit_improvement'],
        'roi_improvement': comparison['roi_improvement'],
        'reports': {
            'pdf': pdf_path,
            'markdown': md_path
        }
    }
    
    # Save summary
    summary_path = os.path.join(version_path, "summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nAnalysis complete. Results saved to {version_path}")
    print(f"Summary saved to {summary_path}")

if __name__ == "__main__":
    main()
