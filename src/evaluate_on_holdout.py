#!/usr/bin/env python
"""
Evaluate a trained regression model on the holdout dataset.

This script loads a trained model, makes predictions on a holdout dataset,
and generates performance metrics and visualizations.

Example usage:
    python src/evaluate_on_holdout.py --model data/processed/regression_modelling/v9/4_regression_model/regression_model.pkl
                                     --holdout data/processed/holdout_all_features.csv 
                                     --output data/processed/holdout_evaluation
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import pickle
import json
import matplotlib.pyplot as plt
import seaborn as sns

# Add the project root to the path to enable relative imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
)

from src.scorecard_regression.evaluation import calculate_regression_metrics
from src.scorecard_regression.profitability import (
    analyze_multiple_thresholds,
    find_optimal_threshold,
    plot_threshold_performance,
    plot_profit_metrics
)

def load_model(model_path):
    """Load a trained model from a pickle file."""
    print(f"Loading model from {model_path}")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

def load_holdout_data(holdout_path, model, target_var='sept_23_repayment_rate'):
    """Load holdout dataset with features matching the model's expected features."""
    print(f"Loading holdout data from {holdout_path}")
    df = pd.read_csv(holdout_path)
    
    # Get the feature names from the model
    if hasattr(model, 'feature_names_in_'):
        feature_names = model.feature_names_in_
        print(f"Using {len(feature_names)} features from trained model")
    else:
        # If model doesn't have feature names, we'll try a basic approach
        print("Model doesn't have feature_names_in_ attribute. Using basic feature selection.")
        feature_names = [col for col in df.columns if col != target_var 
                         and col not in ['client_id', 'duka_name', 'Loan_Type', 'region', 
                                        'area', 'sales_territory', 'contract_start_date', 'month']]
    
    # Check if all required features exist in the holdout data
    missing_features = [feat for feat in feature_names if feat not in df.columns]
    if missing_features:
        print(f"WARNING: {len(missing_features)} features used in training are missing in holdout data:")
        for feat in missing_features[:5]:
            print(f"  - {feat}")
        if len(missing_features) > 5:
            print(f"  - ...and {len(missing_features) - 5} more")
        print("These missing features will be filled with zeros")
        
        # Add missing features with zeros
        for feat in missing_features:
            df[feat] = 0
    
    # Get extra features in holdout not used in training
    extra_features = [col for col in df.columns if col not in feature_names and col != target_var]
    if extra_features:
        print(f"NOTE: {len(extra_features)} features in holdout data were not used in training and will be ignored")
    
    # Select only the features used in training
    X = df[feature_names]
    y = df[target_var]
    
    return X, y, df

def predict_and_evaluate(model, X, y, output_dir):
    """Make predictions and evaluate performance."""
    print("\nMaking predictions on holdout data...")
    y_pred = model.predict(X)
    
    # Calculate metrics
    metrics = calculate_regression_metrics(y, y_pred)
    
    # Print key metrics
    print("\nHoldout Performance:")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  MAE: {metrics['mae']:.4f}")
    print(f"  R²: {metrics['r2']:.4f}")
    print(f"  Mean Error: {metrics['error_mean']:.4f}")
    
    # Create dataframe with actual and predicted values
    df_pred = pd.DataFrame({
        'actual': y,
        'predicted': y_pred,
        'error': y_pred - y
    })
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save metrics to JSON
    metrics_path = os.path.join(output_dir, "holdout_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save predictions to CSV
    df_pred.to_csv(os.path.join(output_dir, "holdout_predictions.csv"), index=False)
    
    # Generate plots
    fig, axes = plt.subplots(2, 1, figsize=(10, 12))
    
    # Actual vs predicted scatter plot
    axes[0].scatter(y, y_pred, alpha=0.5)
    axes[0].plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    axes[0].set_xlabel('Actual')
    axes[0].set_ylabel('Predicted')
    axes[0].set_title('Holdout Set: Actual vs Predicted')
    
    # Error distribution
    sns.histplot(df_pred['error'], kde=True, ax=axes[1])
    axes[1].axvline(0, color='red', linestyle='--')
    axes[1].set_xlabel('Prediction Error')
    axes[1].set_title('Holdout Error Distribution')
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, "holdout_performance.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    return metrics, df_pred

def analyze_profitability(df, predictions, output_dir):
    """Analyze profitability with different thresholds."""
    print("\nPerforming profitability analysis on holdout data...")
    
    # Check if we have loan amount information
    loan_amount_col = 'nominal_contract_value'
    
    # Add predictions to the dataframe
    df_with_pred = df.copy()
    df_with_pred['predicted'] = predictions
    
    # Define thresholds to analyze
    thresholds = [0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    
    # Create output directory
    threshold_dir = os.path.join(output_dir, "threshold_analysis")
    os.makedirs(threshold_dir, exist_ok=True)
    
    # Analyze thresholds
    threshold_results = analyze_multiple_thresholds(
        df_with_pred,
        thresholds=thresholds,
        predicted_col='predicted',
        actual_col='sept_23_repayment_rate',
        loan_amount_col=loan_amount_col,
        gross_margin=0.3,
        output_dir=threshold_dir
    )
    
    # Plot results
    plot_threshold_performance(
        threshold_results['threshold_df'],
        threshold_results['optimal_threshold'],
        output_path=os.path.join(threshold_dir, "threshold_performance.png")
    )
    
    plot_profit_metrics(
        threshold_results['threshold_df'],
        threshold_results['optimal_threshold'],
        output_path=os.path.join(threshold_dir, "profit_metrics.png")
    )
    
    return threshold_results

def main(model_path, holdout_path, output_dir, target_var='sept_23_repayment_rate'):
    """Main function to evaluate model on holdout data."""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    model = load_model(model_path)
    
    # Load holdout data
    X, y, df = load_holdout_data(holdout_path, model, target_var)
    
    # Predict and evaluate
    metrics, df_pred = predict_and_evaluate(model, X, y, output_dir)
    
    # Analyze profitability
    threshold_results = analyze_profitability(df, df_pred['predicted'], output_dir)
    
    # Save overall summary
    summary = {
        'model_path': model_path,
        'holdout_path': holdout_path,
        'holdout_size': len(df),
        'metrics': {
            'rmse': float(metrics['rmse']),
            'mae': float(metrics['mae']),
            'r2': float(metrics['r2']),
            'mse': float(metrics['mse']),
            'error_mean': float(metrics['error_mean']),
            'error_std': float(metrics['error_std'])
        },
        'profitability': {
            'optimal_threshold': float(threshold_results['optimal_threshold']),
            'approval_rate': float(threshold_results['threshold_df'].loc[
                threshold_results['threshold_df']['threshold'] == threshold_results['optimal_threshold'], 
                'approval_rate'
            ].values[0]),
            'actual_repayment_rate': float(threshold_results['threshold_df'].loc[
                threshold_results['threshold_df']['threshold'] == threshold_results['optimal_threshold'], 
                'actual_repayment_rate'
            ].values[0])
        }
    }
    
    # Save summary to JSON
    summary_path = os.path.join(output_dir, "holdout_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n[SUCCESS] Holdout evaluation complete!")
    print(f"Results saved to {output_dir}")
    print(f"Summary saved to {summary_path}")
    
    return summary

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate regression model on holdout data')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model pickle file')
    parser.add_argument('--holdout', type=str, required=True,
                        help='Path to holdout dataset CSV file')
    parser.add_argument('--output', type=str, default="data/processed/holdout_evaluation",
                        help='Output directory for evaluation results')
    parser.add_argument('--target', type=str, default='sept_23_repayment_rate',
                        help='Target variable name')
    
    args = parser.parse_args()
    
    # Run evaluation
    main(args.model, args.holdout, args.output, args.target)
