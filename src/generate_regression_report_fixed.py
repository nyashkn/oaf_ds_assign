#!/usr/bin/env python
"""
Generate a comprehensive PDF report for the regression-based loan repayment prediction model.

This script creates a detailed report using the reporting module from the scorecard_regression package,
presenting model performance, threshold analysis, and profitability metrics in a structured PDF document.

Example usage:
    python src/generate_regression_report_fixed.py --version_dir data/processed/regression_results/v1
                                             --output reports/regression_report.pdf
"""

import os
import sys
import argparse
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Import from the scorecard_regression package (no need for src prefix since we're already in src)
from scorecard_regression.reporting import (
    initialize_regression_report,
    add_executive_summary_section,
    add_data_inspection_section,
    add_model_development_section,
    add_evaluation_section,
    add_threshold_analysis_section,
    add_margin_analysis_section,
    add_holdout_evaluation_section
)

def load_data(version_dir, filename):
    """Load data from a JSON or CSV file."""
    path = os.path.join(version_dir, filename)
    
    if not os.path.exists(path):
        print(f"Warning: File {path} not found")
        return None
    
    if filename.endswith('.json'):
        with open(path, 'r') as f:
            return json.load(f)
    elif filename.endswith('.csv'):
        return pd.read_csv(path)
    else:
        print(f"Warning: Unsupported file format for {filename}")
        return None

def extract_feature_importance(version_dir):
    """Extract feature importance from model results."""
    model_results = load_data(version_dir, "model_results.json")
    
    if model_results and 'feature_importance' in model_results:
        return model_results['feature_importance']
    
    # Alternative: try loading from models/regression directory
    imp_path = os.path.join(version_dir, "models", "regression", "feature_importance.json")
    if os.path.exists(imp_path):
        with open(imp_path, 'r') as f:
            return json.load(f)
    
    return {}

def extract_threshold_metrics(version_dir):
    """Extract threshold analysis metrics."""
    threshold_path = os.path.join(version_dir, "metrics", "threshold_analysis_metrics.csv")
    
    if os.path.exists(threshold_path):
        return pd.read_csv(threshold_path)
    
    return None

def extract_margin_metrics(version_dir):
    """Extract margin analysis metrics."""
    margin_path = os.path.join(version_dir, "metrics", "margin_threshold_metrics.csv")
    
    if os.path.exists(margin_path):
        return pd.read_csv(margin_path)
    
    return None

def prepare_summary_data(version_dir):
    """Prepare executive summary data."""
    # Try to load summary file directly
    summary_path = os.path.join(version_dir, "summary.json")
    if os.path.exists(summary_path):
        with open(summary_path, 'r') as f:
            summary = json.load(f)
    else:
        summary = {}
    
    # Load model evaluation results - check for test metrics in regression/test_metrics.json
    eval_results_path = os.path.join(version_dir, "metrics", "regression", "test_metrics.json")
    if os.path.exists(eval_results_path):
        with open(eval_results_path, 'r') as f:
            eval_results = json.load(f)
    else:
        eval_results = None
        
    # Try getting the training metrics
    train_metrics_path = os.path.join(version_dir, "models", "regression", "training_metrics.json")
    if os.path.exists(train_metrics_path):
        with open(train_metrics_path, 'r') as f:
            train_metrics = json.load(f)
    else:
        train_metrics = None
    
    # Load holdout evaluation if it exists
    holdout_path = os.path.join(version_dir, "metrics", "regression", "holdout_metrics.json")
    if os.path.exists(holdout_path):
        with open(holdout_path, 'r') as f:
            holdout_results = json.load(f)
    else:
        holdout_results = None
    
    # Get threshold metrics
    threshold_metrics = extract_threshold_metrics(version_dir)
    
    # Extract performance metrics
    model_performance = {}
    
    if train_metrics:
        model_performance['train'] = train_metrics
    
    if eval_results and 'regression_metrics' in eval_results:
        model_performance['test'] = eval_results['regression_metrics']
    
    if holdout_results:
        model_performance['holdout'] = holdout_results
    
    # Extract business impact data from threshold analysis
    business_impact = {}
    
    if threshold_metrics is not None:
        # Try to find the optimal threshold row
        if 'is_optimal' in threshold_metrics.columns:
            opt_row = threshold_metrics[threshold_metrics['is_optimal'] == True].iloc[0]
        elif 'is_profit_optimal' in threshold_metrics.columns:
            opt_row = threshold_metrics[threshold_metrics['is_profit_optimal'] == True].iloc[0]
        else:
            # Use the row with the highest profit
            if 'actual_profit' in threshold_metrics.columns:
                opt_idx = threshold_metrics['actual_profit'].idxmax()
                opt_row = threshold_metrics.loc[opt_idx]
            else:
                opt_row = threshold_metrics.iloc[0]
        
        optimal_threshold = opt_row.get('threshold', 0.7)
        business_impact['optimal_threshold'] = optimal_threshold
        business_impact['approval_rate'] = opt_row.get('approval_rate', 0)
        business_impact['repayment_rate'] = opt_row.get('actual_repayment_rate', 0)
        business_impact['expected_profit'] = opt_row.get('actual_profit', 0)
        business_impact['profit_margin'] = opt_row.get('net_profit_margin', 0)
    
    # Extract feature importance
    feature_importance = extract_feature_importance(version_dir)
    
    return {
        'model_performance': model_performance,
        'business_impact': business_impact,
        'feature_importance': feature_importance,
        'summary': summary
    }

def generate_report(version_dir, output_path, logo_path=None):
    """Generate a comprehensive report from model results."""
    print(f"Generating report from {version_dir} to {output_path}")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Initialize report
    # Use the version directory for saving the report instead of reports folder
    version_report_path = os.path.join(version_dir, "reports", "regression_report.pdf")
    os.makedirs(os.path.dirname(version_report_path), exist_ok=True)
    
    report = initialize_regression_report(
        version_path=version_dir,
        title="Loan Repayment Rate Regression Model Report",
        subtitle="OneAcre Fund - Tupande Credit Risk Analysis",
        author="Data Science Team"
    )
    
    # Load data for each report section
    inspection_path = os.path.join(version_dir, "data", "data_inspection.json")
    if os.path.exists(inspection_path):
        with open(inspection_path, 'r') as f:
            inspection_results = json.load(f)
    else:
        inspection_results = None
    
    # Try to load model results
    model_path = os.path.join(version_dir, "models", "regression", "training_metrics.json")
    if os.path.exists(model_path):
        with open(model_path, 'r') as f:
            model_results = json.load(f)
    else:
        model_results = None
    
    # Try to load evaluation results
    eval_path = os.path.join(version_dir, "metrics", "regression", "test_metrics.json")
    if os.path.exists(eval_path):
        with open(eval_path, 'r') as f:
            evaluation_results = json.load(f)
    else:
        evaluation_results = None
    
    # Load threshold results
    threshold_results = {
        'threshold_metrics': extract_threshold_metrics(version_dir),
        'optimal_threshold': 0.70  # Default value if not found
    }
    
    # Try to find optimal threshold from the summary file
    summary_path = os.path.join(version_dir, "summary.json")
    if os.path.exists(summary_path):
        with open(summary_path, 'r') as f:
            summary = json.load(f)
            if 'optimal_thresholds' in summary and 'balanced' in summary['optimal_thresholds']:
                threshold_results['optimal_threshold'] = summary['optimal_thresholds']['balanced']
    
    # Load margin analysis results
    margin_results = {
        'margin_threshold_metrics': extract_margin_metrics(version_dir),
        'optimal_parameters': {
            'margin': 0.16,  # Default value
            'threshold': threshold_results['optimal_threshold'],
            'profit': 0
        }
    }
    
    # Load holdout results if they exist
    holdout_path = os.path.join(version_dir, "metrics", "regression", "holdout_metrics.json")
    if os.path.exists(holdout_path):
        with open(holdout_path, 'r') as f:
            holdout_results = json.load(f)
    else:
        holdout_results = None
    
    # Prepare summary data
    summary_data = prepare_summary_data(version_dir)
    
    # Add report sections
    add_executive_summary_section(report, summary_data)
    
    if inspection_results:
        add_data_inspection_section(report, inspection_results)
    
    if model_results:
        model_data = {
            'training_metrics': model_results,
            'feature_importance': summary_data['feature_importance']
        }
        add_model_development_section(report, model_data)
    
    if evaluation_results:
        add_evaluation_section(report, evaluation_results)
    
    if threshold_results['threshold_metrics'] is not None:
        add_threshold_analysis_section(report, threshold_results)
    
    if margin_results['margin_threshold_metrics'] is not None:
        add_margin_analysis_section(report, margin_results)
    
    if holdout_results:
        add_holdout_evaluation_section(report, holdout_results)
    
    # Add appendix section
    report.add_chapter("Appendix: Detailed Metrics", 
                     "Detailed performance metrics and additional analysis results.")
    
    # Save JSON data in appendix
    if evaluation_results:
        report.add_section("Raw Evaluation Metrics")
        report.add_json_data(evaluation_results, 
                           "Detailed Performance Metrics")
    
    # Save the report to PDF
    output_file = report.save()
    
    if output_file:
        print(f"Report successfully saved to {output_file}")
        return True
    else:
        print("Error: Failed to save report")
        return False

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate Regression Model PDF Report')
    parser.add_argument('--version_dir', type=str, required=True,
                      help='Directory containing model results')
    parser.add_argument('--output', type=str, default="reports/regression_report.pdf",
                      help='Output path for PDF report')
    parser.add_argument('--logo', type=str, 
                      help='Path to logo image for title page')
    
    args = parser.parse_args()
    
    # Generate the report
    generate_report(args.version_dir, args.output, args.logo)
