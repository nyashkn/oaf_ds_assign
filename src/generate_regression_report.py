#!/usr/bin/env python
"""
Generate a comprehensive PDF report for the regression-based loan repayment prediction model.

This script creates a detailed report using the reporting module from the scorecard_regression package,
presenting model performance, threshold analysis, and profitability metrics in a structured PDF document.

Example usage:
    python src/generate_regression_report.py --version_dir data/processed/regression_results/v1
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

from src.scorecard_regression.reporting import (
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
    
    # Alternative: try loading from CSV
    imp_path = os.path.join(version_dir, "feature_importance.csv")
    if os.path.exists(imp_path):
        imp_df = pd.read_csv(imp_path)
        if 'feature' in imp_df.columns and 'importance' in imp_df.columns:
            return dict(zip(imp_df['feature'], imp_df['importance']))
    
    return {}

def extract_threshold_metrics(version_dir):
    """Extract threshold analysis metrics."""
    threshold_path = os.path.join(version_dir, "threshold_metrics.csv")
    
    if os.path.exists(threshold_path):
        return pd.read_csv(threshold_path)
    
    return None

def extract_margin_metrics(version_dir):
    """Extract margin analysis metrics."""
    margin_path = os.path.join(version_dir, "margin_threshold_metrics.csv")
    
    if os.path.exists(margin_path):
        return pd.read_csv(margin_path)
    
    return None

def prepare_summary_data(version_dir):
    """Prepare executive summary data."""
    # Load model evaluation results
    eval_results = load_data(version_dir, "evaluation_results.json")
    holdout_results = load_data(version_dir, "holdout_evaluation.json")
    threshold_metrics = extract_threshold_metrics(version_dir)
    
    # Extract performance metrics
    model_performance = {}
    
    if eval_results and 'metrics' in eval_results:
        model_performance['train'] = eval_results['metrics'].get('train', {})
        model_performance['test'] = eval_results['metrics'].get('test', {})
    
    if holdout_results and 'metrics' in holdout_results:
        model_performance['holdout'] = holdout_results['metrics'].get('holdout', {})
    
    # Extract business impact data
    business_impact = {}
    
    if threshold_metrics is not None and 'optimal_threshold' in threshold_metrics.columns:
        # Find the optimal threshold (assuming it's marked in the data)
        opt_row = threshold_metrics[threshold_metrics['is_optimal'] == True].iloc[0] if 'is_optimal' in threshold_metrics.columns else threshold_metrics.iloc[0]
        
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
        'feature_importance': feature_importance
    }

def generate_report(version_dir, output_path, logo_path=None):
    """Generate a comprehensive report from model results."""
    print(f"Generating report from {version_dir} to {output_path}")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Initialize report
    report = initialize_regression_report(
        version_path=os.path.dirname(output_path),
        title="Loan Repayment Rate Regression Model Report",
        subtitle="OneAcre Fund - Tupande Credit Risk Analysis",
        author="Data Science Team"
    )
    
    # Load data for each report section
    inspection_results = load_data(version_dir, "data_inspection.json")
    model_results = load_data(version_dir, "model_results.json")
    evaluation_results = load_data(version_dir, "evaluation_results.json")
    threshold_results = {
        'threshold_metrics': extract_threshold_metrics(version_dir),
        'optimal_threshold': 0.65  # Default value if not found
    }
    
    if threshold_results['threshold_metrics'] is not None and 'is_optimal' in threshold_results['threshold_metrics'].columns:
        opt_row = threshold_results['threshold_metrics'][threshold_results['threshold_metrics']['is_optimal'] == True]
        if not opt_row.empty:
            threshold_results['optimal_threshold'] = opt_row.iloc[0]['threshold']
    
    margin_results = {
        'margin_threshold_metrics': extract_margin_metrics(version_dir),
        'optimal_parameters': {
            'margin': 0.3,
            'threshold': 0.65,
            'profit': 6000000
        }
    }
    
    if margin_results['margin_threshold_metrics'] is not None and 'actual_profit' in margin_results['margin_threshold_metrics'].columns:
        opt_idx = margin_results['margin_threshold_metrics']['actual_profit'].idxmax()
        opt_row = margin_results['margin_threshold_metrics'].loc[opt_idx]
        margin_results['optimal_parameters'] = {
            'margin': opt_row.get('gross_margin', 0.3),
            'threshold': opt_row.get('threshold', 0.65),
            'profit': opt_row.get('actual_profit', 0)
        }
    
    holdout_results = load_data(version_dir, "holdout_evaluation.json")
    
    # Prepare summary data
    summary_data = prepare_summary_data(version_dir)
    
    # Add report sections
    add_executive_summary_section(report, summary_data)
    
    if inspection_results:
        add_data_inspection_section(report, inspection_results)
    
    if model_results:
        add_model_development_section(report, model_results)
    
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
        report.add_json_data(evaluation_results.get('metrics', {}), 
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
