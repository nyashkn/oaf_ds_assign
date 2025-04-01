"""
Threshold analysis functions for profitability analysis.

This module provides functions to analyze profitability metrics for different
repayment rate thresholds, including single and multiple threshold analyses.
"""

import pandas as pd
import numpy as np
import os
import json
from typing import Dict, List, Optional, Tuple, Any, Union

from .metrics import (
    calculate_loan_metrics,
    calculate_profit_metrics,
    calculate_classification_metrics,
    calculate_business_metrics
)

def analyze_threshold(
    df: pd.DataFrame,
    threshold: float,
    predicted_col: str = 'predicted',
    actual_col: str = 'actual',
    loan_amount_col: str = 'nominal_contract_value',
    gross_margin: float = 0.3
) -> Dict[str, Any]:
    """
    Comprehensively analyze a single threshold.
    
    Args:
        df: DataFrame with actual and predicted repayment rates
        threshold: Repayment rate threshold for classification
        predicted_col: Column name for predicted repayment rates
        actual_col: Column name for actual repayment rates
        loan_amount_col: Column name for loan amounts
        gross_margin: Gross margin percentage
        
    Returns:
        Dictionary with comprehensive threshold analysis results
    """
    # Calculate loan metrics
    loan_metrics = calculate_loan_metrics(
        df, threshold, predicted_col, actual_col, loan_amount_col
    )
    
    # Calculate profit metrics
    profit_metrics = calculate_profit_metrics(
        df, threshold, loan_metrics, predicted_col, actual_col, loan_amount_col, gross_margin
    )
    
    # Calculate classification metrics
    class_metrics = calculate_classification_metrics(loan_metrics)
    
    # Calculate business metrics
    business_metrics = calculate_business_metrics(loan_metrics, profit_metrics)
    
    # Combine all metrics into a comprehensive analysis
    analysis = {
        'threshold': threshold,
        'loan_metrics': loan_metrics,
        'profit_metrics': profit_metrics,
        'classification_metrics': class_metrics,
        'business_metrics': business_metrics
    }
    
    return analysis

def analyze_multiple_thresholds(
    df: pd.DataFrame,
    thresholds: List[float],
    predicted_col: str = 'predicted',
    actual_col: str = 'actual',
    loan_amount_col: str = 'nominal_contract_value',
    gross_margin: float = 0.3,
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Analyze multiple thresholds and find the optimal one.
    
    Args:
        df: DataFrame with actual and predicted repayment rates
        thresholds: List of repayment rate thresholds to evaluate
        predicted_col: Column name for predicted repayment rates
        actual_col: Column name for actual repayment rates
        loan_amount_col: Column name for loan amounts
        gross_margin: Gross margin percentage
        output_dir: Directory to save analysis results
        
    Returns:
        Dictionary with threshold analyses and optimal threshold
    """
    print("\n=== Threshold Analysis ===")
    
    # Analyze each threshold
    threshold_analyses = []
    
    for threshold in thresholds:
        analysis = analyze_threshold(
            df, threshold, predicted_col, actual_col, loan_amount_col, gross_margin
        )
        threshold_analyses.append(analysis)
    
    # Convert to DataFrame for easier analysis
    threshold_data = []
    
    for analysis in threshold_analyses:
        data = {
            'threshold': analysis['threshold'],
            'approval_rate': analysis['business_metrics']['approval_rate'],
            'actual_repayment_rate': analysis['loan_metrics']['actual_repayment_rate'],
            'total_actual_profit': analysis['profit_metrics']['total_actual_profit'],
            'money_left_on_table': analysis['profit_metrics']['money_left_on_table'],
            'accuracy': analysis['classification_metrics']['accuracy'],
            'precision': analysis['classification_metrics']['precision'],
            'recall': analysis['classification_metrics']['recall'],
            'f1_score': analysis['classification_metrics']['f1_score']
        }
        threshold_data.append(data)
    
    threshold_df = pd.DataFrame(threshold_data)
    
    # Find optimal threshold based on total profit
    if len(threshold_df) > 0:
        optimal_idx = threshold_df['total_actual_profit'].idxmax()
        optimal_threshold = threshold_df.loc[optimal_idx, 'threshold']
        optimal_analysis = threshold_analyses[optimal_idx]
        
        print(f"\nOptimal threshold based on total profit: {optimal_threshold:.2f}")
        print(f"  Approval rate: {threshold_df.loc[optimal_idx, 'approval_rate']:.1%}")
        print(f"  Actual repayment rate: {threshold_df.loc[optimal_idx, 'actual_repayment_rate']:.2f}")
        print(f"  Total profit: {threshold_df.loc[optimal_idx, 'total_actual_profit']:.2f}")
        print(f"  Money left on table: {threshold_df.loc[optimal_idx, 'money_left_on_table']:.2f}")
    else:
        optimal_threshold = None
        optimal_analysis = None
        print("\nNo optimal threshold found (empty threshold dataframe)")
    
    # Save results if output directory provided
    if output_dir:
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save threshold metrics to CSV
        threshold_path = os.path.join(output_dir, "threshold_metrics.csv")
        threshold_df.to_csv(threshold_path, index=False)
        
        # Save full analyses
        analyses_summary = {}
        for analysis in threshold_analyses:
            threshold = analysis['threshold']
            analyses_summary[f"threshold_{threshold:.2f}"] = {
                # Convert the most important metrics to simple dictionaries
                'threshold': threshold,
                'loan_counts': {
                    'total_loans': analysis['loan_metrics']['total_loans'],
                    'n_approved': analysis['loan_metrics']['n_approved'],
                    'n_rejected': analysis['loan_metrics']['n_rejected'],
                    'n_good': analysis['loan_metrics']['n_good'],
                    'n_bad': analysis['loan_metrics']['n_bad'],
                    'n_true_pos': analysis['loan_metrics']['n_true_pos'],
                    'n_false_pos': analysis['loan_metrics']['n_false_pos'],
                    'n_true_neg': analysis['loan_metrics']['n_true_neg'],
                    'n_false_neg': analysis['loan_metrics']['n_false_neg']
                },
                'profit_metrics': {
                    'total_actual_profit': analysis['profit_metrics']['total_actual_profit'],
                    'money_left_on_table': analysis['profit_metrics']['money_left_on_table'],
                    'approved_profit': analysis['profit_metrics']['approved_profit']
                },
                'classification_metrics': analysis['classification_metrics'],
                'business_metrics': analysis['business_metrics']
            }
        
        # Save analyses summary
        summary_path = os.path.join(output_dir, "threshold_analyses.json")
        with open(summary_path, 'w') as f:
            json.dump(analyses_summary, f, indent=2)
        
        print(f"\nThreshold analysis saved to {output_dir}")
    
    return {
        'threshold_analyses': threshold_analyses,
        'threshold_df': threshold_df,
        'optimal_threshold': optimal_threshold,
        'optimal_analysis': optimal_analysis
    }

def analyze_cutoff_tradeoffs(
    df: pd.DataFrame,
    target_threshold: float,
    predicted_col: str = 'predicted',
    actual_col: str = 'actual',
    loan_amount_col: str = 'nominal_contract_value',
    gross_margin: float = 0.3,
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Analyze the tradeoff between target threshold and profitability.
    
    Args:
        df: DataFrame with actual and predicted repayment rates
        target_threshold: Target repayment rate threshold for comparison
        predicted_col: Column name for predicted repayment rates
        actual_col: Column name for actual repayment rates
        loan_amount_col: Column name for loan amounts
        gross_margin: Gross margin percentage
        output_dir: Directory to save tradeoff analysis results
        
    Returns:
        Dictionary with tradeoff analysis results
    """
    print(f"\n=== Cutoff Tradeoff Analysis (Target: {target_threshold:.2f}) ===")
    
    # Perform threshold analysis
    analysis = analyze_threshold(
        df, target_threshold, predicted_col, actual_col, loan_amount_col, gross_margin
    )
    
    # Extract key metrics for simplified reporting
    loan_metrics = analysis['loan_metrics']
    profit_metrics = analysis['profit_metrics']
    class_metrics = analysis['classification_metrics']
    business_metrics = analysis['business_metrics']
    
    # Create summary dictionary
    tradeoff_summary = {
        'target_threshold': target_threshold,
        'gross_margin': gross_margin,
        'loan_counts': {
            'total_loans': loan_metrics['total_loans'],
            'good_loans': loan_metrics['n_good'],
            'bad_loans': loan_metrics['n_bad'],
            'true_positives': loan_metrics['n_true_pos'],
            'false_positives': loan_metrics['n_false_pos'],
            'true_negatives': loan_metrics['n_true_neg'],
            'false_negatives': loan_metrics['n_false_neg'],
            'approved_loans': loan_metrics['n_approved'],
            'rejected_loans': loan_metrics['n_rejected']
        },
        'loan_values': {
            'total_value': loan_metrics['good_loan_value'] + loan_metrics['bad_loan_value'],
            'good_loan_value': loan_metrics['good_loan_value'],
            'bad_loan_value': loan_metrics['bad_loan_value'],
            'true_positive_value': loan_metrics['true_pos_value'],
            'false_positive_value': loan_metrics['false_pos_value'],
            'true_negative_value': loan_metrics['true_neg_value'],
            'false_negative_value': loan_metrics['false_neg_value'],
            'approved_value': loan_metrics['approved_value'],
            'rejected_value': loan_metrics['rejected_value']
        },
        'profit_metrics': {
            'total_potential_profit': profit_metrics['total_potential_profit'],
            'total_actual_profit': profit_metrics['total_actual_profit'],
            'true_positive_profit': profit_metrics['true_pos_profit'],
            'false_positive_profit': profit_metrics['false_pos_profit'],
            'false_positive_loss': profit_metrics['false_pos_loss'],
            'true_negative_avoided_loss': profit_metrics['true_neg_avoided_loss'],
            'false_negative_profit': profit_metrics['false_neg_profit'],
            'approved_profit': profit_metrics['approved_profit'],
            'money_left_on_table': profit_metrics['money_left_on_table']
        },
        'classification_metrics': class_metrics,
        'business_metrics': business_metrics
    }
    
    # Print summary
    print("\nLoan Classification Summary:")
    print(f"  Total loans: {loan_metrics['total_loans']}")
    print(f"  Good loans (actual >= {target_threshold:.2f}): {loan_metrics['n_good']} " +
          f"({loan_metrics['n_good']/loan_metrics['total_loans']*100:.1f}%)")
    print(f"  Bad loans (actual < {target_threshold:.2f}): {loan_metrics['n_bad']} " + 
          f"({loan_metrics['n_bad']/loan_metrics['total_loans']*100:.1f}%)")
    
    print("\nModel Performance:")
    print(f"  Accuracy: {class_metrics['accuracy']:.2f}")
    print(f"  Approval rate: {business_metrics['approval_rate']:.2f}")
    print(f"  Profit precision: {business_metrics['profit_precision']:.2f}")
    
    print("\nProfit Analysis:")
    print(f"  Total potential profit: {profit_metrics['total_potential_profit']:.2f}")
    print(f"  Total actual profit: {profit_metrics['total_actual_profit']:.2f}")
    print(f"  Approved loans profit: {profit_metrics['approved_profit']:.2f}")
    print(f"  Money left on table: {profit_metrics['money_left_on_table']:.2f}")
    
    # Save results if output directory provided
    if output_dir:
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save tradeoff summary
        summary_path = os.path.join(output_dir, "tradeoff_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(tradeoff_summary, f, indent=2)
        
        print(f"\nTradeoff analysis saved to {output_dir}")
    
    return {
        'analysis': analysis,
        'tradeoff_summary': tradeoff_summary
    }
