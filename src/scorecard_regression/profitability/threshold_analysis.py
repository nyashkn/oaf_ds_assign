"""
Threshold analysis functions for loan repayment rate prediction.

This module provides functions to analyze the impact of different threshold values
when converting continuous repayment rate predictions into binary loan approval decisions.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Any, Union

from .metrics import (
    calculate_profit_metrics,
    calculate_classification_metrics,
    calculate_loan_metrics,
    calculate_business_metrics
)

from ..constants import DEFAULT_BUSINESS_PARAMS


def analyze_threshold(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    loan_values: np.ndarray,
    threshold: float = 0.8,
    margin: float = DEFAULT_BUSINESS_PARAMS['gross_margin'],
    default_loss_rate: float = DEFAULT_BUSINESS_PARAMS['default_loss_rate'],
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Analyze the impact of a specific threshold on loan approval metrics.
    
    Args:
        y_true: True repayment rates
        y_pred: Predicted repayment rates
        loan_values: Loan amounts
        threshold: Threshold for loan approval
        margin: Gross margin as a decimal (e.g., 0.16 = 16%)
        default_loss_rate: Loss rate on defaulted loans (1.0 = 100%)
        output_path: Path to save analysis results
        
    Returns:
        Dictionary with threshold analysis results
    """
    # Calculate comprehensive business metrics
    metrics = calculate_business_metrics(
        y_true, y_pred, loan_values, threshold, margin, default_loss_rate
    )
    
    # Extract key metrics for easier access
    loan_metrics = metrics['loan_metrics']
    profit_metrics = metrics['profit_metrics']
    class_metrics = metrics['classification_metrics']
    
    # Create a summary of threshold analysis
    summary = {
        'threshold': float(threshold),
        'approval_rate': float(loan_metrics['n_loans']['approval_rate']),
        'average_repayment': {
            'predicted': float(loan_metrics['repayment_rates']['predicted_avg']),
            'actual': float(loan_metrics['repayment_rates']['actual_avg']),
            'difference': float(loan_metrics['repayment_rates']['difference'])
        },
        'profits': {
            'expected': float(profit_metrics['expected_profit']),
            'actual': float(profit_metrics['actual_profit']),
            'difference': float(profit_metrics['actual_profit'] - profit_metrics['expected_profit'])
        },
        'roi': float(profit_metrics['roi']),
        'classification': {
            'accuracy': float(class_metrics['accuracy']),
            'precision': float(class_metrics['precision']),
            'recall': float(class_metrics['recall']),
            'f1_score': float(class_metrics['f1_score'])
        },
        'loan_counts': {
            'approved': int(loan_metrics['n_loans']['approved']),
            'rejected': int(loan_metrics['n_loans']['rejected']),
            'good_approved': int(loan_metrics['loan_categories']['good_approved']['count']),
            'bad_approved': int(loan_metrics['loan_categories']['bad_approved']['count']),
            'good_rejected': int(loan_metrics['loan_categories']['good_rejected']['count']),
            'bad_rejected': int(loan_metrics['loan_categories']['bad_rejected']['count'])
        },
        'loan_values': {
            'approved': float(loan_metrics['loan_values']['approved']),
            'rejected': float(loan_metrics['loan_values']['rejected']),
            'good_approved': float(loan_metrics['loan_categories']['good_approved']['value']),
            'bad_approved': float(loan_metrics['loan_categories']['bad_approved']['value']),
            'good_rejected': float(loan_metrics['loan_categories']['good_rejected']['value']),
            'bad_rejected': float(loan_metrics['loan_categories']['bad_rejected']['value'])
        },
        'parameters': {
            'threshold': float(threshold),
            'margin': float(margin),
            'default_loss_rate': float(default_loss_rate)
        }
    }
    
    # Save results if output_path provided
    if output_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save summary as JSON
        import json
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Threshold analysis saved to {output_path}")
    
    return summary


def analyze_multiple_thresholds(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    loan_values: np.ndarray,
    thresholds: Optional[List[float]] = None,
    margin: float = DEFAULT_BUSINESS_PARAMS['gross_margin'],
    default_loss_rate: float = DEFAULT_BUSINESS_PARAMS['default_loss_rate'],
    output_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Analyze the impact of multiple thresholds on loan approval metrics.
    
    Args:
        y_true: True repayment rates
        y_pred: Predicted repayment rates
        loan_values: Loan amounts
        thresholds: List of thresholds to analyze (if None, uses a default range)
        margin: Gross margin as a decimal (e.g., 0.16 = 16%)
        default_loss_rate: Loss rate on defaulted loans (1.0 = 100%)
        output_path: Path to save analysis results
        
    Returns:
        DataFrame with metrics for each threshold
    """
    # Set default thresholds if none provided
    if thresholds is None:
        # Create a range of thresholds from 0.5 to 0.95 with 0.05 step
        thresholds = np.arange(0.5, 0.96, 0.05).tolist()
    
    # Initialize results list
    results = []
    
    # Analyze each threshold
    for threshold in thresholds:
        # Get metrics for this threshold
        profit_metrics = calculate_profit_metrics(
            y_true, y_pred, loan_values, threshold, margin, default_loss_rate
        )
        
        class_metrics = calculate_classification_metrics(
            y_true, y_pred, threshold
        )
        
        # Calculate average repayment rates for approved loans
        approved_mask = y_pred >= threshold
        
        if np.any(approved_mask):
            actual_avg_repayment = y_true[approved_mask].mean()
            predicted_avg_repayment = y_pred[approved_mask].mean()
        else:
            actual_avg_repayment = 0.0
            predicted_avg_repayment = 0.0
        
        # Combine metrics
        result = {
            'threshold': threshold,
            'approval_rate': profit_metrics['approval_rate'],
            'approved_loan_value': profit_metrics['approved_loan_value'],
            'actual_profit': profit_metrics['actual_profit'],
            'expected_profit': profit_metrics['expected_profit'],
            'actual_loss': profit_metrics['actual_loss'],
            'roi': profit_metrics['roi'],
            'accuracy': class_metrics['accuracy'],
            'precision': class_metrics['precision'],
            'recall': class_metrics['recall'],
            'f1_score': class_metrics['f1_score'],
            'predicted_avg_repayment': predicted_avg_repayment,
            'actual_avg_repayment': actual_avg_repayment
        }
        
        results.append(result)
    
    # Convert to DataFrame
    metrics_df = pd.DataFrame(results)
    
    # Save results if output_path provided
    if output_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save DataFrame as CSV
        metrics_df.to_csv(output_path, index=False)
        
        print(f"Multiple threshold analysis saved to {output_path}")
    
    return metrics_df


def analyze_cutoff_tradeoffs(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    loan_values: np.ndarray,
    thresholds: Optional[List[float]] = None,
    business_params: Optional[Dict[str, float]] = None,
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Analyze the tradeoffs between different metrics when selecting a threshold.
    
    Args:
        y_true: True repayment rates
        y_pred: Predicted repayment rates
        loan_values: Loan amounts
        thresholds: List of thresholds to analyze (if None, uses a default range)
        business_params: Dictionary with business parameters (margin, default_loss_rate)
        output_path: Path to save analysis results
        
    Returns:
        Dictionary with tradeoff analysis results
    """
    # Set default thresholds if none provided
    if thresholds is None:
        # More fine-grained thresholds for detailed analysis
        thresholds = np.arange(0.5, 0.96, 0.02).tolist()
    
    # Set default business parameters if none provided
    if business_params is None:
        business_params = {
            'margin': DEFAULT_BUSINESS_PARAMS['gross_margin'],
            'default_loss_rate': DEFAULT_BUSINESS_PARAMS['default_loss_rate']
        }
    
    # Get metrics for all thresholds
    metrics_df = analyze_multiple_thresholds(
        y_true, y_pred, loan_values, thresholds,
        business_params['margin'], business_params['default_loss_rate']
    )
    
    # Find optimal thresholds for different metrics
    optima = {
        'profit': {
            'threshold': float(metrics_df.loc[metrics_df['actual_profit'].idxmax(), 'threshold']),
            'value': float(metrics_df['actual_profit'].max()),
            'approval_rate': float(metrics_df.loc[metrics_df['actual_profit'].idxmax(), 'approval_rate'])
        },
        'roi': {
            'threshold': float(metrics_df.loc[metrics_df['roi'].idxmax(), 'threshold']),
            'value': float(metrics_df['roi'].max()),
            'approval_rate': float(metrics_df.loc[metrics_df['roi'].idxmax(), 'approval_rate'])
        },
        'f1_score': {
            'threshold': float(metrics_df.loc[metrics_df['f1_score'].idxmax(), 'threshold']),
            'value': float(metrics_df['f1_score'].max()),
            'approval_rate': float(metrics_df.loc[metrics_df['f1_score'].idxmax(), 'approval_rate'])
        }
    }
    
    # Create tradeoff analysis
    tradeoffs = {
        'metrics_by_threshold': metrics_df.to_dict('records'),
        'optimal_thresholds': optima,
        'business_parameters': business_params,
        'recommendations': {
            'profit_focused': optima['profit']['threshold'],
            'roi_focused': optima['roi']['threshold'],
            'balanced': (optima['profit']['threshold'] + optima['roi']['threshold']) / 2
        }
    }
    
    # Calculate sensitivity analysis - how much profit changes with threshold
    profit_sensitivity = []
    
    for i in range(1, len(thresholds)):
        prev_threshold = thresholds[i-1]
        curr_threshold = thresholds[i]
        
        prev_profit = metrics_df[metrics_df['threshold'] == prev_threshold]['actual_profit'].values[0]
        curr_profit = metrics_df[metrics_df['threshold'] == curr_threshold]['actual_profit'].values[0]
        
        threshold_diff = curr_threshold - prev_threshold
        profit_diff = curr_profit - prev_profit
        
        if threshold_diff > 0:
            sensitivity = profit_diff / threshold_diff
        else:
            sensitivity = 0.0
        
        profit_sensitivity.append({
            'threshold_range': f"{prev_threshold:.2f}-{curr_threshold:.2f}",
            'threshold_diff': threshold_diff,
            'profit_diff': profit_diff,
            'sensitivity': sensitivity
        })
    
    tradeoffs['sensitivity_analysis'] = profit_sensitivity
    
    # Save results if output_path provided
    if output_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save tradeoffs as JSON
        import json
        with open(output_path, 'w') as f:
            json.dump(tradeoffs, f, indent=2)
        
        # Also save metrics_df as CSV
        metrics_path = output_path.replace('.json', '_metrics.csv')
        metrics_df.to_csv(metrics_path, index=False)
        
        print(f"Cutoff tradeoff analysis saved to {output_path}")
        print(f"Metrics data saved to {metrics_path}")
    
    return tradeoffs
