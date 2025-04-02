"""
Metrics functions for loan repayment rate prediction profitability.

This module provides functions to calculate various metrics related to the financial
impact of loan repayment rate prediction, including profits, ROI, and business KPIs.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union

from ..constants import DEFAULT_BUSINESS_PARAMS


def calculate_loan_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    loan_values: np.ndarray,
    threshold: float = 0.8,
    margin: float = DEFAULT_BUSINESS_PARAMS['gross_margin'],
    default_loss_rate: float = DEFAULT_BUSINESS_PARAMS['default_loss_rate']
) -> Dict[str, Any]:
    """
    Calculate loan performance metrics based on actual and predicted repayment rates.
    
    Args:
        y_true: True repayment rates
        y_pred: Predicted repayment rates
        loan_values: Loan amounts
        threshold: Threshold for loan approval
        margin: Gross margin as a decimal (e.g., 0.16 = 16%)
        default_loss_rate: Loss rate on defaulted loans (1.0 = 100%)
        
    Returns:
        Dictionary with loan performance metrics
    """
    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    loan_values = np.array(loan_values)
    
    # Identify approved and rejected loans
    approved_mask = y_pred >= threshold
    approved_indices = np.where(approved_mask)[0]
    rejected_indices = np.where(~approved_mask)[0]
    
    # Calculate number of approved and rejected loans
    n_approved = len(approved_indices)
    n_rejected = len(rejected_indices)
    n_total = len(y_true)
    
    # Calculate approval rate
    approval_rate = n_approved / n_total if n_total > 0 else 0.0
    
    # Calculate loan values
    total_loan_value = loan_values.sum()
    approved_loan_value = loan_values[approved_indices].sum() if n_approved > 0 else 0.0
    rejected_loan_value = loan_values[rejected_indices].sum() if n_rejected > 0 else 0.0
    
    # Calculate average loan value
    avg_loan_value = total_loan_value / n_total if n_total > 0 else 0.0
    avg_approved_loan = approved_loan_value / n_approved if n_approved > 0 else 0.0
    avg_rejected_loan = rejected_loan_value / n_rejected if n_rejected > 0 else 0.0
    
    # Calculate predicted and actual repayment rates
    predicted_avg_repayment = y_pred[approved_indices].mean() if n_approved > 0 else 0.0
    actual_avg_repayment = y_true[approved_indices].mean() if n_approved > 0 else 0.0
    
    # Calculate repayment difference
    repayment_difference = actual_avg_repayment - predicted_avg_repayment
    
    # Identify good and bad loans (actual repayment rate >= threshold)
    good_loans_mask = y_true >= threshold
    bad_loans_mask = ~good_loans_mask
    
    good_approved = np.logical_and(good_loans_mask, approved_mask)
    bad_approved = np.logical_and(bad_loans_mask, approved_mask)
    good_rejected = np.logical_and(good_loans_mask, ~approved_mask)
    bad_rejected = np.logical_and(bad_loans_mask, ~approved_mask)
    
    # Count loans in each category
    n_good_approved = np.sum(good_approved)
    n_bad_approved = np.sum(bad_approved)
    n_good_rejected = np.sum(good_rejected)
    n_bad_rejected = np.sum(bad_rejected)
    
    # Calculate loan values in each category
    good_approved_value = loan_values[good_approved].sum() if n_good_approved > 0 else 0.0
    bad_approved_value = loan_values[bad_approved].sum() if n_bad_approved > 0 else 0.0
    good_rejected_value = loan_values[good_rejected].sum() if n_good_rejected > 0 else 0.0
    bad_rejected_value = loan_values[bad_rejected].sum() if n_bad_rejected > 0 else 0.0
    
    # Create result dictionary
    metrics = {
        'n_loans': {
            'total': int(n_total),
            'approved': int(n_approved),
            'rejected': int(n_rejected),
            'approval_rate': float(approval_rate)
        },
        'loan_values': {
            'total': float(total_loan_value),
            'approved': float(approved_loan_value),
            'rejected': float(rejected_loan_value),
            'avg_total': float(avg_loan_value),
            'avg_approved': float(avg_approved_loan),
            'avg_rejected': float(avg_rejected_loan)
        },
        'repayment_rates': {
            'predicted_avg': float(predicted_avg_repayment),
            'actual_avg': float(actual_avg_repayment),
            'difference': float(repayment_difference)
        },
        'loan_categories': {
            'good_approved': {
                'count': int(n_good_approved),
                'value': float(good_approved_value)
            },
            'bad_approved': {
                'count': int(n_bad_approved),
                'value': float(bad_approved_value)
            },
            'good_rejected': {
                'count': int(n_good_rejected),
                'value': float(good_rejected_value)
            },
            'bad_rejected': {
                'count': int(n_bad_rejected),
                'value': float(bad_rejected_value)
            }
        },
        'parameters': {
            'threshold': float(threshold),
            'margin': float(margin),
            'default_loss_rate': float(default_loss_rate)
        }
    }
    
    return metrics


def calculate_profit_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    loan_values: np.ndarray,
    threshold: float = 0.8,
    margin: float = DEFAULT_BUSINESS_PARAMS['gross_margin'],
    default_loss_rate: float = DEFAULT_BUSINESS_PARAMS['default_loss_rate']
) -> Dict[str, float]:
    """
    Calculate profit-related metrics based on actual and predicted repayment rates.
    
    Args:
        y_true: True repayment rates
        y_pred: Predicted repayment rates
        loan_values: Loan amounts
        threshold: Threshold for loan approval
        margin: Gross margin as a decimal (e.g., 0.16 = 16%)
        default_loss_rate: Loss rate on defaulted loans (1.0 = 100%)
        
    Returns:
        Dictionary with profit metrics
    """
    # Get loan metrics
    loan_metrics = calculate_loan_metrics(
        y_true, y_pred, loan_values, threshold, margin, default_loss_rate
    )
    
    # Extract needed values
    approved_indices = np.where(y_pred >= threshold)[0]
    
    if len(approved_indices) == 0:
        # No approvals, no profit
        return {
            'approved_loan_value': 0.0,
            'expected_repayment': 0.0,
            'expected_default': 0.0,
            'expected_profit': 0.0,
            'expected_loss': 0.0,
            'expected_net': 0.0,
            'actual_repayment': 0.0,
            'actual_default': 0.0,
            'actual_profit': 0.0,
            'actual_loss': 0.0,
            'actual_net': 0.0,
            'roi': 0.0,
            'approval_rate': 0.0
        }
    
    # Calculate loan amounts for approved loans
    approved_loan_value = loan_metrics['loan_values']['approved']
    
    # Convert everything to numpy arrays to ensure proper indexing
    if hasattr(y_true, 'values'):  # Check if it's a pandas Series
        y_true_array = y_true.values
    else:
        y_true_array = np.array(y_true)
        
    if hasattr(y_pred, 'values'):
        y_pred_array = y_pred.values
    else:
        y_pred_array = np.array(y_pred)
        
    if hasattr(loan_values, 'values'):
        loan_values_array = loan_values.values
    else:
        loan_values_array = np.array(loan_values)
    
    # Calculate expected repayment and default (based on predicted rates)
    expected_repayment = np.sum(loan_values_array[approved_indices] * y_pred_array[approved_indices])
    expected_default = np.sum(loan_values_array[approved_indices] * (1 - y_pred_array[approved_indices]))
    
    # Calculate actual repayment and default
    actual_repayment = np.sum(loan_values_array[approved_indices] * y_true_array[approved_indices])
    actual_default = np.sum(loan_values_array[approved_indices] * (1 - y_true_array[approved_indices]))
    
    # Calculate profits and losses
    expected_profit = expected_repayment * margin
    expected_loss = expected_default * default_loss_rate
    expected_net = expected_profit - expected_loss
    
    actual_profit = actual_repayment * margin
    actual_loss = actual_default * default_loss_rate
    actual_net = actual_profit - actual_loss
    
    # Calculate ROI (Return on Investment)
    roi = actual_net / approved_loan_value if approved_loan_value > 0 else 0.0
    
    # Create result dictionary
    metrics = {
        'approved_loan_value': float(approved_loan_value),
        'expected_repayment': float(expected_repayment),
        'expected_default': float(expected_default),
        'expected_profit': float(expected_profit),
        'expected_loss': float(expected_loss),
        'expected_net': float(expected_net),
        'actual_repayment': float(actual_repayment),
        'actual_default': float(actual_default),
        'actual_profit': float(actual_profit),
        'actual_loss': float(actual_loss),
        'actual_net': float(actual_net),
        'roi': float(roi),
        'approval_rate': float(loan_metrics['n_loans']['approval_rate'])
    }
    
    return metrics


def calculate_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.8
) -> Dict[str, float]:
    """
    Calculate classification metrics when using threshold on continuous predictions.
    
    Args:
        y_true: True repayment rates
        y_pred: Predicted repayment rates
        threshold: Threshold for loan approval
        
    Returns:
        Dictionary with classification metrics
    """
    # Convert to binary predictions based on threshold
    y_true_bin = (y_true >= threshold).astype(int)
    y_pred_bin = (y_pred >= threshold).astype(int)
    
    # Calculate basic counts
    true_pos = np.sum(np.logical_and(y_true_bin == 1, y_pred_bin == 1))
    true_neg = np.sum(np.logical_and(y_true_bin == 0, y_pred_bin == 0))
    false_pos = np.sum(np.logical_and(y_true_bin == 0, y_pred_bin == 1))
    false_neg = np.sum(np.logical_and(y_true_bin == 1, y_pred_bin == 0))
    
    # Calculate metrics
    total = len(y_true)
    accuracy = (true_pos + true_neg) / total if total > 0 else 0.0
    
    # Precision, recall, specificity, f1
    precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0.0
    recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0.0
    specificity = true_neg / (true_neg + false_pos) if (true_neg + false_pos) > 0 else 0.0
    
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Create result dictionary
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'specificity': float(specificity),
        'f1_score': float(f1_score),
        'true_positive': int(true_pos),
        'true_negative': int(true_neg),
        'false_positive': int(false_pos),
        'false_negative': int(false_neg)
    }
    
    return metrics


def calculate_business_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    loan_values: np.ndarray,
    threshold: float = 0.8,
    margin: float = DEFAULT_BUSINESS_PARAMS['gross_margin'],
    default_loss_rate: float = DEFAULT_BUSINESS_PARAMS['default_loss_rate']
) -> Dict[str, Dict[str, float]]:
    """
    Calculate comprehensive business metrics combining profit and classification.
    
    Args:
        y_true: True repayment rates
        y_pred: Predicted repayment rates
        loan_values: Loan amounts
        threshold: Threshold for loan approval
        margin: Gross margin as a decimal (e.g., 0.16 = 16%)
        default_loss_rate: Loss rate on defaulted loans (1.0 = 100%)
        
    Returns:
        Dictionary with combined business metrics
    """
    # Calculate loan metrics
    loan_metrics = calculate_loan_metrics(
        y_true, y_pred, loan_values, threshold, margin, default_loss_rate
    )
    
    # Calculate profit metrics
    profit_metrics = calculate_profit_metrics(
        y_true, y_pred, loan_values, threshold, margin, default_loss_rate
    )
    
    # Calculate classification metrics
    class_metrics = calculate_classification_metrics(y_true, y_pred, threshold)
    
    # Combine metrics
    combined_metrics = {
        'loan_metrics': loan_metrics,
        'profit_metrics': profit_metrics,
        'classification_metrics': class_metrics,
        'parameters': {
            'threshold': float(threshold),
            'margin': float(margin),
            'default_loss_rate': float(default_loss_rate)
        }
    }
    
    return combined_metrics
