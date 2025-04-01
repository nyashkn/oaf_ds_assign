"""
Metrics calculation functions for profitability analysis.

This module provides functions to calculate various metrics for loan repayment
prediction analysis, including loan, profit, classification, and business metrics.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Any, Optional, Union

def calculate_loan_metrics(
    df: pd.DataFrame,
    threshold: float,
    predicted_col: str = 'predicted',
    actual_col: str = 'actual',
    loan_amount_col: str = 'nominal_contract_value'
) -> Dict[str, Any]:
    """
    Calculate loan-related metrics for a given threshold.
    
    Args:
        df: DataFrame with actual and predicted repayment rates
        threshold: Repayment rate threshold for classification
        predicted_col: Column name for predicted repayment rates
        actual_col: Column name for actual repayment rates
        loan_amount_col: Column name for loan amounts
        
    Returns:
        Dictionary with loan metrics
    """
    # Check if loan amount column exists
    if loan_amount_col not in df.columns:
        df = df.copy()
        df[loan_amount_col] = 1.0
    
    # Categorize loans
    n_approved = sum(df[predicted_col] >= threshold)
    n_rejected = sum(df[predicted_col] < threshold)
    total_loans = len(df)
    
    # Actually good vs bad loans based on actual performance
    good_loans = df[df[actual_col] >= threshold]
    bad_loans = df[df[actual_col] < threshold]
    n_good = len(good_loans)
    n_bad = len(bad_loans)
    
    # Confusion matrix categories
    true_pos = df[(df[predicted_col] >= threshold) & (df[actual_col] >= threshold)]
    false_pos = df[(df[predicted_col] >= threshold) & (df[actual_col] < threshold)]
    true_neg = df[(df[predicted_col] < threshold) & (df[actual_col] < threshold)]
    false_neg = df[(df[predicted_col] < threshold) & (df[actual_col] >= threshold)]
    
    n_true_pos = len(true_pos)
    n_false_pos = len(false_pos)
    n_true_neg = len(true_neg)
    n_false_neg = len(false_neg)
    
    # Loan values by category
    if n_approved > 0:
        approved_df = df[df[predicted_col] >= threshold]
        actual_repayment_rate = approved_df[actual_col].mean()
        predicted_repayment_rate = approved_df[predicted_col].mean()
    else:
        actual_repayment_rate = 0
        predicted_repayment_rate = 0
    
    # Rejected loans metrics
    if n_rejected > 0:
        rejected_df = df[df[predicted_col] < threshold]
        missed_actual_repayment_rate = rejected_df[actual_col].mean()
        n_profitable_rejected = sum(rejected_df[actual_col] >= threshold)
        pct_profitable_rejected = n_profitable_rejected / n_rejected if n_rejected > 0 else 0
    else:
        missed_actual_repayment_rate = 0
        n_profitable_rejected = 0
        pct_profitable_rejected = 0
    
    # Loan values by category
    good_loan_value = good_loans[loan_amount_col].sum() if n_good > 0 else 0
    bad_loan_value = bad_loans[loan_amount_col].sum() if n_bad > 0 else 0
    
    true_pos_value = true_pos[loan_amount_col].sum() if n_true_pos > 0 else 0
    false_pos_value = false_pos[loan_amount_col].sum() if n_false_pos > 0 else 0
    true_neg_value = true_neg[loan_amount_col].sum() if n_true_neg > 0 else 0
    false_neg_value = false_neg[loan_amount_col].sum() if n_false_neg > 0 else 0
    
    approved_value = true_pos_value + false_pos_value
    rejected_value = true_neg_value + false_neg_value
    
    return {
        'threshold': threshold,
        'total_loans': total_loans,
        'n_approved': n_approved,
        'n_rejected': n_rejected,
        'n_good': n_good,
        'n_bad': n_bad,
        'n_true_pos': n_true_pos,
        'n_false_pos': n_false_pos,
        'n_true_neg': n_true_neg,
        'n_false_neg': n_false_neg,
        'n_profitable_rejected': n_profitable_rejected,
        'pct_profitable_rejected': pct_profitable_rejected,
        'actual_repayment_rate': actual_repayment_rate,
        'predicted_repayment_rate': predicted_repayment_rate,
        'missed_actual_repayment_rate': missed_actual_repayment_rate,
        'good_loan_value': good_loan_value,
        'bad_loan_value': bad_loan_value,
        'true_pos_value': true_pos_value,
        'false_pos_value': false_pos_value,
        'true_neg_value': true_neg_value,
        'false_neg_value': false_neg_value,
        'approved_value': approved_value,
        'rejected_value': rejected_value,
        'repayment_prediction_error': predicted_repayment_rate - actual_repayment_rate
    }

def calculate_profit_metrics(
    df: pd.DataFrame,
    threshold: float,
    loan_metrics: Optional[Dict[str, Any]] = None,
    predicted_col: str = 'predicted',
    actual_col: str = 'actual',
    loan_amount_col: str = 'nominal_contract_value',
    gross_margin: float = 0.3
) -> Dict[str, Any]:
    """
    Calculate profit-related metrics for a given threshold.
    
    Args:
        df: DataFrame with actual and predicted repayment rates
        threshold: Repayment rate threshold for classification
        loan_metrics: Pre-calculated loan metrics (optional)
        predicted_col: Column name for predicted repayment rates
        actual_col: Column name for actual repayment rates
        loan_amount_col: Column name for loan amounts
        gross_margin: Gross margin percentage
        
    Returns:
        Dictionary with profit metrics
    """
    # Check if loan amount column exists
    if loan_amount_col not in df.columns:
        df = df.copy()
        df[loan_amount_col] = 1.0
    
    # Calculate profit-related fields
    df = df.copy()
    df['potential_profit'] = df[loan_amount_col] * gross_margin
    df['actual_profit'] = df[loan_amount_col] * df[actual_col] * gross_margin
    df['predicted_profit'] = df[loan_amount_col] * df[predicted_col] * gross_margin
    df['expected_loss'] = df['potential_profit'] - df['predicted_profit']
    
    # Get loan metrics if not provided
    if loan_metrics is None:
        loan_metrics = calculate_loan_metrics(
            df, threshold, predicted_col, actual_col, loan_amount_col
        )
    
    # Extract loan metrics for easier access
    n_true_pos = loan_metrics['n_true_pos']
    n_false_pos = loan_metrics['n_false_pos']
    n_true_neg = loan_metrics['n_true_neg']
    n_false_neg = loan_metrics['n_false_neg']
    
    # Categorize loans based on predicted and actual values
    true_pos = df[(df[predicted_col] >= threshold) & (df[actual_col] >= threshold)]
    false_pos = df[(df[predicted_col] >= threshold) & (df[actual_col] < threshold)]
    true_neg = df[(df[predicted_col] < threshold) & (df[actual_col] < threshold)]
    false_neg = df[(df[predicted_col] < threshold) & (df[actual_col] >= threshold)]
    
    # Total profit metrics
    total_potential_profit = df['potential_profit'].sum()
    total_actual_profit = df['actual_profit'].sum()
    
    # Profit metrics by category
    true_pos_profit = true_pos['actual_profit'].sum() if n_true_pos > 0 else 0
    false_pos_profit = false_pos['actual_profit'].sum() if n_false_pos > 0 else 0
    false_pos_loss = false_pos['potential_profit'].sum() - false_pos['actual_profit'].sum() if n_false_pos > 0 else 0
    
    true_neg_avoided_loss = true_neg['potential_profit'].sum() - true_neg['actual_profit'].sum() if n_true_neg > 0 else 0
    
    false_neg_profit = false_neg['actual_profit'].sum() if n_false_neg > 0 else 0  # Money left on the table
    
    # Calculate metrics for approved loans
    if loan_metrics['n_approved'] > 0:
        approved_df = df[df[predicted_col] >= threshold]
        
        total_loan_value = approved_df[loan_amount_col].sum()
        total_potential_profit = approved_df['potential_profit'].sum()
        total_predicted_profit = approved_df['predicted_profit'].sum()
        total_expected_loss = approved_df['expected_loss'].sum()
        
        avg_loan_value = approved_df[loan_amount_col].mean()
        avg_potential_profit = approved_df['potential_profit'].mean()
        avg_actual_profit = approved_df['actual_profit'].mean()
        avg_predicted_profit = approved_df['predicted_profit'].mean()
        avg_expected_loss = approved_df['expected_loss'].mean()
        
        # Profit calculation based on actual performance
        realized_profit_margin = total_actual_profit / total_loan_value if total_loan_value > 0 else 0
    else:
        total_loan_value = 0
        total_potential_profit = 0
        total_predicted_profit = 0
        total_expected_loss = 0
        avg_loan_value = 0
        avg_potential_profit = 0
        avg_actual_profit = 0
        avg_predicted_profit = 0
        avg_expected_loss = 0
        realized_profit_margin = 0
    
    # Calculate metrics for rejected loans (money left on the table)
    if loan_metrics['n_rejected'] > 0:
        rejected_df = df[df[predicted_col] < threshold]
        
        # Money left on the table (profitable loans that were rejected)
        money_left_on_table = rejected_df[rejected_df[actual_col] >= threshold]['actual_profit'].sum()
        missed_loan_value = rejected_df[loan_amount_col].sum()
        missed_actual_profit = rejected_df['actual_profit'].sum()
    else:
        money_left_on_table = 0
        missed_loan_value = 0
        missed_actual_profit = 0
    
    # Calculate approved loans profit
    approved_profit = true_pos_profit + false_pos_profit
    
    return {
        'gross_margin': gross_margin,
        'total_potential_profit': total_potential_profit,
        'total_actual_profit': df['actual_profit'].sum(),
        'total_predicted_profit': total_predicted_profit,
        'total_expected_loss': total_expected_loss,
        'true_pos_profit': true_pos_profit,
        'false_pos_profit': false_pos_profit,
        'false_pos_loss': false_pos_loss,
        'true_neg_avoided_loss': true_neg_avoided_loss,
        'false_neg_profit': false_neg_profit,  # Money left on table
        'approved_profit': approved_profit,
        'money_left_on_table': money_left_on_table,
        'missed_actual_profit': missed_actual_profit,
        'avg_loan_value': avg_loan_value,
        'avg_potential_profit': avg_potential_profit,
        'avg_actual_profit': avg_actual_profit,
        'avg_predicted_profit': avg_predicted_profit,
        'avg_expected_loss': avg_expected_loss,
        'realized_profit_margin': realized_profit_margin,
    }

def calculate_classification_metrics(loan_metrics: Dict[str, Any]) -> Dict[str, float]:
    """
    Calculate classification performance metrics based on loan metrics.
    
    Args:
        loan_metrics: Dictionary with loan metrics
        
    Returns:
        Dictionary with classification metrics
    """
    n_true_pos = loan_metrics['n_true_pos']
    n_false_pos = loan_metrics['n_false_pos']
    n_true_neg = loan_metrics['n_true_neg']
    n_false_neg = loan_metrics['n_false_neg']
    n_good = loan_metrics['n_good']
    n_bad = loan_metrics['n_bad']
    total_loans = loan_metrics['total_loans']
    
    # Classification metrics
    accuracy = (n_true_pos + n_true_neg) / total_loans if total_loans > 0 else 0
    precision = n_true_pos / (n_true_pos + n_false_pos) if (n_true_pos + n_false_pos) > 0 else 0
    recall = n_true_pos / n_good if n_good > 0 else 0
    specificity = n_true_neg / n_bad if n_bad > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Additional metrics
    false_positive_rate = n_false_pos / n_bad if n_bad > 0 else 0
    false_negative_rate = n_false_neg / n_good if n_good > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'f1_score': f1,
        'false_positive_rate': false_positive_rate,
        'false_negative_rate': false_negative_rate
    }

def calculate_business_metrics(
    loan_metrics: Dict[str, Any],
    profit_metrics: Dict[str, Any]
) -> Dict[str, float]:
    """
    Calculate business-oriented metrics based on loan and profit metrics.
    
    Args:
        loan_metrics: Dictionary with loan metrics
        profit_metrics: Dictionary with profit metrics
        
    Returns:
        Dictionary with business metrics
    """
    n_true_pos = loan_metrics['n_true_pos']
    n_false_pos = loan_metrics['n_false_pos']
    n_true_neg = loan_metrics['n_true_neg']
    n_false_neg = loan_metrics['n_false_neg']
    total_loans = loan_metrics['total_loans']
    
    true_pos_profit = profit_metrics['true_pos_profit']
    false_pos_profit = profit_metrics['false_pos_profit']
    
    # Business metrics
    approval_rate = (n_true_pos + n_false_pos) / total_loans if total_loans > 0 else 0
    rejection_rate = (n_true_neg + n_false_neg) / total_loans if total_loans > 0 else 0
    profit_precision = true_pos_profit / (true_pos_profit + false_pos_profit) if (true_pos_profit + false_pos_profit) > 0 else 0
    
    # ROI-related metrics
    approved_loans_roi = profit_metrics['approved_profit'] / loan_metrics['approved_value'] if loan_metrics['approved_value'] > 0 else 0
    potential_rejected_roi = profit_metrics['money_left_on_table'] / loan_metrics['false_neg_value'] if loan_metrics['false_neg_value'] > 0 else 0
    
    return {
        'approval_rate': approval_rate,
        'rejection_rate': rejection_rate,
        'profit_precision': profit_precision,
        'approved_loans_roi': approved_loans_roi,
        'potential_rejected_roi': potential_rejected_roi
    }
