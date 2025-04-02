"""
Threshold optimization functions for loan repayment rate prediction.

This module provides functions for finding optimal thresholds for
loan approval decisions based on predicted repayment rates.
"""

import numpy as np
import pandas as pd
import os
import json
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from scipy.optimize import minimize_scalar, differential_evolution

from .metrics import calculate_profit_metrics

def profit_function(
    threshold: float,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    loan_values: np.ndarray,
    margin: float = 0.16,
    default_loss_rate: float = 1.0,
    maximize: bool = True
) -> float:
    """
    Calculate profit for a given threshold (for use in optimization).
    
    Args:
        threshold: Threshold for loan approval
        y_true: True repayment rates
        y_pred: Predicted repayment rates
        loan_values: Loan amounts
        margin: Gross margin as a decimal (e.g., 0.16 = 16%)
        default_loss_rate: Loss rate on defaulted loans (1.0 = 100%)
        maximize: If True, return positive profit for maximization
        
    Returns:
        Profit at the given threshold (positive if maximize=True)
    """
    # Calculate profit metrics
    profit_metrics = calculate_profit_metrics(
        y_true, y_pred, loan_values, threshold, margin, default_loss_rate
    )
    
    # Return profit (negated if we want to minimize)
    profit = profit_metrics['actual_profit']
    
    return profit if maximize else -profit


def find_optimal_threshold(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    loan_values: np.ndarray,
    objective: str = 'profit',
    method: str = 'grid',
    margin: float = 0.16,
    default_loss_rate: float = 1.0,
    grid_points: int = 50,
    bounds: Tuple[float, float] = (0.5, 0.95),
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Find the optimal threshold for loan approval decisions.
    
    Args:
        y_true: True repayment rates
        y_pred: Predicted repayment rates
        loan_values: Loan amounts
        objective: Optimization objective ('profit', 'roi', or 'f1')
        method: Optimization method ('grid', 'scipy', or 'evolutionary')
        margin: Gross margin as a decimal (e.g., 0.16 = 16%)
        default_loss_rate: Loss rate on defaulted loans (1.0 = 100%)
        grid_points: Number of points for grid search
        bounds: Bounds for threshold search (min, max)
        output_path: Path to save optimization results
        
    Returns:
        Dictionary with optimization results
    """
    from .metrics import calculate_profit_metrics, calculate_classification_metrics
    
    print(f"\n=== Finding Optimal Threshold ({objective}, {method}) ===")
    
    if objective not in ['profit', 'roi', 'f1']:
        raise ValueError(f"Unsupported objective: {objective}")
    
    if method not in ['grid', 'scipy', 'evolutionary']:
        raise ValueError(f"Unsupported method: {method}")
    
    # Define objective functions
    def profit_obj(threshold):
        metrics = calculate_profit_metrics(
            y_true, y_pred, loan_values, threshold, margin, default_loss_rate
        )
        return -metrics['actual_profit']  # Negate for minimization
    
    def roi_obj(threshold):
        metrics = calculate_profit_metrics(
            y_true, y_pred, loan_values, threshold, margin, default_loss_rate
        )
        roi = metrics['roi'] if metrics['roi'] > 0 else 0
        return -roi  # Negate for minimization
    
    def f1_obj(threshold):
        metrics = calculate_classification_metrics(y_true, y_pred, threshold)
        return -metrics['f1_score']  # Negate for minimization
    
    # Select objective function
    if objective == 'profit':
        obj_function = profit_obj
    elif objective == 'roi':
        obj_function = roi_obj
    else:  # f1
        obj_function = f1_obj
    
    # Optimization using selected method
    if method == 'grid':
        # Grid search
        thresholds = np.linspace(bounds[0], bounds[1], grid_points)
        results = []
        
        for threshold in thresholds:
            # Calculate metrics
            profit_metrics = calculate_profit_metrics(
                y_true, y_pred, loan_values, threshold, margin, default_loss_rate
            )
            class_metrics = calculate_classification_metrics(y_true, y_pred, threshold)
            
            # Store results
            results.append({
                'threshold': float(threshold),
                'profit': float(profit_metrics['actual_profit']),
                'roi': float(profit_metrics['roi']),
                'f1_score': float(class_metrics['f1_score']),
                'approval_rate': float(profit_metrics['approval_rate'])
            })
        
        # Convert to DataFrame for easier analysis
        results_df = pd.DataFrame(results)
        
        # Find optimal threshold
        if objective == 'profit':
            optimal_idx = results_df['profit'].idxmax()
        elif objective == 'roi':
            optimal_idx = results_df['roi'].idxmax()
        else:  # f1
            optimal_idx = results_df['f1_score'].idxmax()
        
        optimal_threshold = results_df.loc[optimal_idx, 'threshold']
        optimal_metrics = {
            'profit': results_df.loc[optimal_idx, 'profit'],
            'roi': results_df.loc[optimal_idx, 'roi'],
            'f1_score': results_df.loc[optimal_idx, 'f1_score'],
            'approval_rate': results_df.loc[optimal_idx, 'approval_rate']
        }
        
    elif method == 'scipy':
        # Scipy optimization
        result = minimize_scalar(
            obj_function,
            bounds=bounds,
            method='bounded'
        )
        
        optimal_threshold = result.x
        
        # Calculate metrics at optimal threshold
        profit_metrics = calculate_profit_metrics(
            y_true, y_pred, loan_values, optimal_threshold, margin, default_loss_rate
        )
        class_metrics = calculate_classification_metrics(y_true, y_pred, optimal_threshold)
        
        optimal_metrics = {
            'profit': float(-obj_function(optimal_threshold) if objective == 'profit' else profit_metrics['actual_profit']),
            'roi': float(-obj_function(optimal_threshold) if objective == 'roi' else profit_metrics['roi']),
            'f1_score': float(-obj_function(optimal_threshold) if objective == 'f1' else class_metrics['f1_score']),
            'approval_rate': float(profit_metrics['approval_rate'])
        }
        
        # Create results for plotting (add some points around optimal)
        thresholds = np.linspace(bounds[0], bounds[1], grid_points)
        results = []
        
        for threshold in thresholds:
            profit_metrics = calculate_profit_metrics(
                y_true, y_pred, loan_values, threshold, margin, default_loss_rate
            )
            class_metrics = calculate_classification_metrics(y_true, y_pred, threshold)
            
            results.append({
                'threshold': float(threshold),
                'profit': float(profit_metrics['actual_profit']),
                'roi': float(profit_metrics['roi']),
                'f1_score': float(class_metrics['f1_score']),
                'approval_rate': float(profit_metrics['approval_rate'])
            })
        
        results_df = pd.DataFrame(results)
        
    else:  # evolutionary
        # Differential evolution (global optimization)
        result = differential_evolution(
            obj_function,
            bounds=[bounds],
            maxiter=100,
            popsize=20,
            seed=42
        )
        
        optimal_threshold = result.x[0]
        
        # Calculate metrics at optimal threshold
        profit_metrics = calculate_profit_metrics(
            y_true, y_pred, loan_values, optimal_threshold, margin, default_loss_rate
        )
        class_metrics = calculate_classification_metrics(y_true, y_pred, optimal_threshold)
        
        optimal_metrics = {
            'profit': float(-obj_function(optimal_threshold) if objective == 'profit' else profit_metrics['actual_profit']),
            'roi': float(-obj_function(optimal_threshold) if objective == 'roi' else profit_metrics['roi']),
            'f1_score': float(-obj_function(optimal_threshold) if objective == 'f1' else class_metrics['f1_score']),
            'approval_rate': float(profit_metrics['approval_rate'])
        }
        
        # Create results for plotting
        thresholds = np.linspace(bounds[0], bounds[1], grid_points)
        results = []
        
        for threshold in thresholds:
            profit_metrics = calculate_profit_metrics(
                y_true, y_pred, loan_values, threshold, margin, default_loss_rate
            )
            class_metrics = calculate_classification_metrics(y_true, y_pred, threshold)
            
            results.append({
                'threshold': float(threshold),
                'profit': float(profit_metrics['actual_profit']),
                'roi': float(profit_metrics['roi']),
                'f1_score': float(class_metrics['f1_score']),
                'approval_rate': float(profit_metrics['approval_rate'])
            })
        
        results_df = pd.DataFrame(results)
    
    # Print results
    print(f"Optimal threshold ({objective}): {optimal_threshold:.4f}")
    print(f"  Profit: {optimal_metrics['profit']:.2f}")
    print(f"  ROI: {optimal_metrics['roi']:.2%}")
    print(f"  F1 Score: {optimal_metrics['f1_score']:.4f}")
    print(f"  Approval Rate: {optimal_metrics['approval_rate']:.2%}")
    
    # Create result dictionary
    result = {
        'optimal_threshold': float(optimal_threshold),
        'optimal_metrics': optimal_metrics,
        'method': method,
        'objective': objective,
        'results': results_df.to_dict('records'),
        'parameters': {
            'margin': margin,
            'default_loss_rate': default_loss_rate,
            'bounds': bounds
        }
    }
    
    # Save results if output_path provided
    if output_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save as JSON
        with open(output_path, 'w') as f:
            # Convert results to list for JSON serialization
            result_to_save = result.copy()
            result_to_save['results'] = results_df.to_dict('records')
            
            json.dump(result_to_save, f, indent=2)
        
        print(f"Optimization results saved to {output_path}")
    
    return result


def advanced_profit_optimization(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    loan_values: np.ndarray,
    margin: float = 0.16,
    default_loss_rate: float = 1.0,
    operating_cost: float = 0.0,
    overhead_allocation: float = 0.0,
    collection_cost_ratio: float = 0.0,
    constraints: Optional[Dict[str, Any]] = None,
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Perform advanced profit optimization with multiple constraints.
    
    Args:
        y_true: True repayment rates
        y_pred: Predicted repayment rates
        loan_values: Loan amounts
        margin: Gross margin as a decimal (e.g., 0.16 = 16%)
        default_loss_rate: Loss rate on defaulted loans (1.0 = 100%)
        operating_cost: Operating cost per loan
        overhead_allocation: Overhead allocation per loan
        collection_cost_ratio: Collection cost as a ratio of defaulted amount
        constraints: Dictionary with optimization constraints
                    (e.g., {'min_approval_rate': 0.3, 'max_default_rate': 0.2})
        output_path: Path to save optimization results
        
    Returns:
        Dictionary with optimization results
    """
    print("\n=== Advanced Profit Optimization ===")
    
    # Default constraints
    default_constraints = {
        'min_approval_rate': 0.0,
        'max_default_rate': 1.0,
        'min_roi': 0.0
    }
    
    # Update with user-provided constraints
    if constraints is None:
        constraints = {}
    
    all_constraints = {**default_constraints, **constraints}
    
    # Define advanced objective function
    def advanced_profit_obj(threshold):
        # Get approved loans
        approved = y_pred >= threshold
        
        if not np.any(approved):
            return 0.0  # No loans approved
        
        # Calculate loan counts
        n_approved = np.sum(approved)
        approval_rate = n_approved / len(y_true)
        
        # Check approval rate constraint
        if approval_rate < all_constraints['min_approval_rate']:
            return -1e10  # Heavy penalty for violating constraint
        
        # Calculate default rate
        defaults = (1 - y_true[approved])
        default_rate = np.mean(defaults)
        
        # Check default rate constraint
        if default_rate > all_constraints['max_default_rate']:
            return -1e10  # Heavy penalty for violating constraint
        
        # Calculate profit components
        repayment_amount = np.sum(loan_values[approved] * y_true[approved])
        default_amount = np.sum(loan_values[approved] * (1 - y_true[approved]))
        
        # Revenue and costs
        revenue = repayment_amount * margin
        default_loss = default_amount * default_loss_rate
        op_cost = n_approved * operating_cost
        overhead = n_approved * overhead_allocation
        collection_cost = default_amount * collection_cost_ratio
        
        # Total profit
        profit = revenue - default_loss - op_cost - overhead - collection_cost
        
        # Calculate ROI
        total_investment = np.sum(loan_values[approved])
        roi = profit / total_investment if total_investment > 0 else 0
        
        # Check ROI constraint
        if roi < all_constraints['min_roi']:
            return -1e10  # Heavy penalty for violating constraint
        
        return profit
    
    # Define threshold range
    thresholds = np.linspace(0.5, 0.95, 100)
    
    # Evaluate profit at each threshold
    results = []
    for threshold in thresholds:
        profit = advanced_profit_obj(threshold)
        
        # Only include valid thresholds (those not violating constraints)
        if profit > -1e9:  # Not a penalty value
            # Get approved loans
            approved = y_pred >= threshold
            
            if np.any(approved):
                # Metrics for this threshold
                n_approved = np.sum(approved)
                approval_rate = n_approved / len(y_true)
                
                defaults = (1 - y_true[approved])
                default_rate = np.mean(defaults)
                
                repayment_amount = np.sum(loan_values[approved] * y_true[approved])
                default_amount = np.sum(loan_values[approved] * (1 - y_true[approved]))
                
                revenue = repayment_amount * margin
                default_loss = default_amount * default_loss_rate
                op_cost = n_approved * operating_cost
                overhead = n_approved * overhead_allocation
                collection_cost = default_amount * collection_cost_ratio
                
                total_profit = revenue - default_loss - op_cost - overhead - collection_cost
                total_investment = np.sum(loan_values[approved])
                roi = total_profit / total_investment if total_investment > 0 else 0
                
                # Add to results
                results.append({
                    'threshold': float(threshold),
                    'profit': float(profit),
                    'approval_rate': float(approval_rate),
                    'default_rate': float(default_rate),
                    'roi': float(roi),
                    'revenue': float(revenue),
                    'default_loss': float(default_loss),
                    'operating_cost': float(op_cost),
                    'overhead': float(overhead),
                    'collection_cost': float(collection_cost)
                })
    
    # Convert to DataFrame
    if results:
        results_df = pd.DataFrame(results)
        
        # Find optimal threshold
        optimal_idx = results_df['profit'].idxmax()
        optimal_threshold = results_df.loc[optimal_idx, 'threshold']
        optimal_profit = results_df.loc[optimal_idx, 'profit']
        
        # Get all metrics at optimal threshold
        optimal_metrics = results_df.loc[optimal_idx].to_dict()
    else:
        # No valid thresholds found
        print("Warning: No thresholds found that satisfy all constraints")
        optimal_threshold = None
        optimal_profit = 0.0
        optimal_metrics = {}
        results_df = pd.DataFrame()
    
    # Print results
    if optimal_threshold is not None:
        print(f"Optimal threshold: {optimal_threshold:.4f}")
        print(f"  Profit: {optimal_profit:.2f}")
        print(f"  Approval Rate: {optimal_metrics.get('approval_rate', 0):.2%}")
        print(f"  Default Rate: {optimal_metrics.get('default_rate', 0):.2%}")
        print(f"  ROI: {optimal_metrics.get('roi', 0):.2%}")
    else:
        print("No optimal threshold found that meets all constraints")
    
    # Create result dictionary
    result = {
        'optimal_threshold': optimal_threshold,
        'optimal_metrics': optimal_metrics,
        'results': results_df.to_dict('records') if not results_df.empty else [],
        'constraints': all_constraints,
        'parameters': {
            'margin': margin,
            'default_loss_rate': default_loss_rate,
            'operating_cost': operating_cost,
            'overhead_allocation': overhead_allocation,
            'collection_cost_ratio': collection_cost_ratio
        }
    }
    
    # Save results if output_path provided
    if output_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save as JSON
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"Advanced optimization results saved to {output_path}")
    
    return result
