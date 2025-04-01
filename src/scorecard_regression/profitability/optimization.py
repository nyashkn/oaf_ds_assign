"""
Profit optimization functions for regression-based loan repayment prediction.

This module provides advanced optimization methods to directly find the threshold
that maximizes profit rather than using a grid search of discrete thresholds.
"""

import pandas as pd
import numpy as np
import os
import json
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from scipy.optimize import minimize_scalar, minimize, differential_evolution

from .metrics import (
    calculate_loan_metrics,
    calculate_profit_metrics
)

def profit_function(
    threshold: float,
    df: pd.DataFrame,
    predicted_col: str = 'predicted',
    actual_col: str = 'actual',
    loan_amount_col: str = 'nominal_contract_value',
    gross_margin: float = 0.3,
    metric: str = 'total_actual_profit'
) -> float:
    """
    Calculate profit for a given threshold (objective function for optimization).
    
    Args:
        threshold: Repayment rate threshold (between 0 and 1)
        df: DataFrame with actual and predicted repayment rates
        predicted_col: Column name for predicted repayment rates
        actual_col: Column name for actual repayment rates
        loan_amount_col: Column name for loan amounts
        gross_margin: Gross margin percentage
        metric: Profit metric to optimize ('total_actual_profit' or 'money_left_on_table')
        
    Returns:
        Negative profit value (for minimization)
    """
    # Ensure threshold is between 0 and 1
    threshold = max(0, min(1, threshold))
    
    # Calculate metrics for this threshold
    loan_metrics = calculate_loan_metrics(
        df, threshold, predicted_col, actual_col, loan_amount_col
    )
    
    profit_metrics = calculate_profit_metrics(
        df, threshold, loan_metrics, predicted_col, actual_col, loan_amount_col, gross_margin
    )
    
    # Default to total_actual_profit if invalid metric
    if metric not in profit_metrics:
        metric = 'total_actual_profit'
    
    # Return negative profit for minimization
    if metric == 'money_left_on_table':
        # For money_left_on_table, we want to minimize it directly
        return profit_metrics[metric]
    else:
        # For other profit metrics, we want to maximize them (minimize negative)
        return -profit_metrics[metric]

def combined_objective(
    threshold: float,
    df: pd.DataFrame,
    predicted_col: str = 'predicted',
    actual_col: str = 'actual',
    loan_amount_col: str = 'nominal_contract_value',
    gross_margin: float = 0.3,
    alpha: float = 0.5
) -> float:
    """
    Combined objective function balancing profit and money left on table.
    
    Args:
        threshold: Repayment rate threshold (between 0 and 1)
        df: DataFrame with actual and predicted repayment rates
        predicted_col: Column name for predicted repayment rates
        actual_col: Column name for actual repayment rates
        loan_amount_col: Column name for loan amounts
        gross_margin: Gross margin percentage
        alpha: Weight for total_actual_profit (1-alpha is weight for money_left_on_table)
        
    Returns:
        Negative weighted combination of profit and money left on table
    """
    # Ensure threshold is between 0 and 1
    threshold = max(0, min(1, threshold))
    
    # Calculate metrics for this threshold
    loan_metrics = calculate_loan_metrics(
        df, threshold, predicted_col, actual_col, loan_amount_col
    )
    
    profit_metrics = calculate_profit_metrics(
        df, threshold, loan_metrics, predicted_col, actual_col, loan_amount_col, gross_margin
    )
    
    # Normalize metrics to [0, 1] scale
    total_profit = profit_metrics['total_actual_profit']
    money_left = profit_metrics['money_left_on_table']
    
    # Combined objective - minimize negative profit and money left on table
    # Higher alpha prioritizes maximizing profit, lower alpha prioritizes minimizing money left on table
    return -(alpha * total_profit - (1 - alpha) * money_left)

def find_optimal_threshold(
    df: pd.DataFrame,
    predicted_col: str = 'predicted',
    actual_col: str = 'actual',
    loan_amount_col: str = 'nominal_contract_value',
    gross_margin: float = 0.3,
    metric: str = 'total_actual_profit',
    method: str = 'bounded',
    initial_guess: float = 0.8,
    bounds: Tuple[float, float] = (0.0, 1.0),
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Find the optimal threshold that maximizes a profit metric using direct optimization.
    
    Args:
        df: DataFrame with actual and predicted repayment rates
        predicted_col: Column name for predicted repayment rates
        actual_col: Column name for actual repayment rates
        loan_amount_col: Column name for loan amounts
        gross_margin: Gross margin percentage
        metric: Profit metric to optimize ('total_actual_profit' or 'money_left_on_table')
        method: Optimization method ('bounded', 'brent', or 'golden')
        initial_guess: Initial threshold value for optimization
        bounds: Bounds for threshold optimization (min, max)
        output_dir: Directory to save optimization results
        
    Returns:
        Dictionary with optimization results
    """
    print("\n=== Profit Optimization ===")
    print(f"Optimizing for '{metric}' using {method} method")
    
    # Define objective function
    def objective(t):
        return profit_function(
            t, df, predicted_col, actual_col, loan_amount_col, gross_margin, metric
        )
    
    # Perform optimization
    try:
        result = minimize_scalar(
            objective,
            bounds=bounds,
            method=method,
            options={'maxiter': 100, 'xatol': 1e-05}
        )
        
        optimal_threshold = result.x
        success = result.success
        
        if metric == 'money_left_on_table':
            optimal_value = result.fun  # Already positive for money_left_on_table
        else:
            optimal_value = -result.fun  # Convert back to positive for profit metrics
        
        print(f"\nOptimization {'successful' if success else 'failed'}")
        print(f"Optimal threshold: {optimal_threshold:.4f}")
        print(f"Optimal {metric}: {optimal_value:.2f}")
        
        # Calculate metrics at optimal threshold
        loan_metrics = calculate_loan_metrics(
            df, optimal_threshold, predicted_col, actual_col, loan_amount_col
        )
        
        profit_metrics = calculate_profit_metrics(
            df, optimal_threshold, loan_metrics, predicted_col, actual_col, loan_amount_col, gross_margin
        )
        
        print(f"Approval rate: {loan_metrics['n_approved']/loan_metrics['total_loans']:.1%}")
        print(f"Actual repayment rate: {loan_metrics['actual_repayment_rate']:.2f}")
        
        # Calculate metrics for various thresholds around the optimum for visualization
        threshold_range = np.linspace(max(0, optimal_threshold - 0.2), 
                                      min(1, optimal_threshold + 0.2), 
                                      41)
        profit_curve = []
        
        for t in threshold_range:
            if metric == 'money_left_on_table':
                value = profit_function(t, df, predicted_col, actual_col, 
                                       loan_amount_col, gross_margin, metric)
            else:
                value = -profit_function(t, df, predicted_col, actual_col, 
                                        loan_amount_col, gross_margin, metric)
            profit_curve.append((t, value))
        
        # Save results if output directory provided
        if output_dir:
            # Create directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Save optimization results
            opt_results = {
                'optimal_threshold': float(optimal_threshold),
                'optimal_value': float(optimal_value),
                'metric': metric,
                'method': method,
                'success': bool(success),
                'n_iterations': int(result.nfev),
                'convergence': dict(result.keys()),
                'profit_metrics': {k: float(v) if isinstance(v, (int, float, np.number)) else v 
                                  for k, v in profit_metrics.items()},
                'loan_metrics': {k: float(v) if isinstance(v, (int, float, np.number)) else v 
                                for k, v in loan_metrics.items()},
                'profit_curve': [[float(t), float(v)] for t, v in profit_curve]
            }
            
            results_path = os.path.join(output_dir, "optimization_results.json")
            with open(results_path, 'w') as f:
                json.dump(opt_results, f, indent=2)
            
            print(f"\nOptimization results saved to {results_path}")
        
        return {
            'optimal_threshold': optimal_threshold,
            'optimal_value': optimal_value,
            'metric': metric,
            'method': method,
            'success': success,
            'n_iterations': result.nfev,
            'function_calls': result.nfev,
            'convergence': result,
            'profit_metrics': profit_metrics,
            'loan_metrics': loan_metrics,
            'profit_curve': profit_curve
        }
        
    except Exception as e:
        print(f"Error in optimization: {str(e)}")
        return {
            'error': str(e),
            'optimal_threshold': initial_guess,
            'success': False
        }

def advanced_profit_optimization(
    df: pd.DataFrame,
    predicted_col: str = 'predicted',
    actual_col: str = 'actual',
    loan_amount_col: str = 'nominal_contract_value',
    gross_margin: float = 0.3,
    alpha_values: Optional[List[float]] = None,
    method: str = 'SLSQP',
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Perform multi-objective optimization balancing profit and money left on table.
    
    Args:
        df: DataFrame with actual and predicted repayment rates
        predicted_col: Column name for predicted repayment rates
        actual_col: Column name for actual repayment rates
        loan_amount_col: Column name for loan amounts
        gross_margin: Gross margin percentage
        alpha_values: List of weights for total_actual_profit 
                      (1-alpha is weight for money_left_on_table)
        method: Optimization method
        output_dir: Directory to save optimization results
        
    Returns:
        Dictionary with optimization results for different alpha values
    """
    print("\n=== Advanced Profit Optimization ===")
    
    if alpha_values is None:
        alpha_values = [0.1, 0.25, 0.5, 0.75, 0.9]
    
    results = {}
    pareto_frontier = []
    
    for alpha in alpha_values:
        print(f"\nOptimizing with alpha = {alpha:.2f}")
        print(f"({alpha:.2f} weight on profit, {1-alpha:.2f} weight on minimizing money left)")
        
        # Define objective function with current alpha
        def objective(t):
            return combined_objective(
                t[0], df, predicted_col, actual_col, loan_amount_col, gross_margin, alpha
            )
        
        # Perform optimization
        try:
            if method == 'differential_evolution':
                result = differential_evolution(
                    objective,
                    bounds=[(0, 1)],
                    maxiter=100,
                    popsize=15,
                    tol=1e-5
                )
            else:
                result = minimize(
                    objective,
                    x0=[0.8],  # Initial guess
                    bounds=[(0, 1)],
                    method=method,
                    options={'maxiter': 100, 'disp': False}
                )
            
            optimal_threshold = result.x[0]
            success = result.success
            
            # Calculate metrics at optimal threshold
            loan_metrics = calculate_loan_metrics(
                df, optimal_threshold, predicted_col, actual_col, loan_amount_col
            )
            
            profit_metrics = calculate_profit_metrics(
                df, optimal_threshold, loan_metrics, predicted_col, actual_col, 
                loan_amount_col, gross_margin
            )
            
            total_profit = profit_metrics['total_actual_profit']
            money_left = profit_metrics['money_left_on_table']
            
            print(f"Optimal threshold: {optimal_threshold:.4f}")
            print(f"Total profit: {total_profit:.2f}")
            print(f"Money left on table: {money_left:.2f}")
            print(f"Approval rate: {loan_metrics['n_approved']/loan_metrics['total_loans']:.1%}")
            
            results[f"alpha_{alpha:.2f}"] = {
                'alpha': alpha,
                'optimal_threshold': optimal_threshold,
                'total_profit': total_profit,
                'money_left_on_table': money_left,
                'approval_rate': loan_metrics['n_approved']/loan_metrics['total_loans'],
                'success': success,
                'profit_metrics': profit_metrics,
                'loan_metrics': loan_metrics
            }
            
            # Add to Pareto frontier
            pareto_frontier.append((optimal_threshold, total_profit, money_left))
            
        except Exception as e:
            print(f"Error in optimization with alpha={alpha}: {str(e)}")
            results[f"alpha_{alpha:.2f}"] = {
                'alpha': alpha,
                'error': str(e),
                'success': False
            }
    
    # Save results if output directory provided
    if output_dir:
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save optimization results
        # Convert to JSON-serializable format
        json_results = {}
        for k, v in results.items():
            json_results[k] = {
                key: (float(val) if isinstance(val, (int, float, np.number)) else val)
                for key, val in v.items()
                if key not in ['profit_metrics', 'loan_metrics']
            }
            
            # Add selected metrics from profit_metrics and loan_metrics
            if 'profit_metrics' in v and v['profit_metrics'] is not None:
                json_results[k]['profit_metrics'] = {
                    metric: float(v['profit_metrics'][metric]) 
                    for metric in ['total_actual_profit', 'money_left_on_table', 'approved_profit'] 
                    if metric in v['profit_metrics']
                }
            
            if 'loan_metrics' in v and v['loan_metrics'] is not None:
                json_results[k]['loan_metrics'] = {
                    metric: float(v['loan_metrics'][metric]) 
                    for metric in ['n_approved', 'n_rejected', 'actual_repayment_rate'] 
                    if metric in v['loan_metrics']
                }
        
        results_path = os.path.join(output_dir, "advanced_optimization_results.json")
        with open(results_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        # Save Pareto frontier
        pareto_path = os.path.join(output_dir, "pareto_frontier.csv")
        pareto_df = pd.DataFrame(pareto_frontier, 
                                columns=['threshold', 'total_profit', 'money_left_on_table'])
        pareto_df.to_csv(pareto_path, index=False)
        
        print(f"\nAdvanced optimization results saved to {output_dir}")
    
    return {
        'results': results,
        'pareto_frontier': pareto_frontier
    }
