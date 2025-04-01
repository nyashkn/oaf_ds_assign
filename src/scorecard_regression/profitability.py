"""
Profitability analysis functions for regression-based loan repayment prediction.

This module provides functions to analyze the profitability and business implications
of different repayment rate thresholds, calculating expected losses and profits.

Note: This is a facade module that re-exports functions from the new modular structure
in the profitability/ package for backward compatibility.
"""

from typing import Dict, List, Optional, Tuple, Any, Union

from .constants import GROSS_MARGIN, DEFAULT_CUTOFF_THRESHOLDS

# Import and re-export functions from the modular structure
from .profitability.metrics import (
    calculate_loan_metrics,
    calculate_profit_metrics,
    calculate_classification_metrics,
    calculate_business_metrics
)

from .profitability.threshold_analysis import (
    analyze_threshold,
    analyze_multiple_thresholds,
    analyze_cutoff_tradeoffs
)

from .profitability.optimization import (
    find_optimal_threshold,
    profit_function,
    advanced_profit_optimization
)

from .profitability.visualization import (
    plot_threshold_performance,
    plot_profit_metrics,
    plot_confusion_matrix,
    create_executive_summary,
    plot_optimization_results,
    plot_pareto_frontier
)

def calculate_profitability(
    df,
    actual_col='actual',
    predicted_col='predicted',
    loan_amount_col='nominal_contract_value',
    gross_margin=GROSS_MARGIN,
    thresholds=None,
    output_dir=None
) -> Dict[str, Any]:
    """
    Calculate profitability metrics for different repayment rate thresholds.
    
    This is a backward-compatible wrapper around analyze_multiple_thresholds.
    
    Args:
        df: DataFrame with actual and predicted repayment rates and loan amounts
        actual_col: Column name for actual repayment rates
        predicted_col: Column name for predicted repayment rates
        loan_amount_col: Column name for loan amounts
        gross_margin: Gross margin percentage (default from constants)
        thresholds: List of repayment rate thresholds to evaluate
        output_dir: Directory to save profitability analysis results
        
    Returns:
        Dictionary with profitability metrics for different thresholds
    """
    # Set default thresholds if not provided
    if thresholds is None:
        thresholds = DEFAULT_CUTOFF_THRESHOLDS
    
    # Call the new implementation
    results = analyze_multiple_thresholds(
        df=df,
        thresholds=thresholds,
        predicted_col=predicted_col,
        actual_col=actual_col,
        loan_amount_col=loan_amount_col,
        gross_margin=gross_margin,
        output_dir=output_dir
    )
    
    # Convert to backward-compatible format
    return {
        'threshold_metrics': results['threshold_analyses'],
        'threshold_df': results['threshold_df'],
        'optimal_threshold': results['optimal_threshold'],
        'gross_margin': gross_margin
    }

# Export all functions from the new structure
__all__ = [
    # Re-exported from metrics.py
    'calculate_loan_metrics',
    'calculate_profit_metrics',
    'calculate_classification_metrics',
    'calculate_business_metrics',
    
    # Re-exported from threshold_analysis.py
    'analyze_threshold',
    'analyze_multiple_thresholds',
    'analyze_cutoff_tradeoffs',
    
    # Re-exported from optimization.py
    'find_optimal_threshold',
    'profit_function',
    'advanced_profit_optimization',
    
    # Re-exported from visualization.py
    'plot_threshold_performance',
    'plot_profit_metrics',
    'plot_confusion_matrix',
    'create_executive_summary',
    'plot_optimization_results',
    'plot_pareto_frontier',
    
    # Backward compatibility function
    'calculate_profitability'
]
