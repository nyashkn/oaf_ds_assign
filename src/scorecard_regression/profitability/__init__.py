"""
Profitability analysis functions for regression-based loan repayment prediction.

This package provides tools to analyze the profitability and business implications
of different repayment rate thresholds, with a focus on optimization.
"""

from .metrics import (
    calculate_loan_metrics,
    calculate_profit_metrics,
    calculate_classification_metrics,
    calculate_business_metrics
)

from .threshold_analysis import (
    analyze_threshold,
    analyze_multiple_thresholds,
    analyze_cutoff_tradeoffs
)

from .optimization import (
    find_optimal_threshold,
    profit_function,
    advanced_profit_optimization
)

from .visualization import (
    plot_threshold_performance,
    plot_profit_metrics,
    plot_confusion_matrix,
    create_executive_summary,
    plot_optimization_results,
    plot_pareto_frontier
)

__all__ = [
    # Metrics functions
    'calculate_loan_metrics',
    'calculate_profit_metrics',
    'calculate_classification_metrics',
    'calculate_business_metrics',
    
    # Threshold analysis functions
    'analyze_threshold',
    'analyze_multiple_thresholds',
    'analyze_cutoff_tradeoffs',
    
    # Optimization functions
    'find_optimal_threshold',
    'profit_function',
    'advanced_profit_optimization',
    
    # Visualization functions
    'plot_threshold_performance',
    'plot_profit_metrics',
    'plot_confusion_matrix',
    'create_executive_summary',
    'plot_optimization_results',
    'plot_pareto_frontier'
]
