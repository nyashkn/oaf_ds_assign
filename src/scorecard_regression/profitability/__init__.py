"""
Profitability analysis subpackage for loan repayment rate prediction.

This subpackage provides tools and utilities for analyzing the profitability
of loan approval decisions based on predicted repayment rates.
"""

# Import metrics functions
from .metrics import (
    calculate_loan_metrics,
    calculate_profit_metrics,
    calculate_classification_metrics,
    calculate_business_metrics
)

# Import threshold analysis functions
from .threshold_analysis import (
    analyze_threshold,
    analyze_multiple_thresholds,
    analyze_cutoff_tradeoffs
)

# Import optimization functions
from .optimization import (
    find_optimal_threshold,
    profit_function,
    advanced_profit_optimization
)

# Import visualization functions
from .visualization import (
    plot_threshold_performance,
    plot_profit_metrics,
    plot_confusion_matrix,
    create_executive_summary,
    plot_optimization_results,
    plot_pareto_frontier
)

# Define what gets imported with "from scorecard_regression.profitability import *"
__all__ = [
    # Metrics
    'calculate_loan_metrics',
    'calculate_profit_metrics',
    'calculate_classification_metrics',
    'calculate_business_metrics',
    
    # Threshold analysis
    'analyze_threshold',
    'analyze_multiple_thresholds',
    'analyze_cutoff_tradeoffs',
    
    # Optimization
    'find_optimal_threshold',
    'profit_function',
    'advanced_profit_optimization',
    
    # Visualization
    'plot_threshold_performance',
    'plot_profit_metrics',
    'plot_confusion_matrix',
    'create_executive_summary',
    'plot_optimization_results',
    'plot_pareto_frontier'
]
