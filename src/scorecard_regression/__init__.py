"""
OneAcre Fund Loan Repayment Rate Prediction - Regression-based Approach

This package provides a complete workflow for regression-based loan repayment rate 
prediction, which differs from traditional binary classification scorecard methods
by directly predicting the continuous repayment rate and then applying thresholds
for loan approval decisions.

Key components:
- Data inspection and preparation
- Variable selection
- Regression modeling
- Threshold optimization
- Profitability analysis
- Reporting and visualization
"""

# Import core modules
from .constants import (
    EXCLUDE_VARS, 
    DATE_PATTERNS,
    DEFAULT_REGRESSION_PARAMS,
    SCALING_METHODS, 
    DEFAULT_BUSINESS_PARAMS,
    VARIABLE_SELECTION
)

from .utils import (
    create_version_path,
    calculate_metrics,
    format_predictions,
    prepare_model_summary,
    save_model,
    load_model,
    create_threshold_prediction,
    normalize_feature_importances
)

from .variable_selection import (
    exclude_leakage_variables,
    partition_data,
    check_multicollinearity,
    select_variables_correlation,
    select_variables_importance,
    select_significant_variables
)

from .modeling_regression import (
    train_regression_model,
    evaluate_regression_model,
    cross_validate_regression,
    compare_regression_vs_classification
)

# Import profitability subpackage
from .profitability import (
    # Metrics functions
    calculate_loan_metrics,
    calculate_profit_metrics,
    calculate_classification_metrics,
    calculate_business_metrics,
    
    # Threshold analysis
    analyze_threshold,
    analyze_multiple_thresholds,
    analyze_cutoff_tradeoffs,
    
    # Optimization
    find_optimal_threshold,
    profit_function,
    advanced_profit_optimization,
    
    # Visualization
    plot_threshold_performance,
    plot_profit_metrics,
    plot_confusion_matrix,
    create_executive_summary,
    plot_optimization_results,
    plot_pareto_frontier
)

# Define what gets imported with "from scorecard_regression import *"
__all__ = [
    # Constants
    'EXCLUDE_VARS', 'DATE_PATTERNS', 'DEFAULT_REGRESSION_PARAMS',
    'SCALING_METHODS', 'DEFAULT_BUSINESS_PARAMS', 'VARIABLE_SELECTION',
    
    # Utilities
    'create_version_path', 'calculate_metrics', 'format_predictions',
    'prepare_model_summary', 'save_model', 'load_model',
    'create_threshold_prediction', 'normalize_feature_importances',
    
    # Variable selection
    'exclude_leakage_variables', 'partition_data', 'check_multicollinearity',
    'select_variables_correlation', 'select_variables_importance',
    'select_significant_variables',
    
    # Modeling
    'train_regression_model', 'evaluate_regression_model',
    'cross_validate_regression', 'compare_regression_vs_classification',
    
    # Metrics
    'calculate_loan_metrics', 'calculate_profit_metrics',
    'calculate_classification_metrics', 'calculate_business_metrics',
    
    # Threshold analysis
    'analyze_threshold', 'analyze_multiple_thresholds', 'analyze_cutoff_tradeoffs',
    
    # Optimization
    'find_optimal_threshold', 'profit_function', 'advanced_profit_optimization',
    
    # Visualization
    'plot_threshold_performance', 'plot_profit_metrics', 'plot_confusion_matrix',
    'create_executive_summary', 'plot_optimization_results', 'plot_pareto_frontend'
]
