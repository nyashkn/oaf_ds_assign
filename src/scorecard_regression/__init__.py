"""
OAF Loan Performance Regression Modeling Package

This package provides a regression approach to loan repayment prediction,
predicting the repayment rate directly rather than using binary classification.
"""

from .scaling import scale_features, apply_scaling_transformation
from .data_inspection import inspect_data, handle_date_like_columns
from .modeling_regression import develop_regression_model, select_regression_model
from .evaluation import evaluate_regression_performance, calculate_regression_metrics, analyze_error_distribution, analyze_performance_by_segment
from .explanation import explain_model, generate_shap_explanation, generate_pdp_plots
from .profitability import analyze_multiple_thresholds, analyze_cutoff_tradeoffs, find_optimal_threshold

__all__ = [
    'inspect_data',
    'handle_date_like_columns',
    'scale_features',
    'apply_scaling_transformation',
    'develop_regression_model',
    'select_regression_model',
    'evaluate_regression_performance',
    'calculate_regression_metrics',
    'analyze_error_distribution',
    'analyze_performance_by_segment',
    'explain_model',
    'generate_shap_explanation',
    'generate_pdp_plots',
    'analyze_multiple_thresholds',
    'analyze_cutoff_tradeoffs',
    'find_optimal_threshold'
]
