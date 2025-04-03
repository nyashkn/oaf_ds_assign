"""
Model Comparison Package for Loan Repayment Prediction Models

This package provides utilities for comparing different loan repayment prediction models,
particularly comparing models that use:
1. Only application-time features
2. Application-time features plus payment history data

Key components:
- Model training and evaluation
- Threshold analysis for different models
- Profitability comparison
- Report generation
"""

from .comparison import (
    APPLICATION_FEATURES,
    SEPTEMBER_PAYMENT_FEATURES,
    BUSINESS_PARAMS,
    load_and_prepare_data,
    filter_features,
    train_model,
    evaluate_model,
    analyze_model_profit,
    compare_models,
    create_comparison_plots,
    save_predictions_for_holdout,
    feature_importance_comparison
)

__all__ = [
    # Feature sets
    'APPLICATION_FEATURES',
    'SEPTEMBER_PAYMENT_FEATURES',
    'BUSINESS_PARAMS',
    
    # Core functions
    'load_and_prepare_data',
    'filter_features',
    'train_model',
    'evaluate_model',
    'analyze_model_profit',
    'compare_models',
    'create_comparison_plots',
    'save_predictions_for_holdout',
    'feature_importance_comparison',
]
