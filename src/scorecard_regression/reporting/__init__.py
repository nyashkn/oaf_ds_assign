"""
Reporting module for the scorecard regression package.

This module provides tools for generating comprehensive PDF reports of regression model results,
including visualizations, performance metrics, threshold analysis, and business impact assessment.
"""

from .report_generator import RegressionReport
from .report_utils import (
    initialize_regression_report,
    add_executive_summary_section,
    add_data_inspection_section, 
    add_model_development_section,
    add_evaluation_section,
    add_threshold_analysis_section,
    add_margin_analysis_section,
    add_holdout_evaluation_section
)
from .visualizations import (
    create_actual_vs_predicted_plot,
    create_error_distribution_plot,
    create_threshold_performance_plot,
    create_profit_curve_plot,
    create_margin_analysis_plot,
    create_metrics_heatmap,
    create_feature_importance_plot
)

__all__ = [
    # Main report generator class
    'RegressionReport',
    
    # Report utility functions
    'initialize_regression_report',
    'add_executive_summary_section',
    'add_data_inspection_section',
    'add_model_development_section',
    'add_evaluation_section',
    'add_threshold_analysis_section',
    'add_margin_analysis_section',
    'add_holdout_evaluation_section',
    
    # Visualization functions
    'create_actual_vs_predicted_plot',
    'create_error_distribution_plot',
    'create_threshold_performance_plot',
    'create_profit_curve_plot',
    'create_margin_analysis_plot',
    'create_metrics_heatmap',
    'create_feature_importance_plot'
]
