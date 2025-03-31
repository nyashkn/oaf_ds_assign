"""
OAF Loan Performance Scorecard Modeling - Modular Package

This package provides a modular approach to credit scorecard development,
with each step implemented in a separate module.
"""

from .utils import create_version_path
from .data_inspection import inspect_data, handle_date_like_columns
from .target_preparation import create_binary_target, find_optimal_cutoff
from .variable_selection import exclude_leakage_variables, partition_data, select_variables
from .binning import perform_woe_binning, apply_woe_transformation
from .modeling import develop_scorecard_model, evaluate_model_performance
from .main import run_modular_scorecard

__all__ = [
    'create_version_path',
    'inspect_data',
    'handle_date_like_columns',
    'create_binary_target',
    'find_optimal_cutoff',
    'exclude_leakage_variables',
    'partition_data',
    'select_variables',
    'perform_woe_binning',
    'apply_woe_transformation',
    'develop_scorecard_model',
    'evaluate_model_performance',
    'run_modular_scorecard'
]
