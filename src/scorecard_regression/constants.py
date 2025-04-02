"""
Constants for regression-based loan repayment rate prediction.

This module provides constants for the scorecard regression package,
including default parameters, excluded variables, and other configuration values.
"""

from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler

# Variables that might leak information about the target
EXCLUDE_VARS = [
    'client_id',
    'month_diff_contract_start_to_nov_23',
    'month_diff_contract_start_to_sept_23',
    'days_diff_contract_start_to_nov_23',
    'days_diff_contract_start_to_sept_23',
    'cumulative_amount_paid',
    'cumulative_amount_paid_start',
    'diff_nov_23_to_sept_23_repayment_rate',
    # 'nominal_contract_value',
    'contract_start_date',
    # These variables directly relate to or derive from the target
    'sept_23_repayment_rate', 
    'nov_23_repayment_rate',
    'months_since_start',
    'days_since_start'
]

# Date patterns to identify date-formatted columns
DATE_PATTERNS = [
    r'^\d{4}-\d{2}$',       # YYYY-MM
    r'^\d{4}-\d{2}-\d{2}$',  # YYYY-MM-DD
    r'^\d{2}/\d{2}/\d{4}$',  # MM/DD/YYYY
    r'^\d{2}/\d{2}/\d{2}$',  # MM/DD/YY
    r'^\d{2}-\d{2}-\d{4}$',  # DD-MM-YYYY
    r'^\d{4}$'               # YYYY
]

# Default business parameters for profitability calculations
DEFAULT_BUSINESS_PARAMS = {
    'gross_margin': 0.16,  # 16% profit margin on repaid loan value
    'default_loss_rate': 1.0,  # 100% loss on defaulted amount
    'target_threshold': 0.8,  # Default threshold for good/bad classification
    'operating_cost_per_loan': 0.0,  # Can be set to a value if applicable
    'overhead_allocation': 0.0,  # Can be set to a value if applicable
    'collection_cost_ratio': 0.0  # Cost of collection efforts as % of loan value
}

# Dictionary of available scalers
SCALING_METHODS = {
    'standard': StandardScaler,
    'robust': RobustScaler,
    'minmax': MinMaxScaler,
    'maxabs': MaxAbsScaler,
    'none': None
}

# Default parameters for regression models
DEFAULT_REGRESSION_PARAMS = {
    'linear': {},  # Default linear regression has no hyperparameters
    'ridge': {
        'alpha': 1.0,
        'fit_intercept': True,
        'random_state': 42
    },
    'lasso': {
        'alpha': 0.01,
        'fit_intercept': True,
        'random_state': 42
    },
    'elasticnet': {
        'alpha': 0.1,
        'l1_ratio': 0.5,
        'fit_intercept': True,
        'random_state': 42
    },
    'histgb': {
        'loss': 'squared_error',
        'learning_rate': 0.1,
        'max_depth': 3,
        'min_samples_leaf': 10,
        'n_estimators': 100,
        'random_state': 42
    },
    'xgboost': {
        'objective': 'reg:squarederror',
        'learning_rate': 0.1,
        'max_depth': 3,
        'min_child_weight': 1,
        'n_estimators': 100,
        'random_state': 42
    },
    'lightgbm': {
        'objective': 'regression',
        'learning_rate': 0.1,
        'max_depth': 3,
        'num_leaves': 31,
        'n_estimators': 100,
        'random_state': 42
    },
    'catboost': {
        'loss_function': 'RMSE',
        'learning_rate': 0.1,
        'depth': 3,
        'iterations': 100,
        'random_seed': 42,
        'verbose': False
    },
    'rf': {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'random_state': 42
    }
}

# Variable selection parameters
VARIABLE_SELECTION = {
    'correlation_threshold': 0.8,  # Maximum correlation allowed between features
    'vif_threshold': 10.0,  # Maximum Variance Inflation Factor allowed for features
    'p_value_threshold': 0.05,  # Maximum p-value for statistical significance
    'importance_threshold': 0.01  # Minimum feature importance to retain a feature
}
