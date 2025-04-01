"""
Constants for regression scorecard modeling package.
"""

# Import and extend constants from the original scorecard package
from src.scorecard.constants import DATE_PATTERNS

# Define variables that should be excluded from regression modeling
EXCLUDE_VARS = [
    # ID/identifier fields
    'client_id',
    'duka_name',  # Store name is text
    
    # Date fields already handled by date column handling
    'contract_start_date',
    'month',
    
    # Categorical fields that need encoding before use
    'Loan_Type',
    'region',
    'area',
    'sales_territory',
    'contract_day_name',
    
    # Time difference fields that may cause leakage
    'month_diff_contract_start_to_nov_23',
    'month_diff_contract_start_to_sept_23',
    'days_diff_contract_start_to_nov_23',
    'days_diff_contract_start_to_sept_23',
    'months_since_start',
    'days_since_start',
    
    # Target-related fields
    'sept_23_repayment_rate',  # Target variable
    'nov_23_repayment_rate',   # Alternative target
    'diff_nov_23_to_sept_23_repayment_rate',  # Derived from targets
    
    # Boolean fields as strings
    'is_weekend',
    'coords_imputed'
]

# Define regression-specific constants
REGRESSION_TARGET_VARS = [
    'sept_23_repayment_rate',
    'nov_23_repayment_rate'
]

# Define regression model parameters
DEFAULT_REGRESSION_PARAMS = {
    'histgb': {
        'max_depth': 3,
        'learning_rate': 0.1,
        'max_iter': 100,
        'l2_regularization': 0.1
    },
    'linear': {
        # LinearRegression doesn't use regularization parameters
    },
    'ridge': {
        'alpha': 0.1
    },
    'lasso': {
        'alpha': 0.1
    },
    'elasticnet': {
        'alpha': 0.1,
        'l1_ratio': 0.5
    },
    'xgboost': {
        'max_depth': 3,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'reg_lambda': 1.0,
        'reg_alpha': 0.0
    },
    'lightgbm': {
        'max_depth': 3,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'reg_lambda': 1.0
    },
    'catboost': {
        'depth': 3,
        'learning_rate': 0.1,
        'iterations': 100,
        'l2_leaf_reg': 3.0
    }
}

# Feature scaling settings
SCALING_METHODS = {
    'robust': 'RobustScaler',
    'standard': 'StandardScaler',
    'minmax': 'MinMaxScaler',
    'maxabs': 'MaxAbsScaler',
    'none': None
}

# Regression performance metrics
REGRESSION_METRICS = [
    'rmse',  # Root Mean Squared Error
    'mae',   # Mean Absolute Error
    'r2',    # Coefficient of Determination
    'mse',   # Mean Squared Error
    'mape',  # Mean Absolute Percentage Error
    'ev'     # Explained Variance
]

# Profitability calculation constants
GROSS_MARGIN = 0.16  # 16% based on OAF's gross margin

# Define cutoff thresholds to evaluate
DEFAULT_CUTOFF_THRESHOLDS = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.98]

# Feature importance visualization settings
SHAP_PLOT_SETTINGS = {
    'max_display': 20,         # Maximum number of features to display
    'plot_type': 'bar',        # Default plot type for summary
    'cmap': 'coolwarm',        # Color map for SHAP values
    'alpha': 0.8               # Transparency
}

# PDP plot settings
PDP_PLOT_SETTINGS = {
    'n_features': 10,          # Top features to create PDPs for
    'grid_resolution': 20,     # Number of points in grid
    'centered': True           # Center PDPs at zero
}
