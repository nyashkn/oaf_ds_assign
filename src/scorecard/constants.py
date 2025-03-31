"""
Constants for scorecard modeling package.
"""

# Define variables that might leak information
EXCLUDE_VARS = [
    'client_id',
    'month_diff_contract_start_to_nov_23',
    'month_diff_contract_start_to_sept_23',
    'days_diff_contract_start_to_nov_23',
    'days_diff_contract_start_to_sept_23'
    'cumulative_amount_paid',
    'nominal_contract_value',
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

