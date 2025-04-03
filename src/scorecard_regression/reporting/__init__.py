"""
Reporting Package for Loan Repayment Prediction Models

This package provides utilities for generating reports and visualizations
for loan repayment prediction models, including:
- Executive summaries
- Performance metrics
- Profitability analysis
- Threshold analysis
- Model comparison
"""

from .model_comparison_report import (
    ModelComparisonReport,
    create_markdown_report
)

__all__ = [
    'ModelComparisonReport',
    'create_markdown_report',
]
