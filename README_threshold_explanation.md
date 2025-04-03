# Threshold Explanation for Loan Repayment Models

This document provides an explanation of how thresholds work in our loan repayment prediction models and how they affect business metrics.

## What are Thresholds?

In our loan repayment prediction models, a **threshold** is a cutoff value that determines whether to approve or reject a loan application:

- If the predicted repayment rate is **≥ threshold**, the loan is approved
- If the predicted repayment rate is **< threshold**, the loan is rejected

For example, with a threshold of 0.80, a loan with a predicted repayment rate of 0.82 would be approved, while a loan with a predicted rate of 0.75 would be rejected.

## Key Metrics in the Analysis

The `metrics_by_threshold` data shows various performance metrics at different threshold values:

### Business Metrics

- **threshold**: The cutoff value for approving loans
- **approval_rate**: Percentage of loans that would be approved at this threshold
- **approved_loan_value**: Total monetary value of approved loans
- **actual_profit**: The realized profit based on actual repayments
- **expected_profit**: The predicted profit based on model predictions
- **actual_loss**: Monetary loss from defaults
- **roi**: Return on Investment (actual profit divided by approved loan value)

### Model Performance Metrics

- **accuracy**: Percentage of correct predictions (approved loans that repay + rejected loans that would default)
- **precision**: Among approved loans, what percentage actually repay above the threshold
- **recall**: Among all good loans (those that repay above threshold), what percentage were approved
- **f1_score**: Harmonic mean of precision and recall
- **predicted_avg_repayment**: Average predicted repayment rate for approved loans
- **actual_avg_repayment**: Average actual repayment rate for approved loans

## How Profit and ROI are Calculated

### Profit Calculation

For each approved loan:
1. **Revenue** = Loan Amount × Actual Repayment Rate × Margin
2. **Loss** = Loan Amount × (1 - Actual Repayment Rate) × Default Loss Rate
3. **Profit** = Revenue - Loss

Where:
- **Margin** is the gross profit margin (default 16%)
- **Default Loss Rate** is the percentage of defaulted amount that becomes a loss (default 100%)

### ROI Calculation

ROI = Actual Profit / Approved Loan Value

A negative ROI means that the losses from defaults exceed the profits from successful repayments.

## Optimal Thresholds

The analysis identifies optimal thresholds for different business objectives:

1. **Profit-focused**: Threshold that maximizes total profit
2. **ROI-focused**: Threshold that maximizes return on investment
3. **Balanced**: A compromise threshold that balances profit and ROI

## Sensitivity Analysis

The sensitivity analysis shows how changes in threshold affect profit:

- **threshold_range**: Range between consecutive thresholds
- **threshold_diff**: Difference between thresholds
- **profit_diff**: Change in profit between thresholds
- **sensitivity**: Profit change per unit of threshold change

Higher sensitivity means profit changes more rapidly at that threshold range.

## Using the Model Comparison Report

We've enhanced the system to generate comprehensive reports comparing two models:

1. **Model 1**: Using only application-time features
2. **Model 2**: Including September payment data

The reports include:
- Performance metrics comparison
- Profitability analysis at different thresholds
- ROI analysis
- Feature importance comparison
- Business recommendations

These reports help understand the trade-offs between different models and threshold settings.

## Running the Enhanced Analysis

Use the `run_modular_regression.py` script to run the complete analysis, including model comparison:

```bash
python run_modular_regression.py --features data/processed/all_features.csv --output data/processed/regression_results
```

The script will generate comprehensive reports in both PDF and Markdown formats in the output directory.
