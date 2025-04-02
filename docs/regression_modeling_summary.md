# Loan Repayment Rate Prediction: Regression vs. Classification Approach

## Executive Summary

This report summarizes the development and evaluation of a regression-based approach for predicting loan repayment rates, as an alternative to the binary classification approach. The regression model achieved excellent performance with R² of 0.961 on the test set and 0.963 on the holdout set, demonstrating strong generalizability. The approach allows for flexible threshold selection after modeling, which enables optimized profit-based decisions.

## Comparison of Approaches

### Binary Classification Approach
- Predicts whether a loan will meet a specific repayment rate threshold (e.g., 0.8)
- Outputs: 0 or 1 (will not meet / will meet threshold)
- Advantages:
  - Simpler conceptual model
  - Directly answers a yes/no business question
- Disadvantages:
  - Loses granularity of information
  - New threshold requires retraining
  - Limited insight on expected loss

### Regression Approach
- Predicts the actual repayment rate value (e.g., 0.75, 0.92)
- Outputs: Continuous value between 0 and 1
- Advantages:
  - Preserves full information about repayment
  - Flexible threshold selection after model training
  - Better expected loss estimation
  - More nuanced risk assessment
  - Enhanced profitability analysis
- Disadvantages:
  - Slightly more complex model interpretation

## Model Performance

### Training & Testing Performance

| Model Type           | Training R² | Test R² | Test RMSE | Test MAE |
|----------------------|------------|---------|-----------|----------|
| Linear Regression    | 0.651      | 0.650   | 0.139     | 0.106    |
| HistGradientBoosting | 0.967      | 0.961   | 0.047     | 0.030    |

The HistGradientBoosting model dramatically outperformed linear regression, indicating significant non-linear relationships in the data.

### Holdout Set Performance
To validate the model's generalizability, we evaluated it on an independent holdout dataset:

| Metric | Value  |
|--------|--------|
| R²     | 0.963  |
| RMSE   | 0.046  |
| MAE    | 0.030  |

The model maintained its excellent performance on the holdout data, confirming its robustness and generalizability.

## Profitability Analysis

The regression approach enables sophisticated profitability analysis by allowing various threshold comparisons across different gross margins.

### Cross-Analysis of Thresholds and Margins

We conducted a comprehensive analysis of how different thresholds and gross margins affect various business metrics:

#### Repayment Rate Across Thresholds and Margins
![Repayment Rate Heatmap](../data/processed/margin_analysis/repayment_rate_heatmap.png)

#### Default Rate Across Thresholds and Margins
![Default Rate Heatmap](../data/processed/margin_analysis/default_rate_heatmap.png)

#### Approval Rate Across Thresholds and Margins
![Approval Rate Heatmap](../data/processed/margin_analysis/approval_rate_heatmap.png)

#### Profit Across Thresholds and Margins
![Profit Heatmap](../data/processed/margin_analysis/actual_profit_heatmap.png)

#### Net Profit Margin Across Thresholds and Margins
![Net Profit Margin Heatmap](../data/processed/margin_analysis/net_profit_margin_heatmap.png)

#### Return on Investment Across Thresholds and Margins
![ROI Heatmap](../data/processed/margin_analysis/return_on_investment_heatmap.png)

### Optimal Threshold (Training)
- **Threshold**: 0.65
- **Approval Rate**: 29.1%
- **Actual Repayment Rate**: 0.79
- **Total Profit**: 6,226,480 KES

### Optimal Threshold (Holdout)
- **Threshold**: 0.65
- **Approval Rate**: 29.3% 
- **Actual Repayment Rate**: 0.79
- **Total Profit**: 5,211,236 KES

### Threshold Comparison (30% Gross Margin)

| Threshold | Approval Rate | Repayment Rate | Default Rate | Net Profit Margin | ROI     |
|-----------|---------------|----------------|--------------|-------------------|---------|
| 0.60      | 35.6%         | 76.0%          | 0.0%         | 22.8%             | -24.0%  |
| 0.65      | 29.3%         | 78.8%          | 0.0%         | 23.6%             | -21.2%  |
| 0.70      | 26.0%         | 80.3%          | 0.0%         | 24.1%             | -19.7%  |
| 0.75      | 16.9%         | 85.4%          | 0.0%         | 25.6%             | -14.6%  |
| 0.80      | 8.2%          | 90.5%          | 0.0%         | 27.2%             | -9.5%   |
| 0.85      | 1.4%          | 95.7%          | 0.0%         | 28.7%             | -4.3%   |
| 0.90      | 0.1%          | 98.5%          | 0.0%         | 29.5%             | -1.5%   |
| 0.95      | 0.0%          | 98.7%          | 0.0%         | 29.6%             | -1.3%   |

> **Note on Customer Lifetime Value**: This analysis does not factor in the customer lifetime value of repeat borrowers. Historically, repeat customers have a lower default rate, as they already value the lending facility. Including this factor would likely improve the profit metrics for lower thresholds that approve more loans.

## Expected Loss Calculation

With the regression approach, we can calculate expected loss more accurately:

```
Expected Loss = Loan Amount × (1 - Predicted Repayment Rate)
```

This formula provides a precise estimation of the expected financial loss for each loan, enabling:
1. Risk-based pricing
2. Loan amount adjustments
3. Differentiation between high-risk and borderline loans

## Key Insights

1. **Model Performance**: The regression approach achieves excellent predictive accuracy with 96.3% of the variance in repayment rates explained by the model.

2. **Consistent Optimal Threshold**: Both training and holdout evaluations identified 0.65 as the optimal threshold for maximizing profit, suggesting this is a robust decision boundary.

3. **Trade-offs**: Higher thresholds lead to better repayment rates but significantly reduce the approval rate, demonstrating the classic risk-reward tradeoff.

4. **Business Impact**: The regression approach provides a more nuanced tool for risk management and profit optimization compared to binary classification.

## Conclusion

The regression-based approach provides significant advantages over binary classification for loan repayment prediction:

1. It preserves the full information about expected repayment rates
2. It enables flexible threshold selection after model training
3. It provides more accurate expected loss estimation
4. It generalizes well to unseen data

For these reasons, we recommend adopting the regression approach for loan repayment prediction, with an optimal threshold of 0.65 for maximizing profit while balancing risk.
