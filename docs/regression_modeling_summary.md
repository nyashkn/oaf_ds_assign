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

The regression approach enables sophisticated profitability analysis by allowing various threshold comparisons:

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

### Threshold Comparison

| Threshold | Approval Rate | Repayment Rate | Expected Loss |
|-----------|---------------|----------------|---------------|
| 0.65      | 29.1%         | 0.79           | 21%           |
| 0.70      | 25.5%         | 0.81           | 19%           |
| 0.75      | 22.4%         | 0.82           | 18%           |
| 0.80      | 19.3%         | 0.84           | 16%           |
| 0.85      | 15.9%         | 0.86           | 14%           |
| 0.90      | 12.1%         | 0.88           | 12%           |
| 0.95      | 7.8%          | 0.90           | 10%           |

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
