# One Acre Fund: Loan Performance Analysis Report

## Executive Summary

This analysis examines the loan performance data from One Acre Fund's lending program, focusing on repayment rates, loan distribution, and key performance indicators.

### Key Findings

1. Portfolio Composition
   - Majority Group Loans
   - Smaller portion of Individual Loans
   - Limited Paygo Loan presence

2. Repayment Performance
   - Average repayment rate below 98% target
   - Significant regional variations
   - Higher performance in Individual Loans

3. Risk Factors
   - Deposit ratio correlation with repayment
   - Loan size impact on performance
   - Regional performance differences

## Detailed Analysis

### 1. Loan Value vs Repayment Rate Distribution
![Loan Value Distribution](../data/processed/detailed_analysis/loan_value_vs_repayment.png)

The hexbin plot above shows the relationship between loan values and repayment rates:
- Density coloring reveals concentration of loans
- Red reference line shows 98% target
- Statistics box provides key metrics
- Higher density in lower loan value ranges

### 2. Deposit Ratio Analysis
![Deposit Analysis](../data/processed/detailed_analysis/deposit_ratio_analysis.png)

Analysis of how deposit ratios correlate with repayment performance:
- Quintile distribution with confidence intervals
- Sample size (n) shown for each group
- Clear trend in repayment rates across deposit levels
- Statistical significance indicated by error bars

### 3. Temporal Analysis
![Monthly Trends](../data/processed/detailed_analysis/monthly_trends.png)

Time series analysis of repayment rates:
- Monthly progression with confidence intervals
- Loan volume overlay shows portfolio growth
- Seasonal patterns and trends
- Volume-weighted performance metrics

### 4. Executive Dashboard
![Executive Dashboard](../data/processed/executive_summary/executive_dashboard.png)

Single-view summary of key metrics:
- Portfolio composition
- Regional performance
- Distribution patterns
- Key performance indicators

## Interactive Analysis

An interactive dashboard is available at `data/processed/executive_summary/interactive_dashboard.html` providing:
- Dynamic filtering capabilities
- Drill-down analysis options
- Detailed tooltips with additional metrics
- Custom view configurations

## Recommendations

1. Risk Management
   - Implement deposit ratio-based risk tiers
   - Consider regional performance in loan approval
   - Monitor loan size correlation with performance

2. Portfolio Strategy
   - Evaluate expansion of Individual Loans
   - Optimize deposit requirements
   - Address regional performance gaps

3. Monitoring Framework
   - Track temporal patterns
   - Monitor deposit ratio effectiveness
   - Regional performance tracking

## Technical Notes

Analysis performed using:
- Python data science stack (pandas, numpy)
- Visualization libraries (matplotlib, seaborn, plotly)
- Statistical analysis tools
- Interactive dashboard capabilities

Code and data processing steps available in:
- `src/01_analysis.v1.py`
- Generated visualizations in `data/processed/`
