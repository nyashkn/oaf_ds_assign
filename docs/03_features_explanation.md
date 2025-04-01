# Feature Categories and Descriptions

## Geographic Levels
Features marked with (*) are computed at three geographic levels:
- region
- area
- sales_territory

## Feature Categories

| Feature Name | Category | Description |
|--------------|----------|-------------|
| deposit_ratio | Basic | Deposit amount / Contract value |
| cumulative_amount_paid | Basic | Total amount paid to date |
| nominal_contract_value | Basic | Original contract amount |
| sept_23_repayment_rate | Performance | Repayment rate as of Sept 2023 |
| nov_23_repayment_rate | Performance | Repayment rate as of Nov 2023 |
| diff_nov_23_to_sept_23_repayment_rate | Performance | Change in repayment rate |
| historical_cum_deposit_* | Historical | Total deposits in geographic area |
| historical_cum_loans_* | Historical | Total loans in geographic area |
| historical_cum_value_* | Historical | Total value in geographic area |
| historical_cum_customers_* | Historical | Unique customers in geographic area |
| deposit_ratio_rank_* | Relative | Deposit ratio percentile |
| contract_value_rank_* | Relative | Contract value percentile |
| historical_loans_rank_* | Relative | Loan count percentile |
| historical_value_rank_* | Relative | Value percentile |
| historical_deposit_rank_* | Relative | Deposit percentile |
| distinct_dukas_* | Infrastructure | Unique retail points |
| distinct_customers_* | Infrastructure | Unique customers |
| distance_to_centroid | Geographic | Distance to regional center |
| distance_to_nairobi | Geographic | Distance to Nairobi |
| distance_to_mombasa | Geographic | Distance to Mombasa |
| distance_to_kisumu | Geographic | Distance to Kisumu |
| coords_imputed | Geographic | Whether coordinates were imputed |
| contract_start_day | Temporal | Day of month (1-31) |
| contract_day_name | Temporal | Day name (Monday-Sunday) |
| contract_month | Temporal | Month number (1-12) |
| contract_quarter | Temporal | Quarter number (1-4) |
| is_weekend | Temporal | Weekend flag (0/1) |
| days_since_start | Temporal | Days from contract start |
| months_since_start | Temporal | Months from contract start |
| days_diff_contract_start_to_sept_23 | Temporal | Days to Sept 2023 |
| days_diff_contract_start_to_nov_23 | Temporal | Days to Nov 2023 |
| month_diff_contract_start_to_sept_23 | Temporal | Months to Sept 2023 |
| month_diff_contract_start_to_nov_23 | Temporal | Months to Nov 2023 |

## Feature Computations

```python
# Basic Metrics
deposit_ratio = pl.col("deposit_amount") / pl.col("nominal_contract_value")

# Performance Metrics
sept_23_repayment_rate = pl.col("cumulative_amount_paid") / pl.col("nominal_contract_value")
nov_23_repayment_rate = pl.col("cumulative_amount_paid") / pl.col("nominal_contract_value")
diff_nov_23_to_sept_23_repayment_rate = pl.col("nov_23_repayment_rate") - pl.col("sept_23_repayment_rate")

# Historical Metrics (computed for each geographic level)
historical_cum_deposit = historical_loans.select(pl.sum("deposit_amount")).item()
historical_cum_loans = len(historical_loans)
historical_cum_value = historical_loans.select(pl.sum("nominal_contract_value")).item()
historical_cum_customers = historical_loans.select(pl.col("client_id").n_unique()).item()

# Relative Position Metrics (computed for each geographic level)
deposit_ratio_rank = len(historical_loans.filter(pl.col("deposit_ratio") < current_deposit_ratio)) / len(historical_loans)
contract_value_rank = len(historical_loans.filter(pl.col("nominal_contract_value") < current_contract_value)) / len(historical_loans)

# Infrastructure Metrics (computed for each geographic level)
distinct_dukas = historical_loans.select(pl.col("duka_name").n_unique()).item()
distinct_customers = historical_loans.select(pl.col("client_id").n_unique()).item()

# Temporal Features
contract_start_day = pl.col("contract_start_date").dt.day()
contract_day_name = day_name_map[pl.col("contract_start_date").dt.weekday()]
contract_month = pl.col("contract_start_date").dt.month()
contract_quarter = pl.col("contract_start_date").dt.quarter()
is_weekend = [1 if (num in [5, 6, 7]) else 0 for num in weekday_nums]

# Time Difference Features
days_since_start = (current_date - pl.col("contract_start_date")).dt.days()
months_since_start = (current_date - pl.col("contract_start_date")).dt.months()
days_diff_contract_start_to_sept_23 = (sept_23_date - pl.col("contract_start_date")).dt.days()
days_diff_contract_start_to_nov_23 = (nov_23_date - pl.col("contract_start_date")).dt.days()
```

Note: Features marked with (*) are computed using the same logic but filtered for different geographic levels (region, area, sales_territory).