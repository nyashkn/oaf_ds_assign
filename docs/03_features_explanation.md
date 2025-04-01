# Feature Descriptions and Computations

- deposit_ratio - Ratio of deposit amount to nominal contract value.

**Computation:**
```python
deposit_ratio = pl.col("deposit_amount") / pl.col("nominal_contract_value")
```

- sept_23_repayment_rate - Repayment rate as of September 2023.

**Computation:**
```python
sept_23_repayment_rate = pl.col("cumulative_amount_paid") / pl.col("nominal_contract_value")
```

- nov_23_repayment_rate - Repayment rate as of November 2023.

**Computation:**
```python
nov_23_repayment_rate = pl.col("cumulative_amount_paid") / pl.col("nominal_contract_value")
```

- diff_nov_23_to_sept_23_repayment_rate - Change in repayment rate between September and November 2023.

**Computation:**
```python
diff_nov_23_to_sept_23_repayment_rate = pl.col("nov_23_repayment_rate") - pl.col("sept_23_repayment_rate")
```

- historical_cum_deposit_region - Total deposit amount in the region for all loans prior to current loan's start date.

**Computation:**
```python
historical_cum_deposit = historical_loans.select(pl.sum("deposit_amount")).item()
```

- historical_cum_loans_region - Count of total loans issued in the same region before the current loan's start date.

**Computation:**
```python
loan_count = len(historical_loans)
```

- historical_cum_value_region - Total contract value in the region for all loans prior to current loan.

**Computation:**
```python
contract_value_sum = historical_loans.select(pl.sum("nominal_contract_value")).item()
```

- historical_cum_customers_region - Count of unique customers in the region before current loan.

**Computation:**
```python
unique_customers = historical_loans.select(pl.col("client_id").n_unique()).item()
```

- deposit_ratio_rank_region - Percentile rank of loan's deposit ratio within historical loans in the region.

**Computation:**
```python
deposit_ratio_rank = len(historical_loans.filter(pl.col("deposit_ratio") < current_deposit_ratio)) / len(historical_loans)
```

- contract_value_rank_region - Percentile rank of loan amount within historical loans in the region.

**Computation:**
```python
contract_value_rank = len(historical_loans.filter(pl.col("nominal_contract_value") < current_contract_value)) / len(historical_loans)
```

- distinct_dukas_region - Count of unique dukas (retail points) in the region before current loan.

**Computation:**
```python
distinct_dukas = historical_loans.select(pl.col("duka_name").n_unique()).item()
```

- distinct_customers_region - Count of unique customers in the region before current loan.

**Computation:**
```python
distinct_customers = historical_loans.select(pl.col("client_id").n_unique()).item()
```

- distance_to_centroid - Distance to regional center in kilometers.

**Computation:**
```python
# Computed using geospatial coordinates and distance formula
# Specific implementation depends on coordinate system used
```

- distance_to_nairobi - Distance to Nairobi in kilometers.

**Computation:**
```python
# Computed using geospatial coordinates and distance formula
```

- distance_to_mombasa - Distance to Mombasa in kilometers.

**Computation:**
```python
# Computed using geospatial coordinates and distance formula
```

- distance_to_kisumu - Distance to Kisumu in kilometers.

**Computation:**
```python
# Computed using geospatial coordinates and distance formula
```

- coords_imputed - Boolean flag indicating whether coordinates were imputed.

**Computation:**
```python
# True if coordinates were imputed, False otherwise
```

- contract_start_day - Day of month when contract started.

**Computation:**
```python
contract_start_day = pl.col("contract_start_date").dt.day()
```

- contract_day_name - Name of the day when contract started.

**Computation:**
```python
contract_day_name = day_name_map[pl.col("contract_start_date").dt.weekday()]
```

- contract_month - Month when contract started.

**Computation:**
```python
contract_month = pl.col("contract_start_date").dt.month()
```

- contract_quarter - Quarter when contract started.

**Computation:**
```python
contract_quarter = pl.col("contract_start_date").dt.quarter()
```

- is_weekend - Boolean flag indicating if contract started on weekend.

**Computation:**
```python
is_weekend = [1 if (num in [5, 6, 7]) else 0 for num in weekday_nums]
```

- days_since_start - Number of days elapsed since contract start date.

**Computation:**
```python
days_since_start = (current_date - pl.col("contract_start_date")).dt.days()
```

- months_since_start - Number of months elapsed since contract start date.

**Computation:**
```python
months_since_start = (current_date - pl.col("contract_start_date")).dt.months()
```

Note: Features ending with _region are also computed for area and sales_territory levels using the same logic but filtered for different geographic levels.