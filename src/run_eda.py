from eda_analysis import create_sample_data, enhanced_repayment_eda
import pandas as pd

# Load your data
df = pd.read_csv('../data/raw/training_loan_processed.csv')

# Run full analysis
enhanced_repayment_eda(df, output_dir='repayment_analysis')

# Or run with sample
enhanced_repayment_eda(df, sample_size=5000, output_dir='../data/processed/eda_analysis')