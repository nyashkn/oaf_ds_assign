import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set style for static plots
plt.style.use('seaborn-v0_8')  # Updated style name for newer matplotlib
sns.set_palette("husl")

# Create directories if they don't exist
Path("data/processed").mkdir(parents=True, exist_ok=True)

def load_data():
    """
    Load and perform initial data preprocessing
    """
    # Load training and holdout datasets
    train_df = pd.read_csv("data/raw/training_loan_processed.csv")
    holdout_df = pd.read_csv("data/raw/holdout_loan_processed.csv")
    
    # Convert date columns to datetime
    date_columns = ['contract_start_date']
    for df in [train_df, holdout_df]:
        for col in date_columns:
            df[col] = pd.to_datetime(df[col])
    
    return train_df, holdout_df

def analyze_distributions(df):
    """
    Analyze and plot distributions of key variables
    """
    # Create a directory for distribution plots
    plot_dir = Path("data/processed/distribution_plots")
    plot_dir.mkdir(exist_ok=True)
    
    # Numerical variables to analyze
    num_vars = [
        'nominal_contract_value',
        'deposit_amount',
        'cumulative_amount_paid_start'
    ]
    
    # Create distribution plots for numerical variables
    for var in num_vars:
        fig = plt.figure(figsize=(12, 6))
        
        # Distribution plot
        plt.subplot(1, 2, 1)
        sns.histplot(data=df, x=var, kde=True)
        plt.title(f'Distribution of {var}')
        plt.xticks(rotation=45)
        
        # Box plot
        plt.subplot(1, 2, 2)
        sns.boxplot(data=df, y=var)
        plt.title(f'Box Plot of {var}')
        
        plt.tight_layout()
        plt.savefig(plot_dir / f"{var}_distribution.png")
        plt.close()
    
    # Categorical variables analysis
    cat_vars = ['Loan_Type', 'region', 'area', 'sales_territory']
    
    for var in cat_vars:
        plt.figure(figsize=(12, 6))
        value_counts = df[var].value_counts()
        
        # Bar plot
        sns.barplot(x=value_counts.index, y=value_counts.values)
        plt.title(f'Distribution of {var}')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(plot_dir / f"{var}_distribution.png")
        plt.close()
    
    # Save summary statistics
    summary_stats = df[num_vars].describe()
    summary_stats.to_csv(plot_dir / "numerical_summary_stats.csv")
    
    # Calculate and save correlation matrix
    correlation_matrix = df[num_vars].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Matrix of Numerical Variables')
    plt.tight_layout()
    plt.savefig(plot_dir / "correlation_matrix.png")
    plt.close()

def analyze_repayment_distribution(df):
    """
    Analyze repayment patterns and identify potential default thresholds
    """
    plot_dir = Path("data/processed/repayment_analysis")
    plot_dir.mkdir(exist_ok=True)
    
    # Calculate repayment rate (cumulative_amount_paid_start / nominal_contract_value)
    df['repayment_rate'] = df['cumulative_amount_paid_start'] / df['nominal_contract_value']
    
    # Plot overall repayment rate distribution
    plt.figure(figsize=(12, 6))
    sns.histplot(data=df, x='repayment_rate', bins=50, kde=True)
    plt.title('Distribution of Repayment Rates')
    plt.axvline(x=0.98, color='r', linestyle='--', label='98% Target')
    plt.legend()
    plt.savefig(plot_dir / "repayment_rate_distribution.png")
    plt.close()
    
    # Repayment rate by loan type
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='Loan_Type', y='repayment_rate')
    plt.title('Repayment Rate by Loan Type')
    plt.axhline(y=0.98, color='r', linestyle='--', label='98% Target')
    plt.legend()
    plt.savefig(plot_dir / "repayment_rate_by_loan_type.png")
    plt.close()
    
    # Repayment rate by region
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='region', y='repayment_rate')
    plt.title('Repayment Rate by Region')
    plt.xticks(rotation=45)
    plt.axhline(y=0.98, color='r', linestyle='--', label='98% Target')
    plt.legend()
    plt.savefig(plot_dir / "repayment_rate_by_region.png")
    plt.close()
    
    # Calculate and save repayment statistics
    repayment_stats = df.groupby('Loan_Type')['repayment_rate'].describe()
    repayment_stats.to_csv(plot_dir / "repayment_stats_by_loan_type.csv")
    
    # Find potential elbow points in repayment rate distribution
    sorted_rates = np.sort(df['repayment_rate'].values)
    n_points = len(sorted_rates)
    x = np.arange(n_points)
    
    plt.figure(figsize=(12, 6))
    plt.plot(x/n_points, sorted_rates)
    plt.title('Cumulative Distribution of Repayment Rates')
    plt.xlabel('Proportion of Loans')
    plt.ylabel('Repayment Rate')
    plt.grid(True)
    plt.savefig(plot_dir / "repayment_rate_cumulative.png")
    plt.close()

def analyze_time_series_patterns(df):
    """
    Analyze temporal patterns in repayment behavior
    """
    plot_dir = Path("data/processed/time_series_analysis")
    plot_dir.mkdir(exist_ok=True)
    
    # Add month since contract start
    df['months_since_start'] = (
        pd.to_datetime('2023-09-01') - df['contract_start_date']
    ).dt.days / 30
    
    # Plot repayment rate vs months since contract start
    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=df, x='months_since_start', y='repayment_rate', alpha=0.5)
    plt.title('Repayment Rate vs Months Since Contract Start')
    plt.axhline(y=0.98, color='r', linestyle='--', label='98% Target')
    plt.legend()
    plt.savefig(plot_dir / "repayment_vs_time.png")
    plt.close()
    
    # Monthly averages
    monthly_avg = df.groupby(df['contract_start_date'].dt.to_period('M'))[
        'repayment_rate'
    ].mean().reset_index()
    monthly_avg['contract_start_date'] = monthly_avg['contract_start_date'].astype(str)
    
    plt.figure(figsize=(12, 6))
    plt.plot(monthly_avg['contract_start_date'], monthly_avg['repayment_rate'], marker='o')
    plt.title('Average Repayment Rate by Contract Start Month')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_dir / "monthly_repayment_trends.png")
    plt.close()
    
    # Repayment progression by loan type
    plt.figure(figsize=(12, 6))
    for loan_type in df['Loan_Type'].unique():
        subset = df[df['Loan_Type'] == loan_type]
        sns.regplot(
            data=subset,
            x='months_since_start',
            y='repayment_rate',
            label=loan_type,
            scatter=False
        )
    plt.title('Repayment Progression by Loan Type')
    plt.axhline(y=0.98, color='r', linestyle='--', label='98% Target')
    plt.legend()
    plt.savefig(plot_dir / "repayment_progression_by_loan_type.png")
    plt.close()
    
    # Save temporal statistics
    temporal_stats = df.groupby('months_since_start').agg({
        'repayment_rate': ['mean', 'std', 'count']
    }).round(4)
    temporal_stats.to_csv(plot_dir / "temporal_repayment_stats.csv")

def main():
    """
    Main execution function
    """
    print("Loading data...")
    train_df, holdout_df = load_data()
    
    print("\nAnalyzing distributions...")
    analyze_distributions(train_df)
    
    print("\nAnalyzing repayment distributions...")
    analyze_repayment_distribution(train_df)
    
    print("\nAnalyzing time series patterns...")
    analyze_time_series_patterns(train_df)
    
    print("\nAnalysis complete. Results saved in data/processed/")

if __name__ == "__main__":
    main()
