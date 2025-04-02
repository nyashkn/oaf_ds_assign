import pandas as pd
import numpy as np
from ydata_profiling import ProfileReport
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm

def enhanced_repayment_eda(df, sample_size=None, output_dir='repayment_eda_output', random_state=42):
    """
    Comprehensive EDA for repayment rate analysis with time-based comparisons,
    cure rates, and target achievement metrics.
    
    Args:
        df: DataFrame with loan repayment data
        sample_size: Optional sample size to use (e.g., 5000). If None, uses full dataset
        output_dir: Directory to save outputs
        random_state: Random seed for reproducible sampling
    """
    # Take a random sample if sample_size is specified
    if sample_size is not None and sample_size < len(df):
        print(f"Taking a random sample of {sample_size} records from {len(df)} total records")
        df = df.sample(n=sample_size, random_state=random_state)
        output_dir = f"{output_dir}_sample_{sample_size}"
        print(f"Results will be saved to {output_dir}")
    else:
        print(f"Using full dataset with {len(df)} records")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("Starting enhanced repayment rate analysis...")
    
    # Rest of your function remains the same
    # ... (all the existing code)

    # 1. Data preparation with progress tracking
    steps = ["Data preparation", "Feature engineering", "Univariate analysis", 
             "Target metrics", "Categorical analysis", "Multivariate analysis", 
             "Time-based analysis", "PDF report generation", "CSV exports"]
    
    with tqdm(total=len(steps), desc="EDA Progress") as pbar:
        # Step 1: Data preparation
        df['contract_start_date'] = pd.to_datetime(df['contract_start_date'])
        df['days_since_start'] = (pd.Timestamp.now() - df['contract_start_date']).dt.days
        pbar.update(1)
        
        # Step 2: Feature engineering - Add your specific metrics
        # Ratio calculations
        df['deposit_ratio'] = df['deposit_amount'] / df['nominal_contract_value']
        df['sept_23_repayment_rate'] = df['cumulative_amount_paid_start'] / df['nominal_contract_value']
        df['nov_23_repayment_rate'] = df['cumulative_amount_paid'] / df['nominal_contract_value']
        
        # Target achievement indicators
        df['sept_below_target'] = df['sept_23_repayment_rate'] < 0.98
        df['nov_below_target'] = df['nov_23_repayment_rate'] < 0.98
        df['cured'] = (df['nov_23_repayment_rate'] >= 0.98) & (df['sept_23_repayment_rate'] < 0.98)
        
        # Improvement calculation
        df['repayment_improvement'] = df['nov_23_repayment_rate'] - df['sept_23_repayment_rate']
        
        # Calculate repayment rate quantiles and create buckets
        df['sept_repayment_bucket'] = pd.qcut(df['sept_23_repayment_rate'], 5, 
                                            labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
        df['nov_repayment_bucket'] = pd.qcut(df['nov_23_repayment_rate'], 5, 
                                           labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
        pbar.update(1)
        
        # Step 3: Univariate statistics
        numerical_vars = ['deposit_ratio', 'sept_23_repayment_rate', 'nov_23_repayment_rate', 
                         'repayment_improvement', 'cumulative_amount_paid', 'nominal_contract_value', 
                         'deposit_amount', 'days_since_start']
        
        univariate_stats = {}
        for var in numerical_vars:
            stats = df[var].describe(percentiles=[0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
            univariate_stats[var] = stats
        
        # Convert to DataFrame for export later
        univariate_df = pd.DataFrame(univariate_stats)
        pbar.update(1)
        
        # Step 4: Target metrics - Portfolio level KPIs
        portfolio_metrics = {
            'sept_below_target': df['sept_below_target'].mean(),
            'nov_below_target': df['nov_below_target'].mean(),
            'pct_cured': df['cured'].mean(),
            'avg_sept_repayment': df['sept_23_repayment_rate'].mean(),
            'avg_nov_repayment': df['nov_23_repayment_rate'].mean(),
            'overall_cure_rate': df['nov_23_repayment_rate'].mean() - df['sept_23_repayment_rate'].mean(),
            'total_contracts': len(df),
            'total_contract_value': df['nominal_contract_value'].sum(),
            'total_collected': df['cumulative_amount_paid'].sum(),
            'overall_collection_rate': df['cumulative_amount_paid'].sum() / df['nominal_contract_value'].sum()
        }
        
        # Create time-based progress table
        overall_progress = pd.DataFrame({
            'Time Point': ['Initial', 'September 2023', 'November 2023'],
            'Average Repayment Rate': [
                df['deposit_ratio'].mean(),
                df['sept_23_repayment_rate'].mean(),
                df['nov_23_repayment_rate'].mean()
            ],
            'Below Target (%)': [
                1.0,  # All accounts below target initially
                df['sept_below_target'].mean() * 100,
                df['nov_below_target'].mean() * 100
            ],
            'Total Collected': [
                df['deposit_amount'].sum(),
                df['cumulative_amount_paid_start'].sum(),
                df['cumulative_amount_paid'].sum()
            ]
        })
        
        # Add the cure rate calculation requested
        portfolio_metrics['overall_cure_rate'] = (
            overall_progress['Average Repayment Rate'].iloc[2] - 
            overall_progress['Average Repayment Rate'].iloc[1]
        )
        pbar.update(1)
        
        # Step 5: Categorical variables analysis
        categorical_vars = ['Loan_Type', 'region', 'area', 'sales_territory']
        cat_stats = {}
        
        for cat in categorical_vars:
            if cat in df.columns:
                # Group by category and calculate key metrics
                cat_group = df.groupby(cat).agg({
                    'client_id': 'count',
                    'sept_23_repayment_rate': ['mean', 'median'],
                    'nov_23_repayment_rate': ['mean', 'median'],
                    'sept_below_target': 'mean',
                    'nov_below_target': 'mean',
                    'cured': 'mean',
                    'nominal_contract_value': ['mean', 'sum'],
                    'cumulative_amount_paid': ['mean', 'sum']
                })
                
                # Calculate frequency percentage
                cat_group[('client_id', 'percentage')] = cat_group[('client_id', 'count')] / len(df) * 100
                
                # Store for later export
                cat_stats[cat] = cat_group
        pbar.update(1)
        
        # Step 6: Multivariate analysis
        # Correlation matrix of key metrics
        target_vars = ['deposit_ratio', 'sept_23_repayment_rate', 'nov_23_repayment_rate', 
                       'repayment_improvement', 'sept_below_target', 'nov_below_target', 'cured']
        
        # Create correlation matrix focusing on target variables
        all_vars = numerical_vars + [v for v in target_vars if v not in numerical_vars]
        corr_matrix = df[all_vars].corr()
        
        # Create cross-tabulations
        cross_tabs = {}
        if 'Loan_Type' in df.columns and 'region' in df.columns:
            for metric in ['sept_23_repayment_rate', 'nov_23_repayment_rate', 'cured']:
                cross_tab = df.pivot_table(
                    values=metric,
                    index='Loan_Type',
                    columns='region',
                    aggfunc=['mean', 'count']
                )
                cross_tabs[metric] = cross_tab
        pbar.update(1)
        
        # Step 7: Time-based analysis
        # Temporal progression by cohort
        df['contract_month'] = df['contract_start_date'].dt.to_period('M')
        
        # Time-based cohort analysis
        time_cohort = df.groupby('contract_month').agg({
            'client_id': 'count',
            'sept_23_repayment_rate': ['mean', 'median'],
            'nov_23_repayment_rate': ['mean', 'median'],
            'sept_below_target': 'mean',
            'nov_below_target': 'mean',
            'cured': 'mean',
            'nominal_contract_value': 'sum',
            'cumulative_amount_paid': 'sum'
        })
        
        # Calculate improvement by cohort
        time_cohort[('improvement', 'mean')] = (
            time_cohort[('nov_23_repayment_rate', 'mean')] - 
            time_cohort[('sept_23_repayment_rate', 'mean')]
        )
        pbar.update(1)
        
        # Step 8: Generate profile report with ydata-profiling
        # For large samples, we might want to reduce the depth to speed up processing
        minimal_mode = sample_size is not None and sample_size >= 10000
        
        profile = ProfileReport(
            df, 
            title=f"Loan Repayment Analysis - Sept vs Nov 2023{' (Sample)' if sample_size else ''}",
            explorative=not minimal_mode,
            minimal=minimal_mode,
            correlations={
                "pearson": {"calculate": True},
                "spearman": {"calculate": True},
                "kendall": {"calculate": True},
                "phi_k": {"calculate": not minimal_mode},  # Computationally expensive
                "cramers": {"calculate": not minimal_mode}  # Computationally expensive
            },
            interactions={
                "continuous": not minimal_mode  # Disable for large samples to improve performance
            }
        )
        
        # Save as HTML (can be printed to PDF)
        profile.to_file(f"{output_dir}/repayment_profile_report.html")
        pbar.update(1)
        
        # Step 9: Custom visualizations and CSV exports
        # Prepare a multi-page PDF with key visualizations
        plt.figure(figsize=(12, 8))
        
        # 1. Distribution comparison of Sept vs Nov repayment rates
        plt.subplot(2, 2, 1)
        sns.histplot(df['sept_23_repayment_rate'], kde=True, color='blue', alpha=0.5, label='Sept')
        sns.histplot(df['nov_23_repayment_rate'], kde=True, color='red', alpha=0.5, label='Nov')
        plt.title('Distribution of Repayment Rates: Sept vs Nov')
        plt.legend()
        
        # 2. Cure rates by region
        plt.subplot(2, 2, 2)
        if 'region' in df.columns:
            cure_by_region = df.groupby('region')['cured'].mean().sort_values(ascending=False)
            sns.barplot(x=cure_by_region.index, y=cure_by_region.values)
            plt.xticks(rotation=45)
            plt.title('Cure Rate by Region')
        
        # 3. Repayment improvement vs initial repayment
        plt.subplot(2, 2, 3)
        sns.scatterplot(x='sept_23_repayment_rate', y='repayment_improvement', data=df)
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        plt.title('Improvement vs Initial Repayment Rate')
        
        # 4. Repayment rate progression over time
        plt.subplot(2, 2, 4)
        sns.lineplot(data=overall_progress, x='Time Point', y='Average Repayment Rate', marker='o')
        plt.axhline(y=0.98, color='r', linestyle='--', alpha=0.5, label='Target (98%)')
        plt.xticks(rotation=45)
        plt.title('Repayment Rate Progression')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/repayment_time_analysis.pdf")
        
        # Create a second visualization page focusing on categorical breakdowns
        if len(categorical_vars) > 0 and all(cat in df.columns for cat in categorical_vars[:2]):
            plt.figure(figsize=(12, 10))
            
            # 1. Loan type performance
            plt.subplot(2, 2, 1)
            loan_perf = df.groupby('Loan_Type').agg({
                'sept_23_repayment_rate': 'mean',
                'nov_23_repayment_rate': 'mean'
            })
            loan_perf.plot(kind='bar', ax=plt.gca())
            plt.title('Repayment by Loan Type')
            plt.xticks(rotation=45)
            
            # 2. Region performance
            plt.subplot(2, 2, 2)
            region_perf = df.groupby('region').agg({
                'sept_23_repayment_rate': 'mean',
                'nov_23_repayment_rate': 'mean'
            })
            region_perf.plot(kind='bar', ax=plt.gca())
            plt.title('Repayment by Region')
            plt.xticks(rotation=45)
            
            # 3. Deposit ratio vs final repayment
            plt.subplot(2, 2, 3)
            sns.scatterplot(x='deposit_ratio', y='nov_23_repayment_rate', data=df)
            plt.axhline(y=0.98, color='r', linestyle='--', alpha=0.5)
            plt.title('Final Repayment vs Deposit Ratio')
            
            # 4. Cohort analysis
            plt.subplot(2, 2, 4)
            cohort_data = time_cohort[[('sept_23_repayment_rate', 'mean'), 
                                      ('nov_23_repayment_rate', 'mean')]]
            cohort_data.columns = ['Sept', 'Nov']
            cohort_data.plot(kind='bar', ax=plt.gca())
            plt.title('Repayment by Contract Month Cohort')
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/repayment_categorical_analysis.pdf")
        
        # Save all analysis objects to CSV
        univariate_df.to_csv(f"{output_dir}/univariate_statistics.csv")
        pd.DataFrame(portfolio_metrics, index=[0]).to_csv(f"{output_dir}/portfolio_metrics.csv")
        overall_progress.to_csv(f"{output_dir}/repayment_progression.csv")
        corr_matrix.to_csv(f"{output_dir}/correlation_matrix.csv")
        time_cohort.to_csv(f"{output_dir}/cohort_analysis.csv")
        
        for cat, data in cat_stats.items():
            data.to_csv(f"{output_dir}/analysis_by_{cat}.csv")
            
        for metric, data in cross_tabs.items():
            data.to_csv(f"{output_dir}/crosstab_{metric}.csv")
        
        # Additional CSV for transition matrix - Sept to Nov repayment buckets
        transition = pd.crosstab(
            df['sept_repayment_bucket'], 
            df['nov_repayment_bucket'],
            normalize='index'
        )
        transition.to_csv(f"{output_dir}/repayment_bucket_transition.csv")
        
        # Export augmented dataset
        df.to_csv(f"{output_dir}/augmented_repayment_data.csv", index=False)
        pbar.update(1)
    
    print(f"EDA completed! Reports and data saved to {output_dir}/")
    
    return df