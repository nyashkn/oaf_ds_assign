import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# OAF brand colors
OAF_GREEN = '#1B5E20'  # Primary green
OAF_BROWN = '#8B4513'  # Accent color
OAF_GRAY = '#58595B'   # Dark gray
OAF_LIGHT_GREEN = '#81C784'  # Light green
OAF_BLUE = '#1976D2'   # Blue for accent

# Set style for static plots
plt.style.use('seaborn-v0_8')
sns.set_palette([OAF_GREEN, OAF_BLUE, OAF_BROWN, OAF_LIGHT_GREEN])

def load_and_preprocess_data():
    """Load and perform initial data preprocessing"""
    # Create directories
    for dir_path in ['data/processed/executive_summary', 'data/processed/detailed_analysis']:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    # Load training dataset
    df = pd.read_csv("data/raw/training_loan_processed.csv")
    
    # Convert date columns
    df['contract_start_date'] = pd.to_datetime(df['contract_start_date'])
    
    # Calculate derived metrics
    df['repayment_rate'] = df['cumulative_amount_paid_start'] / df['nominal_contract_value']
    df['deposit_ratio'] = df['deposit_amount'] / df['nominal_contract_value']
    
    return df

def create_executive_dashboard(df):
    """Create executive summary dashboard"""
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle('One Acre Fund: Loan Performance Analysis\nExecutive Summary', 
                fontsize=16, fontweight='bold', y=0.95)
    
    # 1. Key Metrics Overview (Top Left)
    ax1 = plt.subplot(2, 2, 1)
    metrics = {
        'Average Loan Value': f"KES {df['nominal_contract_value'].mean():,.0f}",
        'Median Deposit Ratio': f"{df['deposit_ratio'].median():.1%}",
        'Average Repayment Rate': f"{df['repayment_rate'].mean():.1%}",
        'Total Active Loans': f"{len(df):,}"
    }
    
    # Create text box with metrics
    metrics_text = '\n'.join([f"{k}: {v}" for k, v in metrics.items()])
    ax1.text(0.5, 0.5, metrics_text, 
            ha='center', va='center',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor=OAF_GREEN),
            fontsize=12)
    ax1.axis('off')
    ax1.set_title('Key Performance Indicators', fontsize=12, fontweight='bold')
    
    # 2. Loan Type Distribution (Top Right)
    ax2 = plt.subplot(2, 2, 2)
    loan_type_counts = df['Loan_Type'].value_counts()
    ax2.pie(loan_type_counts.values, labels=loan_type_counts.index,
            autopct='%1.1f%%', colors=sns.color_palette([OAF_GREEN, OAF_BLUE, OAF_BROWN]))
    ax2.set_title('Loan Portfolio Composition', fontsize=12, fontweight='bold')
    
    # 3. Repayment Rate Distribution (Bottom Left)
    ax3 = plt.subplot(2, 2, 3)
    sns.histplot(data=df, x='repayment_rate', bins=30, ax=ax3)
    ax3.axvline(x=0.98, color='red', linestyle='--', label='98% Target')
    ax3.set_title('Repayment Rate Distribution', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Repayment Rate')
    ax3.set_ylabel('Count')
    ax3.legend()
    
    # 4. Regional Performance (Bottom Right)
    ax4 = plt.subplot(2, 2, 4)
    regional_performance = df.groupby('region')['repayment_rate'].agg(['mean', 'count']).reset_index()
    regional_performance = regional_performance.sort_values('mean', ascending=True)
    
    bars = ax4.barh(regional_performance['region'], 
                    regional_performance['mean'],
                    color=OAF_GREEN)
    
    # Add percentage labels
    for bar in bars:
        width = bar.get_width()
        ax4.text(width, bar.get_y() + bar.get_height()/2,
                f'{width:.1%}', ha='left', va='center')
    
    ax4.set_title('Regional Repayment Performance', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Average Repayment Rate')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('data/processed/executive_summary/executive_dashboard.png', 
                dpi=300, bbox_inches='tight')
    plt.close()

def create_detailed_analysis(df):
    """Create detailed analysis visualizations"""
    
    # 1. Loan Value vs Repayment Rate (Hexbin plot for better density visualization)
    plt.figure(figsize=(12, 8))
    plt.hexbin(df['nominal_contract_value'], df['repayment_rate'],
               gridsize=30, cmap='YlOrRd')
    plt.colorbar(label='Count')
    plt.axhline(y=0.98, color='red', linestyle='--', label='98% Target')
    
    # Add summary statistics as text
    stats_text = (
        f"Mean Loan: {df['nominal_contract_value'].mean():,.0f} KES\n"
        f"Median Loan: {df['nominal_contract_value'].median():,.0f} KES\n"
        f"Mean Repayment: {df['repayment_rate'].mean():.1%}\n"
        f"Loans Meeting Target: {(df['repayment_rate'] >= 0.98).mean():.1%}"
    )
    plt.text(0.02, 0.98, stats_text,
             transform=plt.gca().transAxes,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.xlabel('Loan Value (KES)')
    plt.ylabel('Repayment Rate')
    plt.title('Loan Value vs Repayment Rate Distribution')
    plt.legend()
    plt.tight_layout()
    plt.savefig('data/processed/detailed_analysis/loan_value_vs_repayment.png')
    plt.close()
    
    # 2. Deposit Ratio Analysis with Error Bars
    plt.figure(figsize=(12, 6))
    deposit_bins = pd.qcut(df['deposit_ratio'], q=5)
    deposit_stats = df.groupby(deposit_bins, observed=True)['repayment_rate'].agg(['mean', 'std', 'count'])
    deposit_stats['ci'] = 1.96 * deposit_stats['std'] / np.sqrt(deposit_stats['count'])
    
    plt.bar(range(len(deposit_stats)), deposit_stats['mean'],
            yerr=deposit_stats['ci'],
            color=OAF_GREEN,
            capsize=5)
    plt.xticks(range(len(deposit_stats)),
               [f'{b.left:.0%}-{b.right:.0%}' for b in deposit_stats.index],
               rotation=45)
    
    # Add count labels
    for i, (_, row) in enumerate(deposit_stats.iterrows()):
        plt.text(i, row['mean'],
                f'n={int(row["count"]):,}',
                ha='center', va='bottom')
    
    plt.xlabel('Deposit Ratio Range')
    plt.ylabel('Average Repayment Rate')
    plt.title('Repayment Performance by Deposit Ratio Quintiles\nwith 95% Confidence Intervals')
    plt.tight_layout()
    plt.savefig('data/processed/detailed_analysis/deposit_ratio_analysis.png')
    plt.close()
    
    # 3. Time Series Analysis with Loan Counts
    plt.figure(figsize=(12, 8))
    
    # Calculate months since start for each loan
    df['months_since_start'] = (df['contract_start_date'] - df['contract_start_date'].min()).dt.days / 30
    
    # Create monthly bins
    df['month_bin'] = pd.qcut(df['months_since_start'], q=12, labels=[f'Month {i+1}' for i in range(12)])
    
    # Calculate statistics by month bin
    monthly_stats = df.groupby('month_bin', observed=True).agg({
        'repayment_rate': ['mean', 'std', 'count'],
        'nominal_contract_value': 'sum'
    }).round(4)
    
    # Create subplot for dual axis
    fig, ax1 = plt.subplots(figsize=(12, 8))
    ax2 = ax1.twinx()
    
    # Plot average repayment rate with confidence intervals
    x = range(len(monthly_stats))
    means = monthly_stats[('repayment_rate', 'mean')]
    stds = monthly_stats[('repayment_rate', 'std')]
    counts = monthly_stats[('repayment_rate', 'count')]
    ci = 1.96 * stds / np.sqrt(counts)
    
    ax1.plot(x, means, 'o-', color=OAF_GREEN, label='Repayment Rate')
    ax1.fill_between(x, means - ci, means + ci, color=OAF_GREEN, alpha=0.2)
    
    # Plot loan counts on secondary axis
    ax2.bar(x, counts, alpha=0.2, color=OAF_BLUE, label='Number of Loans')
    
    # Customize axes
    ax1.set_xlabel('Time Period')
    ax1.set_ylabel('Average Repayment Rate')
    ax2.set_ylabel('Number of Loans')
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(monthly_stats.index, rotation=45)
    
    # Add legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.title('Repayment Rate Trends Over Time\nwith 95% Confidence Intervals')
    plt.tight_layout()
    plt.savefig('data/processed/detailed_analysis/monthly_trends.png')
    plt.close()

def create_interactive_dashboard(df):
    """Create interactive Plotly dashboard"""
    
    # Create subplot figure
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Loan Value Distribution',
                       'Repayment Rate by Region',
                       'Monthly Trends',
                       'Deposit vs Repayment Analysis'),
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    # 1. Loan Value Distribution
    fig.add_trace(
        go.Histogram(x=df['nominal_contract_value'],
                     name='Loan Value',
                     nbinsx=30),
        row=1, col=1
    )
    
    # 2. Regional Performance
    regional_perf = df.groupby('region')['repayment_rate'].mean().sort_values()
    fig.add_trace(
        go.Bar(x=regional_perf.values,
               y=regional_perf.index,
               orientation='h',
               name='Regional Performance'),
        row=1, col=2
    )
    
    # 3. Monthly Trends
    monthly_perf = df.groupby(df['contract_start_date'].dt.to_period('M'))['repayment_rate'].mean()
    fig.add_trace(
        go.Scatter(x=[str(x) for x in monthly_perf.index],
                   y=monthly_perf.values,
                   mode='lines+markers',
                   name='Monthly Trend'),
        row=2, col=1
    )
    
    # 4. Deposit vs Repayment
    fig.add_trace(
        go.Scatter(x=df['deposit_ratio'],
                   y=df['repayment_rate'],
                   mode='markers',
                   marker=dict(color=OAF_GREEN),
                   name='Deposit vs Repayment'),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        title_text='One Acre Fund: Interactive Loan Analysis Dashboard',
        showlegend=False,
        height=800,
        width=1200
    )
    
    # Save as HTML
    fig.write_html('data/processed/executive_summary/interactive_dashboard.html')

def main():
    """Main execution function"""
    print("Loading and preprocessing data...")
    df = load_and_preprocess_data()
    
    print("\nCreating executive dashboard...")
    create_executive_dashboard(df)
    
    print("\nGenerating detailed analysis...")
    create_detailed_analysis(df)
    
    print("\nCreating interactive dashboard...")
    create_interactive_dashboard(df)
    
    print("\nAnalysis complete. Results saved in data/processed/")

if __name__ == "__main__":
    main()
