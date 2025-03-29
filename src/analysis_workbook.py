# ---
# title: OAF Loan Performance Analysis
# description: Interactive analysis of One Acre Fund loan data
# ---

import marimo as mo
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Add src directory to Python path for imports
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

# Import our analysis module
from src import analysis

# Set up markdown output
md = mo.md

# Show introduction
md("""
# One Acre Fund: Loan Performance Analysis
Interactive analysis notebook for exploring loan performance data.
""")

# ---- Data Loading and Preprocessing ----

@mo.Cell
def load_raw_data():
    """Load raw loan data"""
    df = analysis.load_data("data/raw/training_loan_processed.csv")
    return df

@mo.Cell
def process_data():
    """Preprocess loan data"""
    df = analysis.preprocess_data(load_raw_data())
    return df

# Show initial data summary
@mo.Cell
def show_data_summary():
    df = process_data()
    stats = analysis.get_summary_statistics(df)
    
    summary_md = f"""
    ## Data Overview
    
    **Portfolio Size**
    - Total Loans: {stats['loan_count']:,}
    - Average Loan Value: KES {stats['avg_loan_value']:,.0f}
    - Median Loan Value: KES {stats['median_loan_value']:,.0f}
    
    **Repayment Performance**
    - Average Rate: {stats['avg_repayment_rate']:.1%}
    - Target Achievement: {stats['target_achievement_rate']:.1%}
    - Median Deposit Ratio: {stats['median_deposit_ratio']:.1%}
    
    **Portfolio Composition**
    """
    
    for loan_type, count in stats['loan_type_counts'].items():
        summary_md += f"\n- {loan_type}: {count:,} loans"
    
    return md(summary_md)

# ---- Interactive Controls ----

# Region filter
region_selector = mo.ui.dropdown(
    options=process_data()["region"].unique().tolist(),
    value=[],
    label="Filter by Region",
    multiple=True
)

# Loan type filter
loan_type_selector = mo.ui.dropdown(
    options=process_data()["Loan_Type"].unique().tolist(),
    value=[],
    label="Filter by Loan Type",
    multiple=True
)

# Time period selector for temporal analysis
time_period_selector = mo.ui.slider(
    4, 12, value=6,
    label="Number of Time Periods"
)

@mo.Cell
def get_filtered_data():
    """Apply filters to the dataset"""
    df = process_data()
    
    if region_selector.value:
        df = analysis.filter_data(df, regions=region_selector.value)
    
    if loan_type_selector.value:
        df = analysis.filter_data(df, loan_types=loan_type_selector.value)
    
    return df

# ---- Portfolio Analysis ----

md("## Portfolio Composition")

@mo.Cell
def show_portfolio_composition():
    fig = analysis.plot_loan_portfolio_composition(get_filtered_data())
    return mo.output.mpl(fig)

md("## Repayment Performance")

@mo.Cell
def show_repayment_distribution():
    fig = analysis.plot_repayment_distribution(get_filtered_data())
    return mo.output.mpl(fig)

md("## Regional Analysis")

@mo.Cell
def show_regional_performance():
    fig = analysis.plot_regional_performance(get_filtered_data())
    return mo.output.mpl(fig)

md("## Loan Value Analysis")

@mo.Cell
def show_loan_value_analysis():
    fig = analysis.plot_loan_value_repayment(get_filtered_data())
    return mo.output.mpl(fig)

md("## Deposit Ratio Impact")

@mo.Cell
def show_deposit_analysis():
    fig, stats = analysis.analyze_deposit_ratio(get_filtered_data())
    return (
        mo.output.mpl(fig),
        mo.output.dataframe(stats.round(4))
    )

md("## Temporal Trends")

@mo.Cell
def show_temporal_analysis():
    fig, stats = analysis.analyze_temporal_trends(
        get_filtered_data(), 
        n_bins=time_period_selector.value
    )
    return (
        mo.output.mpl(fig),
        mo.output.dataframe(stats.round(4))
    )

md("## Client Segmentation")

@mo.Cell
def show_client_segments():
    fig, stats = analysis.segment_clients(get_filtered_data())
    return (
        mo.output.mpl(fig),
        mo.output.dataframe(stats.round(4))
    )

md("## Geographic Patterns")

@mo.Cell
def show_geographic_analysis():
    fig, stats = analysis.analyze_geographic_patterns(get_filtered_data())
    return (
        mo.output.mpl(fig),
        mo.output.dataframe(stats.round(4))
    )

# ---- Executive Summary ----

md("## Executive Dashboard")

@mo.Cell
def show_executive_dashboard():
    fig = analysis.create_executive_dashboard(get_filtered_data())
    return mo.output.mpl(fig)
