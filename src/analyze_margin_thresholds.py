#!/usr/bin/env python
"""
Analyze profitability across different gross margins and thresholds.

This script evaluates how different gross margins and prediction thresholds
affect profitability and business metrics.

Example usage:
    python src/analyze_margin_thresholds.py --predictions data/processed/holdout_evaluation/holdout_predictions.csv
                                           --data data/processed/holdout_all_features.csv
                                           --output data/processed/margin_analysis
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple

# Add the project root to the path to enable relative imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def calculate_profit_metrics(
    df: pd.DataFrame,
    threshold: float,
    gross_margin: float,
    predicted_col: str = 'predicted',
    actual_col: str = 'actual',
    loan_amount_col: str = 'nominal_contract_value'
) -> Dict[str, float]:
    """
    Calculate profit metrics for a given threshold and gross margin.
    
    Args:
        df: DataFrame with predicted and actual values
        threshold: Threshold for approval
        gross_margin: Gross margin percentage
        predicted_col: Column name for predictions
        actual_col: Column name for actual values
        loan_amount_col: Column name for loan amounts
        
    Returns:
        Dictionary with profit metrics
    """
    # Filter loans that would be approved based on threshold
    approved = df[df[predicted_col] >= threshold].copy()
    n_approved = len(approved)
    n_total = len(df)
    
    if n_approved == 0:
        return {
            'threshold': threshold,
            'gross_margin': gross_margin,
            'approval_rate': 0.0,
            'repayment_rate': 0.0,
            'default_rate': 0.0,
            'avg_loan_amount': 0.0,
            'total_loans': 0.0,
            'total_repaid': 0.0,
            'actual_profit': 0.0,
            'money_left_on_table': 0.0,
            'net_profit_margin': 0.0,
            'return_on_investment': 0.0
        }
    
    # Calculate approval rate
    approval_rate = n_approved / n_total
    
    # Calculate actual repayment rate of approved loans
    actual_repayment_rate = approved[actual_col].mean()
    
    # Calculate default rate (loans with repayment rate < 0.5)
    default_rate = (approved[actual_col] < 0.5).mean()
    
    # Calculate financial metrics
    total_loan_amount = approved[loan_amount_col].sum()
    avg_loan_amount = approved[loan_amount_col].mean()
    
    # Amount actually repaid
    total_repaid = approved[loan_amount_col].sum() * actual_repayment_rate
    
    # Actual profit (considering gross margin)
    actual_profit = total_repaid * gross_margin
    
    # Money left on table - the profit missed from rejected loans that would have repaid
    rejected = df[df[predicted_col] < threshold]
    would_have_repaid = rejected[loan_amount_col].sum() * rejected[actual_col].mean() * gross_margin
    money_left_on_table = would_have_repaid
    
    # Calculate net profit margin
    net_profit_margin = actual_profit / total_loan_amount if total_loan_amount > 0 else 0
    
    # Calculate return on investment
    return_on_investment = total_repaid / total_loan_amount - 1 if total_loan_amount > 0 else 0
    
    return {
        'threshold': threshold,
        'gross_margin': gross_margin,
        'approval_rate': float(approval_rate),
        'repayment_rate': float(actual_repayment_rate),
        'default_rate': float(default_rate),
        'avg_loan_amount': float(avg_loan_amount),
        'total_loans': float(total_loan_amount),
        'total_repaid': float(total_repaid),
        'actual_profit': float(actual_profit),
        'money_left_on_table': float(money_left_on_table),
        'net_profit_margin': float(net_profit_margin),
        'return_on_investment': float(actual_repayment_rate) - 1.0
    }

def analyze_margins_and_thresholds(
    predictions_df: pd.DataFrame,
    full_df: pd.DataFrame,
    thresholds: List[float],
    gross_margins: List[float],
    output_dir: str
) -> pd.DataFrame:
    """
    Analyze profitability across different margins and thresholds.
    
    Args:
        predictions_df: DataFrame with predictions
        full_df: Full dataset with loan amounts
        thresholds: List of thresholds to evaluate
        gross_margins: List of gross margins to evaluate
        output_dir: Directory to save results
        
    Returns:
        DataFrame with results for all combinations
    """
    # Merge predictions with full data to get loan amounts
    df = predictions_df.copy()
    
    # Check if we need to merge with full_df to get loan amounts
    if 'nominal_contract_value' not in df.columns:
        # Create temporary client_id for merging if not available
        if 'client_id' not in df.columns:
            print("WARNING: No client_id column for merging with full data.")
            print("Using row index as temporary client_id.")
            df['temp_id'] = df.index
            merge_key = 'temp_id'
            full_df['temp_id'] = full_df.index
        else:
            merge_key = 'client_id'
        
        # Merge to get loan amount
        df = pd.merge(
            df,
            full_df[['nominal_contract_value', merge_key]],
            on=merge_key,
            how='left'
        )
    
    # Create results for all combinations
    results = []
    
    for margin in gross_margins:
        for threshold in thresholds:
            metrics = calculate_profit_metrics(
                df,
                threshold=threshold,
                gross_margin=margin,
                predicted_col='predicted',
                actual_col='actual',
                loan_amount_col='nominal_contract_value'
            )
            results.append(metrics)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save to CSV
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "margin_threshold_analysis.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")
    
    return results_df

def create_heatmap(
    results_df: pd.DataFrame,
    metric: str,
    output_dir: str,
    title: str = None
) -> None:
    """
    Create a heatmap of a metric across margins and thresholds.
    
    Args:
        results_df: DataFrame with results
        metric: Metric to visualize
        output_dir: Directory to save plot
        title: Optional title for the plot
    """
    # Pivot the data for heatmap
    pivot_df = results_df.pivot(
        index='gross_margin',
        columns='threshold',
        values=metric
    )
    
    # Create plot
    plt.figure(figsize=(12, 8))
    heatmap = sns.heatmap(
        pivot_df,
        annot=True,
        fmt='.3f' if metric in ['repayment_rate', 'default_rate', 'approval_rate', 'net_profit_margin', 'return_on_investment'] else '.0f',
        cmap='viridis',
        linewidths=0.5
    )
    
    # Format
    if title is None:
        title = f"{metric.replace('_', ' ').title()} by Threshold and Margin"
    
    plt.title(title, fontsize=14)
    plt.xlabel('Predicted Repayment Rate Threshold', fontsize=12)
    plt.ylabel('Gross Margin', fontsize=12)
    
    # Add formatting to display as percentages if appropriate
    if metric in ['repayment_rate', 'default_rate', 'approval_rate', 'net_profit_margin', 'return_on_investment']:
        # Format annotation texts to show percentages
        for t in heatmap.texts:
            try:
                value = float(t.get_text())
                t.set_text(f"{value:.1%}")
            except ValueError:
                pass
    
    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, f"{metric}_heatmap.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Heatmap for {metric} saved to {plot_path}")

def generate_business_recommendation(results_df: pd.DataFrame, output_dir: str) -> Dict:
    """
    Generate business recommendations based on the analysis.
    
    Args:
        results_df: DataFrame with results
        output_dir: Directory to save recommendations
        
    Returns:
        Dictionary with recommendations
    """
    # Filter for high repayment rate (close to 98%)
    high_repayment = results_df[results_df['repayment_rate'] >= 0.90].copy()
    
    # If no thresholds meet this criteria, find the highest
    if len(high_repayment) == 0:
        high_repayment = results_df.sort_values('repayment_rate', ascending=False).head(5)
    
    # Find the combination with the best profit among high repayment options
    best_profit_row = high_repayment.sort_values('actual_profit', ascending=False).iloc[0]
    
    # Find the combination with the lowest default rate
    low_default_row = results_df.sort_values('default_rate').iloc[0]
    
    # Find the combination with optimal balance (weighted score)
    results_df['balance_score'] = (
        results_df['repayment_rate'] * 0.4 +
        (1 - results_df['default_rate']) * 0.3 +
        results_df['approval_rate'] * 0.2 +
        results_df['net_profit_margin'] * 0.1
    )
    balanced_row = results_df.sort_values('balance_score', ascending=False).iloc[0]
    
    recommendations = {
        "best_profit": {
            "threshold": float(best_profit_row['threshold']),
            "gross_margin": float(best_profit_row['gross_margin']),
            "repayment_rate": float(best_profit_row['repayment_rate']),
            "default_rate": float(best_profit_row['default_rate']),
            "approval_rate": float(best_profit_row['approval_rate']),
            "profit": float(best_profit_row['actual_profit']),
            "roi": float(best_profit_row['return_on_investment'])
        },
        "lowest_default": {
            "threshold": float(low_default_row['threshold']),
            "gross_margin": float(low_default_row['gross_margin']),
            "repayment_rate": float(low_default_row['repayment_rate']),
            "default_rate": float(low_default_row['default_rate']),
            "approval_rate": float(low_default_row['approval_rate']),
            "profit": float(low_default_row['actual_profit']),
            "roi": float(low_default_row['return_on_investment'])
        },
        "balanced": {
            "threshold": float(balanced_row['threshold']),
            "gross_margin": float(balanced_row['gross_margin']),
            "repayment_rate": float(balanced_row['repayment_rate']),
            "default_rate": float(balanced_row['default_rate']),
            "approval_rate": float(balanced_row['approval_rate']),
            "profit": float(balanced_row['actual_profit']),
            "roi": float(balanced_row['return_on_investment'])
        }
    }
    
    # Save to JSON
    os.makedirs(output_dir, exist_ok=True)
    rec_path = os.path.join(output_dir, "business_recommendations.json")
    with open(rec_path, 'w') as f:
        json.dump(recommendations, f, indent=2)
    
    print(f"Business recommendations saved to {rec_path}")
    return recommendations

def main(predictions_path: str, data_path: str, output_dir: str):
    """Main function to run the analysis."""
    # Load predictions
    predictions_df = pd.read_csv(predictions_path)
    
    # Load full data (for loan amounts)
    full_df = pd.read_csv(data_path)
    
    # Set thresholds and margins to analyze
    thresholds = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    gross_margins = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    
    # Analyze all combinations
    results_df = analyze_margins_and_thresholds(
        predictions_df,
        full_df,
        thresholds,
        gross_margins,
        output_dir
    )
    
    # Create heatmaps for key metrics
    for metric in ['repayment_rate', 'default_rate', 'approval_rate', 
                  'actual_profit', 'net_profit_margin', 'return_on_investment']:
        create_heatmap(results_df, metric, output_dir)
    
    # Generate business recommendations
    recommendations = generate_business_recommendation(results_df, output_dir)
    
    # Print summary
    print("\n=== Analysis Summary ===")
    print(f"Analyzed {len(thresholds)} thresholds × {len(gross_margins)} margins = {len(thresholds) * len(gross_margins)} combinations")
    print("\nBest for High Repayment Rate & Profit:")
    print(f"Threshold: {recommendations['best_profit']['threshold']:.2f}, Margin: {recommendations['best_profit']['gross_margin']:.2f}")
    print(f"Repayment Rate: {recommendations['best_profit']['repayment_rate']:.1%}")
    print(f"Default Rate: {recommendations['best_profit']['default_rate']:.1%}")
    print(f"Approval Rate: {recommendations['best_profit']['approval_rate']:.1%}")
    
    print("\nBest for Lowest Default Rate:")
    print(f"Threshold: {recommendations['lowest_default']['threshold']:.2f}, Margin: {recommendations['lowest_default']['gross_margin']:.2f}")
    print(f"Default Rate: {recommendations['lowest_default']['default_rate']:.1%}")
    print(f"Repayment Rate: {recommendations['lowest_default']['repayment_rate']:.1%}")
    print(f"Approval Rate: {recommendations['lowest_default']['approval_rate']:.1%}")
    
    print("\nBest Balanced Approach:")
    print(f"Threshold: {recommendations['balanced']['threshold']:.2f}, Margin: {recommendations['balanced']['gross_margin']:.2f}")
    print(f"Repayment Rate: {recommendations['balanced']['repayment_rate']:.1%}")
    print(f"Default Rate: {recommendations['balanced']['default_rate']:.1%}")
    print(f"Approval Rate: {recommendations['balanced']['approval_rate']:.1%}")
    
    return results_df, recommendations

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Analyze margins and thresholds')
    parser.add_argument('--predictions', type=str, required=True,
                       help='Path to predictions CSV file')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to full dataset CSV file')
    parser.add_argument('--output', type=str, default="data/processed/margin_analysis",
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Run the analysis
    main(args.predictions, args.data, args.output)
