"""
Threshold and Profitability Metrics Explainer

This module provides a clear explanation of how thresholds and profitability metrics
work in the loan repayment prediction models, using real examples and visualizations.
It explains the key concepts of:

1. Thresholds and their role in loan approvals
2. ROI and profit calculations
3. The trade-offs between different threshold settings
4. How to interpret the metrics in business contexts

Use this module for educational purposes or to better understand model outputs.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional, Union

# Import the metrics functions to demonstrate
from .profitability.metrics import (
    calculate_loan_metrics,
    calculate_profit_metrics,
    calculate_business_metrics
)

# Set default plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("colorblind")


def explain_threshold_concept(output_dir: Optional[str] = None) -> None:
    """
    Explain the concept of thresholds in loan repayment prediction.
    
    Args:
        output_dir: Directory to save explanatory visualizations
    """
    print("\n=== Threshold Concept Explanation ===")
    print(
        "In loan repayment prediction, a threshold is a cutoff value for the predicted\n"
        "repayment rate that determines which loans to approve or reject:\n"
        "- If predicted repayment rate ≥ threshold → Loan is approved\n"
        "- If predicted repayment rate < threshold → Loan is rejected\n\n"
        "For example, with a threshold of 0.7 (70%):\n"
        "- A loan predicted to repay 75% would be approved\n"
        "- A loan predicted to repay 65% would be rejected"
    )
    
    # Create example data for visualization
    np.random.seed(42)
    n_loans = 100
    
    # Generate sample predicted repayment rates
    predictions = np.clip(np.random.normal(0.7, 0.15, n_loans), 0, 1)
    
    # Loan amounts between 5,000 and 20,000
    loan_amounts = np.random.uniform(5000, 20000, n_loans)
    
    # Create a DataFrame for easy plotting
    df = pd.DataFrame({
        'predicted_repayment_rate': predictions,
        'loan_amount': loan_amounts
    })
    
    # Sort by predicted rate for clearer visualization
    df = df.sort_values('predicted_repayment_rate')
    
    # Add loan indices for plotting
    df['loan_index'] = range(len(df))
    
    # Visualize the threshold concept
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Plot 1: Threshold separating approved and rejected loans
        plt.figure(figsize=(12, 6))
        
        # Example threshold at 0.7
        threshold = 0.7
        
        # Color loans by approval status
        colors = ['#ff7f0e' if rate < threshold else '#1f77b4' for rate in df['predicted_repayment_rate']]
        
        plt.scatter(df['loan_index'], df['predicted_repayment_rate'], c=colors, alpha=0.7)
        plt.axhline(y=threshold, color='red', linestyle='--', label=f'Threshold = {threshold}')
        
        # Label the areas
        plt.text(len(df)*0.5, threshold+0.05, 'APPROVED LOANS', ha='center', fontsize=12, color='#1f77b4')
        plt.text(len(df)*0.5, threshold-0.05, 'REJECTED LOANS', ha='center', fontsize=12, color='#ff7f0e')
        
        plt.xlabel('Loan Index (sorted by predicted repayment rate)')
        plt.ylabel('Predicted Repayment Rate')
        plt.title('Threshold Concept: Loan Approval Decision')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        
        plt.savefig(os.path.join(output_dir, "threshold_concept.png"), dpi=300)
        plt.close()
        
        # Plot 2: How different thresholds affect approval rates
        plt.figure(figsize=(12, 6))
        
        thresholds = np.linspace(0.5, 0.9, 9)
        approval_rates = []
        
        for t in thresholds:
            approval_rate = np.mean(df['predicted_repayment_rate'] >= t)
            approval_rates.append(approval_rate)
        
        plt.plot(thresholds, approval_rates, 'o-', linewidth=2)
        
        # Add annotations
        for i, t in enumerate(thresholds):
            plt.annotate(f"{approval_rates[i]:.1%}", 
                        (t, approval_rates[i]), 
                        xytext=(0, 10),
                        textcoords='offset points',
                        ha='center')
        
        plt.xlabel('Threshold')
        plt.ylabel('Approval Rate')
        plt.title('How Threshold Affects Approval Rate')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        plt.savefig(os.path.join(output_dir, "threshold_approval_rates.png"), dpi=300)
        plt.close()


def explain_profit_calculation(output_dir: Optional[str] = None) -> None:
    """
    Explain how profit is calculated in the loan repayment prediction model.
    
    Args:
        output_dir: Directory to save explanatory visualizations
    """
    print("\n=== Profit Calculation Explanation ===")
    print(
        "Profit calculation in loan repayment prediction considers:\n"
        "1. Approved loan values: The total amount of money loaned out\n"
        "2. Actual repayment: How much of the loaned money is repaid\n"
        "3. Margin: The profit made on repaid amounts (typically 16%)\n"
        "4. Default loss: The loss on defaulted amounts (typically 100%)\n\n"
        "For each approved loan:\n"
        "- Profit = repaid amount × margin\n"
        "- Loss = defaulted amount × default loss rate\n"
        "- Net = profit - loss\n\n"
        "The business aims to maximize total net profit across all loans."
    )
    
    # Create example for clearer explanation
    # Simple example with one loan
    loan_amount = 10000
    repayment_rate = 0.8  # 80% repayment
    margin = 0.16  # 16% profit margin
    default_loss_rate = 1.0  # 100% loss on defaulted amount
    
    # Calculate profit components
    repaid_amount = loan_amount * repayment_rate
    defaulted_amount = loan_amount * (1 - repayment_rate)
    
    profit = repaid_amount * margin
    loss = defaulted_amount * default_loss_rate
    net = profit - loss
    
    print("\nExample for a $10,000 loan with 80% repayment rate:")
    print(f"  Repaid amount: ${repaid_amount:.2f}")
    print(f"  Defaulted amount: ${defaulted_amount:.2f}")
    print(f"  Profit (16% of repaid): ${profit:.2f}")
    print(f"  Loss (100% of defaulted): ${loss:.2f}")
    print(f"  Net profit: ${net:.2f}")
    print(f"  ROI: {net/loan_amount:.1%}")
    
    # Create profit waterfall chart
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Plot: Profit waterfall chart
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Define the steps in the waterfall
        labels = ['Loan Amount', 'Repaid Amount', 'Profit (16%)', 'Defaulted Amount', 'Loss (100%)', 'Net Profit']
        values = [loan_amount, repaid_amount, profit, defaulted_amount, -loss, net]
        
        # Create colors for the bars
        colors = ['#1f77b4', '#1f77b4', '#2ca02c', '#ff7f0e', '#d62728', '#8c564b']
        
        # Create the waterfall chart
        bottom = 0
        for i, (label, value, color) in enumerate(zip(labels, values, colors)):
            if i == 0:  # First bar (Loan Amount)
                ax.bar(label, value, bottom=0, color=color)
                bottom = value
            elif i == len(labels) - 1:  # Last bar (Net Profit)
                ax.bar(label, value, bottom=0, color=color)
            else:
                # For intermediate bars
                if value >= 0:
                    ax.bar(label, value, bottom=0, color=color)
                else:
                    ax.bar(label, value, bottom=0, color=color)
        
        # Add value labels
        for i, (label, value) in enumerate(zip(labels, values)):
            if i == 0:  # Loan Amount
                ax.text(i, value/2, f"${value:.0f}", ha='center', va='center')
            elif i == 1:  # Repaid Amount
                ax.text(i, value/2, f"${value:.0f}\n({repayment_rate:.0%} of loan)", ha='center', va='center')
            elif i == 2:  # Profit
                ax.text(i, value/2, f"${value:.0f}\n({margin:.0%} of repaid)", ha='center', va='center')
            elif i == 3:  # Defaulted Amount
                ax.text(i, value/2, f"${value:.0f}\n({1-repayment_rate:.0%} of loan)", ha='center', va='center')
            elif i == 4:  # Loss
                ax.text(i, value/2, f"${value:.0f}\n({default_loss_rate:.0%} of defaulted)", ha='center', va='center', color='white')
            else:  # Net Profit
                ax.text(i, value/2, f"${value:.0f}\n(ROI: {net/loan_amount:.1%})", ha='center', va='center')
        
        # Plot formatting
        ax.set_title('Profit Calculation for a $10,000 Loan with 80% Repayment')
        ax.set_ylabel('Amount ($)')
        ax.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        plt.savefig(os.path.join(output_dir, "profit_calculation.png"), dpi=300)
        plt.close()


def explain_threshold_profitability_tradeoff(output_dir: Optional[str] = None) -> None:
    """
    Explain the tradeoff between threshold, approval rate, profit, and ROI.
    
    Args:
        output_dir: Directory to save explanatory visualizations
    """
    print("\n=== Threshold-Profitability Tradeoff Explanation ===")
    print(
        "There is a fundamental tradeoff in threshold selection:\n"
        "- Lower thresholds: More loans approved → Higher total profit but lower ROI\n"
        "- Higher thresholds: Fewer loans approved → Lower total profit but higher ROI\n\n"
        "This tradeoff occurs because:\n"
        "1. Lower thresholds include more marginal loans that increase absolute profit\n"
        "   but may decrease the profit per dollar invested (ROI)\n"
        "2. Higher thresholds only approve the safest loans, which have better ROI\n"
        "   but reduce the total loan volume and absolute profit\n\n"
        "The optimal threshold depends on business priorities (profit vs. ROI)"
    )
    
    # Create example data for visualization
    np.random.seed(42)
    n_loans = 200
    
    # Generate sample predicted repayment rates
    pred_rates = np.clip(np.random.normal(0.7, 0.15, n_loans), 0, 1)
    
    # Actual rates correlate with predictions but have some noise
    noise = np.random.normal(0, 0.1, n_loans)
    true_rates = np.clip(pred_rates + noise, 0, 1)
    
    # Loan amounts between 5,000 and 20,000
    loan_amounts = np.random.uniform(5000, 20000, n_loans)
    
    # Business parameters
    margin = 0.16
    default_loss_rate = 1.0
    
    # Calculate metrics for different thresholds
    thresholds = np.linspace(0.5, 0.9, 9)
    results = []
    
    for threshold in thresholds:
        metrics = calculate_business_metrics(
            true_rates, pred_rates, loan_amounts, 
            threshold=threshold,
            margin=margin,
            default_loss_rate=default_loss_rate
        )
        
        # Extract key metrics
        approval_rate = metrics['loan_metrics']['n_loans']['approval_rate']
        profit = metrics['profit_metrics']['actual_profit']
        roi = metrics['profit_metrics']['roi']
        
        results.append({
            'threshold': threshold,
            'approval_rate': approval_rate,
            'profit': profit,
            'roi': roi
        })
    
    # Convert to DataFrame for plotting
    results_df = pd.DataFrame(results)
    
    # Print the results table
    print("\nMetrics at Different Thresholds:")
    
    for _, row in results_df.iterrows():
        print(f"  Threshold = {row['threshold']:.2f}: "
              f"Approval Rate = {row['approval_rate']:.1%}, "
              f"Profit = ${row['profit']:.2f}, "
              f"ROI = {row['roi']:.2%}")
    
    # Create visualizations
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Plot 1: Approval Rate vs Threshold
        plt.figure(figsize=(12, 6))
        plt.plot(results_df['threshold'], results_df['approval_rate'], 'o-', linewidth=2)
        
        # Add annotations
        for _, row in results_df.iterrows():
            plt.annotate(f"{row['approval_rate']:.1%}", 
                        (row['threshold'], row['approval_rate']), 
                        xytext=(0, 10),
                        textcoords='offset points',
                        ha='center')
        
        plt.xlabel('Threshold')
        plt.ylabel('Approval Rate')
        plt.title('How Threshold Affects Approval Rate')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        plt.savefig(os.path.join(output_dir, "threshold_vs_approval.png"), dpi=300)
        plt.close()
        
        # Plot 2: Profit vs Threshold
        plt.figure(figsize=(12, 6))
        plt.plot(results_df['threshold'], results_df['profit'], 'o-', linewidth=2, color='#2ca02c')
        
        # Add annotations
        for _, row in results_df.iterrows():
            plt.annotate(f"${row['profit']:.0f}", 
                        (row['threshold'], row['profit']), 
                        xytext=(0, 10),
                        textcoords='offset points',
                        ha='center')
        
        plt.xlabel('Threshold')
        plt.ylabel('Profit ($)')
        plt.title('How Threshold Affects Profit')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        plt.savefig(os.path.join(output_dir, "threshold_vs_profit.png"), dpi=300)
        plt.close()
        
        # Plot 3: ROI vs Threshold
        plt.figure(figsize=(12, 6))
        plt.plot(results_df['threshold'], results_df['roi'], 'o-', linewidth=2, color='#ff7f0e')
        
        # Add annotations
        for _, row in results_df.iterrows():
            plt.annotate(f"{row['roi']:.1%}", 
                        (row['threshold'], row['roi']), 
                        xytext=(0, 10),
                        textcoords='offset points',
                        ha='center')
        
        plt.xlabel('Threshold')
        plt.ylabel('ROI')
        plt.title('How Threshold Affects ROI')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        plt.savefig(os.path.join(output_dir, "threshold_vs_roi.png"), dpi=300)
        plt.close()
        
        # Plot 4: Profit vs ROI with Threshold values
        plt.figure(figsize=(12, 6))
        plt.scatter(results_df['roi'], results_df['profit'], s=100)
        
        # Add threshold annotations
        for _, row in results_df.iterrows():
            plt.annotate(f"{row['threshold']:.2f}", 
                        (row['roi'], row['profit']), 
                        xytext=(10, 0),
                        textcoords='offset points')
        
        plt.xlabel('ROI')
        plt.ylabel('Profit ($)')
        plt.title('Profit-ROI Tradeoff at Different Thresholds')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        plt.savefig(os.path.join(output_dir, "profit_roi_tradeoff.png"), dpi=300)
        plt.close()


def explain_real_world_example(threshold_data: Dict[str, Any], output_dir: Optional[str] = None) -> None:
    """
    Explain the profit and ROI calculations using real-world examples from the data.
    
    Args:
        threshold_data: Dictionary of metrics at different thresholds
        output_dir: Directory to save visualizations
    """
    print("\n=== Real World Example Explanation ===")
    print(
        "Let's analyze some real model results to understand how thresholds\n"
        "affect business metrics in practice:\n"
    )
    
    # Format the real-world data for printing
    print("Metrics from Real Model Data:")
    
    metrics_list = threshold_data.get('metrics_by_threshold', [])
    for metrics in metrics_list[:5]:  # Show first 5 thresholds
        threshold = metrics.get('threshold', 0)
        approval_rate = metrics.get('approval_rate', 0)
        loan_value = metrics.get('approved_loan_value', 0)
        profit = metrics.get('actual_profit', 0)
        loss = metrics.get('actual_loss', 0)
        roi = metrics.get('roi', 0)
        
        print(f"  Threshold = {threshold:.2f}:")
        print(f"    Approval Rate: {approval_rate:.2%}")
        print(f"    Approved Loan Value: ${loan_value:.2f}")
        print(f"    Actual Profit: ${profit:.2f}")
        print(f"    Actual Loss: ${loss:.2f}")
        print(f"    ROI: {roi:.2%}")
        print("")
    
    # Plot visualizations
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert to DataFrame for easier plotting
        metrics_df = pd.DataFrame(metrics_list)
        
        # Plot 1: Profit vs Threshold from real data
        plt.figure(figsize=(12, 6))
        plt.plot(metrics_df['threshold'], metrics_df['actual_profit'], 'o-', linewidth=2, color='#2ca02c')
        
        plt.xlabel('Threshold')
        plt.ylabel('Profit ($)')
        plt.title('Real Model Data: Profit vs Threshold')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        plt.savefig(os.path.join(output_dir, "real_threshold_vs_profit.png"), dpi=300)
        plt.close()
        
        # Plot 2: ROI vs Threshold from real data
        plt.figure(figsize=(12, 6))
        plt.plot(metrics_df['threshold'], metrics_df['roi'], 'o-', linewidth=2, color='#ff7f0e')
        
        plt.xlabel('Threshold')
        plt.ylabel('ROI')
        plt.title('Real Model Data: ROI vs Threshold')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        plt.savefig(os.path.join(output_dir, "real_threshold_vs_roi.png"), dpi=300)
        plt.close()
        
        # Plot 3: Approval Rate vs Threshold from real data
        plt.figure(figsize=(12, 6))
        plt.plot(metrics_df['threshold'], metrics_df['approval_rate'], 'o-', linewidth=2, color='#1f77b4')
        
        plt.xlabel('Threshold')
        plt.ylabel('Approval Rate')
        plt.title('Real Model Data: Approval Rate vs Threshold')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        plt.savefig(os.path.join(output_dir, "real_threshold_vs_approval.png"), dpi=300)
        plt.close()
        
        # Plot 4: Combined metrics from real data
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # Plot profit on primary y-axis
        ax1.plot(metrics_df['threshold'], metrics_df['actual_profit'], 'o-', color='#2ca02c', label='Profit ($)')
        ax1.set_xlabel('Threshold')
        ax1.set_ylabel('Profit ($)', color='#2ca02c')
        ax1.tick_params(axis='y', labelcolor='#2ca02c')
        
        # Create a secondary y-axis for ROI
        ax2 = ax1.twinx()
        ax2.plot(metrics_df['threshold'], metrics_df['roi'], 's-', color='#ff7f0e', label='ROI')
        ax2.set_ylabel('ROI', color='#ff7f0e')
        ax2.tick_params(axis='y', labelcolor='#ff7f0e')
        
        # Add a third y-axis for approval rate
        ax3 = ax1.twinx()
        ax3.spines['right'].set_position(('outward', 60))
        ax3.plot(metrics_df['threshold'], metrics_df['approval_rate'], '^-', color='#1f77b4', label='Approval Rate')
        ax3.set_ylabel('Approval Rate', color='#1f77b4')
        ax3.tick_params(axis='y', labelcolor='#1f77b4')
        
        # Create a combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        lines3, labels3 = ax3.get_legend_handles_labels()
        lines = lines1 + lines2 + lines3
        labels = labels1 + labels2 + labels3
        ax1.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3)
        
        plt.title('Real Model Data: Combined Metrics vs Threshold')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        plt.savefig(os.path.join(output_dir, "real_combined_metrics.png"), dpi=300)
        plt.close()
        
        # Plot 5: Profit-ROI tradeoff from real data
        plt.figure(figsize=(12, 6))
        plt.scatter(metrics_df['roi'], metrics_df['actual_profit'], s=100)
        
        # Add threshold annotations
        for _, row in metrics_df.iterrows():
            if row['threshold'] <= 0.7:  # Limit annotations to make the plot cleaner
                plt.annotate(f"{row['threshold']:.2f}", 
                            (row['roi'], row['actual_profit']), 
                            xytext=(10, 0),
                            textcoords='offset points')
        
        plt.xlabel('ROI')
        plt.ylabel('Profit ($)')
        plt.title('Real Model Data: Profit-ROI Tradeoff')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        plt.savefig(os.path.join(output_dir, "real_profit_roi_tradeoff.png"), dpi=300)
        plt.close()


def explain_optimal_threshold_selection(threshold_data: Dict[str, Any]) -> None:
    """
    Explain how to choose optimal thresholds based on business priorities.
    
    Args:
        threshold_data: Dictionary of metrics at different thresholds
    """
    print("\n=== Optimal Threshold Selection Explanation ===")
    
    # Extract recommendations and optimal thresholds
    recommendations = threshold_data.get('recommendations', {})
    optimal = threshold_data.get('optimal_thresholds', {})
    
    print(
        "The optimal threshold depends on business priorities:\n"
        f"1. Profit-focused: Threshold = {optimal.get('profit', {}).get('threshold', 0):.2f}\n"
        f"   - Maximizes absolute profit (${optimal.get('profit', {}).get('value', 0):.2f})\n"
        f"   - Approval rate: {optimal.get('profit', {}).get('approval_rate', 0):.1%}\n\n"
        
        f"2. ROI-focused: Threshold = {optimal.get('roi', {}).get('threshold', 0):.2f}\n"
        f"   - Maximizes return on investment ({optimal.get('roi', {}).get('value', 0):.2%})\n"
        f"   - Approval rate: {optimal.get('roi', {}).get('approval_rate', 0):.1%}\n\n"
        
        f"3. Balanced approach: Threshold = {recommendations.get('balanced', 0):.2f}\n"
        "   - Compromise between profit and ROI\n\n"
        
        "Business considerations for threshold selection:\n"
        "- Capital constraints: Limited capital favors higher thresholds for better ROI\n"
        "- Growth targets: Expansion goals may favor lower thresholds for higher volume\n"
        "- Risk tolerance: Conservative approaches favor higher thresholds\n"
        "- Market conditions: Tighter economic conditions may warrant higher thresholds"
    )


def explain_sensitivity_analysis(threshold_data: Dict[str, Any], output_dir: Optional[str] = None) -> None:
    """
    Explain the sensitivity analysis for threshold changes.
    
    Args:
        threshold_data: Dictionary containing sensitivity analysis data
        output_dir: Directory to save visualizations
    """
    print("\n=== Threshold Sensitivity Analysis Explanation ===")
    print(
        "Sensitivity analysis examines how changes in threshold values affect profit:\n"
        "- High sensitivity means small threshold changes significantly impact profit\n"
        "- Low sensitivity indicates stable profit across threshold changes\n\n"
        "This analysis helps identify 'sweet spots' where the threshold is most stable."
    )
    
    # Extract sensitivity analysis data
    sensitivity_data = threshold_data.get('sensitivity_analysis', [])
    
    if sensitivity_data:
        # Convert to DataFrame for easier manipulation
        sens_df = pd.DataFrame(sensitivity_data)
        
        # Print some examples
        print("\nSensitivity Analysis Examples:")
        for i, row in sens_df.iloc[:5].iterrows():
            print(f"  Threshold change {row['threshold_range']}: "
                  f"Profit changes by ${row['profit_diff']:.2f} "
                  f"(Sensitivity: {row['sensitivity']:.2f})")
        
        # Create visualization
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
            plt.figure(figsize=(12, 6))
            plt.bar(sens_df['threshold_range'], sens_df['sensitivity'].abs(), color='#1f77b4')
            
            plt.xlabel('Threshold Range')
            plt.ylabel('Sensitivity (absolute value)')
            plt.title('Threshold Sensitivity Analysis')
            plt.xticks(rotation=45)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            plt.savefig(os.path.join(output_dir, "threshold_sensitivity.png"), dpi=300)
            plt.close()


def main(example_data_path: Optional[str] = None, output_dir: str = "data/processed/threshold_explanation"):
    """
    Run all explanation functions to create a comprehensive understanding of thresholds
    and profitability metrics.
    
    Args:
        example_data_path: Path to example threshold metrics data (JSON)
        output_dir: Directory to save all explanation visualizations
    """
    print("===== THRESHOLD AND PROFITABILITY METRICS EXPLANATION =====")
    print(
        "This script explains how thresholds work in loan repayment prediction models\n"
        "and how they affect profitability metrics like ROI and profit."
    )
    
    # Basic concept explanations
    explain_threshold_concept(output_dir)
    explain_profit_calculation(output_dir)
    explain_threshold_profitability_tradeoff(output_dir)
    
    # Load example data if provided
    threshold_data = {}
    if example_data_path:
        try:
            import json
            with open(example_data_path, 'r') as f:
                threshold_data = json.load(f)
                
            # Examples with real data
            explain_real_world_example(threshold_data, output_dir)
            explain_optimal_threshold_selection(threshold_data)
            explain_sensitivity_analysis(threshold_data, output_dir)
        except Exception as e:
            print(f"Error loading example data: {e}")
    
    print("\n===== EXPLANATION COMPLETE =====")
    print(f"Visualizations saved to: {output_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Explain threshold and profitability metrics')
    parser.add_argument('--data', default='data/processed/model_comparison/model1_tradeoffs.json',
                       help='Path to example threshold metrics data (JSON)')
    parser.add_argument('--output', default='data/processed/threshold_explanation',
                       help='Directory to save explanation visualizations')
    
    args = parser.parse_args()
    main(args.data, args.output)
