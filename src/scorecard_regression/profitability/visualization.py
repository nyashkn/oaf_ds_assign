"""
Visualization functions for profitability analysis.

This module provides various plotting functions to visualize the results of
threshold analysis and profit optimization.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Dict, List, Optional, Tuple, Any, Union

def plot_threshold_performance(
    threshold_df: pd.DataFrame,
    optimal_threshold: float,
    output_path: Optional[str] = None,
    dpi: int = 300
) -> plt.Figure:
    """
    Plot approval rate and actual repayment rate vs threshold.
    
    Args:
        threshold_df: DataFrame with threshold analysis results
        optimal_threshold: Optimal threshold value
        output_path: Path to save the plot
        dpi: DPI for saved plot
        
    Returns:
        Matplotlib Figure object
    """
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plot approval rate on left y-axis
    ax1.plot(threshold_df['threshold'], threshold_df['approval_rate'] * 100, 
             'b-', marker='o', label='Approval Rate (%)')
    ax1.set_xlabel('Repayment Rate Threshold')
    ax1.set_ylabel('Approval Rate (%)', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    
    # Plot actual repayment rate on right y-axis
    ax2 = ax1.twinx()
    ax2.plot(threshold_df['threshold'], threshold_df['actual_repayment_rate'] * 100, 
             'g-', marker='s', label='Actual Repayment Rate (%)')
    ax2.set_ylabel('Actual Repayment Rate (%)', color='g')
    ax2.tick_params(axis='y', labelcolor='g')
    
    # Add vertical line for optimal threshold
    plt.axvline(x=optimal_threshold, color='r', linestyle='--', 
               label=f'Optimal Threshold = {optimal_threshold:.2f}')
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
    
    plt.title('Approval Rate and Actual Repayment Rate vs Threshold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot if output path provided
    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    
    return fig

def plot_profit_metrics(
    threshold_df: pd.DataFrame,
    optimal_threshold: float,
    output_path: Optional[str] = None,
    dpi: int = 300
) -> plt.Figure:
    """
    Plot profit metrics vs threshold.
    
    Args:
        threshold_df: DataFrame with threshold analysis results
        optimal_threshold: Optimal threshold value
        output_path: Path to save the plot
        dpi: DPI for saved plot
        
    Returns:
        Matplotlib Figure object
    """
    plt.figure(figsize=(12, 8))
    
    plt.plot(threshold_df['threshold'], threshold_df['total_actual_profit'], 
             'b-', marker='o', label='Total Actual Profit')
    plt.plot(threshold_df['threshold'], threshold_df['money_left_on_table'], 
             'r-', marker='s', label='Money Left on Table')
    
    # Add vertical line for optimal threshold
    plt.axvline(x=optimal_threshold, color='g', linestyle='--', 
               label=f'Optimal Threshold = {optimal_threshold:.2f}')
    
    plt.xlabel('Repayment Rate Threshold')
    plt.ylabel('Profit')
    plt.title('Profit Metrics vs Threshold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    # Save plot if output path provided
    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    
    return plt.gcf()

def plot_confusion_matrix(
    loan_metrics: Dict[str, Any],
    target_threshold: float,
    output_path: Optional[str] = None,
    dpi: int = 300
) -> plt.Figure:
    """
    Plot confusion matrix for loan classification.
    
    Args:
        loan_metrics: Dictionary with loan metrics
        target_threshold: Target repayment rate threshold
        output_path: Path to save the plot
        dpi: DPI for saved plot
        
    Returns:
        Matplotlib Figure object
    """
    # Extract confusion matrix elements
    n_true_pos = loan_metrics['n_true_pos']
    n_false_pos = loan_metrics['n_false_pos']
    n_true_neg = loan_metrics['n_true_neg']
    n_false_neg = loan_metrics['n_false_neg']
    
    # Create confusion matrix data
    confusion_data = [
        [n_true_pos, n_false_neg],
        [n_false_pos, n_true_neg]
    ]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(confusion_data, annot=True, fmt="d", cmap="Blues", cbar=False,
               xticklabels=[f"Actual ≥ {target_threshold:.2f}", f"Actual < {target_threshold:.2f}"],
               yticklabels=[f"Predicted ≥ {target_threshold:.2f}", f"Predicted < {target_threshold:.2f}"])
    plt.title(f"Loan Classification Confusion Matrix (Threshold = {target_threshold:.2f})")
    plt.ylabel("Predicted")
    plt.xlabel("Actual")
    
    # Add percentage annotations
    total = sum([sum(row) for row in confusion_data])
    for i in range(2):
        for j in range(2):
            text = ax.texts[i*2 + j]
            current_value = confusion_data[i][j]
            percentage = current_value / total * 100 if total > 0 else 0
            text.set_text(f"{current_value}\n({percentage:.1f}%)")
    
    plt.tight_layout()
    
    # Save plot if output path provided
    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    
    return fig

def create_executive_summary(
    loan_metrics: Dict[str, Any],
    profit_metrics: Dict[str, Any],
    class_metrics: Dict[str, float],
    target_threshold: float,
    output_path: Optional[str] = None,
    dpi: int = 300
) -> plt.Figure:
    """
    Create an executive summary visualization of profitability analysis.
    
    Args:
        loan_metrics: Dictionary with loan metrics
        profit_metrics: Dictionary with profit metrics
        class_metrics: Dictionary with classification metrics
        target_threshold: Target repayment rate threshold
        output_path: Path to save the plot
        dpi: DPI for saved plot
        
    Returns:
        Matplotlib Figure object
    """
    # Extract metrics
    n_good = loan_metrics['n_good']
    n_bad = loan_metrics['n_bad']
    total_loans = loan_metrics['total_loans']
    
    n_true_pos = loan_metrics['n_true_pos']
    n_false_pos = loan_metrics['n_false_pos']
    n_true_neg = loan_metrics['n_true_neg']
    n_false_neg = loan_metrics['n_false_neg']
    
    # Create executive summary plot
    fig = plt.figure(figsize=(15, 10))
    
    # Define grid layout
    gs = plt.GridSpec(2, 3, figure=fig)
    
    # Panel 1: Loan Classification
    ax1 = fig.add_subplot(gs[0, 0])
    labels = ['Good Loans', 'Bad Loans']
    sizes = [n_good, n_bad]
    colors = ['#4ECDC4', '#FF6B6B']
    ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax1.set_title(f'Loan Quality (Threshold = {target_threshold:.2f})')
    
    # Panel 2: Model Performance
    ax2 = fig.add_subplot(gs[0, 1])
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    values = [
        class_metrics['accuracy'], 
        class_metrics['precision'], 
        class_metrics['recall'], 
        class_metrics['f1_score']
    ]
    ax2.bar(metrics, values, color='#1A535C')
    ax2.set_ylim(0, 1)
    ax2.set_title('Classification Metrics')
    for i, v in enumerate(values):
        ax2.text(i, v + 0.02, f"{v:.2f}", ha='center')
    
    # Panel 3: Approval/Rejection Rate
    ax3 = fig.add_subplot(gs[0, 2])
    labels = ['Approved', 'Rejected']
    sizes = [n_true_pos + n_false_pos, n_true_neg + n_false_neg]
    colors = ['#4ECDC4', '#FF6B6B']
    ax3.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax3.set_title('Loan Decisions')
    
    # Panel 4: Profit Analysis (spans bottom row)
    ax4 = fig.add_subplot(gs[1, :])
    
    # Extract profit metrics for visualization
    true_pos_profit = profit_metrics['true_pos_profit']
    false_pos_profit = profit_metrics['false_pos_profit']
    money_left_on_table = profit_metrics['money_left_on_table']
    total_actual_profit = profit_metrics['total_actual_profit']
    
    # Set up bar chart
    profit_categories = ['Profit from\nGood Loans', 'Profit from\nBad Loans', 
                         'Money Left\non Table', 'Total\nProfit']
    profit_values = [true_pos_profit, false_pos_profit, money_left_on_table, total_actual_profit]
    profit_colors = ['#4ECDC4', '#FF6B6B', '#F4A261', '#2A9D8F']
    
    bars = ax4.bar(profit_categories, profit_values, color=profit_colors)
    
    # Add labels
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{height:.0f}', ha='center', va='bottom')
    
    ax4.set_title('Profit Analysis')
    ax4.set_ylabel('Amount')
    ax4.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add approval rate annotation
    approval_rate = (n_true_pos + n_false_pos) / total_loans if total_loans > 0 else 0
    
    # Add title and important conclusions
    plt.suptitle(f"Profitability Analysis: Threshold = {target_threshold:.2f}", fontsize=16, y=0.98)
    
    bottom_text = (
        f"Approval Rate: {approval_rate:.1%} • "
        f"Expected Profit per Loan: {total_actual_profit/total_loans:.2f} • "
        f"Profit from Approved: {(true_pos_profit + false_pos_profit):.0f} • "
        f"Money Left on Table: {money_left_on_table:.0f}"
    )
    
    plt.figtext(0.5, 0.01, bottom_text, ha="center", fontsize=12, 
               bbox={"facecolor":"#E9C46A", "alpha":0.5, "pad":5})
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save plot if output path provided
    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    
    return fig

def plot_optimization_results(
    optimization_results: Dict[str, Any],
    metric: str = 'total_actual_profit',
    output_path: Optional[str] = None,
    dpi: int = 300
) -> plt.Figure:
    """
    Plot optimization results including the profit curve and optimal threshold.
    
    Args:
        optimization_results: Dictionary with optimization results
        metric: Profit metric optimized
        output_path: Path to save the plot
        dpi: DPI for saved plot
        
    Returns:
        Matplotlib Figure object
    """
    # Extract results
    optimal_threshold = optimization_results['optimal_threshold']
    profit_curve = optimization_results['profit_curve']
    
    # Convert profit curve to arrays
    threshold_values = [t for t, _ in profit_curve]
    profit_values = [v for _, v in profit_curve]
    
    # Create plot
    plt.figure(figsize=(12, 6))
    
    plt.plot(threshold_values, profit_values, 'b-', linewidth=2)
    plt.axvline(x=optimal_threshold, color='r', linestyle='--', 
                label=f'Optimal Threshold = {optimal_threshold:.4f}')
    
    # Add points for each evaluated threshold
    plt.scatter(threshold_values, profit_values, color='b', s=30)
    
    # Add a larger marker for the optimal point
    optimal_value_idx = min(range(len(threshold_values)), 
                           key=lambda i: abs(threshold_values[i] - optimal_threshold))
    optimal_value = profit_values[optimal_value_idx]
    plt.scatter([optimal_threshold], [optimal_value], color='r', s=100, zorder=5)
    
    # Add text annotation for optimal point
    plt.annotate(f'Optimal: ({optimal_threshold:.4f}, {optimal_value:.2f})',
                xy=(optimal_threshold, optimal_value),
                xytext=(optimal_threshold + 0.02, optimal_value + 0.1 * (max(profit_values) - min(profit_values))),
                arrowprops=dict(facecolor='black', arrowstyle='->'),
                fontsize=10)
    
    metric_label = "Profit" if metric == "total_actual_profit" else "Value"
    plt.xlabel('Threshold')
    plt.ylabel(f'{metric_label}')
    plt.title(f'Optimization Results: {metric}')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    
    # Save plot if output path provided
    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    
    return plt.gcf()

def plot_pareto_frontier(
    pareto_frontier: List[Tuple[float, float, float]],
    output_path: Optional[str] = None,
    dpi: int = 300
) -> plt.Figure:
    """
    Plot the Pareto frontier from multi-objective optimization.
    
    Args:
        pareto_frontier: List of (threshold, profit, money_left) tuples
        output_path: Path to save the plot
        dpi: DPI for saved plot
        
    Returns:
        Matplotlib Figure object
    """
    # Extract values
    thresholds = [t for t, _, _ in pareto_frontier]
    profits = [p for _, p, _ in pareto_frontier]
    money_left = [m for _, _, m in pareto_frontier]
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot profit vs money left
    sc = ax1.scatter(profits, money_left, c=thresholds, cmap='viridis', 
                    s=100, alpha=0.8)
    
    # Add colorbar for thresholds
    cbar = plt.colorbar(sc, ax=ax1)
    cbar.set_label('Threshold')
    
    # Add labels to each point
    for i, (t, p, m) in enumerate(pareto_frontier):
        ax1.annotate(f"{t:.2f}", (p, m), fontsize=8)
    
    ax1.set_xlabel('Total Profit')
    ax1.set_ylabel('Money Left on Table')
    ax1.set_title('Pareto Frontier: Profit vs Money Left on Table')
    ax1.grid(True, alpha=0.3)
    
    # Plot metrics vs threshold
    ax2.plot(thresholds, profits, 'b-o', label='Total Profit')
    ax2.set_xlabel('Threshold')
    ax2.set_ylabel('Total Profit', color='b')
    ax2.tick_params(axis='y', labelcolor='b')
    
    ax2_2 = ax2.twinx()
    ax2_2.plot(thresholds, money_left, 'r-o', label='Money Left on Table')
    ax2_2.set_ylabel('Money Left on Table', color='r')
    ax2_2.tick_params(axis='y', labelcolor='r')
    
    # Create legend
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper center')
    
    ax2.set_title('Metrics vs Threshold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot if output path provided
    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    
    return fig
