"""
Visualization functions for loan repayment rate prediction profitability.

This module provides functions to visualize profitability metrics, threshold
performance, and other business analytics for loan approval decisions.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any, Union
import matplotlib.patches as mpatches

# Set Seaborn style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

def plot_threshold_performance(
    metrics_df: pd.DataFrame,
    output_path: Optional[str] = None,
    title: Optional[str] = None,
    highlight_threshold: Optional[float] = None,
    figsize: Tuple[int, int] = (12, 10)
) -> plt.Figure:
    """
    Plot key metrics vs threshold to visualize performance across thresholds.
    
    Args:
        metrics_df: DataFrame with metrics at different thresholds
                   (should include 'threshold', 'profit', 'roi', etc.)
        output_path: Path to save the plot
        title: Custom title for the plot
        highlight_threshold: Specific threshold to highlight
        figsize: Figure size as (width, height)
        
    Returns:
        Matplotlib Figure object
    """
    # Create figure with subplots
    fig, axs = plt.subplots(2, 2, figsize=figsize)
    
    # Profit vs Threshold
    axs[0, 0].plot(metrics_df['threshold'], metrics_df['actual_profit'] 
                  if 'actual_profit' in metrics_df.columns else metrics_df['profit'], 
                  'o-', color='#2C7BB6')
    axs[0, 0].set_xlabel('Threshold')
    axs[0, 0].set_ylabel('Profit')
    axs[0, 0].set_title('Profit vs Threshold')
    axs[0, 0].grid(True, linestyle='--', alpha=0.7)
    
    # ROI vs Threshold
    roi_col = 'roi'
    if roi_col in metrics_df.columns:
        # Convert to percentage
        roi_values = metrics_df[roi_col] * 100
        axs[0, 1].plot(metrics_df['threshold'], roi_values, 'o-', color='#D7191C')
        axs[0, 1].set_xlabel('Threshold')
        axs[0, 1].set_ylabel('ROI (%)')
        axs[0, 1].set_title('ROI vs Threshold')
        axs[0, 1].grid(True, linestyle='--', alpha=0.7)
    
    # Approval Rate vs Threshold
    approval_col = 'approval_rate'
    if approval_col in metrics_df.columns:
        # Convert to percentage
        approval_values = metrics_df[approval_col] * 100
        axs[1, 0].plot(metrics_df['threshold'], approval_values, 'o-', color='#1A9641')
        axs[1, 0].set_xlabel('Threshold')
        axs[1, 0].set_ylabel('Approval Rate (%)')
        axs[1, 0].set_title('Approval Rate vs Threshold')
        axs[1, 0].grid(True, linestyle='--', alpha=0.7)
    
    # F1 Score vs Threshold
    f1_col = 'f1_score'
    if f1_col in metrics_df.columns:
        axs[1, 1].plot(metrics_df['threshold'], metrics_df[f1_col], 'o-', color='#7570B3')
        axs[1, 1].set_xlabel('Threshold')
        axs[1, 1].set_ylabel('F1 Score')
        axs[1, 1].set_title('F1 Score vs Threshold')
        axs[1, 1].grid(True, linestyle='--', alpha=0.7)
    
    # Highlight specific threshold if provided
    if highlight_threshold is not None:
        for ax in axs.flat:
            ax.axvline(x=highlight_threshold, color='r', linestyle='--', alpha=0.7,
                      label=f'Threshold = {highlight_threshold:.2f}')
            
            # Add a legend
            ax.legend(loc='best')
    
    # Set a main title
    if title:
        fig.suptitle(title, fontsize=16, y=1.02)
    else:
        fig.suptitle('Threshold Performance Analysis', fontsize=16, y=1.02)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if output_path provided
    if output_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save figure
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Threshold performance plot saved to {output_path}")
    
    return fig


def plot_profit_metrics(
    metrics_df: pd.DataFrame,
    output_path: Optional[str] = None,
    title: Optional[str] = None,
    highlight_threshold: Optional[float] = None,
    figsize: Tuple[int, int] = (12, 10)
) -> plt.Figure:
    """
    Create a more detailed profit metrics visualization.
    
    Args:
        metrics_df: DataFrame with metrics at different thresholds
        output_path: Path to save the plot
        title: Custom title for the plot
        highlight_threshold: Specific threshold to highlight
        figsize: Figure size as (width, height)
        
    Returns:
        Matplotlib Figure object
    """
    # Create figure with subplots
    fig, axs = plt.subplots(2, 2, figsize=figsize)
    
    # Helper function to find the best threshold for a metric
    def find_best_threshold(metric_col, maximize=True):
        if metric_col not in metrics_df.columns:
            return None
        
        if maximize:
            idx = metrics_df[metric_col].idxmax()
        else:
            idx = metrics_df[metric_col].idxmin()
            
        return metrics_df.loc[idx, 'threshold']
    
    # Profit vs Threshold with both actual and expected
    profit_ax = axs[0, 0]
    if 'actual_profit' in metrics_df.columns:
        profit_ax.plot(metrics_df['threshold'], metrics_df['actual_profit'], 'o-', 
                      color='#2C7BB6', label='Actual Profit')
    
    if 'expected_profit' in metrics_df.columns:
        profit_ax.plot(metrics_df['threshold'], metrics_df['expected_profit'], 's--', 
                      color='#5AAE61', label='Expected Profit')
    
    if 'actual_profit' in metrics_df.columns:
        best_profit = find_best_threshold('actual_profit')
        if best_profit is not None:
            profit_ax.axvline(x=best_profit, color='#2C7BB6', linestyle='--', alpha=0.7,
                           label=f'Best Profit Threshold = {best_profit:.2f}')
    
    profit_ax.set_xlabel('Threshold')
    profit_ax.set_ylabel('Profit')
    profit_ax.set_title('Profit vs Threshold')
    profit_ax.grid(True, linestyle='--', alpha=0.7)
    profit_ax.legend(loc='best')
    
    # ROI vs Profit 
    roi_profit_ax = axs[0, 1]
    if 'roi' in metrics_df.columns and ('actual_profit' in metrics_df.columns or 'profit' in metrics_df.columns):
        profit_col = 'actual_profit' if 'actual_profit' in metrics_df.columns else 'profit'
        roi_profit_ax.scatter(metrics_df[profit_col], metrics_df['roi'] * 100, 
                           c=metrics_df['threshold'], cmap='viridis', s=50)
        roi_profit_ax.set_xlabel('Profit')
        roi_profit_ax.set_ylabel('ROI (%)')
        roi_profit_ax.set_title('ROI vs Profit')
        
        # Add colorbar
        cbar = plt.colorbar(plt.cm.ScalarMappable(cmap='viridis'), ax=roi_profit_ax)
        cbar.set_label('Threshold')
        
        # Add labels for min and max thresholds
        min_threshold = metrics_df['threshold'].min()
        max_threshold = metrics_df['threshold'].max()
        
        min_idx = metrics_df['threshold'].idxmin()
        max_idx = metrics_df['threshold'].idxmax()
        
        roi_profit_ax.annotate(f'Min Threshold: {min_threshold:.2f}',
                           xy=(metrics_df.loc[min_idx, profit_col], 
                               metrics_df.loc[min_idx, 'roi'] * 100),
                           xytext=(10, 10),
                           textcoords='offset points',
                           arrowprops=dict(arrowstyle='->'))
        
        roi_profit_ax.annotate(f'Max Threshold: {max_threshold:.2f}',
                           xy=(metrics_df.loc[max_idx, profit_col], 
                               metrics_df.loc[max_idx, 'roi'] * 100),
                           xytext=(10, -20),
                           textcoords='offset points',
                           arrowprops=dict(arrowstyle='->'))
    
    # Actual vs Expected Profit Difference
    diff_ax = axs[1, 0]
    if 'actual_profit' in metrics_df.columns and 'expected_profit' in metrics_df.columns:
        # Calculate difference
        metrics_df['profit_diff'] = metrics_df['actual_profit'] - metrics_df['expected_profit']
        
        diff_ax.plot(metrics_df['threshold'], metrics_df['profit_diff'], 'o-', color='#D73027')
        diff_ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        diff_ax.set_xlabel('Threshold')
        diff_ax.set_ylabel('Actual - Expected Profit')
        diff_ax.set_title('Profit Prediction Gap')
        diff_ax.grid(True, linestyle='--', alpha=0.7)
    
    # Approval Rate and ROI
    combo_ax = axs[1, 1]
    if 'approval_rate' in metrics_df.columns and 'roi' in metrics_df.columns:
        # Plot approval rate
        color1 = '#1A9641'
        ln1 = combo_ax.plot(metrics_df['threshold'], metrics_df['approval_rate'] * 100, 
                         'o-', color=color1, label='Approval Rate (%)')
        combo_ax.set_xlabel('Threshold')
        combo_ax.set_ylabel('Approval Rate (%)', color=color1)
        combo_ax.tick_params(axis='y', labelcolor=color1)
        
        # Create a twin axis for ROI
        combo_ax2 = combo_ax.twinx()
        color2 = '#D73027'
        ln2 = combo_ax2.plot(metrics_df['threshold'], metrics_df['roi'] * 100, 
                          's-', color=color2, label='ROI (%)')
        combo_ax2.set_ylabel('ROI (%)', color=color2)
        combo_ax2.tick_params(axis='y', labelcolor=color2)
        
        # Combine legends
        lns = ln1 + ln2
        labs = [l.get_label() for l in lns]
        combo_ax.legend(lns, labs, loc='upper left')
        
        combo_ax.set_title('Approval Rate and ROI vs Threshold')
        combo_ax.grid(True, linestyle='--', alpha=0.7)
    
    # Set a main title
    if title:
        fig.suptitle(title, fontsize=16, y=1.02)
    else:
        fig.suptitle('Detailed Profit Metrics Analysis', fontsize=16, y=1.02)
    
    # Highlight specific threshold if provided
    if highlight_threshold is not None:
        for ax in [profit_ax, diff_ax, combo_ax]:
            ax.axvline(x=highlight_threshold, color='purple', linestyle='--', alpha=0.7,
                      label=f'Threshold = {highlight_threshold:.2f}')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if output_path provided
    if output_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save figure
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Profit metrics plot saved to {output_path}")
    
    return fig


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.8,
    normalize: bool = True,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Plot confusion matrix for loan approval decisions at a given threshold.
    
    Args:
        y_true: True repayment rates
        y_pred: Predicted repayment rates
        threshold: Threshold for loan approval
        normalize: Whether to normalize the confusion matrix
        output_path: Path to save the plot
        figsize: Figure size as (width, height)
        
    Returns:
        Matplotlib Figure object
    """
    # Convert continuous values to binary using threshold
    y_true_bin = (y_true >= threshold).astype(int)
    y_pred_bin = (y_pred >= threshold).astype(int)
    
    # Calculate confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true_bin, y_pred_bin)
    
    # Normalize if requested
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.round(cm, 2)
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Use seaborn heatmap
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', cmap='Blues',
               xticklabels=['Reject', 'Approve'],
               yticklabels=['Bad Loan', 'Good Loan'],
               ax=ax)
    
    # Add labels
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f'Confusion Matrix (Threshold = {threshold:.2f})')
    
    # Add descriptive labels in the corners
    fontdict = {'fontsize': 9, 'fontweight': 'bold', 'color': 'darkslategray'}
    
    # True Negative (bottom-left)
    ax.text(0.2, 1.7, 'True Negative\n(Correct Rejection)', 
           ha='center', va='center', fontdict=fontdict)
    
    # False Positive (bottom-right)
    ax.text(1.2, 1.7, 'False Positive\n(Type I Error)', 
           ha='center', va='center', fontdict=fontdict)
    
    # False Negative (top-left)
    ax.text(0.2, 0.7, 'False Negative\n(Type II Error)', 
           ha='center', va='center', fontdict=fontdict)
    
    # True Positive (top-right)
    ax.text(1.2, 0.7, 'True Positive\n(Correct Approval)', 
           ha='center', va='center', fontdict=fontdict)
    
    plt.tight_layout()
    
    # Save figure if output_path provided
    if output_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save figure
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix plot saved to {output_path}")
    
    return fig


def create_executive_summary(
    metrics: Dict[str, Any],
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 10)
) -> plt.Figure:
    """
    Create an executive summary visualization of loan repayment prediction.
    
    Args:
        metrics: Dictionary with various metrics from threshold analysis
        output_path: Path to save the plot
        figsize: Figure size as (width, height)
        
    Returns:
        Matplotlib Figure object
    """
    # Create figure
    fig, axs = plt.subplots(2, 2, figsize=figsize)
    
    # Extract key metrics
    threshold = metrics.get('optimal_thresholds', {}).get('profit_focused', 0.8)
    profit = metrics.get('profit_metrics', {}).get('actual_profit', 0)
    roi = metrics.get('profit_metrics', {}).get('roi', 0) * 100  # Convert to percentage
    approval_rate = metrics.get('loan_metrics', {}).get('n_loans', {}).get('approval_rate', 0) * 100
    
    # Get loan counts
    loan_counts = metrics.get('loan_metrics', {}).get('loan_categories', {})
    
    # Loan approval decision pie chart
    axs[0, 0].set_title('Loan Approval Decisions')
    
    good_approved = loan_counts.get('good_approved', {}).get('count', 0)
    bad_approved = loan_counts.get('bad_approved', {}).get('count', 0)
    good_rejected = loan_counts.get('good_rejected', {}).get('count', 0)
    bad_rejected = loan_counts.get('bad_rejected', {}).get('count', 0)
    
    sizes = [good_approved, bad_approved, good_rejected, bad_rejected]
    labels = ['Good Approved', 'Bad Approved', 'Good Rejected', 'Bad Rejected']
    colors = ['#1A9641', '#D73027', '#A6D96A', '#FC8D59']
    
    axs[0, 0].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    axs[0, 0].axis('equal')
    
    # Metrics summary
    axs[0, 1].axis('off')  # Turn off axis
    metrics_text = (
        f"Optimal Threshold: {threshold:.2f}\n\n"
        f"Profit: {profit:.2f}\n\n"
        f"ROI: {roi:.1f}%\n\n"
        f"Approval Rate: {approval_rate:.1f}%\n\n"
    )
    
    axs[0, 1].text(0.5, 0.5, metrics_text, 
                  ha='center', va='center', 
                  fontsize=14,
                  bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=1'))
    
    # Loan value distribution
    loan_values = metrics.get('loan_metrics', {}).get('loan_values', {})
    
    approved_value = loan_values.get('approved', 0)
    rejected_value = loan_values.get('rejected', 0)
    
    axs[1, 0].set_title('Loan Value Distribution')
    axs[1, 0].bar([0, 1], [approved_value, rejected_value], 
                 color=['#66C2A5', '#FC8D62'])
    axs[1, 0].set_xticks([0, 1])
    axs[1, 0].set_xticklabels(['Approved', 'Rejected'])
    axs[1, 0].set_ylabel('Loan Value')
    
    # Add values on top of bars
    for i, v in enumerate([approved_value, rejected_value]):
        axs[1, 0].text(i, v, f"{v:.0f}", ha='center', va='bottom')
    
    # Expected vs actual repayment
    predicted_repayment = metrics.get('repayment_rates', {}).get('predicted_avg', 0) * 100
    actual_repayment = metrics.get('repayment_rates', {}).get('actual_avg', 0) * 100
    
    axs[1, 1].set_title('Expected vs Actual Repayment Rate')
    axs[1, 1].bar([0, 1], [predicted_repayment, actual_repayment], 
                 color=['#8DA0CB', '#66C2A5'])
    axs[1, 1].set_xticks([0, 1])
    axs[1, 1].set_xticklabels(['Expected', 'Actual'])
    axs[1, 1].set_ylabel('Repayment Rate (%)')
    axs[1, 1].set_ylim([0, 100])
    
    # Add values on top of bars
    for i, v in enumerate([predicted_repayment, actual_repayment]):
        axs[1, 1].text(i, v, f"{v:.1f}%", ha='center', va='bottom')
    
    # Set a main title
    fig.suptitle('Loan Repayment Prediction Executive Summary', fontsize=16, y=0.98)
    
    plt.tight_layout()
    
    # Save figure if output_path provided
    if output_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save figure
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Executive summary plot saved to {output_path}")
    
    return fig


def plot_optimization_results(
    optimization_results: Dict[str, Any],
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """
    Visualize the results of threshold optimization.
    
    Args:
        optimization_results: Results from optimization functions
        output_path: Path to save the plot
        figsize: Figure size as (width, height)
        
    Returns:
        Matplotlib Figure object
    """
    # Extract data from optimization results
    if 'results' not in optimization_results:
        raise ValueError("Optimization results must contain 'results' key")
    
    # Convert to DataFrame if necessary
    if isinstance(optimization_results['results'], list):
        results_df = pd.DataFrame(optimization_results['results'])
    else:
        results_df = optimization_results['results']
    
    # Get optimal threshold
    optimal_threshold = optimization_results.get('optimal_threshold', None)
    
    # Create figure with subplots
    fig, axs = plt.subplots(2, 2, figsize=figsize)
    
    # Plot optimization objective
    objective = optimization_results.get('objective', 'profit')
    
    # Helper function to select color based on objective
    def get_objective_color(obj):
        colors = {
            'profit': '#2C7BB6',
            'roi': '#D73027',
            'f1': '#7570B3'
        }
        return colors.get(obj, '#000000')
    
    # Main objective plot
    ax_obj = axs[0, 0]
    objective_col = 'profit' if objective == 'profit' else objective
    if objective_col in results_df.columns:
        ax_obj.plot(results_df['threshold'], results_df[objective_col], 'o-', 
                   color=get_objective_color(objective))
        
        if optimal_threshold is not None:
            # Find value at optimal threshold
            opt_row = results_df[results_df['threshold'] == optimal_threshold]
            if not opt_row.empty:
                opt_value = opt_row[objective_col].values[0]
                
                # Highlight optimal point
                ax_obj.plot([optimal_threshold], [opt_value], 'o', 
                           color='red', markersize=10)
                
                # Add vertical line
                ax_obj.axvline(x=optimal_threshold, color='red', linestyle='--', alpha=0.7,
                            label=f'Optimal: {optimal_threshold:.2f}')
                
        ax_obj.set_xlabel('Threshold')
        ax_obj.set_ylabel(objective.capitalize())
        ax_obj.set_title(f'{objective.capitalize()} vs Threshold')
        ax_obj.grid(True, linestyle='--', alpha=0.7)
        ax_obj.legend()
    
    # Other metrics (approval rate, ROI if not objective)
    ax_other = axs[0, 1]
    
    # Always include approval rate
    if 'approval_rate' in results_df.columns:
        color = '#1A9641'
        ln1 = ax_other.plot(results_df['threshold'], results_df['approval_rate'] * 100, 
                         'o-', color=color, label='Approval Rate (%)')
        ax_other.set_xlabel('Threshold')
        ax_other.set_ylabel('Approval Rate (%)', color=color)
        ax_other.tick_params(axis='y', labelcolor=color)
    
    # Add ROI if it's not the main objective
    if objective != 'roi' and 'roi' in results_df.columns:
        # Create a twin axis
        ax_roi = ax_other.twinx()
        color = '#D73027'
        ln2 = ax_roi.plot(results_df['threshold'], results_df['roi'] * 100, 
                       's-', color=color, label='ROI (%)')
        ax_roi.set_ylabel('ROI (%)', color=color)
        ax_roi.tick_params(axis='y', labelcolor=color)
        
        # Combine legends if both metrics are plotted
        if 'approval_rate' in results_df.columns:
            lns = ln1 + ln2
            labs = [l.get_label() for l in lns]
            ax_other.legend(lns, labs, loc='upper left')
        else:
            ax_roi.legend()
    else:
        # Just add the legend for approval rate
        if 'approval_rate' in results_df.columns:
            ax_other.legend()
    
    # Add title
    ax_other.set_title('Key Metrics vs Threshold')
    ax_other.grid(True, linestyle='--', alpha=0.7)
    
    # Highlight optimal threshold if provided
    if optimal_threshold is not None:
        ax_other.axvline(x=optimal_threshold, color='red', linestyle='--', alpha=0.7)
    
    # Optimization method performance
    ax_method = axs[1, 0]
    
    method = optimization_results.get('method', 'grid')
    ax_method.text(0.5, 0.5, 
                 f"Optimization Method: {method.capitalize()}\n\n"
                 f"Objective: {objective.capitalize()}\n\n"
                 f"Optimal Threshold: {optimal_threshold:.4f}" if optimal_threshold else "No optimal threshold found",
                 ha='center', va='center',
                 fontsize=12,
                 bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=1'))
    ax_method.axis('off')
    
    # Constraints and parameters
    ax_params = axs[1, 1]
    
    # Get constraints
    constraints = optimization_results.get('constraints', {})
    parameters = optimization_results.get('parameters', {})
    
    # Combine all parameters
    all_params = {}
    if constraints:
        all_params.update({f"constraint_{k}": v for k, v in constraints.items()})
    if parameters:
        all_params.update(parameters)
    
    # Create a string with all parameters
    params_str = "\n".join([f"{k}: {v}" for k, v in all_params.items()])
    
    ax_params.text(0.5, 0.5, 
                  f"Parameters:\n\n{params_str}",
                  ha='center', va='center',
                  fontsize=10,
                  bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=1'))
    ax_params.axis('off')
    
    # Set a main title
    fig.suptitle(f'Threshold Optimization Results ({method.capitalize()})', fontsize=16, y=0.98)
    
    plt.tight_layout()
    
    # Save figure if output_path provided
    if output_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save figure
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Optimization results plot saved to {output_path}")
    
    return fig


def plot_pareto_frontier(
    metrics_df: pd.DataFrame,
    x_col: str = 'roi',
    y_col: str = 'profit',
    output_path: Optional[str] = None,
    highlight_threshold: Optional[float] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Plot the Pareto frontier for multiple optimization objectives.
    
    Args:
        metrics_df: DataFrame with metrics at different thresholds
        x_col: Column to use for x-axis
        y_col: Column to use for y-axis
        output_path: Path to save the plot
        highlight_threshold: Specific threshold to highlight
        figsize: Figure size as (width, height)
        
    Returns:
        Matplotlib Figure object
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Check if columns exist
    if x_col not in metrics_df.columns or y_col not in metrics_df.columns:
        raise ValueError(f"Columns {x_col} and/or {y_col} not found in metrics DataFrame")
    
    # Scale ROI to percentage if needed
    x_data = metrics_df[x_col]
    if x_col == 'roi':
        x_data = x_data * 100  # Convert to percentage
    
    # Create scatter plot with threshold as color
    scatter = ax.scatter(
        x_data, 
        metrics_df[y_col],
        c=metrics_df['threshold'],
        cmap='viridis',
        s=60,
        alpha=0.8
    )
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Threshold')
    
    # Identify Pareto-optimal points
    # A point is Pareto-optimal if no other point dominates it
    # (i.e., no other point is better in both dimensions)
    pareto_points = []
    
    # Sort by x_col (ascending) to find Pareto frontier
    sorted_df = metrics_df.sort_values(x_col)
    
    # Initialize with the point having the highest y_col value
    best_y = float('-inf')
    
    for idx, row in sorted_df.iterrows():
        if row[y_col] > best_y:
            best_y = row[y_col]
            pareto_points.append((row[x_col], row[y_col], row['threshold']))
    
    # Convert to arrays for plotting
    pareto_x = [p[0] for p in pareto_points]
    pareto_y = [p[1] for p in pareto_points]
    pareto_thresholds = [p[2] for p in pareto_points]
    
    # If x is ROI, convert to percentage for Pareto points too
    if x_col == 'roi':
        pareto_x = [x * 100 for x in pareto_x]
    
    # Plot Pareto frontier
    ax.plot(pareto_x, pareto_y, 'r--', linewidth=2, label='Pareto Frontier')
    
    # Highlight Pareto points
    ax.scatter(pareto_x, pareto_y, c='red', s=100, zorder=5, alpha=0.8)
    
    # Annotate Pareto points with thresholds
    for i, (x, y, t) in enumerate(zip(pareto_x, pareto_y, pareto_thresholds)):
        ax.annotate(
            f'{t:.2f}',
            xy=(x, y),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=8
        )
    
    # Highlight specific threshold if provided
    if highlight_threshold is not None:
        highlight_row = metrics_df[metrics_df['threshold'] == highlight_threshold]
        if not highlight_row.empty:
            x_val = highlight_row[x_col].values[0]
            if x_col == 'roi':
                x_val = x_val * 100
            
            y_val = highlight_row[y_col].values[0]
            
            ax.scatter(
                [x_val], 
                [y_val],
                c='purple',
                s=150,
                marker='*',
                label=f'Selected Threshold ({highlight_threshold:.2f})',
                zorder=10
            )
    
    # Set labels and title
    ax.set_xlabel(f"{x_col.upper() if x_col == 'roi' else x_col.capitalize()} {'(%)' if x_col == 'roi' else ''}")
    ax.set_ylabel(y_col.capitalize())
    ax.set_title(f"Pareto Frontier: {y_col.capitalize()} vs {x_col.upper() if x_col == 'roi' else x_col.capitalize()}")
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add legend
    ax.legend(loc='best')
    
    plt.tight_layout()
    
    # Save figure if output_path provided
    if output_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save figure
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Pareto frontier plot saved to {output_path}")
    
    return fig
