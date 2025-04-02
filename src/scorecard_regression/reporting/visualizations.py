"""
Visualization functions for the regression reporting module.

This module provides functions to create standardized visualizations for report generation.
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any

# Set default style
plt.style.use('default')
sns.set_theme(style="whitegrid")

# Customize matplotlib defaults for report-quality figures
plt.rcParams['figure.figsize'] = (8, 6)
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 16

def create_actual_vs_predicted_plot(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
    title: str = "Actual vs Predicted Values",
    sample_points: Optional[int] = 1000,
    add_diagonal: bool = True,
    add_error_bands: bool = True,
    alpha: float = 0.5,
    figsize: Tuple[int, int] = (8, 6),
    color: str = "#3d85c6",
    output_path: Optional[str] = None
) -> plt.Figure:
    """
    Create a scatter plot of actual vs predicted values.
    
    Args:
        y_true: Array of actual values
        y_pred: Array of predicted values
        title: Plot title
        sample_points: Number of points to sample (to avoid overcrowding)
        add_diagonal: Whether to add a 45-degree diagonal line
        add_error_bands: Whether to add error bands
        alpha: Transparency of points
        figsize: Figure size
        color: Point color
        output_path: Path to save the figure
        
    Returns:
        Matplotlib figure object
    """
    # Convert to numpy arrays
    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)
    
    # Subsample points if needed
    if sample_points and len(y_true_np) > sample_points:
        indices = np.random.choice(len(y_true_np), sample_points, replace=False)
        y_true_np = y_true_np[indices]
        y_pred_np = y_pred_np[indices]
    
    # Calculate error
    error = y_pred_np - y_true_np
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot scatter points
    scatter = ax.scatter(y_true_np, y_pred_np, alpha=alpha, color=color, edgecolor='none')
    
    # Add a 45-degree diagonal line
    if add_diagonal:
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        lims = [min(x_min, y_min), max(x_max, y_max)]
        ax.plot(lims, lims, 'r--', alpha=0.8, lw=2, label='Perfect prediction')
        ax.set_xlim(lims)
        ax.set_ylim(lims)
    
    # Add error bands
    if add_error_bands:
        # Calculate standard deviation of error
        error_std = np.std(error)
        # Add bands at ±1σ and ±2σ
        for n_std in [1, 2]:
            # Upper band
            upper_x = np.linspace(min(y_true_np), max(y_true_np), 100)
            upper_y = upper_x + n_std * error_std
            ax.plot(upper_x, upper_y, 'k--', alpha=0.3, lw=1)
            
            # Lower band
            lower_x = np.linspace(min(y_true_np), max(y_true_np), 100)
            lower_y = lower_x - n_std * error_std
            ax.plot(lower_x, lower_y, 'k--', alpha=0.3, lw=1)
            
            # Add label for the first band only
            if n_std == 1:
                ax.text(upper_x[-1], upper_y[-1], f'+{n_std}σ', 
                       ha='right', va='bottom', fontsize=9, alpha=0.6)
                ax.text(lower_x[-1], lower_y[-1], f'-{n_std}σ', 
                       ha='right', va='top', fontsize=9, alpha=0.6)
    
    # Calculate metrics to display
    r2 = np.corrcoef(y_true_np, y_pred_np)[0, 1]**2
    rmse = np.sqrt(np.mean((y_pred_np - y_true_np)**2))
    mae = np.mean(np.abs(y_pred_np - y_true_np))
    
    # Create textbox with metrics
    metric_text = f"$R^2$: {r2:.4f}\nRMSE: {rmse:.4f}\nMAE: {mae:.4f}"
    props = dict(boxstyle='round', facecolor='white', alpha=0.7)
    ax.text(0.05, 0.95, metric_text, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', bbox=props)
    
    # Set labels and title
    ax.set_xlabel('Actual Values')
    ax.set_ylabel('Predicted Values')
    ax.set_title(title)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure if path provided
    if output_path:
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig

def create_error_distribution_plot(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
    title: str = "Error Distribution",
    bins: int = 50,
    add_kde: bool = True,
    add_statistics: bool = True,
    figsize: Tuple[int, int] = (8, 6),
    color: str = "#3d85c6",
    output_path: Optional[str] = None
) -> plt.Figure:
    """
    Create a histogram of prediction errors.
    
    Args:
        y_true: Array of actual values
        y_pred: Array of predicted values
        title: Plot title
        bins: Number of histogram bins
        add_kde: Whether to add a KDE curve
        add_statistics: Whether to add error statistics
        figsize: Figure size
        color: Main color for the plot
        output_path: Path to save the figure
        
    Returns:
        Matplotlib figure object
    """
    # Convert to numpy arrays
    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)
    
    # Calculate error
    error = y_pred_np - y_true_np
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot histogram with KDE
    if add_kde:
        sns.histplot(error, bins=bins, kde=True, color=color, ax=ax)
    else:
        sns.histplot(error, bins=bins, color=color, ax=ax)
    
    # Add vertical line at zero
    ax.axvline(0, color='r', linestyle='--', alpha=0.7, label='Zero Error')
    
    # Add error statistics
    if add_statistics:
        mean_error = np.mean(error)
        std_error = np.std(error)
        
        # Add vertical lines at mean and ±1σ
        ax.axvline(mean_error, color='green', linestyle='-', alpha=0.7, label=f'Mean Error: {mean_error:.4f}')
        ax.axvline(mean_error + std_error, color='k', linestyle=':', alpha=0.7, label=f'+1σ: {mean_error + std_error:.4f}')
        ax.axvline(mean_error - std_error, color='k', linestyle=':', alpha=0.7, label=f'-1σ: {mean_error - std_error:.4f}')
        
        # Create textbox with statistics
        stats_text = (f"Mean: {mean_error:.4f}\n"
                      f"Std Dev: {std_error:.4f}\n"
                      f"Min: {np.min(error):.4f}\n"
                      f"Max: {np.max(error):.4f}")
        props = dict(boxstyle='round', facecolor='white', alpha=0.7)
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
              verticalalignment='top', bbox=props)
    
    # Set labels and title
    ax.set_xlabel('Prediction Error (Predicted - Actual)')
    ax.set_ylabel('Frequency')
    ax.set_title(title)
    
    # Add legend
    ax.legend()
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure if path provided
    if output_path:
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig

def create_threshold_performance_plot(
    threshold_df: pd.DataFrame,
    optimal_threshold: float,
    metrics: List[str] = ['actual_repayment_rate', 'approval_rate', 'default_rate'],
    title: str = "Performance Metrics Across Thresholds",
    figsize: Tuple[int, int] = (10, 6),
    output_path: Optional[str] = None
) -> plt.Figure:
    """
    Create a line plot of performance metrics across thresholds.
    
    Args:
        threshold_df: DataFrame with threshold analysis results
        optimal_threshold: Optimal threshold value
        metrics: List of metrics to plot
        title: Plot title
        figsize: Figure size
        output_path: Path to save the figure
        
    Returns:
        Matplotlib figure object
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Define colors for each metric
    colors = {
        'actual_repayment_rate': '#28a745',  # Green
        'approval_rate': '#3d85c6',          # Blue
        'default_rate': '#dc3545',           # Red
        'total_profit': '#f39c12',           # Orange
        'net_profit_margin': '#9b59b6',      # Purple
        'money_left_on_table': '#95a5a6'     # Gray
    }
    
    # Plot each metric
    for metric in metrics:
        if metric in threshold_df.columns:
            ax.plot(threshold_df['threshold'], threshold_df[metric], marker='o', 
                   label=metric.replace('_', ' ').title(), color=colors.get(metric, 'black'))
    
    # Add vertical line at optimal threshold
    ax.axvline(optimal_threshold, color='red', linestyle='--', alpha=0.7, 
              label=f'Optimal Threshold: {optimal_threshold:.2f}')
    
    # Set labels and title
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Metric Value')
    ax.set_title(title)
    
    # Format y-axis as percentage if metrics are rates
    rate_metrics = ['actual_repayment_rate', 'approval_rate', 'default_rate', 'net_profit_margin']
    if set(metrics).issubset(set(rate_metrics)):
        ax.set_ylim(0, 1)
        ax.set_ylabel('Rate (percentage)')
        # Format y-axis as percentage
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add legend
    ax.legend(loc='best')
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure if path provided
    if output_path:
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig

def create_profit_curve_plot(
    threshold_df: pd.DataFrame,
    optimal_threshold: float,
    profit_col: str = 'actual_profit',
    title: str = "Profit Curve",
    figsize: Tuple[int, int] = (10, 6),
    output_path: Optional[str] = None
) -> plt.Figure:
    """
    Create a profit curve plot showing profit vs threshold.
    
    Args:
        threshold_df: DataFrame with threshold analysis results
        optimal_threshold: Optimal threshold value
        profit_col: Column name for profit values
        title: Plot title
        figsize: Figure size
        output_path: Path to save the figure
        
    Returns:
        Matplotlib figure object
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot profit curve
    ax.plot(threshold_df['threshold'], threshold_df[profit_col], marker='o', 
           label=profit_col.replace('_', ' ').title(), color='#28a745')
    
    # Add vertical line at optimal threshold
    ax.axvline(optimal_threshold, color='red', linestyle='--', alpha=0.7, 
              label=f'Optimal Threshold: {optimal_threshold:.2f}')
    
    # Find profit at optimal threshold - find the closest threshold instead of exact match
    # to avoid issues with floating point precision
    idx = (threshold_df['threshold'] - optimal_threshold).abs().idxmin()
    optimal_profit = threshold_df.loc[idx, profit_col]
    
    # Add point at optimal threshold
    ax.scatter([optimal_threshold], [optimal_profit], color='red', s=100, zorder=5)
    
    # Add annotation for optimal profit
    ax.annotate(f'Optimal Profit: {optimal_profit:,.0f}', 
               xy=(optimal_threshold, optimal_profit),
               xytext=(optimal_threshold + 0.05, optimal_profit * 0.9),
               arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
               bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))
    
    # Set labels and title
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Profit')
    ax.set_title(title)
    
    # Format y-axis for large values
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:,.0f}'))
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add legend
    ax.legend(loc='best')
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure if path provided
    if output_path:
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig

def create_margin_analysis_plot(
    margin_threshold_df: pd.DataFrame,
    metric: str = 'actual_profit',
    title: str = "Profit by Margin and Threshold",
    cmap: str = 'viridis',
    figsize: Tuple[int, int] = (10, 8),
    output_path: Optional[str] = None
) -> plt.Figure:
    """
    Create a heatmap showing a metric across thresholds and margins.
    
    Args:
        margin_threshold_df: DataFrame with margin and threshold analysis results
        metric: Metric to visualize (column in DataFrame)
        title: Plot title
        cmap: Colormap for heatmap
        figsize: Figure size
        output_path: Path to save the figure
        
    Returns:
        Matplotlib figure object
    """
    # Pivot data for heatmap
    pivot_df = margin_threshold_df.pivot(
        index='gross_margin',
        columns='threshold',
        values=metric
    )
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    sns.heatmap(
        pivot_df,
        annot=True,
        fmt='.4g',
        cmap=cmap,
        linewidths=0.5,
        ax=ax,
        cbar_kws={'label': metric.replace('_', ' ').title()}
    )
    
    # Set labels and title
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Gross Margin')
    ax.set_title(title)
    
    # Format y-axis as percentage if values are small
    if pivot_df.index.max() <= 1:
        ax.set_yticklabels([f'{float(y):.0%}' for y in pivot_df.index])
    
    # Find optimal combination
    if 'actual_profit' in margin_threshold_df.columns:
        # Get the optimal combination (highest profit)
        optimal_row = margin_threshold_df.loc[margin_threshold_df['actual_profit'].idxmax()]
        optimal_margin = optimal_row['gross_margin']
        optimal_threshold = optimal_row['threshold']
        
        # Find the position in the heatmap
        optimal_margin_idx = pivot_df.index.get_loc(optimal_margin)
        optimal_threshold_idx = pivot_df.columns.get_loc(optimal_threshold)
        
        # Add a red rectangle around the optimal cell
        ax.add_patch(plt.Rectangle(
            (optimal_threshold_idx, optimal_margin_idx),
            1, 1, fill=False, edgecolor='red', lw=2
        ))
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure if path provided
    if output_path:
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig

def create_metrics_heatmap(
    metrics: Dict[str, Dict[str, float]],
    title: str = "Performance Metrics",
    cmap: str = 'viridis',
    figsize: Tuple[int, int] = (10, 8),
    output_path: Optional[str] = None
) -> plt.Figure:
    """
    Create a heatmap of performance metrics for different datasets.
    
    Args:
        metrics: Nested dictionary of metrics {dataset: {metric: value}}
        title: Plot title
        cmap: Colormap for heatmap
        figsize: Figure size
        output_path: Path to save the figure
        
    Returns:
        Matplotlib figure object
    """
    # Convert to DataFrame
    df = pd.DataFrame(metrics).T
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    sns.heatmap(
        df,
        annot=True,
        fmt='.4f',
        cmap=cmap,
        linewidths=0.5,
        ax=ax
    )
    
    # Set labels and title
    ax.set_xlabel('Metric')
    ax.set_ylabel('Dataset')
    ax.set_title(title)
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure if path provided
    if output_path:
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig

def create_feature_importance_plot(
    importances: Dict[str, float],
    title: str = "Feature Importance",
    n_features: int = 20,
    figsize: Tuple[int, int] = (10, 8),
    color: str = "#3d85c6",
    output_path: Optional[str] = None
) -> plt.Figure:
    """
    Create a horizontal bar plot of feature importances.
    
    Args:
        importances: Dictionary of feature importances {feature: importance}
        title: Plot title
        n_features: Number of top features to include
        figsize: Figure size
        color: Bar color
        output_path: Path to save the figure
        
    Returns:
        Matplotlib figure object
    """
    # Convert to DataFrame and sort
    df = pd.DataFrame({
        'feature': list(importances.keys()),
        'importance': list(importances.values())
    }).sort_values('importance', ascending=False)
    
    # Limit to top n features
    if len(df) > n_features:
        df = df.head(n_features)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot horizontal bars
    bars = ax.barh(df['feature'], df['importance'], color=color)
    
    # Add value labels
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{width:.4f}', ha='left', va='center')
    
    # Set labels and title
    ax.set_xlabel('Importance')
    ax.set_title(title)
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure if path provided
    if output_path:
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig
