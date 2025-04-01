"""
Enhanced plotting functions for scorecard modeling using Seaborn.

This module provides improved visualizations for WOE binning and other scorecard
analytics using Seaborn's advanced styling capabilities with Tupande theme.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Union
import re
import os
import textwrap
from matplotlib.ticker import MaxNLocator

# Tupande theme colors
TUPANDE_BLUE = '#007bff'
TUPANDE_RED = '#dc3545'
TUPANDE_GREEN = '#28a745'
TUPANDE_GRAY = '#6c757d'
TUPANDE_LIGHT_BLUE = '#4dabf7'
TUPANDE_LIGHT_RED = '#f86a7e'

# Set default style
plt.style.use('default')
sns.set_theme(style="whitegrid")

def create_woe_plot(
    bin_df: pd.DataFrame, 
    var_name: str, 
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 7),
    palette: Dict[str, str] = {'bad': TUPANDE_RED, 'good': TUPANDE_GREEN},
    dpi: int = 100,
    show_group_details: bool = True,
    max_bin_label_length: int = 15,
    group_large_bins: bool = True,
    max_bins_display: int = 10
) -> plt.Figure:
    """
    Create an enhanced WOE plot for a variable using Seaborn.
    
    Args:
        bin_df: Binning DataFrame for a single variable
        var_name: Name of the variable
        output_path: Optional path to save the plot
        figsize: Figure size as (width, height)
        palette: Color palette for good/bad counts
        dpi: Resolution of the output image
        show_group_details: Whether to show detailed group information for categorical variables
        
    Returns:
        Matplotlib Figure object
    """
    # Check if we have valid binning data
    if bin_df is None or len(bin_df) == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, f"No valid binning data for {var_name}", 
                ha='center', va='center', fontsize=12)
        if output_path:
            plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        return fig
    
    # Extract key information
    iv_value = bin_df['bin_iv'].sum()
    is_categorical = False
    group_details = {}
    
    # Handle large number of bins by grouping if needed
    need_bin_grouping = False
    if group_large_bins and len(bin_df) > max_bins_display:
        need_bin_grouping = True
    
    # Check if this is a categorical variable with bin group names like 'bin_0', 'bin_1', etc.
    if 'bin' in bin_df.columns:
        bin_values = bin_df['bin'].values
        if bin_values[0] == 'missing':
            bin_values = bin_values[1:]
        
        # Check for categorical bins like 'bin_0', 'bin_1', etc.
        pattern = re.compile(r'bin_\d+')
        if any(isinstance(b, str) and pattern.match(b) for b in bin_values if isinstance(b, str)):
            is_categorical = True
            
            # For categorical variables, extract the actual values in each bin
            if 'variable' in bin_df.columns and 'bin_description' in bin_df.columns:
                for _, row in bin_df.iterrows():
                    if isinstance(row['bin'], str) and pattern.match(row['bin']):
                        bin_name = row['bin']
                        description = row['bin_description'] 
                        if isinstance(description, str) and ":" in description:
                            try:
                                # Extract categories from description like "category:cat1,cat2,cat3"
                                categories = description.split(":", 1)[1].strip()
                                group_details[bin_name] = categories
                            except:
                                group_details[bin_name] = description
    
    # Create figure
    fig, ax1 = plt.subplots(figsize=figsize)
    
    # Calculate count percentages
    bin_df['total_count'] = bin_df['good'] + bin_df['bad']
    bin_df['good_pct'] = bin_df['good'] / bin_df['total_count']
    bin_df['bad_pct'] = bin_df['bad'] / bin_df['total_count']
    bin_df['total_pct'] = bin_df['total_count'] / bin_df['total_count'].sum()
    
    # Get x-axis labels
    if is_categorical:
        x_labels = bin_df['bin'].tolist()
    else:
        # For numeric variables, we'll format the bins as ranges
        x_labels = []
        for _, row in bin_df.iterrows():
            if row['bin'] == 'missing':
                x_labels.append('missing')
            elif isinstance(row['bin'], str) and '[' in row['bin'] and ']' in row['bin']:
                x_labels.append(row['bin'])
            else:
                # Check if 'lower' and 'upper' columns exist in the dataframe
                if 'lower' in row.index and 'upper' in row.index:
                    lower = row['lower']
                    upper = row['upper']
                    if pd.isna(lower):
                        bin_label = f"< {upper:.2f}"
                    elif pd.isna(upper):
                        bin_label = f"> {lower:.2f}"
                    else:
                        bin_label = f"[{lower:.2f},{upper:.2f})"
                # Fallback if lower/upper columns don't exist
                else:
                    # Use bin value directly or as a string representation
                    bin_label = str(row['bin'])
                x_labels.append(bin_label)
    
    # Build the dataset for plotting
    plot_data = []
    for i, row in bin_df.iterrows():
        plot_data.append({'bin': x_labels[i], 'count': row['bad'], 'type': 'bad', 
                         'pct': row['bad_pct'], 'woe': row['woe'], 'total': row['total_count'],
                         'total_pct': row['total_pct']})
        plot_data.append({'bin': x_labels[i], 'count': row['good'], 'type': 'good', 
                         'pct': row['good_pct'], 'woe': row['woe'], 'total': row['total_count'],
                         'total_pct': row['total_pct']})
    
    plot_df = pd.DataFrame(plot_data)
    
    # Adjust the bin labels for better display if too many or too long
    if need_bin_grouping:
        # Create simplified labels for plotting
        simple_labels = []
        for i, label in enumerate(x_labels):
            if len(label) > max_bin_label_length:
                simple_labels.append(f"Group_{i+1}")
                group_details[f"Group_{i+1}"] = label
            else:
                simple_labels.append(label)
        
        # Update the plot data with simplified labels
        for i, item in enumerate(plot_data):
            item_idx = i // 2  # Each bin has 2 entries (good/bad)
            if item_idx < len(simple_labels):
                item['bin'] = simple_labels[item_idx]
        
        # Update DataFrame
        plot_df = pd.DataFrame(plot_data)
        
        # For grouped bins, always show group details
        show_group_details = True
        
    # Create stacked bar chart with Tupande theme colors
    sns.barplot(
        x='bin', 
        y='count', 
        hue='type',
        data=plot_df,
        palette=palette,
        ax=ax1
    )
    
    # Add count labels on bars
    for i, row in bin_df.iterrows():
        # Skip if counts are too small to show labels
        if row['total_count'] < 5:
            continue
            
        # Position for bad count (top of the stacked bar)
        ax1.text(
            i, 
            row['total_count'] + 3, 
            f"{row['bad_pct']*100:.1f}%, {row['bad']}",
            ha='center', 
            va='bottom',
            fontsize=9,
            fontweight='bold'
        )
        
        # Position for good count (middle of the good portion)
        if row['good'] > 5:  # Only if there are enough good samples
            ax1.text(
                i, 
                row['good'] / 2, 
                f"{row['good_pct']*100:.1f}%, {row['good']}",
                ha='center', 
                va='center',
                fontsize=9,
                fontweight='bold',
                color='white' if row['good_pct'] > 0.3 else 'black'
            )
    
    # Create a second y-axis for default rate values
    ax2 = ax1.twinx()
    
    # Calculate default rates (bad rates) as percentages for each bin
    default_rates = bin_df['bad_pct'].values * 100
    
    # Plot default rate line with Tupande theme blue
    default_line = ax2.plot(range(len(x_labels)), default_rates, 'o-', 
                        color=TUPANDE_BLUE, linewidth=2.5, markersize=8)
    
    # Add default rate labels
    for i, rate in enumerate(default_rates):
        ax2.text(i, rate + 2, f"{rate:.1f}%", ha='center', fontsize=9, 
                fontweight='bold', color=TUPANDE_BLUE)
    
    # Set axis labels and title
    ax1.set_xlabel('Bin', fontsize=12)
    ax1.set_ylabel('Bin Count Distribution', fontsize=12)
    ax2.set_ylabel('Default Rate (%)', fontsize=12, color=TUPANDE_BLUE)
    
    # Set y-axis range for default rate to start from 0
    ax2.set_ylim(0, max(default_rates) * 1.2)
    
    title = f"{var_name} (IV:{iv_value:.3f}) WOE Binning Plot"
    plt.title(title, fontsize=14, pad=20)
    
    # Customize tick labels
    if len(x_labels) > 5:
        ax1.set_xticklabels(x_labels, rotation=45, ha='right')
    else:
        ax1.set_xticklabels(x_labels)
    
    # Set tick colors for the second y-axis
    ax2.tick_params(axis='y', colors='#375E97')
    
    # Add legend
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles, ['Bad', 'Good'], loc='upper right')
    
    # Add group details for categorical variables or grouped bins
    if show_group_details and group_details:
        details_text = "Group Mappings:\n"
        for bin_name, categories in group_details.items():
            # Wrap long category lists
            wrapped_cats = textwrap.fill(str(categories), width=80)
            details_text += f"{bin_name}: {wrapped_cats}\n"
        
        # Add a styled text box at the bottom of the plot with Tupande theme
        plt.figtext(0.1, 0.01, details_text, fontsize=9, 
                   bbox=dict(facecolor=f"{TUPANDE_BLUE}15", alpha=0.9,
                             edgecolor=TUPANDE_BLUE, boxstyle="round,pad=0.5"))
        
        # Make room for the text box - adjust based on number of groups
        extra_space = min(0.3, 0.1 + 0.02 * len(group_details))
        plt.subplots_adjust(bottom=extra_space)
    
    # Grid and styling
    ax1.grid(True, linestyle='--', alpha=0.7)
    fig.patch.set_facecolor('#f8f9fa')  # Light background for the figure
    plt.tight_layout()
    
    # Save plot if output path provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    
    return fig

def create_woe_plots_grid(
    bins: Dict[str, pd.DataFrame],
    output_dir: str,
    rows: int = 3, 
    cols: int = 2,
    figsize: Tuple[int, int] = (18, 24),
    dpi: int = 120,
    min_iv: float = 0.0,
    summary_grid: bool = True,
    group_large_bins: bool = True,
    max_bins_display: int = 10
) -> List[str]:
    """
    Create a grid of WOE plots for multiple variables with Tupande theme styling.
    
    Args:
        bins: Dictionary of bin DataFrames by variable
        output_dir: Directory to save the plots
        rows: Number of rows in the grid
        cols: Number of columns in the grid
        figsize: Figure size as (width, height)
        dpi: Resolution of the output image
        min_iv: Minimum information value to include in plots
        summary_grid: Whether to create a summary grid of all plots
        group_large_bins: Whether to group large bins for better display
        max_bins_display: Maximum number of bins to display before grouping
        
    Returns:
        List of paths to the saved plot files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Sort variables by information value (descending)
    sorted_vars = []
    for var, bin_df in bins.items():
        iv = bin_df['bin_iv'].sum() if 'bin_iv' in bin_df.columns else 0
        # Filter by minimum IV value
        if iv >= min_iv:
            sorted_vars.append((var, iv))
    
    sorted_vars.sort(key=lambda x: x[1], reverse=True)
    
    # Generate individual plots and collect paths
    plot_paths = []
    for var, iv in sorted_vars:
        plot_path = os.path.join(output_dir, f"{var}_woe_plot.png")
        create_woe_plot(
            bins[var], 
            var, 
            plot_path,
            palette={'bad': TUPANDE_RED, 'good': TUPANDE_GREEN},
            group_large_bins=group_large_bins,
            max_bins_display=max_bins_display
        )
        plot_paths.append(plot_path)
    
    # Create a summary grid of all plots if requested
    if summary_grid and sorted_vars:
        # Calculate grid dimensions
        n_plots = len(sorted_vars)
        grid_rows = min(rows, (n_plots + cols - 1) // cols)
        grid_cols = min(cols, n_plots)
        
        # Create grid figure
        fig, axes = plt.subplots(grid_rows, grid_cols, figsize=figsize)
        fig.suptitle('WOE Binning Analysis - Top Variables by Information Value', 
                    fontsize=20, fontweight='bold', color=TUPANDE_BLUE)
        
        # Flatten axes array for easier indexing if multiple rows and columns
        if grid_rows > 1 and grid_cols > 1:
            axes = axes.flatten()
        elif grid_rows == 1 and grid_cols > 1:
            axes = axes  # Already a 1D array
        elif grid_rows > 1 and grid_cols == 1:
            axes = axes.flatten()
        else:
            axes = [axes]  # Single subplot
        
        # Add each variable plot to the grid
        for i, (var, iv) in enumerate(sorted_vars[:grid_rows*grid_cols]):
            if i < len(axes):
                # Add image to subplot
                img = plt.imread(os.path.join(output_dir, f"{var}_woe_plot.png"))
                axes[i].imshow(img)
                axes[i].set_title(f"{var} (IV: {iv:.3f})", fontsize=12, color=TUPANDE_BLUE)
                axes[i].axis('off')
        
        # Hide any unused subplots
        for i in range(len(sorted_vars), len(axes)):
            axes[i].axis('off')
        
        # Add a footer with timestamp
        plt.figtext(0.5, 0.01, f"Generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
                   ha='center', fontsize=10, color=TUPANDE_GRAY)
        
        # Adjust layout and save
        plt.tight_layout()
        plt.subplots_adjust(top=0.95, bottom=0.05)
        summary_path = os.path.join(output_dir, "woe_plots_summary.png")
        plt.savefig(summary_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        
        # Add summary path to returned paths
        plot_paths.append(summary_path)
    
    return plot_paths
