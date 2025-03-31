"""
Helper functions for report generation in the scorecard modeling process.
"""

import os
import json
import pandas as pd
from typing import Dict, Optional
from .report_generator import ScorecardReport

def initialize_modeling_report(version_path: str) -> ScorecardReport:
    """
    Initialize a new scorecard modeling report.
    
    Args:
        version_path: Path to the version directory
        
    Returns:
        ScorecardReport object
    """
    report_path = os.path.join(version_path, "modeling_report.pdf")
    return ScorecardReport(report_path)

def add_data_inspection_section(report: ScorecardReport, inspection_results: Dict):
    """
    Add data inspection results to the report.
    
    Args:
        report: ScorecardReport object
        inspection_results: Results from data inspection
    """
    report.add_section("Data Inspection", "Summary of data quality checks and variable characteristics")
    
    # Variable types
    if 'variable_types' in inspection_results:
        var_type_data = [[var, type_] for var, type_ in inspection_results['variable_types'].items()]
        report.add_table(var_type_data, ['Variable', 'Type'], "Variable Types")
    
    # Missing values
    if 'missing_rates' in inspection_results:
        missing_data = [[var, f"{rate:.2%}"] for var, rate in inspection_results['missing_rates'].items()]
        report.add_table(missing_data, ['Variable', 'Missing Rate'], "Missing Value Analysis")
    
    # Date-like columns
    if 'date_like_columns' in inspection_results:
        if inspection_results['date_like_columns']:
            report.add_subsection("Date-like Columns", 
                                "The following columns were identified as containing date values:\n" + 
                                ", ".join(inspection_results['date_like_columns']))

def add_target_preparation_section(report: ScorecardReport, cutoff_info: Dict, binary_target_stats: Dict):
    """
    Add target preparation information to the report.
    
    Args:
        report: ScorecardReport object
        cutoff_info: Information about the optimal cutoff selection
        binary_target_stats: Statistics about the binary target variable
    """
    report.add_section("Target Preparation", "Analysis and preparation of the target variable")
    
    # Cutoff selection
    report.add_subsection("Optimal Cutoff Selection", 
                         f"Selected cutoff: {cutoff_info['optimal_cutoff']:.3f}")
    
    # Target distribution
    if 'distribution' in binary_target_stats:
        dist_data = [
            ['Good', f"{binary_target_stats['distribution']['good_count']:,}",
             f"{binary_target_stats['distribution']['good_pct']:.1%}"],
            ['Bad', f"{binary_target_stats['distribution']['bad_count']:,}",
             f"{binary_target_stats['distribution']['bad_pct']:.1%}"]
        ]
        report.add_table(dist_data, ['Class', 'Count', 'Percentage'], "Target Distribution")

def add_variable_selection_section(report: ScorecardReport, selection_results: Dict):
    """
    Add variable selection results to the report.
    
    Args:
        report: ScorecardReport object
        selection_results: Results from variable selection process
    """
    report.add_section("Variable Selection", "Feature selection process and results")
    
    # Selected variables
    if 'selected_variables' in selection_results:
        report.add_subsection("Selected Variables", 
                            f"Number of selected variables: {len(selection_results['selected_variables'])}")
        # Get IV values from information_values.csv
        iv_path = os.path.join(selection_results['output_dir'], 'information_values.csv')
        if os.path.exists(iv_path):
            iv_data = pd.read_csv(iv_path)
            # Print column names for debugging
            print("Available columns:", iv_data.columns.tolist())
            # Assuming the first column is the variable name and second is the IV value
            var_data = [[row[0], f"{row[1]:.4f}"] for _, row in iv_data.iterrows()]
            report.add_table(var_data, ['Variable', 'Information Value'], "Variable Importance")
    
    # Excluded variables
    if 'excluded_variables' in selection_results:
        report.add_subsection("Excluded Variables", 
                            "The following variables were excluded from modeling:\n" + 
                            ", ".join(selection_results['excluded_variables']))

def add_woe_binning_section(report: ScorecardReport, binning_summary: Dict, woe_plot_dir: Optional[str] = None):
    """
    Add WOE binning results to the report.
    
    Args:
        report: ScorecardReport object
        binning_summary: Summary of binning results
        woe_plot_dir: Directory containing WOE plot images
    """
    report.add_section("WOE Binning", "Weight of Evidence binning analysis")
    
    # Binning summary
    if 'bin_stats' in binning_summary:
        bin_data = [[var, stats['n_bins'], f"{stats['iv']:.4f}"] 
                   for var, stats in binning_summary['bin_stats'].items()]
        report.add_table(bin_data, ['Variable', 'Number of Bins', 'IV'], 
                        "Binning Summary")
    
    # WOE plots
    if woe_plot_dir and os.path.isdir(woe_plot_dir):
        # Get all PNG files from the WOE plots directory
        plot_files = [os.path.join(woe_plot_dir, f) for f in os.listdir(woe_plot_dir) 
                     if f.endswith('_woe_plot.png')]
        
        if plot_files:
            # Extract variable names from filenames
            var_names = [os.path.basename(f).replace('_woe_plot.png', '') for f in plot_files]
            captions = [f"WOE Binning - {var}" for var in var_names]
            
            # Add the plots to the report
            report.add_subsection("WOE Binning Plots", 
                               "Relationship between binned variables and Weight of Evidence")
            report.add_woe_plots_grid(plot_files, captions)
        else:
            print(f"Warning: No WOE plot images found in {woe_plot_dir}")
    elif woe_plot_dir:
        print(f"Warning: WOE plot directory not found: {woe_plot_dir}")

def add_model_development_section(report: ScorecardReport, model_results: Dict):
    """
    Add model development results to the report.
    
    Args:
        report: ScorecardReport object
        model_results: Results from model development
    """
    report.add_section("Model Development", "Scorecard model development and results")
    
    # Model configuration
    config = {
        'Classifier Type': model_results.get('classifier_type', 'Unknown'),
        'Handle Missing': str(model_results.get('handle_missing', False))
    }
    if 'classifier_params' in model_results:
        config.update(model_results['classifier_params'])
    
    # Model coefficients
    if 'coefficients' in model_results:
        coef_data = [[var, f"{coef:.4f}"] for var, coef in model_results['coefficients'].items()]
        report.add_table(coef_data, ['Variable', 'Coefficient'], "Model Coefficients")
    
    # Add model summary
    model_info = {
        'config': config,
        'metrics': model_results.get('metrics', {}),
        'feature_importance': model_results.get('feature_importance', {})
    }
    report.add_model_summary(model_info)

def add_performance_evaluation_section(report: ScorecardReport, performance_results: Dict):
    """
    Add model performance evaluation results to the report.
    
    Args:
        report: ScorecardReport object
        performance_results: Results from model performance evaluation
    """
    report.add_section("Performance Evaluation", "Model performance metrics and analysis")
    
    # Training performance
    if 'train_metrics' in performance_results:
        train_data = [[metric, f"{value:.4f}"] 
                     for metric, value in performance_results['train_metrics'].items()]
        report.add_table(train_data, ['Metric', 'Value'], "Training Performance")
    
    # Testing performance
    if 'test_metrics' in performance_results:
        test_data = [[metric, f"{value:.4f}"] 
                    for metric, value in performance_results['test_metrics'].items()]
        report.add_table(test_data, ['Metric', 'Value'], "Testing Performance")
    
    # PSI
    if 'psi' in performance_results:
        report.add_subsection("Population Stability Index", 
                            f"PSI: {performance_results['psi']:.4f}")
    
    # Performance plots
    if 'performance_plots' in performance_results:
        for plot_name, fig in performance_results['performance_plots'].items():
            report.add_plot(fig, plot_name)
