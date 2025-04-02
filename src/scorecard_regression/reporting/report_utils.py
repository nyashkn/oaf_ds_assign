"""
Helper functions for report generation in the regression modeling process.

This module provides specialized functions for adding structured sections to regression reports.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Any, Union

from .report_generator import RegressionReport
from .visualizations import (
    create_actual_vs_predicted_plot,
    create_error_distribution_plot,
    create_threshold_performance_plot,
    create_profit_curve_plot,
    create_margin_analysis_plot,
    create_metrics_heatmap,
    create_feature_importance_plot
)

def initialize_regression_report(
    version_path: str,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    author: Optional[str] = None
) -> RegressionReport:
    """
    Initialize a new regression modeling report.
    
    Args:
        version_path: Path to the version directory
        title: Optional custom title
        subtitle: Optional custom subtitle
        author: Optional author or team name
        
    Returns:
        RegressionReport object
    """
    # Create report path in the version directory's reports folder
    report_path = os.path.join(version_path, "reports", "regression_modeling_report.pdf")
    os.makedirs(os.path.join(version_path, "reports"), exist_ok=True)
    
    # Set defaults if not provided
    title = title or "Loan Repayment Rate Regression Model Report"
    subtitle = subtitle or "OneAcre Fund - Tupande Credit Risk Analysis"
    author = author or "Data Science Team"
    
    # Create report
    return RegressionReport(
        output_path=report_path,
        title=title,
        subtitle=subtitle,
        author=author
    )

def add_executive_summary_section(
    report: RegressionReport,
    summary_data: Dict[str, Any],
    include_plots: bool = True
) -> None:
    """
    Add executive summary section to the report.
    
    Args:
        report: RegressionReport object
        summary_data: Dictionary with summary information
        include_plots: Whether to include summary plots
    """
    # Add chapter
    report.add_chapter(
        "Executive Summary",
        "Overview of key model performance metrics and business recommendations."
    )
    
    # Add model performance summary
    if 'model_performance' in summary_data:
        perf = summary_data['model_performance']
        report.add_section("Model Performance Summary")
        
        # Create metrics table
        metrics_data = []
        if 'train' in perf:
            metrics_data.append(['Training', 
                                f"{perf['train'].get('r2', 0):.4f}", 
                                f"{perf['train'].get('rmse', 0):.4f}", 
                                f"{perf['train'].get('mae', 0):.4f}"])
        if 'test' in perf:
            metrics_data.append(['Testing', 
                               f"{perf['test'].get('r2', 0):.4f}", 
                               f"{perf['test'].get('rmse', 0):.4f}", 
                               f"{perf['test'].get('mae', 0):.4f}"])
        if 'holdout' in perf:
            metrics_data.append(['Holdout', 
                                f"{perf['holdout'].get('r2', 0):.4f}", 
                                f"{perf['holdout'].get('rmse', 0):.4f}", 
                                f"{perf['holdout'].get('mae', 0):.4f}"])
            
        report.add_table(
            data=metrics_data,
            headers=['Dataset', 'R²', 'RMSE', 'MAE'],
            title="Performance Metrics",
            highlight_max=[1],  # Highlight best R² value
            highlight_min=[2, 3]  # Highlight best RMSE and MAE values
        )
    
    # Add business impact summary
    if 'business_impact' in summary_data:
        impact = summary_data['business_impact']
        report.add_section("Business Impact")
        
        # Format and add thresholds and margins
        if 'optimal_threshold' in impact:
            report.add_paragraph(
                f"The optimal threshold for loan approval is <b>{impact['optimal_threshold']:.2f}</b>, "
                f"which yields a repayment rate of {impact.get('repayment_rate', 0):.1%} "
                f"with an approval rate of {impact.get('approval_rate', 0):.1%}."
            )
        
        if 'expected_profit' in impact:
            report.add_paragraph(
                f"Expected profit at this threshold is <b>{impact['expected_profit']:,.0f} KES</b> "
                f"with a net profit margin of {impact.get('profit_margin', 0):.1%}."
            )
    
    # Add feature importance summary
    if 'feature_importance' in summary_data and include_plots:
        report.add_section("Key Drivers")
        
        # Get top 5 features
        importance = summary_data['feature_importance']
        top_features = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5])
        
        # Create a small feature importance table
        feature_data = [[feat, f"{imp:.4f}"] for feat, imp in top_features.items()]
        report.add_table(
            data=feature_data,
            headers=['Feature', 'Importance'],
            title="Top 5 Features"
        )
        
        # Add explanatory text
        if feature_data:
            top_feature = feature_data[0][0]
            report.add_paragraph(
                f"The most important predictor of repayment rate is <b>{top_feature}</b>, "
                f"followed by {', '.join([row[0] for row in feature_data[1:]])}."
            )
    
    report.add_page_break()

def add_data_inspection_section(
    report: RegressionReport,
    inspection_results: Dict[str, Any],
    include_distributions: bool = True
) -> None:
    """
    Add data inspection results to the report.
    
    Args:
        report: RegressionReport object
        inspection_results: Results from data inspection
        include_distributions: Whether to include feature distributions
    """
    report.add_chapter(
        "Data Inspection and Preparation",
        "Analysis of data quality, missing values, and preprocessing steps."
    )
    
    # Dataset overview
    if 'shape' in inspection_results:
        report.add_section("Dataset Overview")
        rows, cols = inspection_results['shape']
        report.add_paragraph(
            f"The dataset contains <b>{rows:,}</b> loans with <b>{cols}</b> features."
        )
    
    # Missing values
    if 'missing_values' in inspection_results:
        report.add_section("Missing Values Analysis")
        
        # Create missing values table
        missing_data = []
        for col, count in inspection_results['missing_values'].items():
            if count > 0:
                pct = inspection_results.get('missing_pct', {}).get(col, 0)
                missing_data.append([col, count, f"{pct:.2%}"])
        
        if missing_data:
            # Sort by count descending
            missing_data.sort(key=lambda x: x[1], reverse=True)
            # Take top 10
            missing_data = missing_data[:10]
            
            report.add_table(
                data=missing_data,
                headers=['Feature', 'Missing Count', 'Missing %'],
                title="Top Features with Missing Values"
            )
            
            # Add text about how missing values were handled
            if 'missing_handling' in inspection_results:
                handling = inspection_results['missing_handling']
                methods = set(handling.values())
                report.add_paragraph(
                    f"Missing values were handled using the following methods: {', '.join(methods)}."
                )
        else:
            report.add_paragraph("No missing values found in the dataset.")
    
    # Feature types
    if 'dtypes' in inspection_results:
        report.add_section("Feature Types")
        
        # Get counts of each type
        type_counts = {}
        for dtype in inspection_results['dtypes'].values():
            if 'float' in dtype:
                type_counts['Numeric (float)'] = type_counts.get('Numeric (float)', 0) + 1
            elif 'int' in dtype:
                type_counts['Numeric (int)'] = type_counts.get('Numeric (int)', 0) + 1
            elif 'object' in dtype or 'string' in dtype:
                type_counts['Categorical'] = type_counts.get('Categorical', 0) + 1
            elif 'date' in dtype or 'time' in dtype:
                type_counts['Date/Time'] = type_counts.get('Date/Time', 0) + 1
            elif 'bool' in dtype:
                type_counts['Boolean'] = type_counts.get('Boolean', 0) + 1
            else:
                type_counts['Other'] = type_counts.get('Other', 0) + 1
        
        # Create feature types table
        type_data = [[type_name, count] for type_name, count in type_counts.items()]
        report.add_table(
            data=type_data,
            headers=['Type', 'Count'],
            title="Feature Types Distribution"
        )
        
    # Target distribution
    if 'target_stats' in inspection_results:
        report.add_section("Target Variable Distribution")
        
        stats = inspection_results['target_stats']
        stats_data = [
            ['Mean', f"{stats.get('mean', 0):.4f}"],
            ['Median', f"{stats.get('median', 0):.4f}"],
            ['Min', f"{stats.get('min', 0):.4f}"],
            ['Max', f"{stats.get('max', 0):.4f}"],
            ['Std Dev', f"{stats.get('std', 0):.4f}"]
        ]
        
        report.add_table(
            data=stats_data,
            headers=['Statistic', 'Value'],
            title="Target Variable Statistics"
        )
        
        # Add percentile information if available
        if 'percentiles' in stats:
            percentiles = stats['percentiles']
            percentile_data = []
            for p in [0, 25, 50, 75, 90, 95, 99, 100]:
                if p in percentiles:
                    percentile_data.append([f"{p}%", f"{percentiles[p]:.4f}"])
            
            if percentile_data:
                report.add_table(
                    data=percentile_data,
                    headers=['Percentile', 'Value'],
                    title="Target Percentiles"
                )
    
    # Preprocessing steps
    if 'preprocessing' in inspection_results:
        report.add_section("Preprocessing Steps")
        
        steps = inspection_results['preprocessing']
        if isinstance(steps, list):
            report.add_bullet_list(steps)
        else:
            report.add_paragraph(str(steps))
    
    report.add_page_break()

def add_model_development_section(
    report: RegressionReport,
    model_results: Dict[str, Any],
    include_code: bool = False
) -> None:
    """
    Add model development results to the report.
    
    Args:
        report: RegressionReport object
        model_results: Results from model development
        include_code: Whether to include model code
    """
    report.add_chapter(
        "Model Development",
        "Regression model development process and hyperparameter tuning."
    )
    
    # Model configuration
    if 'model_config' in model_results:
        report.add_section("Model Configuration")
        
        config = model_results['model_config']
        config_data = [[key, str(value)] for key, value in config.items()]
        
        report.add_table(
            data=config_data,
            headers=['Parameter', 'Value'],
            title="Model Configuration"
        )
    
    # Hyperparameter tuning
    if 'hyperparameter_tuning' in model_results:
        report.add_section("Hyperparameter Tuning")
        
        tuning = model_results['hyperparameter_tuning']
        
        # Best parameters
        if 'best_params' in tuning:
            best_params = tuning['best_params']
            params_data = [[param, str(value)] for param, value in best_params.items()]
            
            report.add_table(
                data=params_data,
                headers=['Parameter', 'Value'],
                title="Best Hyperparameters"
            )
        
        # CV results
        if 'cv_results' in tuning:
            cv_results = tuning['cv_results']
            if isinstance(cv_results, list) and len(cv_results) > 0:
                # Show top 5 results
                cv_results = sorted(cv_results, key=lambda x: x.get('score', 0), reverse=True)[:5]
                cv_data = []
                
                # Extract parameters and scores
                for result in cv_results:
                    params_str = ', '.join([f"{k}={v}" for k, v in result.get('params', {}).items()])
                    cv_data.append([
                        params_str,
                        f"{result.get('score', 0):.4f}",
                        f"{result.get('std', 0):.4f}"
                    ])
                
                report.add_table(
                    data=cv_data,
                    headers=['Parameters', 'CV Score', 'Std Dev'],
                    title="Cross-validation Results (Top 5)"
                )
    
    # Feature importance
    if 'feature_importance' in model_results:
        report.add_section("Feature Importance")
        
        importance = model_results['feature_importance']
        
        # Create feature importance table (top 10)
        importance_items = sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
        importance_data = [[feat, f"{imp:.4f}"] for feat, imp in importance_items]
        
        report.add_table(
            data=importance_data,
            headers=['Feature', 'Importance'],
            title="Feature Importance (Top 10)"
        )
        
        # Add feature importance plot
        if len(importance) > 0:
            fig = create_feature_importance_plot(
                importance,
                title="Feature Importance",
                n_features=20
            )
            report.add_plot(
                figure=fig,
                caption="Feature importance ranking showing the relative influence of each variable."
            )
            plt.close(fig)
    
    # Model code
    if include_code and 'model_code' in model_results:
        report.add_section("Model Implementation")
        report.add_code_block(model_results['model_code'], language="python")
    
    report.add_page_break()

def add_evaluation_section(
    report: RegressionReport,
    evaluation_results: Dict[str, Any],
    include_predictions: bool = True
) -> None:
    """
    Add model evaluation results to the report.
    
    Args:
        report: RegressionReport object
        evaluation_results: Results from model evaluation
        include_predictions: Whether to include prediction plots
    """
    report.add_chapter(
        "Model Evaluation",
        "Comprehensive model performance metrics and error analysis."
    )
    
    # Performance metrics
    if 'metrics' in evaluation_results:
        report.add_section("Performance Metrics")
        
        metrics = evaluation_results['metrics']
        
        # Create metrics table for train/test sets
        metrics_data = []
        for dataset, dataset_metrics in metrics.items():
            metrics_row = [dataset.capitalize()]
            
            # Add common regression metrics
            for metric in ['r2', 'rmse', 'mae']:
                metrics_row.append(f"{dataset_metrics.get(metric, 0):.4f}")
            
            metrics_data.append(metrics_row)
        
        report.add_table(
            data=metrics_data,
            headers=['Dataset', 'R²', 'RMSE', 'MAE'],
            title="Regression Metrics",
            highlight_max=[1],     # Highlight best R²
            highlight_min=[2, 3]   # Highlight best RMSE and MAE
        )
    
    # Detailed metrics
    if 'detailed_metrics' in evaluation_results:
        report.add_section("Detailed Performance Analysis")
        
        detailed = evaluation_results['detailed_metrics']
        
        # Create detailed metrics table
        detailed_data = []
        for metric, value in detailed.items():
            detailed_data.append([metric.replace('_', ' ').title(), f"{value:.4f}"])
        
        if detailed_data:
            report.add_table(
                data=detailed_data,
                headers=['Metric', 'Value'],
                title="Additional Metrics"
            )
    
    # Prediction plots
    if include_predictions and 'predictions' in evaluation_results:
        report.add_section("Prediction Analysis")
        
        predictions = evaluation_results['predictions']
        
        # Actual vs Predicted plot
        if 'y_true' in predictions and 'y_pred' in predictions:
            fig = create_actual_vs_predicted_plot(
                y_true=predictions['y_true'],
                y_pred=predictions['y_pred'],
                title="Actual vs Predicted Repayment Rates"
            )
            report.add_plot(
                figure=fig,
                caption="Scatter plot showing actual vs predicted values with error bands."
            )
            plt.close(fig)
        
        # Error distribution plot
        if 'y_true' in predictions and 'y_pred' in predictions:
            fig = create_error_distribution_plot(
                y_true=predictions['y_true'],
                y_pred=predictions['y_pred'],
                title="Error Distribution"
            )
            report.add_plot(
                figure=fig,
                caption="Distribution of prediction errors with statistical metrics."
            )
            plt.close(fig)
    
    # Residual analysis
    if 'residuals' in evaluation_results:
        report.add_section("Residual Analysis")
        
        residuals = evaluation_results['residuals']
        report.add_paragraph(
            "Analysis of model residuals (prediction errors) to assess model quality and assumptions."
        )
        
        # Add residual plots if available
        if 'residual_plots' in residuals and isinstance(residuals['residual_plots'], list):
            for i, plot_path in enumerate(residuals['residual_plots']):
                if os.path.exists(plot_path):
                    report.add_plot(
                        image_path=plot_path,
                        caption=f"Residual analysis plot {i+1}: {os.path.basename(plot_path)}"
                    )
    
    report.add_page_break()

def add_threshold_analysis_section(
    report: RegressionReport,
    threshold_results: Dict[str, Any]
) -> None:
    """
    Add threshold analysis results to the report.
    
    Args:
        report: RegressionReport object
        threshold_results: Results from threshold analysis
    """
    report.add_chapter(
        "Threshold Analysis",
        "Analysis of model performance across different prediction thresholds."
    )
    
    # Summary of threshold analysis
    report.add_section("Threshold Analysis Overview")
    
    # Add explanation
    report.add_paragraph(
        "In regression-based repayment prediction, we can apply various thresholds to the "
        "predicted repayment rates to determine loan approval. This analysis examines the "
        "trade-offs between different threshold choices."
    )
    
    # Add optimal threshold information
    if 'optimal_threshold' in threshold_results:
        optimal = threshold_results['optimal_threshold']
        report.add_subsection("Optimal Threshold")
        
        report.add_paragraph(
            f"The optimal threshold for loan approval is <b>{optimal:.2f}</b>. "
            f"This threshold maximizes expected profit while balancing risk."
        )
    
    # Threshold metrics table
    if 'threshold_metrics' in threshold_results and isinstance(threshold_results['threshold_metrics'], pd.DataFrame):
        report.add_section("Metrics Across Thresholds")
        
        df = threshold_results['threshold_metrics']
        
        # Extract key metrics at different thresholds
        thresholds = sorted(df['threshold'].unique())
        
        metrics_data = []
        for threshold in thresholds:
            row_data = df[df['threshold'] == threshold].iloc[0]
            
            metrics_row = [
                f"{threshold:.2f}",
                f"{row_data.get('approval_rate', 0):.1%}",
                f"{row_data.get('actual_repayment_rate', 0):.1%}",
                f"{row_data.get('default_rate', 0):.1%}"
            ]
            
            # Add profit metrics if available
            if 'actual_profit' in row_data:
                metrics_row.append(f"{row_data['actual_profit']:,.0f}")
            
            if 'net_profit_margin' in row_data:
                metrics_row.append(f"{row_data['net_profit_margin']:.1%}")
                
            if 'return_on_investment' in row_data:
                metrics_row.append(f"{row_data['return_on_investment']:.1%}")
            
            metrics_data.append(metrics_row)
        
        # Determine headers based on available columns
        headers = ['Threshold', 'Approval Rate', 'Repayment Rate', 'Default Rate']
        if 'actual_profit' in df.columns:
            headers.append('Profit')
        if 'net_profit_margin' in df.columns:
            headers.append('Profit Margin')
        if 'return_on_investment' in df.columns:
            headers.append('ROI')
        
        report.add_table(
            data=metrics_data,
            headers=headers,
            title="Performance Metrics by Threshold"
        )
    
    # Threshold performance plot
    if 'threshold_metrics' in threshold_results and 'optimal_threshold' in threshold_results:
        df = threshold_results['threshold_metrics']
        optimal = threshold_results['optimal_threshold']
        
        metrics_to_plot = ['actual_repayment_rate', 'approval_rate', 'default_rate']
        available_metrics = [m for m in metrics_to_plot if m in df.columns]
        
        if available_metrics:
            fig = create_threshold_performance_plot(
                threshold_df=df,
                optimal_threshold=optimal,
                metrics=available_metrics,
                title="Performance Metrics by Threshold"
            )
            report.add_plot(
                figure=fig,
                caption="Comparison of key performance metrics across different thresholds."
            )
            plt.close(fig)
        
        # Add profit curve if available
        if 'actual_profit' in df.columns:
            fig = create_profit_curve_plot(
                threshold_df=df,
                optimal_threshold=optimal,
                profit_col='actual_profit',
                title="Profit by Threshold"
            )
            report.add_plot(
                figure=fig,
                caption="Total profit at different threshold levels, with optimal threshold highlighted."
            )
            plt.close(fig)
    
    # Business implications
    report.add_section("Business Implications")
    
    report.add_paragraph(
        "Different thresholds represent different business strategies:"
    )
    
    implications = [
        "<b>Low threshold (e.g., 0.6)</b>: More inclusive approach with higher approval rate but lower average repayment rate. "
        "May be appropriate for customer acquisition focus.",
        
        "<b>Medium threshold (e.g., 0.7-0.8)</b>: Balanced approach that typically maximizes total profit by accepting "
        "more moderate-risk loans while maintaining reasonable repayment rates.",
        
        "<b>High threshold (e.g., 0.9+)</b>: Conservative approach with very high repayment rates but low approval rates. "
        "May be appropriate for risk-minimization strategy."
    ]
    
    report.add_bullet_list(implications)
    
    report.add_page_break()

def add_margin_analysis_section(
    report: RegressionReport,
    margin_results: Dict[str, Any]
) -> None:
    """
    Add margin analysis results to the report.
    
    Args:
        report: RegressionReport object
        margin_results: Results from margin analysis
    """
    report.add_chapter(
        "Gross Margin Analysis",
        "Analysis of loan profitability across different gross margins and thresholds."
    )
    
    # Margin analysis overview
    report.add_section("Margin Analysis Overview")
    
    # Add explanation
    report.add_paragraph(
        "This section analyzes how different gross margins affect profitability across various "
        "approval thresholds. The gross margin represents the difference between the loan value "
        "and the cost of goods sold (input costs)."
    )
    
    # Add optimal parameters
    if 'optimal_parameters' in margin_results:
        optimal = margin_results['optimal_parameters']
        
        report.add_subsection("Optimal Parameters")
        
        report.add_paragraph(
            f"The optimal gross margin is <b>{optimal.get('margin', 0):.1%}</b> with a threshold of "
            f"<b>{optimal.get('threshold', 0):.2f}</b>. This combination yields an expected profit of "
            f"<b>{optimal.get('profit', 0):,.0f} KES</b>."
        )
    
    # Margin-threshold heatmaps
    if 'margin_threshold_metrics' in margin_results and isinstance(margin_results['margin_threshold_metrics'], pd.DataFrame):
        report.add_section("Cross-Analysis of Thresholds and Margins")
        
        df = margin_results['margin_threshold_metrics']
        
        # Create separate heatmaps for different metrics
        metrics_to_plot = [
            ('actual_repayment_rate', 'Repayment Rate by Margin and Threshold', 'viridis'),
            ('default_rate', 'Default Rate by Margin and Threshold', 'rocket_r'),  # Reversed colormap for default rate
            ('approval_rate', 'Approval Rate by Margin and Threshold', 'viridis'),
            ('actual_profit', 'Profit by Margin and Threshold', 'viridis'),
            ('net_profit_margin', 'Net Profit Margin by Margin and Threshold', 'viridis'),
            ('return_on_investment', 'ROI by Margin and Threshold', 'viridis')
        ]
        
        for metric, title, cmap in metrics_to_plot:
            if metric in df.columns:
                fig = create_margin_analysis_plot(
                    margin_threshold_df=df,
                    metric=metric,
                    title=title,
                    cmap=cmap
                )
                report.add_plot(
                    figure=fig,
                    caption=f"Heatmap showing {metric.replace('_', ' ')} across different margins and thresholds."
                )
                plt.close(fig)
    
    # Margin sensitivity analysis
    if 'margin_sensitivity' in margin_results:
        report.add_section("Margin Sensitivity Analysis")
        
        sensitivity = margin_results['margin_sensitivity']
        
        # Add sensitivity table
        if isinstance(sensitivity, list):
            sensitivity_data = []
            for item in sensitivity:
                margin = item.get('margin', 0)
                base_profit = item.get('base_profit', 0)
                new_profit = item.get('new_profit', 0)
                change_pct = (new_profit - base_profit) / base_profit if base_profit != 0 else 0
                
                sensitivity_data.append([
                    f"{margin:.1%}",
                    f"{base_profit:,.0f}",
                    f"{new_profit:,.0f}",
                    f"{change_pct:.1%}"
                ])
            
            if sensitivity_data:
                report.add_table(
                    data=sensitivity_data,
                    headers=['Margin', 'Base Profit', 'New Profit', 'Change %'],
                    title="Profit Sensitivity to Margin Changes"
                )
        
        # Add sensitivity explanation
        report.add_paragraph(
            "This analysis shows how profits change with different gross margins. "
            "Higher margins generally increase profitability, but may affect sales volume in practice."
        )
    
    report.add_page_break()

def add_holdout_evaluation_section(
    report: RegressionReport,
    holdout_results: Dict[str, Any]
) -> None:
    """
    Add holdout validation results to the report.
    
    Args:
        report: RegressionReport object
        holdout_results: Results from holdout validation
    """
    report.add_chapter(
        "Holdout Validation",
        "Validation of model performance on independent holdout data."
    )
    
    # Holdout performance metrics
    if 'metrics' in holdout_results:
        report.add_section("Holdout Performance Metrics")
        
        metrics = holdout_results['metrics']
        
        # Create metrics comparison table
        metrics_data = []
        for dataset in ['train', 'test', 'holdout']:
            if dataset in metrics:
                dataset_metrics = metrics[dataset]
                metrics_row = [dataset.capitalize()]
                
                # Add common regression metrics
                for metric in ['r2', 'rmse', 'mae']:
                    metrics_row.append(f"{dataset_metrics.get(metric, 0):.4f}")
                
                metrics_data.append(metrics_row)
        
        report.add_table(
            data=metrics_data,
            headers=['Dataset', 'R²', 'RMSE', 'MAE'],
            title="Performance Comparison",
            highlight_max=[1],     # Highlight best R²
            highlight_min=[2, 3]   # Highlight best RMSE and MAE
        )
    
    # Holdout performance plots
    if 'predictions' in holdout_results:
        report.add_section("Holdout Prediction Analysis")
        
        predictions = holdout_results['predictions']
        
        # Actual vs Predicted plot
        if 'y_true' in predictions and 'y_pred' in predictions:
            fig = create_actual_vs_predicted_plot(
                y_true=predictions['y_true'],
                y_pred=predictions['y_pred'],
                title="Holdout: Actual vs Predicted Repayment Rates"
            )
            report.add_plot(
                figure=fig,
                caption="Scatter plot of actual vs predicted values on the holdout dataset."
            )
            plt.close(fig)
        
        # Error distribution plot
        if 'y_true' in predictions and 'y_pred' in predictions:
            fig = create_error_distribution_plot(
                y_true=predictions['y_true'],
                y_pred=predictions['y_pred'],
                title="Holdout: Error Distribution"
            )
