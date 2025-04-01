"""
Evaluation functions for regression-based loan repayment prediction.

This module provides evaluation metrics and analysis specifically for regression models,
replacing the classification evaluation metrics in the original scorecard package.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from typing import Dict, List, Optional, Tuple, Any, Union
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score, explained_variance_score,
    mean_absolute_percentage_error
)
from sklearn.calibration import calibration_curve

def calculate_regression_metrics(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    thresholds: Optional[List[float]] = None
) -> Dict[str, Any]:
    """
    Calculate a comprehensive set of regression performance metrics.
    
    Args:
        y_true: Actual target values
        y_pred: Predicted target values
        thresholds: Optional list of thresholds for binary classification analysis
        
    Returns:
        Dictionary with calculated metrics
    """
    # Standard regression metrics
    metrics = {
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred),
        'mse': mean_squared_error(y_true, y_pred),
        'ev': explained_variance_score(y_true, y_pred)
    }
    
    # Calculate MAPE safely (handling zero values)
    non_zero_mask = np.abs(y_true) > 1e-10  # Small threshold to avoid division by near-zero
    if np.any(non_zero_mask):
        mape = mean_absolute_percentage_error(
            y_true[non_zero_mask], y_pred[non_zero_mask]
        )
        metrics['mape'] = float(mape) * 100  # Convert to percentage
    else:
        metrics['mape'] = np.nan
    
    # Error distribution statistics
    errors = y_pred - y_true
    metrics['error_mean'] = float(np.mean(errors))
    metrics['error_std'] = float(np.std(errors))
    metrics['error_median'] = float(np.median(errors))
    metrics['error_min'] = float(np.min(errors))
    metrics['error_max'] = float(np.max(errors))
    
    # Absolute error statistics
    abs_errors = np.abs(errors)
    metrics['abs_error_mean'] = float(np.mean(abs_errors))
    metrics['abs_error_median'] = float(np.median(abs_errors))
    metrics['abs_error_90pct'] = float(np.percentile(abs_errors, 90))
    metrics['abs_error_95pct'] = float(np.percentile(abs_errors, 95))
    
    # Binary classification metrics at different thresholds (if provided)
    if thresholds is not None:
        threshold_metrics = {}
        for threshold in thresholds:
            y_true_binary = (y_true >= threshold).astype(int)
            y_pred_binary = (y_pred >= threshold).astype(int)
            
            # Calculate confusion matrix elements
            tp = np.sum((y_true_binary == 1) & (y_pred_binary == 1))
            tn = np.sum((y_true_binary == 0) & (y_pred_binary == 0))
            fp = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
            fn = np.sum((y_true_binary == 1) & (y_pred_binary == 0))
            
            # Calculate metrics from confusion matrix
            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            threshold_metrics[f'threshold_{threshold:.2f}'] = {
                'tp': int(tp),
                'tn': int(tn),
                'fp': int(fp),
                'fn': int(fn),
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'specificity': float(specificity),
                'f1_score': float(f1)
            }
        
        metrics['threshold_metrics'] = threshold_metrics
    
    return metrics

def evaluate_regression_performance(
    train_actual: pd.Series,
    test_actual: pd.Series,
    train_pred: np.ndarray,
    test_pred: np.ndarray,
    thresholds: Optional[List[float]] = None,
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Evaluate regression model performance on training and testing sets.
    
    Args:
        train_actual: Actual target values for training set
        test_actual: Actual target values for testing set
        train_pred: Predicted values for training set
        test_pred: Predicted values for testing set
        thresholds: Optional list of thresholds for binary classification analysis
        output_dir: Directory to save evaluation results
        
    Returns:
        Dictionary with performance metrics and plots
    """
    print("\n=== Regression Performance Evaluation ===")
    
    # Calculate metrics for training and testing sets
    train_metrics = calculate_regression_metrics(train_actual, train_pred, thresholds)
    test_metrics = calculate_regression_metrics(test_actual, test_pred, thresholds)
    
    # Print key metrics
    print("\nTraining Performance:")
    print(f"  RMSE: {train_metrics['rmse']:.4f}")
    print(f"  MAE: {train_metrics['mae']:.4f}")
    print(f"  R²: {train_metrics['r2']:.4f}")
    print(f"  Mean Error: {train_metrics['error_mean']:.4f}")
    
    print("\nTesting Performance:")
    print(f"  RMSE: {test_metrics['rmse']:.4f}")
    print(f"  MAE: {test_metrics['mae']:.4f}")
    print(f"  R²: {test_metrics['r2']:.4f}")
    print(f"  Mean Error: {test_metrics['error_mean']:.4f}")
    
    # Create dataframes with actual and predicted values
    train_df = pd.DataFrame({
        'actual': train_actual,
        'predicted': train_pred,
        'error': train_pred - train_actual,
        'dataset': 'Train'
    })
    
    test_df = pd.DataFrame({
        'actual': test_actual,
        'predicted': test_pred,
        'error': test_pred - test_actual,
        'dataset': 'Test'
    })
    
    # Combine for overall performance assessment
    combined_df = pd.concat([train_df, test_df])
    
    # Generate plots if output directory is provided
    plots = {}
    
    if output_dir:
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Plot actual vs. predicted with error distribution
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Actual vs predicted scatter plot (train)
        axes[0, 0].scatter(train_actual, train_pred, alpha=0.5)
        axes[0, 0].plot([train_actual.min(), train_actual.max()], 
                        [train_actual.min(), train_actual.max()], 'r--')
        axes[0, 0].set_xlabel('Actual')
        axes[0, 0].set_ylabel('Predicted')
        axes[0, 0].set_title('Training Set: Actual vs Predicted')
        
        # Actual vs predicted scatter plot (test)
        axes[0, 1].scatter(test_actual, test_pred, alpha=0.5)
        axes[0, 1].plot([test_actual.min(), test_actual.max()], 
                         [test_actual.min(), test_actual.max()], 'r--')
        axes[0, 1].set_xlabel('Actual')
        axes[0, 1].set_ylabel('Predicted')
        axes[0, 1].set_title('Testing Set: Actual vs Predicted')
        
        # Error distribution (train)
        sns.histplot(train_df['error'], kde=True, ax=axes[1, 0])
        axes[1, 0].axvline(0, color='red', linestyle='--')
        axes[1, 0].set_xlabel('Prediction Error')
        axes[1, 0].set_title('Training Error Distribution')
        
        # Error distribution (test)
        sns.histplot(test_df['error'], kde=True, ax=axes[1, 1])
        axes[1, 1].axvline(0, color='red', linestyle='--')
        axes[1, 1].set_xlabel('Prediction Error')
        axes[1, 1].set_title('Testing Error Distribution')
        
        plt.tight_layout()
        
        # Save plot
        perf_plot_path = os.path.join(output_dir, "regression_performance.png")
        plt.savefig(perf_plot_path, dpi=300)
        plt.close()
        plots['performance'] = perf_plot_path
        
        # Create residual plots
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Residuals vs predicted (train)
        axes[0].scatter(train_pred, train_df['error'], alpha=0.5)
        axes[0].axhline(0, color='red', linestyle='--')
        axes[0].set_xlabel('Predicted Value')
        axes[0].set_ylabel('Residual')
        axes[0].set_title('Training Set: Residuals vs Predicted')
        
        # Residuals vs predicted (test)
        axes[1].scatter(test_pred, test_df['error'], alpha=0.5)
        axes[1].axhline(0, color='red', linestyle='--')
        axes[1].set_xlabel('Predicted Value')
        axes[1].set_ylabel('Residual')
        axes[1].set_title('Testing Set: Residuals vs Predicted')
        
        plt.tight_layout()
        
        # Save plot
        residual_plot_path = os.path.join(output_dir, "regression_residuals.png")
        plt.savefig(residual_plot_path, dpi=300)
        plt.close()
        plots['residuals'] = residual_plot_path
        
        # Create calibration plots if thresholds are provided
        if thresholds is not None:
            fig, ax = plt.subplots(figsize=(10, 8))
            
            for threshold in thresholds:
                # Calculate fraction of positives and mean predicted probability
                train_true_binary = (train_actual >= threshold).astype(int)
                train_prob = train_pred / 1.0  # Normalize for probabilistic interpretation
                
                test_true_binary = (test_actual >= threshold).astype(int)
                test_prob = test_pred / 1.0  # Normalize for probabilistic interpretation
                
                # Create calibration curves
                train_fraction, train_mean_pred = calibration_curve(
                    train_true_binary, train_prob, n_bins=10, strategy='uniform'
                )
                
                test_fraction, test_mean_pred = calibration_curve(
                    test_true_binary, test_prob, n_bins=10, strategy='uniform'
                )
                
                # Plot calibration curves
                ax.plot(train_mean_pred, train_fraction, 's-',
                        label=f'Train (threshold={threshold:.2f})')
                ax.plot(test_mean_pred, test_fraction, 'o-',
                        label=f'Test (threshold={threshold:.2f})')
            
            # Add reference line
            ax.plot([0, 1], [0, 1], 'k:', label='Perfectly calibrated')
            
            ax.set_xlabel('Mean predicted value')
            ax.set_ylabel('Fraction of positives')
            ax.set_title('Calibration plots at different thresholds')
            ax.legend(loc='best')
            
            # Save plot
            calib_plot_path = os.path.join(output_dir, "calibration_plots.png")
            plt.savefig(calib_plot_path, dpi=300)
            plt.close()
            plots['calibration'] = calib_plot_path
        
        # Save metrics to JSON
        metrics = {
            'train': train_metrics,
            'test': test_metrics
        }
        
        metrics_path = os.path.join(output_dir, "regression_metrics.json")
        with open(metrics_path, 'w') as f:
            # Use a custom encoder to handle NumPy types
            class NpEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, np.integer):
                        return int(obj)
                    if isinstance(obj, np.floating):
                        return float(obj)
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    return super(NpEncoder, self).default(obj)
            
            json.dump(metrics, f, indent=2, cls=NpEncoder)
        
        print(f"\nPerformance evaluation results saved to {output_dir}")
    
    return {
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'train_df': train_df,
        'test_df': test_df,
        'combined_df': combined_df,
        'plots': plots
    }

def analyze_performance_by_segment(
    df: pd.DataFrame,
    segment_col: str,
    actual_col: str = 'actual',
    predicted_col: str = 'predicted',
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Analyze model performance by segments (e.g., regions, loan types).
    
    Args:
        df: DataFrame with actual and predicted values
        segment_col: Column name for segmentation
        actual_col: Column name for actual values
        predicted_col: Column name for predicted values
        output_dir: Directory to save segment analysis results
        
    Returns:
        Dictionary with segment performance metrics
    """
    print(f"\n=== Performance Analysis by {segment_col} ===")
    
    # Group by segment and calculate metrics
    segment_metrics = {}
    
    for segment, group in df.groupby(segment_col):
        # Skip segments with too few samples
        if len(group) < 10:
            print(f"Skipping segment '{segment}' - too few samples ({len(group)})")
            continue
            
        metrics = calculate_regression_metrics(
            group[actual_col].values, 
            group[predicted_col].values
        )
        
        segment_metrics[segment] = metrics
        
        print(f"\nSegment: {segment} ({len(group)} samples)")
        print(f"  RMSE: {metrics['rmse']:.4f}")
        print(f"  MAE: {metrics['mae']:.4f}")
        print(f"  R²: {metrics['r2']:.4f}")
    
    # Create summary dataframe
    summary_data = []
    
    for segment, metrics in segment_metrics.items():
        summary_data.append({
            'segment': segment,
            'n_samples': len(df[df[segment_col] == segment]),
            'rmse': metrics['rmse'],
            'mae': metrics['mae'],
            'r2': metrics['r2'],
            'error_mean': metrics['error_mean'],
            'error_std': metrics['error_std']
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Generate plots if output directory is provided
    if output_dir and len(summary_df) > 0:
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Plot performance metrics by segment
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Sort by count for better visualization
        summary_sorted = summary_df.sort_values('n_samples', ascending=False)
        
        # Sample count
        sns.barplot(x='segment', y='n_samples', data=summary_sorted, ax=axes[0, 0])
        axes[0, 0].set_title(f'Sample Count by {segment_col}')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].set_xlabel('')
        
        # RMSE
        sns.barplot(x='segment', y='rmse', data=summary_sorted, ax=axes[0, 1])
        axes[0, 1].set_title(f'RMSE by {segment_col}')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].set_xlabel('')
        
        # MAE
        sns.barplot(x='segment', y='mae', data=summary_sorted, ax=axes[1, 0])
        axes[1, 0].set_title(f'MAE by {segment_col}')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].set_xlabel('')
        
        # R²
        sns.barplot(x='segment', y='r2', data=summary_sorted, ax=axes[1, 1])
        axes[1, 1].set_title(f'R² by {segment_col}')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].set_xlabel('')
        
        plt.tight_layout()
        
        # Save plot
        segment_plot_path = os.path.join(output_dir, f"performance_by_{segment_col}.png")
        plt.savefig(segment_plot_path, dpi=300)
        plt.close()
        
        # Save summary to CSV
        summary_path = os.path.join(output_dir, f"metrics_by_{segment_col}.csv")
        summary_df.to_csv(summary_path, index=False)
        
        print(f"\nSegment analysis for {segment_col} saved to {output_dir}")
    
    return {
        'segment_metrics': segment_metrics,
        'summary_df': summary_df
    }

def analyze_error_distribution(
    df: pd.DataFrame,
    actual_col: str = 'actual',
    predicted_col: str = 'predicted',
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Analyze the distribution of prediction errors.
    
    Args:
        df: DataFrame with actual and predicted values
        actual_col: Column name for actual values
        predicted_col: Column name for predicted values
        output_dir: Directory to save error analysis results
        
    Returns:
        Dictionary with error distribution statistics
    """
    print("\n=== Error Distribution Analysis ===")
    
    # Calculate error
    df = df.copy()
    df['error'] = df[predicted_col] - df[actual_col]
    df['abs_error'] = np.abs(df['error'])
    df['pct_error'] = np.where(
        np.abs(df[actual_col]) > 1e-10,
        100 * df['error'] / df[actual_col],
        np.nan
    )
    
    # Calculate error statistics
    stats = {
        'mean_error': float(df['error'].mean()),
        'median_error': float(df['error'].median()),
        'std_error': float(df['error'].std()),
        'mean_abs_error': float(df['abs_error'].mean()),
        'median_abs_error': float(df['abs_error'].median()),
        'p90_abs_error': float(df['abs_error'].quantile(0.9)),
        'p95_abs_error': float(df['abs_error'].quantile(0.95)),
        'p99_abs_error': float(df['abs_error'].quantile(0.99)),
        'min_error': float(df['error'].min()),
        'max_error': float(df['error'].max()),
        'mean_pct_error': float(df['pct_error'].mean()),
        'median_pct_error': float(df['pct_error'].median())
    }
    
    # Print key statistics
    print(f"Mean Error: {stats['mean_error']:.4f}")
    print(f"Median Error: {stats['median_error']:.4f}")
    print(f"Mean Absolute Error: {stats['mean_abs_error']:.4f}")
    print(f"90th Percentile Absolute Error: {stats['p90_abs_error']:.4f}")
    print(f"Mean Percentage Error: {stats['mean_pct_error']:.2f}%")
    
    # Generate plots if output directory is provided
    if output_dir:
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Create error distribution plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Error distribution
        sns.histplot(df['error'], kde=True, ax=axes[0, 0])
        axes[0, 0].axvline(0, color='red', linestyle='--')
        axes[0, 0].set_xlabel('Prediction Error')
        axes[0, 0].set_title('Error Distribution')
        
        # Absolute error distribution
        sns.histplot(df['abs_error'], kde=True, ax=axes[0, 1])
        axes[0, 1].set_xlabel('Absolute Prediction Error')
        axes[0, 1].set_title('Absolute Error Distribution')
        
        # Error vs actual
        axes[1, 0].scatter(df[actual_col], df['error'], alpha=0.5)
        axes[1, 0].axhline(0, color='red', linestyle='--')
        axes[1, 0].set_xlabel('Actual Value')
        axes[1, 0].set_ylabel('Error')
        axes[1, 0].set_title('Error vs Actual Value')
        
        # Percentage error distribution (excluding outliers)
        pct_error_filtered = df['pct_error'].dropna()
        pct_error_filtered = pct_error_filtered[
            (pct_error_filtered > np.percentile(pct_error_filtered, 1)) &
            (pct_error_filtered < np.percentile(pct_error_filtered, 99))
        ]
        
        if len(pct_error_filtered) > 0:
            sns.histplot(pct_error_filtered, kde=True, ax=axes[1, 1])
            axes[1, 1].axvline(0, color='red', linestyle='--')
            axes[1, 1].set_xlabel('Percentage Error')
            axes[1, 1].set_title('Percentage Error Distribution (1-99 percentile)')
        else:
            axes[1, 1].text(0.5, 0.5, 'Insufficient data for percentage error plot',
                           ha='center', va='center')
        
        plt.tight_layout()
        
        # Save plot
        error_plot_path = os.path.join(output_dir, "error_distribution.png")
        plt.savefig(error_plot_path, dpi=300)
        plt.close()
        
        # Save statistics to JSON
        stats_path = os.path.join(output_dir, "error_statistics.json")
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"\nError distribution analysis saved to {output_dir}")
    
    return {
        'error_stats': stats,
        'df_with_errors': df
    }
