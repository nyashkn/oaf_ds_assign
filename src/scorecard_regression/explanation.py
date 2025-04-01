"""
Model explanation functions for regression-based loan repayment prediction.

This module provides functions for explaining model predictions using SHAP,
partial dependence plots, and other interpretability techniques.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import pickle
from typing import Dict, List, Optional, Tuple, Any, Union
import warnings

# Import optional model explanation packages with fallbacks
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("SHAP not available. SHAP explanations will not be available.")

try:
    from pdpbox import pdp
    PDPBOX_AVAILABLE = True
except ImportError:
    PDPBOX_AVAILABLE = False
    warnings.warn("PDPbox not available. Partial dependence plots will not be available.")

# Defer eli5 import to runtime
ELI5_AVAILABLE = False

try:
    from pycebox.ice import ice, ice_plot
    PYCEBOX_AVAILABLE = True
except ImportError:
    PYCEBOX_AVAILABLE = False
    warnings.warn("PyCEbox not available. ICE plots will not be available.")

from .constants import SHAP_PLOT_SETTINGS, PDP_PLOT_SETTINGS

def explain_model(
    model,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series = None,
    y_test: pd.Series = None,
    feature_names: Optional[List[str]] = None,
    top_features: int = 20,
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate comprehensive model explanations using multiple techniques.
    
    Args:
        model: Trained regression model
        X_train: Training features
        X_test: Testing features
        y_train: Training target values (optional)
        y_test: Testing target values (optional)
        feature_names: Feature names (if not included in X_train/X_test)
        top_features: Number of top features to analyze in detail
        output_dir: Directory to save explanation outputs
        
    Returns:
        Dictionary with explanation results
    """
    print("\n=== Model Explanation ===")
    
    # Create dictionary to store results
    explanation_results = {}
    
    # Use DataFrame columns as feature names if not provided
    if feature_names is None and isinstance(X_train, pd.DataFrame):
        feature_names = X_train.columns.tolist()
    
    # If X is DataFrame, convert to numpy for some operations
    X_train_values = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
    X_test_values = X_test.values if isinstance(X_test, pd.DataFrame) else X_test
    
    # Create output directory if provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # 1. Get feature importances from the model (if available)
    feature_importances = get_feature_importances(model, feature_names)
    explanation_results['feature_importances'] = feature_importances
    
    if feature_importances:
        print("\nFeature Importances (Top 10):")
        for feature, importance in sorted(
            feature_importances.items(), 
            key=lambda x: abs(x[1]), 
            reverse=True
        )[:10]:
            print(f"  {feature}: {importance:.4f}")
            
        # Save feature importances
        if output_dir:
            importances_df = pd.DataFrame({
                'feature': list(feature_importances.keys()),
                'importance': list(feature_importances.values())
            }).sort_values('importance', ascending=False)
            
            # Create bar plot
            plt.figure(figsize=(12, 8))
            sns.barplot(x='importance', y='feature', data=importances_df.head(top_features))
            plt.title('Feature Importances')
            plt.tight_layout()
            
            # Save plot and data
            importances_path = os.path.join(output_dir, "feature_importances.png")
            plt.savefig(importances_path, dpi=300)
            plt.close()
            
            importances_csv_path = os.path.join(output_dir, "feature_importances.csv")
            importances_df.to_csv(importances_csv_path, index=False)
    
    # 2. Generate SHAP explanations
    if SHAP_AVAILABLE:
        try:
            print("\nGenerating SHAP explanations...")
            shap_results = generate_shap_explanation(
                model, X_train, X_test,
                feature_names=feature_names,
                output_dir=output_dir,
                max_display=top_features
            )
            explanation_results['shap'] = shap_results
            print("SHAP explanations generated successfully")
        except Exception as e:
            print(f"Error generating SHAP explanations: {str(e)}")
            explanation_results['shap'] = None
    else:
        explanation_results['shap'] = None
    
    # 3. Generate Partial Dependence Plots
    if PDPBOX_AVAILABLE:
        # Select top features for PDP analysis based on feature importance
        if feature_importances and len(feature_importances) > 0:
            top_features_list = [
                feature for feature, _ in sorted(
                    feature_importances.items(),
                    key=lambda x: abs(x[1]),
                    reverse=True
                )[:PDP_PLOT_SETTINGS['n_features']]
            ]
        else:
            # If no feature importances available, use top N features by index
            top_features_list = feature_names[:PDP_PLOT_SETTINGS['n_features']] if feature_names else []
        
        # Generate PDPs for top features
        if top_features_list:
            try:
                print("\nGenerating Partial Dependence Plots...")
                pdp_results = generate_pdp_plots(
                    model, X_train, top_features_list,
                    output_dir=output_dir
                )
                explanation_results['pdp'] = pdp_results
                print("Partial Dependence Plots generated successfully")
            except Exception as e:
                print(f"Error generating Partial Dependence Plots: {str(e)}")
                explanation_results['pdp'] = None
        else:
            explanation_results['pdp'] = None
    else:
        explanation_results['pdp'] = None
    
    # 4. Generate ICE Plots if available
    if PYCEBOX_AVAILABLE and feature_importances:
        try:
            print("\nGenerating ICE Plots...")
            # Select top features for ICE analysis based on feature importance
            top_features_ice = [
                feature for feature, _ in sorted(
                    feature_importances.items(),
                    key=lambda x: abs(x[1]),
                    reverse=True
                )[:min(5, len(feature_importances))]  # Limit to fewer features for ICE plots
            ]
            
            ice_results = generate_ice_plots(
                model, X_test, top_features_ice,
                output_dir=output_dir
            )
            explanation_results['ice'] = ice_results
            print("ICE Plots generated successfully")
        except Exception as e:
            print(f"Error generating ICE Plots: {str(e)}")
            explanation_results['ice'] = None
    else:
        explanation_results['ice'] = None
    
    # 5. Generate ELI5 explanations
    if ELI5_AVAILABLE:
        try:
            print("\nGenerating ELI5 explanations...")
            eli5_results = generate_eli5_explanation(
                model, X_train, feature_names,
                output_dir=output_dir
            )
            explanation_results['eli5'] = eli5_results
            print("ELI5 explanations generated successfully")
        except Exception as e:
            print(f"Error generating ELI5 explanations: {str(e)}")
            explanation_results['eli5'] = None
    else:
        explanation_results['eli5'] = None
    
    # Save all explanation results
    if output_dir:
        # Save serializable results
        explanation_summary = {
            'top_features': top_features_list if 'top_features_list' in locals() else [],
            'explanation_methods': {
                'feature_importances': feature_importances is not None,
                'shap': explanation_results['shap'] is not None,
                'pdp': explanation_results['pdp'] is not None,
                'ice': explanation_results['ice'] is not None,
                'eli5': explanation_results['eli5'] is not None
            }
        }
        
        summary_path = os.path.join(output_dir, "explanation_summary.json")
        with open(summary_path, 'w') as f:
            # Handle non-serializable values
            summary_json = explanation_summary.copy()
            # Convert any non-serializable items
            if 'top_features' in summary_json:
                summary_json['top_features'] = [str(f) for f in summary_json['top_features']]
            
            json.dump(summary_json, f, indent=2)
        
        print(f"\nExplanation results saved to {output_dir}")
    
    return explanation_results

def get_feature_importances(model, feature_names: List[str]) -> Optional[Dict[str, float]]:
    """
    Extract feature importances from the model if available.
    
    Args:
        model: Trained model
        feature_names: Feature names
        
    Returns:
        Dictionary with feature importances or None if not available
    """
    # For models with .feature_importances_ attribute
    if hasattr(model, 'feature_importances_'):
        return dict(zip(feature_names, model.feature_importances_))
    
    # For linear models with .coef_ attribute
    elif hasattr(model, 'coef_'):
        if model.coef_.ndim > 1:
            return dict(zip(feature_names, model.coef_[0]))
        else:
            return dict(zip(feature_names, model.coef_))
    
    # For XGBoost
    elif hasattr(model, 'get_booster') and hasattr(model.get_booster(), 'get_score'):
        try:
            importance_dict = model.get_booster().get_score(importance_type='gain')
            # XGBoost feature names might be different, so map them to original names
            return {feature_names[int(key.replace('f', ''))]: value 
                    for key, value in importance_dict.items()}
        except:
            return None
    
    # For LightGBM
    elif hasattr(model, 'booster_') and hasattr(model.booster_, 'feature_importance'):
        try:
            return dict(zip(feature_names, model.booster_.feature_importance(importance_type='gain')))
        except:
            return None
    
    return None

def generate_shap_explanation(
    model,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    feature_names: Optional[List[str]] = None,
    n_samples: int = 100,
    output_dir: Optional[str] = None,
    max_display: int = 20
) -> Dict[str, Any]:
    """
    Generate SHAP explanations for the model.
    
    Args:
        model: Trained model
        X_train: Training features
        X_test: Testing features
        feature_names: Feature names (if not included in X_train/X_test)
        n_samples: Number of samples to use for SHAP approximation
        output_dir: Directory to save SHAP plots
        max_display: Maximum number of features to display in summary plots
        
    Returns:
        Dictionary with SHAP explanation results
    """
    if not SHAP_AVAILABLE:
        print("SHAP not available. Skipping SHAP explanations.")
        return None
    
    # If too many samples, use a subset for SHAP computation
    X_train_sample = X_train
    if len(X_train) > n_samples:
        X_train_sample = X_train.sample(n_samples, random_state=42)
    
    X_test_sample = X_test
    if len(X_test) > n_samples:
        X_test_sample = X_test.sample(n_samples, random_state=42)
    
    # Try to choose appropriate explainer
    try:
        # Tree explainer for tree-based models
        if hasattr(model, 'estimators_') or hasattr(model, 'get_booster') or hasattr(model, 'booster_'):
            explainer = shap.TreeExplainer(model)
        # Linear explainer for linear models
        elif hasattr(model, 'coef_'):
            explainer = shap.LinearExplainer(model, X_train_sample)
        # Default to Kernel explainer
        else:
            explainer = shap.KernelExplainer(model.predict, X_train_sample)
        
        # Calculate SHAP values
        print("  Calculating SHAP values...")
        shap_values = explainer.shap_values(X_test_sample)
        
        # Store mean absolute SHAP values as feature importance
        shap_importances = pd.DataFrame({
            'feature': feature_names,
            'importance': np.abs(shap_values).mean(axis=0)
        }).sort_values('importance', ascending=False)
        
        # Generate and save plots if output directory provided
        if output_dir:
            plots_dir = os.path.join(output_dir, "shap_plots")
            os.makedirs(plots_dir, exist_ok=True)
            
            # Summary plot
            plt.figure(figsize=(12, 8))
            shap.summary_plot(
                shap_values, 
                X_test_sample,
                feature_names=feature_names,
                max_display=max_display,
                show=False,
                cmap=SHAP_PLOT_SETTINGS['cmap'],
                alpha=SHAP_PLOT_SETTINGS['alpha']
            )
            plt.tight_layout()
            summary_path = os.path.join(plots_dir, "shap_summary.png")
            plt.savefig(summary_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Bar summary plot
            plt.figure(figsize=(12, 8))
            shap.summary_plot(
                shap_values,
                X_test_sample,
                feature_names=feature_names,
                plot_type='bar',
                max_display=max_display,
                show=False
            )
            plt.tight_layout()
            bar_path = os.path.join(plots_dir, "shap_bar.png")
            plt.savefig(bar_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Individual SHAP dependency plots for top features
            for feature in shap_importances['feature'].head(10):
                plt.figure(figsize=(10, 6))
                shap.dependence_plot(
                    feature,
                    shap_values,
                    X_test_sample,
                    feature_names=feature_names,
                    show=False
                )
                plt.tight_layout()
                feature_path = os.path.join(plots_dir, f"shap_dependence_{feature}.png")
                plt.savefig(feature_path, dpi=300, bbox_inches='tight')
                plt.close()
            
            # Save SHAP importances as CSV
            importance_path = os.path.join(plots_dir, "shap_importances.csv")
            shap_importances.to_csv(importance_path, index=False)
            
            print(f"  SHAP plots saved to {plots_dir}")
            
        # Create SHAP waterfall plot for a sample loan
        if output_dir:
            # Select a sample with close to median prediction value
            sample_idx = 0
            try:
                predictions = model.predict(X_test)
                median_value = np.median(predictions)
                distances = np.abs(predictions - median_value)
                sample_idx = np.argmin(distances)
                
                # Create waterfall plot
                plt.figure(figsize=(12, 8))
                shap.plots.waterfall(
                    shap.Explanation(
                        values=shap_values[sample_idx],
                        base_values=explainer.expected_value,
                        data=X_test_sample.iloc[sample_idx],
                        feature_names=feature_names
                    ),
                    max_display=10,
                    show=False
                )
                plt.tight_layout()
                waterfall_path = os.path.join(plots_dir, "shap_waterfall.png")
                plt.savefig(waterfall_path, dpi=300, bbox_inches='tight')
                plt.close()
            except Exception as e:
                print(f"  Warning: Could not create SHAP waterfall plot: {str(e)}")
        
        return {
            'shap_values': shap_values,
            'explainer': explainer,
            'shap_importances': shap_importances
        }
        
    except Exception as e:
        print(f"  Error in SHAP explanation generation: {str(e)}")
        return None

def generate_pdp_plots(
    model,
    X: pd.DataFrame,
    top_features: List[str],
    num_grid_points: int = 20,
    output_dir: Optional[str] = None,
    n_samples: int = 1000
) -> Dict[str, Any]:
    """
    Generate Partial Dependence Plots for the model.
    
    Args:
        model: Trained model
        X: Feature data
        top_features: List of features to generate PDPs for
        num_grid_points: Number of grid points for PDP
        output_dir: Directory to save PDP plots
        n_samples: Number of samples to use for PDP calculation
        
    Returns:
        Dictionary with PDP results
    """
    if not PDPBOX_AVAILABLE:
        print("PDPbox not available. Skipping Partial Dependence Plots.")
        return None
    
    # Sample data if needed
    X_sample = X
    if len(X) > n_samples:
        X_sample = X.sample(n_samples, random_state=42)
    
    pdp_results = {}
    
    # Generate and save plots if output directory provided
    if output_dir:
        plots_dir = os.path.join(output_dir, "pdp_plots")
        os.makedirs(plots_dir, exist_ok=True)
    
    for feature in top_features:
        try:
            # Skip if feature is not in dataframe
            if feature not in X.columns:
                continue
                
            # Generate PDP for the feature
            pdp_feature = pdp.pdp_isolate(
                model=model,
                dataset=X_sample,
                model_features=X.columns,
                feature=feature,
                num_grid_points=num_grid_points
            )
            
            pdp_results[feature] = pdp_feature
            
            # Create and save plot
            if output_dir:
                fig, axes = plt.subplots(1, 2, figsize=(16, 6))
                
                # PDP plot
                pdp.pdp_plot(
                    pdp_isolate_out=pdp_feature,
                    feature_name=feature,
                    plot_lines=True,
                    center=PDP_PLOT_SETTINGS['centered'],
                    plot_pts_dist=True,
                    ax=axes[0]
                )
                
                # Distribution plot
                if pd.api.types.is_numeric_dtype(X[feature]):
                    sns.histplot(X[feature], kde=True, ax=axes[1])
                    axes[1].set_title(f'Distribution of {feature}')
                else:
                    X[feature].value_counts().plot(kind='bar', ax=axes[1])
                    axes[1].set_title(f'Frequency of {feature}')
                    plt.xticks(rotation=45)
                
                plt.tight_layout()
                pdp_path = os.path.join(plots_dir, f"pdp_{feature}.png")
                plt.savefig(pdp_path, dpi=300, bbox_inches='tight')
                plt.close()
        
        except Exception as e:
            print(f"  Warning: Could not create PDP for {feature}: {str(e)}")
    
    # Create a grid of PDPs for top features
    if output_dir and len(pdp_results) > 1:
        try:
            n_plots = min(9, len(pdp_results))
            n_cols = 3
            n_rows = (n_plots + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
            axes = axes.flatten()
            
            for i, feature in enumerate(list(pdp_results.keys())[:n_plots]):
                # Skip if too many plots
                if i >= len(axes):
                    break
                
                pdp.pdp_plot(
                    pdp_isolate_out=pdp_results[feature],
                    feature_name=feature,
                    center=PDP_PLOT_SETTINGS['centered'],
                    plot_lines=True,
                    plot_pts_dist=False,
                    ax=axes[i]
                )
            
            # Hide unused subplots
            for j in range(i+1, len(axes)):
                axes[j].set_visible(False)
            
            plt.tight_layout()
            grid_path = os.path.join(plots_dir, "pdp_grid.png")
            plt.savefig(grid_path, dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"  Warning: Could not create PDP grid: {str(e)}")
    
    if output_dir:
        print(f"  PDP plots saved to {plots_dir}")
        
    return pdp_results

def generate_ice_plots(
    model,
    X: pd.DataFrame,
    top_features: List[str],
    n_samples: int = 100,
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate Individual Conditional Expectation (ICE) plots.
    
    Args:
        model: Trained model
        X: Feature data
        top_features: List of features to generate ICE plots for
        n_samples: Number of samples to plot
        output_dir: Directory to save ICE plots
        
    Returns:
        Dictionary with ICE results
    """
    if not PYCEBOX_AVAILABLE:
        print("PyCEbox not available. Skipping ICE plots.")
        return None
    
    # Sample data if needed
    X_sample = X
    if len(X) > n_samples:
        X_sample = X.sample(n_samples, random_state=42)
    
    ice_results = {}
    
    # Generate and save plots if output directory provided
    if output_dir:
        plots_dir = os.path.join(output_dir, "ice_plots")
        os.makedirs(plots_dir, exist_ok=True)
    
    for feature in top_features:
        try:
            # Skip if feature is not in dataframe
            if feature not in X.columns:
                continue
                
            # Generate ICE dataframe
            ice_df = ice(
                X=X_sample,
                column=feature,
                predict=model.predict,
                num_grid_points=20
            )
            
            ice_results[feature] = ice_df
            
            # Create and save plot
            if output_dir:
                plt.figure(figsize=(10, 6))
                ice_plot(
                    ice_df=ice_df,
                    linewidth=0.5,
                    c="#1f77b4",
                    alpha=0.2
                )
                plt.axhline(y=np.mean(model.predict(X)), color='r', linestyle='--', label='Mean prediction')
                plt.grid(True, linestyle='--', alpha=0.5)
                plt.title(f'ICE Plot for {feature}')
                plt.xlabel(feature)
                plt.ylabel('Predicted Repayment Rate')
                plt.legend()
                
                plt.tight_layout()
                ice_path = os.path.join(plots_dir, f"ice_{feature}.png")
                plt.savefig(ice_path, dpi=300, bbox_inches='tight')
                plt.close()
        
        except Exception as e:
            print(f"  Warning: Could not create ICE plot for {feature}: {str(e)}")
    
    if output_dir:
        print(f"  ICE plots saved to {plots_dir}")
        
    return ice_results

def generate_eli5_explanation(
    model,
    X: pd.DataFrame,
    feature_names: List[str],
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate ELI5 explanations for the model.
    
    Args:
        model: Trained model
        X: Feature data
        feature_names: Feature names
        output_dir: Directory to save ELI5 explanations
        
    Returns:
        Dictionary with ELI5 results
    """
    # Try to import eli5 at runtime
    try:
        import eli5
        from eli5.sklearn import explain_weights
        eli5_available = True
    except ImportError:
        print("ELI5 not available. Skipping ELI5 explanations.")
        return None
    
    try:
        # Generate ELI5 explanation
        explanation = explain_weights(model, feature_names=feature_names)
        
        # Convert to HTML
        html = eli5.format_as_html(explanation)
        
        # Save HTML if output directory provided
        if output_dir:
            html_path = os.path.join(output_dir, "eli5_explanation.html")
            with open(html_path, 'w') as f:
                f.write(html)
            
            # Also save text version
            text = eli5.format_as_text(explanation)
            text_path = os.path.join(output_dir, "eli5_explanation.txt")
            with open(text_path, 'w') as f:
                f.write(text)
            
            print(f"  ELI5 explanations saved to {output_dir}")
        
        return {
            'explanation': explanation,
            'html': html
        }
    
    except Exception as e:
        print(f"  Warning: Could not generate ELI5 explanation: {str(e)}")
        return None
