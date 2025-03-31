"""
Scorecard modeling and evaluation functions.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import scorecardpy as sc
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from typing import Dict, List, Optional, Tuple, Any, Union

def get_classifier(
    classifier_type: str = 'logistic',
    handle_missing: bool = False,
    **kwargs
) -> Union[LogisticRegression, Pipeline]:
    """
    Get a classifier based on specified type and configuration.
    
    Args:
        classifier_type: Type of classifier ('logistic' or 'histgb')
        handle_missing: Whether to include an imputer in the pipeline
        **kwargs: Additional arguments for the classifier
        
    Returns:
        Configured classifier or pipeline
    """
    if classifier_type == 'logistic':
        clf = LogisticRegression(
            penalty=kwargs.get('penalty', 'l1'),
            C=kwargs.get('C', 0.9),
            solver=kwargs.get('solver', 'saga'),
            random_state=kwargs.get('random_state', 42)
        )
    elif classifier_type == 'histgb':
        clf = HistGradientBoostingClassifier(
            max_iter=kwargs.get('max_iter', 100),
            learning_rate=kwargs.get('learning_rate', 0.1),
            max_depth=kwargs.get('max_depth', 3),
            random_state=kwargs.get('random_state', 42)
        )
    else:
        raise ValueError(f"Unsupported classifier type: {classifier_type}")
    
    if handle_missing:
        return Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('classifier', clf)
        ])
    
    return clf

def develop_scorecard_model(
    train_woe: pd.DataFrame,
    test_woe: pd.DataFrame,
    bins: Dict,
    target_var: str,
    output_dir: Optional[str] = None,
    classifier_type: str = 'logistic',
    handle_missing: bool = False,
    classifier_params: Optional[Dict] = None,
    random_state: int = 42
) -> Dict:
    """
    Develop a scorecard model using specified classifier.
    
    Args:
        train_woe: WOE-transformed training data
        test_woe: WOE-transformed testing data
        bins: WOE binning information
        target_var: Target variable name
        output_dir: Directory to save model results
        classifier_type: Type of classifier to use
        handle_missing: Whether to handle missing values in pipeline
        classifier_params: Additional parameters for classifier
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary with scorecard model and results
    """
    print("\n=== Scorecard Development ===")
    print(f"Using classifier: {classifier_type}")
    
    # Prepare data for modeling
    X_train = train_woe.drop(columns=[target_var])
    y_train = train_woe[target_var]
    X_test = test_woe.drop(columns=[target_var])
    y_test = test_woe[target_var]
    
    # Get classifier with specified configuration
    params = classifier_params or {}
    params['random_state'] = random_state
    clf = get_classifier(classifier_type, handle_missing, **params)
    
    # Fit model
    clf.fit(X_train, y_train)
    
    # Get model coefficients if available
    if hasattr(clf, 'coef_'):
        coef = clf.coef_[0]
    elif hasattr(clf, 'named_steps') and hasattr(clf.named_steps['classifier'], 'coef_'):
        coef = clf.named_steps['classifier'].coef_[0]
    else:
        coef = np.zeros(len(X_train.columns))  # Placeholder for models without coefficients
    
    coef_df = pd.DataFrame({
        'variable': X_train.columns,
        'coefficient': coef
    }).sort_values('coefficient', ascending=False)
    
    print("\nModel coefficients (top 10):")
    for _, row in coef_df.head(10).iterrows():
        print(f"  {row['variable']}: {row['coefficient']:.4f}")
    
    # Calculate predicted probabilities
    train_pred = clf.predict_proba(X_train)[:, 1]
    test_pred = clf.predict_proba(X_test)[:, 1]
    
    # Create scorecard (only for logistic regression)
    if classifier_type == 'logistic':
        if handle_missing:
            card = sc.scorecard(bins, clf.named_steps['classifier'], X_train.columns)
        else:
            card = sc.scorecard(bins, clf, X_train.columns)
        
        # Apply scorecard to get scores
        train_score = sc.scorecard_ply(train_woe, card, print_step=0)
        test_score = sc.scorecard_ply(test_woe, card, print_step=0)
        
        print(f"\nScorecard created with {len(card)} components")
        print(f"Score ranges: Train [{train_score['score'].min()}, {train_score['score'].max()}], "
              f"Test [{test_score['score'].min()}, {test_score['score'].max()}]")
    else:
        card = None
        train_score = pd.DataFrame({'score': train_pred})
        test_score = pd.DataFrame({'score': test_pred})
        print("\nNote: Scorecard not created for non-logistic classifier")
    
    # Save results if output_dir provided
    if output_dir:
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        if card is not None:
            # Save scorecard as CSV (combine all components)
            card_df = pd.concat(card.values())
            card_path = os.path.join(output_dir, "scorecard.csv")
            card_df.to_csv(card_path, index=False)
            print(f"\nScorecard saved to {card_path}")
        
        # Save scores
        train_score_path = os.path.join(output_dir, "train_scores.csv")
        test_score_path = os.path.join(output_dir, "test_scores.csv")
        
        train_score.to_csv(train_score_path, index=False)
        test_score.to_csv(test_score_path, index=False)
        
        # Save model coefficients
        coef_path = os.path.join(output_dir, "model_coefficients.csv")
        coef_df.to_csv(coef_path, index=False)
        
        print(f"Training scores saved to {train_score_path}")
        print(f"Testing scores saved to {test_score_path}")
        print(f"Model coefficients saved to {coef_path}")
    
    # Results dictionary
    return {
        'card': card,
        'scorecard_df': pd.concat(card.values()) if card else None,
        'model': clf,
        'coefficients': coef_df,
        'predictions': {
            'train': train_pred,
            'test': test_pred
        },
        'scores': {
            'train': train_score,
            'test': test_score
        }
    }

def evaluate_model_performance(
    train_actual: pd.Series,
    test_actual: pd.Series,
    train_pred: np.ndarray,
    test_pred: np.ndarray,
    output_dir: Optional[str] = None
) -> Dict:
    """
    Evaluate model performance using various metrics.
    
    Args:
        train_actual: Actual target values for training set
        test_actual: Actual target values for testing set
        train_pred: Predicted probabilities for training set
        test_pred: Predicted probabilities for testing set
        output_dir: Directory to save evaluation results
        
    Returns:
        Dictionary with performance metrics
    """
    print("\n=== Model Performance Evaluation ===")
    
    # Calculate performance metrics using scorecardpy
    train_perf = sc.perf_eva(train_actual, train_pred, title="Train", show_plot=False)
    test_perf = sc.perf_eva(test_actual, test_pred, title="Test", show_plot=False)
    
    # Print key metrics
    print("\nTraining Performance:")
    print(f"  KS: {train_perf['KS']:.4f}")
    print(f"  AUC: {train_perf['AUC']:.4f}")
    print(f"  Gini: {train_perf['Gini']:.4f}")
    
    print("\nTesting Performance:")
    print(f"  KS: {test_perf['KS']:.4f}")
    print(f"  AUC: {test_perf['AUC']:.4f}")
    print(f"  Gini: {test_perf['Gini']:.4f}")
    
    # Calculate PSI
    score = {
        'train': pd.DataFrame({'score': train_pred}),
        'test': pd.DataFrame({'score': test_pred})
    }
    label = {
        'train': pd.Series(train_actual),
        'test': pd.Series(test_actual)
    }
    
    try:
        psi_result = sc.perf_psi(
            score=score,
            label=label,
            return_distr_dat=True,  # Ensure we get distribution data
            show_plot=False  # Prevent interactive plots
        )
    except Exception as e:
        print(f"Warning: PSI calculation failed - {str(e)}")
        psi_result = {
            'psi': pd.DataFrame({'PSI': [float('nan')]}),
            'pic': None
        }
    
    print(f"\nPopulation Stability Index (PSI): {psi_result['psi']['PSI'].values[0]:.4f}")
    
    # Save results if output_dir provided
    if output_dir:
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save performance metrics
        metrics = {
            'train': {
                'KS': float(train_perf['KS']),
                'AUC': float(train_perf['AUC']),
                'Gini': float(train_perf['Gini'])
            },
            'test': {
                'KS': float(test_perf['KS']),
                'AUC': float(test_perf['AUC']),
                'Gini': float(test_perf['Gini'])
            },
            'psi': float(psi_result['psi']['PSI'].values[0])
        }
        
        metrics_path = os.path.join(output_dir, "performance_metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Save plots
        if 'pic' in train_perf:
            train_plot_path = os.path.join(output_dir, "train_performance.png")
            train_perf['pic'].savefig(train_plot_path)
            plt.close(train_perf['pic'])
        
        if 'pic' in test_perf:
            test_plot_path = os.path.join(output_dir, "test_performance.png")
            test_perf['pic'].savefig(test_plot_path)
            plt.close(test_perf['pic'])
        
        if 'pic' in psi_result:
            psi_plot_path = os.path.join(output_dir, "psi_plot.png")
            list(psi_result['pic'].values())[0].savefig(psi_plot_path)
            plt.close(list(psi_result['pic'].values())[0])
        
        print(f"\nPerformance metrics saved to {metrics_path}")
        print("Performance plots saved to output directory")
    
    # Return performance metrics
    return {
        'train_perf': train_perf,
        'test_perf': test_perf,
        'psi': psi_result
    }
