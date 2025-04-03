"""
Run the Threshold and Profitability Metrics Explainer

This script runs the threshold explainer module to generate visualizations
and explanations of how loan repayment thresholds affect profitability metrics.

Usage:
    python src/run_threshold_explainer.py [--data PATH] [--output PATH]

Options:
    --data PATH    Path to sample metrics data (default: model1_tradeoffs.json)
    --output PATH  Directory to save visualizations and explanations
"""

import os
import json
import argparse
from src.scorecard_regression.threshold_explanation import (
    explain_threshold_concept,
    explain_profit_calculation,
    explain_threshold_profitability_tradeoff,
    explain_real_world_example,
    explain_optimal_threshold_selection,
    explain_sensitivity_analysis
)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run threshold and profitability metrics explainer")
    parser.add_argument("--data", default="data/processed/model_comparison/model1_tradeoffs.json",
                      help="Path to sample metrics data")
    parser.add_argument("--output", default="data/processed/threshold_explanation",
                      help="Directory to save visualizations and explanations")
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output, exist_ok=True)
    
    print("===== THRESHOLD AND PROFITABILITY METRICS EXPLANATION =====")
    print(
        "This script explains how thresholds work in loan repayment prediction models\n"
        "and how they affect profitability metrics like ROI and profit."
    )
    
    # Run basic concept explanations
    explain_threshold_concept(args.output)
    explain_profit_calculation(args.output)
    explain_threshold_profitability_tradeoff(args.output)
    
    # Try to load real data for more detailed examples
    print("\nAttempting to load sample metrics data for real-world examples...")
    threshold_data = {}
    try:
        with open(args.data, 'r') as f:
            threshold_data = json.load(f)
            print(f"Successfully loaded sample data from {args.data}")
        
        # Run detailed explanations with real data
        explain_real_world_example(threshold_data, args.output)
        explain_optimal_threshold_selection(threshold_data)
        explain_sensitivity_analysis(threshold_data, args.output)
        
        # Save a copy of the explanations to a text file
        with open(os.path.join(args.output, "threshold_explanation.txt"), "w") as f:
            f.write("===== THRESHOLD AND PROFITABILITY METRICS EXPLANATION =====\n\n")
            f.write("This document explains the key concepts of threshold-based loan approval and its impact on profitability.\n\n")
            
            # Include business parameters
            f.write("Business Parameters:\n")
            f.write(f"- Profit Margin: {threshold_data.get('business_parameters', {}).get('margin', 0.16):.0%}\n")
            f.write(f"- Default Loss Rate: {threshold_data.get('business_parameters', {}).get('default_loss_rate', 1.0):.0%}\n\n")
            
            # Include recommendations
            recommendations = threshold_data.get('recommendations', {})
            f.write("Recommended Thresholds:\n")
            f.write(f"- Profit-focused: {recommendations.get('profit_focused', 0.5):.2f}\n")
            f.write(f"- ROI-focused: {recommendations.get('roi_focused', 0.8):.2f}\n")
            f.write(f"- Balanced approach: {recommendations.get('balanced', 0.65):.2f}\n\n")
            
            # Summarize optimal thresholds
            optimal = threshold_data.get('optimal_thresholds', {})
            f.write("Optimal Thresholds by Metric:\n")
            for metric, details in optimal.items():
                f.write(f"- {metric.capitalize()} optimizing threshold: {details.get('threshold', 0):.2f}\n")
                f.write(f"  Value: {details.get('value', 0):.4f}\n")
                f.write(f"  Approval rate: {details.get('approval_rate', 0):.1%}\n")
            
            f.write("\nFor visual explanations, see the PNG files in this directory.\n")
        
        print(f"\nExplanations complete. Generated visualizations saved to: {args.output}")
        print(f"Summary explanation saved to: {os.path.join(args.output, 'threshold_explanation.txt')}")
        
    except Exception as e:
        print(f"Error loading or processing sample data: {e}")
        print("Basic explanations were generated, but detailed examples with real data were not available.")
    
    print("\nTo understand threshold concepts and profitability metrics, review the generated visualizations.")
    print("For further understanding, the full source code in src/scorecard_regression/threshold_explanation.py")
    print("provides detailed explanations of each concept.")

if __name__ == "__main__":
    main()
