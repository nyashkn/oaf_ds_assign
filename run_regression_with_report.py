#!/usr/bin/env python3
"""
Run modular regression with missing value handling and generate a complete report.
"""

import os
import subprocess
import sys
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import json
from pathlib import Path

def run_regression_pipeline():
    """Run the regression pipeline with missing value handling and report generation."""
    print("=" * 80)
    print("Running OAF Loan Repayment Rate Regression Modeling with Report")
    print("=" * 80)
    
    # Step 1: Load data
    print("\n[STEP 1] Loading data...")
    features_path = "data/processed/all_features.csv"
    df = pd.read_csv(features_path)
    print(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Step 2: Handle missing values
    print("\n[STEP 2] Handling missing values...")
    
    # Create a copy to avoid modifying the original
    df_imputed = df.copy()
    
    # Identify numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col != 'client_id' and col in df.columns]
    
    # Impute missing values for numeric columns
    imputer = SimpleImputer(strategy='mean')
    df_imputed[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    
    # Save imputed dataset
    imputed_path = "data/processed/all_features_imputed.csv"
    os.makedirs(os.path.dirname(imputed_path), exist_ok=True)
    df_imputed.to_csv(imputed_path, index=False)
    print(f"Imputed data saved to: {imputed_path}")
    
    # Step 3: Run modular regression
    print("\n[STEP 3] Running modular regression...")
    cmd = [
        "python", "run_modular_regression.py",
        "--features", imputed_path,
        "--target", "sept_23_repayment_rate",
        "--sample", "5000",
        "--plots"
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print("Regression modeling completed successfully")
        
        # Find the most recent version directory
        base_path = "data/processed/regression_modelling"
        existing_versions = [d for d in os.listdir(base_path) 
                            if os.path.isdir(os.path.join(base_path, d)) and d.startswith('v')]
        
        # Extract version numbers and find highest
        version_numbers = [int(v[1:]) for v in existing_versions if v[1:].isdigit()]
        if not version_numbers:
            print("Error: No version directories found")
            return
            
        latest_version = f"v{max(version_numbers)}"
        version_path = os.path.join(base_path, latest_version)
        
        print(f"Using results from: {version_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error running regression: {e}")
        return
    
    # Step 4: Generate report
    print("\n[STEP 4] Generating PDF report...")
    reports_dir = "reports"
    os.makedirs(reports_dir, exist_ok=True)
    report_output = os.path.join(reports_dir, "loan_repayment_regression_report.pdf")
    
    report_cmd = [
        "python", "src/generate_regression_report_fixed.py",
        "--version_dir", version_path,
        "--output", report_output
    ]
    
    try:
        subprocess.run(report_cmd, check=True)
        print(f"Report successfully generated at: {report_output}")
    except subprocess.CalledProcessError as e:
        print(f"Error generating report: {e}")
        return
    
    print("\n[COMPLETE] Regression analysis and reporting completed")
    print(f"Review the full report at: {report_output}")
    
    return {
        "version_path": version_path,
        "report_path": report_output
    }

if __name__ == "__main__":
    run_regression_pipeline()
