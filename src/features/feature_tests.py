# src/features/feature_tests.py

import time
import traceback
from typing import List, Dict, Any, Callable, Optional
import polars as pl
import pandas as pd
import numpy as np
import datetime

from .feature_registry import feature_registry

def test(feature_name=None, description=None):
    """
    Decorator to register a test for a feature
    
    Args:
        feature_name: Name of the feature to test
        description: Test description
    """
    def decorator(func):
        func._is_feature_test = True
        func._feature_name = feature_name
        func._test_description = description
        return func
    return decorator

def run_tests(feature_name: Optional[str] = None, verbose: bool = True) -> Dict[str, Any]:
    """
    Run all tests for a feature or all features
    
    Args:
        feature_name: Specific feature to test, or None for all tests
        verbose: Whether to print detailed test results
        
    Returns:
        Dictionary with test results
    """
    import inspect
    import sys
    
    # Find all test functions
    test_functions = []
    for module_name, module in sys.modules.items():
        if module_name.startswith('__'):
            continue
            
        for name, obj in inspect.getmembers(module):
            if inspect.isfunction(obj) and hasattr(obj, '_is_feature_test'):
                if feature_name is None or obj._feature_name == feature_name:
                    test_functions.append(obj)
    
    # Run tests
    results = {"passed": [], "failed": []}
    
    for test_func in test_functions:
        feature = test_func._feature_name
        test_name = test_func.__name__
        description = test_func._test_description or ""
        
        if verbose:
            print(f"Running test: {test_name} for feature '{feature}'")
            if description:
                print(f"  Description: {description}")
        
        start_time = time.time()
        
        try:
            test_func()
            execution_time = time.time() - start_time
            
            # Record test result
            if feature:
                feature_registry.record_test_result(
                    feature, test_name, True, "", execution_time
                )
            
            results["passed"].append({
                "feature": feature,
                "test_name": test_name,
                "execution_time": execution_time
            })
            
            if verbose:
                print(f"  ✓ PASSED in {execution_time:.3f}s")
                
        except Exception as e:
            execution_time = time.time() - start_time
            error_message = str(e)
            trace = traceback.format_exc()
            
            # Record test result
            if feature:
                feature_registry.record_test_result(
                    feature, test_name, False, error_message, execution_time
                )
            
            results["failed"].append({
                "feature": feature,
                "test_name": test_name,
                "error": error_message,
                "trace": trace,
                "execution_time": execution_time
            })
            
            if verbose:
                print(f"  ✗ FAILED in {execution_time:.3f}s")
                print(f"    Error: {error_message}")
    
    # Summary
    if verbose:
        print(f"\nTest Results: {len(results['passed'])} passed, {len(results['failed'])} failed")
    
    return results

def generate_test_report(results: Dict[str, Any], output_path: Optional[str] = None) -> str:
    """Generate a test report in Markdown format"""
    report = "# Feature Test Report\n\n"
    
    passed = len(results["passed"])
    failed = len(results["failed"])
    total = passed + failed
    
    report += f"**Summary:** {passed}/{total} passed ({passed/total*100:.1f}%)\n\n"
    
    # Failed tests
    if failed > 0:
        report += "## Failed Tests\n\n"
        for test in results["failed"]:
            report += f"### {test['test_name']} ({test['feature']})\n\n"
            report += f"**Error:** {test['error']}\n\n"
            report += "**Stack Trace:**\n\n```\n"
            report += test['trace']
            report += "```\n\n"
    
    # Passed tests
    if passed > 0:
        report += "## Passed Tests\n\n"
        report += "| Feature | Test | Execution Time |\n"
        report += "|---------|------|---------------|\n"
        
        for test in results["passed"]:
            report += f"| {test['feature']} | {test['test_name']} | {test['execution_time']:.3f}s |\n"
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(report)
        print(f"Test report written to {output_path}")
    
    return report

# Sample tests

@test(feature_name="deposit_ratio", description="Test deposit ratio calculation")
def test_deposit_ratio_basic():
    """Test basic deposit ratio calculation"""
    test_df = pl.DataFrame({
        "deposit_amount": [100, 200, 500],
        "nominal_contract_value": [1000, 1000, 2000]
    })
    result = feature_registry.apply_feature(test_df, "deposit_ratio")
    expected = [0.1, 0.2, 0.25]
    actual = result["deposit_ratio"].to_list()
    assert all(abs(a - e) < 1e-6 for a, e in zip(actual, expected)), \
        f"Expected {expected} but got {actual}"

@test(feature_name="historical_cum_loans_region", description="Test historical loans count")
def test_historical_cum_loans_region():
    """Test historical cumulative loans for a region"""
    # Create test data with different regions and dates
    test_df = pl.DataFrame({
        "client_id": ["A", "B", "C", "D", "E"],
        "region": ["West", "West", "East", "West", None],
        "contract_start_date": [
            datetime.datetime(2023, 1, 1),
            datetime.datetime(2023, 1, 15), 
            datetime.datetime(2023, 1, 20),
            datetime.datetime(2023, 2, 1),
            datetime.datetime(2023, 2, 15)
        ]
    })
    
    result = feature_registry.apply_feature(test_df, "historical_cum_loans_region")
    
    # Expected: 
    # A: First loan in West region = 0 previous loans
    # B: Second loan in West region after A = 1 previous loan
    # C: First loan in East region = 0 previous loans
    # D: Third loan in West region after A and B = 2 previous loans
    # E: First loan with null region = 0 previous loans
    expected = [0, 1, 0, 2, 0]
    actual = result["historical_cum_loans_region"].to_list()
    
    assert expected == actual, f"Expected {expected} but got {actual}"