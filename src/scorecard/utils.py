"""
Utility functions for scorecard modeling.
"""

import os

def create_version_path(base_path: str = "data/processed/scorecard_modelling") -> str:
    """
    Create a new version directory for the current modeling iteration.
    
    Args:
        base_path: Base directory for scorecard modeling outputs
        
    Returns:
        Path to the new version directory
    """
    # Ensure base path exists
    os.makedirs(base_path, exist_ok=True)
    
    # Find highest existing version
    existing_versions = [d for d in os.listdir(base_path) 
                        if os.path.isdir(os.path.join(base_path, d)) and d.startswith('v')]
    
    if not existing_versions:
        new_version = "v0"
    else:
        # Extract version numbers and find highest
        version_numbers = [int(v[1:]) for v in existing_versions if v[1:].isdigit()]
        if not version_numbers:
            new_version = "v0"
        else:
            new_version = f"v{max(version_numbers) + 1}"
    
    # Create new version directory
    version_path = os.path.join(base_path, new_version)
    os.makedirs(version_path, exist_ok=True)
    
    print(f"Created version directory: {version_path}")
    return version_path
