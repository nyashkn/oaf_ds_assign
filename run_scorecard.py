#!/usr/bin/env python
"""
Wrapper script to run the modular scorecard with the correct Python path.

This ensures the src directory is in the Python path for imports.
"""

import os
import sys

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import and run the main module
from src.run_scorecard import main

if __name__ == "__main__":
    sys.exit(main())
