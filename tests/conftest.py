"""
pytest configuration for tensor debugger tests.

Sets up test environment and common fixtures.
"""

import pytest
import sys
import os

# Add parent directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure pytest to show more detailed output
def pytest_configure(config):
    """Configure pytest settings"""
    config.option.verbose = True