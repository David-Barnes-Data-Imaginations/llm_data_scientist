"""
Main source package for the data scientist project.

This package contains modules for client interactions, utility functions,
and data handling operations.
"""

# Import main modules
from . import client
from . import utils
from . import data

__all__ = [
    'client',
    'utils',
    'data'
]
