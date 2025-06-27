"""
Client module for the data scientist project.

This module provides functionality for setting up LLM configurations
and working with code agents.
"""

from .agent import CustomAgent
from .telemetry import TelemetryManager
from . import ui

__all__ = [
    'CustomAgent',
    'TelemetryManager',
    'ui'
]
