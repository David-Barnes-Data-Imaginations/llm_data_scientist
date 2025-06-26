"""
Client module for the data scientist project.

This module provides functionality for interacting with MCP clients,
setting up LLM configurations, and working with code agents.
"""

from .mcp_client import create_mcp_client, list_tools
from .agent import CustomAgent
from .telemetry import TelemetryManager

__all__ = [
    'create_mcp_client',
    'list_tools',
    'CustomAgent',
    'TelemetryManager'
]
