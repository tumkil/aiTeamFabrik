"""
SoftwareTeamFabrik package initialization.

This package provides the core functionality for the autonomous AI-driven
software development factory.
"""

# Import main components for easy access
from .main import app as factory_app
from .adapters.api import app as api_app

__all__ = ["factory_app", "api_app"]
