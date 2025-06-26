"""
Splitwise Clone Backend Package

This package contains the FastAPI backend for a Splitwise clone application.
"""

import os
from pathlib import Path

# Package metadata
__version__ = "1.0.0"
__title__ = "Splitwise Clone Backend"
__description__ = "FastAPI backend for expense sharing application"

# Package-level imports for easier access
from .database import get_db
from .main import app

# Package-level constants
ROOT_DIR = Path(__file__).parent
CONFIG_DIR = ROOT_DIR / "config"

# Environment-based configuration
ENV = os.getenv("ENVIRONMENT", "development")

# Export public API
__all__ = [
    "app",
    "get_db",
    "__version__",
    "__title__",
    "__description__"
]
