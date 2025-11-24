"""
Bloomberg Analysis Package

A comprehensive equity factor analysis package for Bloomberg data.
Provides tools for factor construction, cross-sectional regression,
peer analysis, clustering, and portfolio construction.

Author: Derivative Trading Strategy Team
Date: November 2025
"""

from .data_loader import DataLoader, read_excel_advanced
from .factor_analysis import FactorAnalyzer, zscore_matrix
from .regression import CrossSectionalRegressor
from .peer_analysis import PeerAnalyzer
from .clustering import FactorClusterer
from .portfolio import PortfolioConstructor
from .utils import setup_plotting, check_dependencies, plot_series

__version__ = "1.0.0"
__author__ = "Derivative Trading Strategy Team"

# Package-level imports for convenience
__all__ = [
    "DataLoader",
    "FactorAnalyzer", 
    "CrossSectionalRegressor",
    "PeerAnalyzer",
    "FactorClusterer", 
    "PortfolioConstructor",
    "setup_plotting",
    "check_dependencies",
    "zscore_matrix",
    "read_excel_advanced"
]