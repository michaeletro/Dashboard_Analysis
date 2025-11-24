"""
Bond series analysis module for fixed income analytics.
"""

from .bond_analyzer import BondAnalyzer, create_synthetic_bond_data, calculate_bond_portfolio_risk

__version__ = "1.0.0"
__all__ = ["BondAnalyzer", "create_synthetic_bond_data", "calculate_bond_portfolio_risk"]