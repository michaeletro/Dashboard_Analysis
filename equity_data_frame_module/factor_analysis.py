"""
Factor analysis utilities for Bloomberg equity analysis.
"""

import pandas as pd
import numpy as np

def zscore_matrix(X):
    """
    Compute z-score standardization using linear algebra.
    
    Parameters:
    -----------
    X : numpy.ndarray
        Input matrix of shape (n_samples, n_features)
    
    Returns:
    --------
    Z : numpy.ndarray
        Standardized matrix
    mu : numpy.ndarray
        Vector of means
    sigma : numpy.ndarray
        Vector of standard deviations
    """
    mu = np.nanmean(X, axis=0)
    sigma = np.nanstd(X, axis=0, ddof=1)
    
    # Avoid division by zero
    sigma = np.where(sigma == 0, 1, sigma)
    
    Z = (X - mu) / sigma
    return Z, mu, sigma

class FactorAnalyzer:
    """Constructs multi-factor scores using advanced financial metrics."""
    
    def __init__(self):
        self.factor_definitions = {
            'value': {
                'metrics': ['PE Ratio (TTM)', 'Forward PE', 'Price/Sales (TTM)', 'EV/EBITDA (TTM)', 'Price/Book (TTM)'],
                'invert': True,  # Lower values = better (higher factor score)
                'description': 'Cheaper stocks get higher scores'
            },
            'quality': {
                'positive_metrics': ['ROE (%)', 'ROA (%)', 'Net Margin (%)', 'Current Ratio'],
                'negative_metrics': ['Debt/Equity'],  # Lower is better
                'description': 'Higher profitability and efficiency'
            },
            'momentum': {
                'metrics': ['1W % Change', '1M % Change'],
                'scale': 0.01,  # Convert percentage to decimal
                'description': 'Recent price performance'
            },
            'growth': {
                'metrics': ['Revenue Growth YoY (%)', 'EPS Growth YoY (%)'],
                'description': 'Revenue and earnings growth'
            },
            'size': {
                'metrics': ['Market Cap (USD Billion)'],
                'transform': 'log',  # Log transformation for size
                'description': 'Market capitalization'
            }
        }
    
    def construct_factor_scores(self, df):
        """
        Construct multi-factor scores using advanced financial metrics.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input DataFrame with financial metrics
            
        Returns:
        --------
        dict : Dictionary with factor names as keys and score arrays as values
        """
        print("🔬 Constructing Factor Scores...")
        print("=" * 50)
        
        factor_scores = {}
        
        # 1. VALUE FACTOR
        factor_scores['ValueScore'] = self._compute_value_factor(df)
        
        # 2. QUALITY FACTOR
        factor_scores['QualityScore'] = self._compute_quality_factor(df)
        
        # 3. MOMENTUM FACTOR
        factor_scores['MomentumScore'] = self._compute_momentum_factor(df)
        
        # 4. GROWTH FACTOR
        factor_scores['GrowthScore'] = self._compute_growth_factor(df)
        
        # 5. SIZE FACTOR
        factor_scores['SizeScore'] = self._compute_size_factor(df)
        
        return factor_scores
    
    def _compute_value_factor(self, df):
        """Compute value factor scores."""
        value_metrics = self.factor_definitions['value']['metrics']
        available_value = [col for col in value_metrics if col in df.columns]
        
        if available_value:
            V = df[available_value].values
            V_inv = -V  # Invert so cheaper = higher score
            Z_V, _, _ = zscore_matrix(V_inv)
            value_score = np.nanmean(Z_V, axis=1)
            print(f"✓ Value Factor: {len(available_value)} metrics")
        else:
            value_score = np.zeros(len(df))
            print("⚠ Value Factor: No metrics available")
        
        return value_score
    
    def _compute_quality_factor(self, df):
        """Compute quality factor scores."""
        quality_def = self.factor_definitions['quality']
        available_quality = [col for col in quality_def['positive_metrics'] if col in df.columns]
        available_quality_neg = [col for col in quality_def['negative_metrics'] if col in df.columns]
        
        if available_quality or available_quality_neg:
            Q_pos = df[available_quality].values if available_quality else np.empty((len(df), 0))
            Q_neg = -df[available_quality_neg].values if available_quality_neg else np.empty((len(df), 0))
            
            Q = np.concatenate([Q_pos, Q_neg], axis=1) if Q_pos.shape[1] > 0 or Q_neg.shape[1] > 0 else np.zeros((len(df), 1))
            Z_Q, _, _ = zscore_matrix(Q)
            quality_score = np.nanmean(Z_Q, axis=1)
            print(f"✓ Quality Factor: {len(available_quality)} positive + {len(available_quality_neg)} negative metrics")
        else:
            quality_score = np.zeros(len(df))
            print("⚠ Quality Factor: No metrics available")
        
        return quality_score
    
    def _compute_momentum_factor(self, df):
        """Compute momentum factor scores."""
        momentum_def = self.factor_definitions['momentum']
        momentum_metrics = momentum_def['metrics']
        available_momentum = [col for col in momentum_metrics if col in df.columns]
        
        if available_momentum:
            M = df[available_momentum].values * momentum_def['scale']  # Convert to decimal
            Z_M, _, _ = zscore_matrix(M)
            momentum_score = np.nanmean(Z_M, axis=1)
            print(f"✓ Momentum Factor: {len(available_momentum)} metrics")
        else:
            momentum_score = np.zeros(len(df))
            print("⚠ Momentum Factor: No metrics available")
        
        return momentum_score
    
    def _compute_growth_factor(self, df):
        """Compute growth factor scores."""
        growth_def = self.factor_definitions['growth']
        growth_metrics = growth_def['metrics']
        available_growth = [col for col in growth_metrics if col in df.columns]
        
        if available_growth:
            G = df[available_growth].values
            Z_G, _, _ = zscore_matrix(G)
            growth_score = np.nanmean(Z_G, axis=1)
            print(f"✓ Growth Factor: {len(available_growth)} metrics")
        else:
            growth_score = np.zeros(len(df))
            print("⚠ Growth Factor: No metrics available")
        
        return growth_score
    
    def _compute_size_factor(self, df):
        """Compute size factor scores."""
        size_def = self.factor_definitions['size']
        
        if size_def['metrics'][0] in df.columns:
            size_log = np.log(df[size_def['metrics'][0]].values + 1e-6)  # Add small constant to avoid log(0)
            Z_S, _, _ = zscore_matrix(size_log.reshape(-1, 1))
            size_score = Z_S[:, 0]
            print("✓ Size Factor: Log market cap")
        else:
            size_score = np.zeros(len(df))
            print("⚠ Size Factor: No market cap data")
        
        return size_score