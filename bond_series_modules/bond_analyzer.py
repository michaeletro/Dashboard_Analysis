"""
Bond series analysis module for fixed income analytics.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional


class BondAnalyzer:
    """Analyzes bond portfolio characteristics and risk metrics."""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.bonds_data = None
        
    def load_data(self, df_bonds: pd.DataFrame) -> None:
        """Load bond data for analysis."""
        self.bonds_data = df_bonds.copy()
        if self.verbose:
            print(f"📊 Loaded {len(df_bonds)} bonds for analysis")
    
    def calculate_duration_statistics(self) -> Dict[str, float]:
        """Calculate portfolio duration and convexity statistics."""
        if self.bonds_data is None:
            raise ValueError("No bond data loaded")
        
        duration = self.bonds_data['Duration'].values
        weights = self.bonds_data['Weight'].values if 'Weight' in self.bonds_data.columns else np.ones(len(duration)) / len(duration)
        
        portfolio_duration = np.average(duration, weights=weights)
        duration_variance = np.average((duration - portfolio_duration)**2, weights=weights)
        
        return {
            'portfolio_duration': portfolio_duration,
            'duration_std': np.sqrt(duration_variance),
            'min_duration': duration.min(),
            'max_duration': duration.max(),
            'avg_duration': duration.mean()
        }
    
    def yield_curve_analysis(self) -> pd.DataFrame:
        """Analyze yield curve positioning and spreads."""
        if self.bonds_data is None:
            raise ValueError("No bond data loaded")
        
        # Handle missing maturity data
        maturity_col = None
        for col in ['Maturity_Years', 'Duration']:
            if col in self.bonds_data.columns:
                maturity_col = col
                break
        
        if maturity_col is None:
            print("Warning: No maturity data found, using duration as proxy")
            # Create approximate maturity from duration
            maturity_values = self.bonds_data['Duration'] * 1.2  # Rough approximation
        else:
            maturity_values = self.bonds_data[maturity_col]
        
        # Group by maturity buckets
        maturity_buckets = pd.cut(maturity_values, 
                                bins=[0, 1, 3, 5, 10, 30], 
                                labels=['<1Y', '1-3Y', '3-5Y', '5-10Y', '10Y+'])
        
        # Build aggregation dict based on available columns
        agg_dict = {}
        if 'Yield' in self.bonds_data.columns:
            agg_dict['Yield'] = ['mean', 'std', 'count']
        if 'Duration' in self.bonds_data.columns:
            agg_dict['Duration'] = 'mean'
        if 'Credit_Spread' in self.bonds_data.columns:
            agg_dict['Credit_Spread'] = 'mean'
        
        if not agg_dict:
            # Fallback if no recognized columns
            return pd.DataFrame()
        
        try:
            curve_stats = self.bonds_data.groupby(maturity_buckets).agg(agg_dict).round(4)
            return curve_stats
        except Exception as e:
            print(f"Warning: Yield curve analysis failed: {e}")
            return pd.DataFrame()
    
    def credit_analysis(self) -> Dict[str, pd.DataFrame]:
        """Analyze credit quality distribution and spreads."""
        if self.bonds_data is None:
            raise ValueError("No bond data loaded")
        
        # Credit rating distribution
        if 'Rating' in self.bonds_data.columns:
            rating_dist = pd.DataFrame({'Count': self.bonds_data['Rating'].value_counts().sort_index()})
        else:
            rating_dist = pd.DataFrame()
        
        # Sector analysis if available
        if 'Sector' in self.bonds_data.columns:
            agg_dict = {'Yield': 'mean'} if 'Yield' in self.bonds_data.columns else {}
            if 'Duration' in self.bonds_data.columns:
                agg_dict['Duration'] = 'mean'
            if 'Credit_Spread' in self.bonds_data.columns:
                agg_dict['Credit_Spread'] = 'mean'
            
            if agg_dict:
                sector_stats = self.bonds_data.groupby('Sector').agg(agg_dict).round(4)
            else:
                sector_stats = pd.DataFrame()
        else:
            sector_stats = pd.DataFrame()
        
        return {
            'rating_distribution': rating_dist,
            'sector_analysis': sector_stats
        }


def create_synthetic_bond_data(n_bonds: int = 100, seed: int = 42) -> pd.DataFrame:
    """Create synthetic bond portfolio data for demonstration."""
    np.random.seed(seed)
    
    # Rating categories with different yield levels
    ratings = ['AAA', 'AA+', 'AA', 'AA-', 'A+', 'A', 'A-', 'BBB+', 'BBB', 'BBB-']
    rating_spreads = [0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.1, 2.5, 3.0, 3.5]  # spread over treasury
    
    sectors = ['Government', 'Corporate', 'Financial', 'Utility', 'Industrial', 'Technology']
    
    # Generate bond characteristics
    maturities = np.random.exponential(7, n_bonds)  # Average 7-year maturity
    maturities = np.clip(maturities, 0.25, 30)  # Between 3 months and 30 years
    
    # Duration calculation: higher coupon = lower duration
    # Modified duration approximation: Duration ≈ (1 - (1 + y)^(-T)) / y for bonds
    # Simplified: use maturity adjusted for coupon effect
    coupon_rates = np.random.uniform(0.01, 0.08, n_bonds)  # 1% to 8% coupons
    duration_adjustment = 1 / (1 + coupon_rates/2)  # Semi-annual coupon effect
    durations = maturities * duration_adjustment * np.random.uniform(0.85, 0.95, n_bonds)
    
    # Base treasury yield curve (simplified)
    treasury_yields = 0.02 + 0.025 * (1 - np.exp(-maturities / 5))  # Rising curve
    
    # Assign ratings and credit spreads
    assigned_ratings = np.random.choice(ratings, n_bonds, p=[0.05, 0.08, 0.12, 0.15, 0.15, 0.15, 0.12, 0.08, 0.07, 0.03])
    credit_spreads = np.array([rating_spreads[ratings.index(r)] for r in assigned_ratings]) / 100
    
    # Add random noise to spreads
    credit_spreads *= np.random.uniform(0.8, 1.2, n_bonds)
    
    df = pd.DataFrame({
        'Bond_ID': [f'BOND_{i:03d}' for i in range(1, n_bonds + 1)],
        'Issuer': [f'Issuer_{i}' for i in range(1, n_bonds + 1)],
        'Maturity_Years': maturities,
        'Duration': durations,
        'Treasury_Yield': treasury_yields,
        'Credit_Spread': credit_spreads,
        'Yield': treasury_yields + credit_spreads,
        'Rating': assigned_ratings,
        'Sector': np.random.choice(sectors, n_bonds),
        'Face_Value': np.random.uniform(1000, 10000, n_bonds),  # Bond face values
        'Market_Value': np.random.uniform(95, 105, n_bonds),    # Price as % of face value
        'Weight': np.random.dirichlet(np.ones(n_bonds))         # Portfolio weights sum to 1
    })
    
    # Add some time series data for yield changes
    df['1D_Yield_Change'] = np.random.normal(0, 0.05, n_bonds)  # 5bp daily volatility
    df['1W_Yield_Change'] = np.random.normal(0, 0.15, n_bonds)  # 15bp weekly
    df['1M_Yield_Change'] = np.random.normal(0, 0.30, n_bonds)  # 30bp monthly
    
    return df


def calculate_bond_portfolio_risk(df_bonds: pd.DataFrame) -> Dict[str, float]:
    """Calculate portfolio-level risk metrics."""
    weights = np.asarray(df_bonds['Weight'].values, dtype=float)
    durations = np.asarray(df_bonds['Duration'].values, dtype=float)
    yields = np.asarray(df_bonds['Yield'].values, dtype=float)
    
    # Portfolio duration (DV01)
    portfolio_duration = np.sum(weights * durations)
    
    # Average yield
    portfolio_yield = np.sum(weights * yields)
    
    # Credit quality metrics
    if 'Rating' in df_bonds.columns:
        # Simple credit score: AAA=1, AA+=2, ..., BBB-=10
        rating_map = {'AAA': 1, 'AA+': 2, 'AA': 3, 'AA-': 4, 'A+': 5, 'A': 6, 'A-': 7, 'BBB+': 8, 'BBB': 9, 'BBB-': 10}
        credit_scores = df_bonds['Rating'].map(rating_map).fillna(10)
        avg_credit_quality = np.average(np.asarray(credit_scores.values, dtype=float), weights=weights)
    else:
        avg_credit_quality = 5.0  # Default to A-rated
    
    # Calculate face value sum
    face_value_col = 'Face_Value' if 'Face_Value' in df_bonds.columns else None
    if face_value_col:
        total_face_value = float(df_bonds[face_value_col].sum())
    else:
        total_face_value = float(len(df_bonds) * 1000)  # Assume $1000 face value
    
    return {
        'portfolio_duration': float(portfolio_duration),
        'portfolio_yield': float(portfolio_yield),
        'average_credit_quality': float(avg_credit_quality),
        'total_face_value': total_face_value,
        'number_of_bonds': len(df_bonds)
    }