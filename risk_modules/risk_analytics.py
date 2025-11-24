"""
Risk Management Module - Using Existing Portfolio Data
Enhanced risk analytics for equity and bond portfolios using current data structure.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Tuple, Optional, List
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


class PortfolioRiskManager:
    """Advanced risk analytics using existing portfolio data structure."""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        
    def calculate_portfolio_var(
        self, 
        returns: np.ndarray, 
        confidence_level: float = 0.05,
        method: str = 'historical'
    ) -> Dict[str, float]:
        """
        Calculate Value at Risk using multiple methodologies.
        
        Args:
            returns: Array of portfolio returns (use 1W or 1M % Change columns)
            confidence_level: VaR confidence level (default 5% = 95% VaR)
            method: 'historical', 'parametric', or 'monte_carlo'
        """
        results = {}
        
        # Historical VaR
        if method in ['historical', 'all']:
            historical_var = np.percentile(returns, confidence_level * 100)
            results['Historical_VaR'] = historical_var
            
        # Parametric VaR (assumes normal distribution)
        if method in ['parametric', 'all']:
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            parametric_var = mean_return + norm.ppf(confidence_level) * std_return
            results['Parametric_VaR'] = parametric_var
            
        # Expected Shortfall (Conditional VaR)
        if method in ['historical', 'all']:
            var_threshold = results.get('Historical_VaR', np.percentile(returns, confidence_level * 100))
            tail_returns = returns[returns <= var_threshold]
            expected_shortfall = np.mean(tail_returns) if len(tail_returns) > 0 else var_threshold
            results['Expected_Shortfall'] = expected_shortfall
            
        return results
    
    def factor_risk_attribution(
        self, 
        df_equity: pd.DataFrame,
        portfolio_weights: Optional[np.ndarray] = None
    ) -> pd.DataFrame:
        """
        Decompose portfolio risk by factor exposures using existing factor scores.
        
        Args:
            df_equity: DataFrame with factor scores (ValueScore, QualityScore, etc.)
            portfolio_weights: Portfolio weights (equal weight if None)
        """
        factor_cols = [col for col in df_equity.columns if col.endswith('Score')]
        
        if len(factor_cols) == 0:
            raise ValueError("No factor score columns found")
            
        if portfolio_weights is None:
            portfolio_weights = np.ones(len(df_equity)) / len(df_equity)
            
        # Calculate portfolio factor exposures
        factor_exposures = {}
        for factor in factor_cols:
            exposure = np.sum(portfolio_weights * df_equity[factor].values)
            factor_exposures[factor] = exposure
            
        # Calculate factor risk contributions
        factor_data = df_equity[factor_cols].values
        factor_cov = np.cov(factor_data.T)
        
        # Portfolio factor loadings
        portfolio_factors = factor_data.T @ portfolio_weights
        
        # Risk contributions
        risk_contributions = {}
        total_var = portfolio_factors.T @ factor_cov @ portfolio_factors
        
        for i, factor in enumerate(factor_cols):
            marginal_contrib = 2 * (factor_cov @ portfolio_factors)[i] * portfolio_factors[i]
            risk_contributions[f'{factor}_Risk_Contrib'] = marginal_contrib / total_var if total_var > 0 else 0
            
        # Combine results
        attribution_df = pd.DataFrame([{
            'Factor': factor.replace('Score', ''),
            'Exposure': factor_exposures[factor],
            'Risk_Contribution': risk_contributions[f'{factor}_Risk_Contrib']
        } for factor in factor_cols])
        
        return attribution_df
    
    def sector_concentration_analysis(
        self, 
        df_equity: pd.DataFrame,
        portfolio_weights: Optional[np.ndarray] = None
    ) -> pd.DataFrame:
        """
        Analyze sector concentration risk using existing sector classifications.
        """
        if 'Sector' not in df_equity.columns:
            raise ValueError("Sector column not found in data")
            
        if portfolio_weights is None:
            portfolio_weights = np.ones(len(df_equity)) / len(df_equity)
            
        # Calculate sector weights
        df_analysis = df_equity.copy()
        df_analysis['Weight'] = portfolio_weights
        
        sector_analysis = df_analysis.groupby('Sector').agg({
            'Weight': 'sum',
            '1M % Change': ['mean', 'std', 'count'],
            'ValueScore': 'mean',
            'QualityScore': 'mean'
        }).round(4)
        
        # Flatten column names
        sector_analysis.columns = ['_'.join(col).strip() for col in sector_analysis.columns]
        sector_analysis = sector_analysis.reset_index()
        
        # Calculate concentration metrics
        weights = sector_analysis['Weight_sum'].values
        sector_analysis['HHI_Contribution'] = weights ** 2
        
        # Add risk metrics
        if '1M % Change_std' in sector_analysis.columns:
            sector_analysis['Risk_Contribution'] = (
                sector_analysis['Weight_sum'] * sector_analysis['1M % Change_std']
            )
            
        return sector_analysis
    
    def correlation_regime_analysis(
        self, 
        df_equity: pd.DataFrame,
        lookback_periods: List[int] = [20, 60, 120]
    ) -> Dict[str, np.ndarray]:
        """
        Analyze correlation regimes using rolling windows of existing return data.
        """
        if '1M % Change' not in df_equity.columns:
            raise ValueError("1M % Change column not found for correlation analysis")
            
        # Create returns matrix by sector
        if 'Sector' in df_equity.columns:
            sector_returns = df_equity.pivot_table(
                values='1M % Change',
                index=df_equity.index,
                columns='Sector',
                aggfunc='mean'
            )
        else:
            # Use individual stock returns
            return_cols = [col for col in df_equity.columns if '% Change' in col]
            sector_returns = df_equity[return_cols]
            
        correlation_regimes = {}
        
        for period in lookback_periods:
            if len(sector_returns) >= period:
                # Calculate rolling correlations
                rolling_corr = sector_returns.rolling(window=period).corr()
                
                # Extract average correlation for each period
                period_corrs = []
                for i in range(period, len(sector_returns)):
                    corr_matrix = rolling_corr.iloc[i*len(sector_returns.columns):(i+1)*len(sector_returns.columns)]
                    avg_corr = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()
                    period_corrs.append(avg_corr)
                    
                correlation_regimes[f'{period}D_Average_Correlation'] = np.array(period_corrs)
                
        return correlation_regimes
    
    def stress_test_scenarios(
        self, 
        df_equity: pd.DataFrame,
        df_bonds: Optional[pd.DataFrame] = None,
        portfolio_weights_equity: Optional[np.ndarray] = None,
        portfolio_weights_bonds: Optional[np.ndarray] = None
    ) -> Dict[str, Dict]:
        """
        Stress test portfolio using predefined scenarios based on existing data.
        """
        scenarios = {}
        
        # Market Crash Scenario (-30% equity market)
        if '1M % Change' in df_equity.columns:
            base_returns = df_equity['1M % Change'].values
            market_crash = base_returns - 30  # Assume 30% crash
            
            if portfolio_weights_equity is not None:
                portfolio_crash_return = np.sum(portfolio_weights_equity * market_crash)
            else:
                portfolio_crash_return = np.mean(market_crash)
                
            scenarios['Market_Crash'] = {
                'equity_return': portfolio_crash_return,
                'description': 'Market crash scenario (-30%)'
            }
            
        # High Volatility Scenario (2x historical volatility)
        if '1M % Change' in df_equity.columns:
            base_vol = np.std(df_equity['1M % Change'].values)
            high_vol_returns = np.random.normal(
                np.mean(df_equity['1M % Change'].values),
                2 * base_vol,
                len(df_equity)
            )
            
            if portfolio_weights_equity is not None:
                portfolio_vol_return = np.sum(portfolio_weights_equity * high_vol_returns)
            else:
                portfolio_vol_return = np.mean(high_vol_returns)
                
            scenarios['High_Volatility'] = {
                'equity_return': portfolio_vol_return,
                'description': 'High volatility scenario (2x historical vol)'
            }
            
        # Interest Rate Shock for Bonds
        if df_bonds is not None and 'Duration' in df_bonds.columns:
            # 200 basis point rate shock
            rate_shock = 2.0  # 200 bps
            bond_price_changes = -df_bonds['Duration'].values * rate_shock
            
            if portfolio_weights_bonds is not None:
                portfolio_bond_impact = np.sum(portfolio_weights_bonds * bond_price_changes)
            else:
                portfolio_bond_impact = np.mean(bond_price_changes)
                
            scenarios['Interest_Rate_Shock'] = {
                'bond_return': portfolio_bond_impact,
                'description': 'Interest rate shock (+200 bps)'
            }
            
        return scenarios
    
    def outlier_detection_analysis(
        self, 
        df_equity: pd.DataFrame,
        z_score_threshold: float = 3.0
    ) -> pd.DataFrame:
        """
        Detect outlier securities using existing financial metrics.
        """
        # Select numerical columns for outlier detection
        numeric_cols = df_equity.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove percentage columns and scores for base analysis
        base_metrics = [col for col in numeric_cols if not any(x in col.lower() for x in ['%', 'score', 'change'])]
        
        if len(base_metrics) == 0:
            base_metrics = ['P/E (TTM)', 'P/B (TTM)', 'ROE (%)', 'Debt/Equity']
            base_metrics = [col for col in base_metrics if col in df_equity.columns]
            
        outlier_results = df_equity.copy()
        
        # Calculate Z-scores for each metric
        for col in base_metrics:
            if col in df_equity.columns:
                mean_val = df_equity[col].mean()
                std_val = df_equity[col].std()
                
                if std_val > 0:
                    z_scores = np.abs((df_equity[col] - mean_val) / std_val)
                    outlier_results[f'{col}_ZScore'] = z_scores
                    outlier_results[f'{col}_Outlier'] = z_scores > z_score_threshold
                    
        # Overall outlier score (number of metrics that are outliers)
        outlier_cols = [col for col in outlier_results.columns if col.endswith('_Outlier')]
        outlier_results['Total_Outlier_Count'] = outlier_results[outlier_cols].sum(axis=1)
        
        # Return summary of outliers
        outlier_summary = outlier_results[
            outlier_results['Total_Outlier_Count'] > 0
        ].sort_values('Total_Outlier_Count', ascending=False)
        
        return outlier_summary[['Ticker', 'Company Name', 'Sector'] + outlier_cols + ['Total_Outlier_Count']]


class BondRiskManager:
    """Enhanced bond risk analytics using existing bond data structure."""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        
    def duration_risk_analysis(
        self, 
        df_bonds: pd.DataFrame,
        rate_shock_scenarios: List[float] = [-1.0, -0.5, 0.5, 1.0, 2.0]
    ) -> pd.DataFrame:
        """
        Analyze bond portfolio sensitivity to interest rate changes.
        """
        if 'Duration' not in df_bonds.columns:
            raise ValueError("Duration column required for interest rate risk analysis")
            
        results = []
        
        for shock in rate_shock_scenarios:
            # Price change = -Duration × ΔRate
            price_changes = -df_bonds['Duration'].values * shock
            
            # Add convexity adjustment if available
            if 'Convexity' in df_bonds.columns:
                convexity_adj = 0.5 * df_bonds['Convexity'].values * (shock ** 2)
                price_changes += convexity_adj
                
            # Portfolio level impact
            if 'Weight' in df_bonds.columns:
                portfolio_impact = np.sum(df_bonds['Weight'].values * price_changes)
            else:
                portfolio_impact = np.mean(price_changes)
                
            results.append({
                'Rate_Shock_bps': shock * 100,
                'Portfolio_Price_Change_%': portfolio_impact,
                'Max_Individual_Impact_%': np.max(price_changes),
                'Min_Individual_Impact_%': np.min(price_changes)
            })
            
        return pd.DataFrame(results)
    
    def credit_risk_concentration(
        self, 
        df_bonds: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Analyze credit risk concentration using existing credit data.
        """
        concentration_metrics = {}
        
        # Rating concentration
        if 'Rating' in df_bonds.columns:
            if 'Weight' in df_bonds.columns:
                rating_weights = df_bonds.groupby('Rating')['Weight'].sum()
            else:
                rating_counts = df_bonds['Rating'].value_counts()
                rating_weights = rating_counts / rating_counts.sum()
                
            concentration_metrics['Rating_HHI'] = (rating_weights ** 2).sum()
            
        # Sector concentration
        if 'Sector' in df_bonds.columns:
            if 'Weight' in df_bonds.columns:
                sector_weights = df_bonds.groupby('Sector')['Weight'].sum()
            else:
                sector_counts = df_bonds['Sector'].value_counts()
                sector_weights = sector_counts / sector_counts.sum()
                
            concentration_metrics['Sector_HHI'] = (sector_weights ** 2).sum()
            
        # Credit spread analysis
        if 'Credit_Spread' in df_bonds.columns:
            concentration_metrics['Avg_Credit_Spread'] = df_bonds['Credit_Spread'].mean()
            concentration_metrics['Credit_Spread_Std'] = df_bonds['Credit_Spread'].std()
            
        # Default probability analysis
        if 'Probability of Default (1Y %)' in df_bonds.columns:
            concentration_metrics['Avg_Default_Prob'] = df_bonds['Probability of Default (1Y %)'].mean()
            concentration_metrics['Max_Default_Prob'] = df_bonds['Probability of Default (1Y %)'].max()
            
        return pd.DataFrame([concentration_metrics])
    
    def liquidity_risk_assessment(
        self, 
        df_bonds: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Assess liquidity risk using trading volume and bid-ask spreads.
        """
        liquidity_metrics = []
        
        for idx, bond in df_bonds.iterrows():
            metrics = {'Bond_ID': bond.get('Bond_ID', f'Bond_{idx}')}
            
            # Trading volume analysis
            if 'Trading Volume (USD)' in df_bonds.columns:
                volume = bond['Trading Volume (USD)']
                metrics['Trading_Volume_USD'] = volume
                
                # Volume-based liquidity score (higher volume = more liquid)
                total_volume = df_bonds['Trading Volume (USD)'].sum()
                if total_volume > 0:
                    metrics['Volume_Liquidity_Score'] = volume / total_volume
                    
            # Bid-ask spread analysis
            if 'Bid Price' in df_bonds.columns and 'Ask Price' in df_bonds.columns:
                bid = bond['Bid Price']
                ask = bond['Ask Price']
                mid_price = (bid + ask) / 2
                
                if mid_price > 0:
                    bid_ask_spread = ((ask - bid) / mid_price) * 100
                    metrics['Bid_Ask_Spread_%'] = bid_ask_spread
                    
                    # Lower spread = higher liquidity
                    metrics['Spread_Liquidity_Score'] = 1 / (1 + bid_ask_spread)
                    
            # Time since last trade
            if 'Last Trade Date' in df_bonds.columns:
                # This would require date parsing in practice
                metrics['Has_Recent_Trade'] = pd.notna(bond['Last Trade Date'])
                
            liquidity_metrics.append(metrics)
            
        return pd.DataFrame(liquidity_metrics)