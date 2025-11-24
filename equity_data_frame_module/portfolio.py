"""
Portfolio construction and backtesting utilities.
"""

import pandas as pd
import numpy as np

class PortfolioConstructor:
    """Constructs long-short portfolios based on multi-factor scores."""
    
    def __init__(self, default_weights={'Value': 0.4, 'Quality': 0.3, 'Momentum': 0.2, 'Growth': 0.1}):
        self.default_weights = default_weights
        self.portfolio_results_ = None
        self.long_portfolio_ = None
        self.short_portfolio_ = None
    
    def construct_portfolio(self, df, weights=None):
        """
        Construct long-short portfolio based on multi-factor scores.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input DataFrame with factor scores
        weights : dict or None
            Factor weights for portfolio construction. If None, uses default_weights
            
        Returns:
        --------
        dict : Dictionary containing portfolio analysis results
        """
        if weights is None:
            weights = self.default_weights
        
        print("📈 Portfolio Construction & Backtesting")
        print("=" * 50)
        
        # Create combined score using factor weights
        available_factors = [f'{name}Score' for name in weights.keys() if f'{name}Score' in df.columns]
        
        if not available_factors:
            print("⚠ No factors available for portfolio construction")
            return None
        
        combined_score = np.zeros(len(df))
        total_weight = 0
        
        print("📊 Factor Weights:")
        for factor_name, weight in weights.items():
            factor_col = f'{factor_name}Score'
            if factor_col in df.columns:
                combined_score += weight * df[factor_col].values
                total_weight += weight
                print(f"  • {factor_name}: {weight:.1%}")
        
        # Normalize to ensure weights sum to 1
        if total_weight > 0:
            combined_score /= total_weight
        
        df = df.copy()
        df['CombinedScore'] = combined_score
        
        # Define quintile portfolios
        quintiles = pd.qcut(combined_score, 5, labels=['Q1_Worst', 'Q2_Bad', 'Q3_Neutral', 'Q4_Good', 'Q5_Best'])
        df['Quintile'] = quintiles
        
        # Long portfolio: Top quintile (Q5)
        # Short portfolio: Bottom quintile (Q1)
        long_portfolio = df[df['Quintile'] == 'Q5_Best'].copy()
        short_portfolio = df[df['Quintile'] == 'Q1_Worst'].copy()
        
        print(f"\n🎯 Portfolio Construction:")
        print(f"  Long Portfolio:  {len(long_portfolio)} stocks")
        print(f"  Short Portfolio: {len(short_portfolio)} stocks")
        
        # Calculate portfolio performance
        portfolio_results = self._calculate_portfolio_performance(df, long_portfolio, short_portfolio)
        
        # Analyze portfolio characteristics
        self._analyze_portfolio_characteristics(df, long_portfolio, short_portfolio)
        
        # Store results
        self.portfolio_results_ = portfolio_results
        self.long_portfolio_ = long_portfolio
        self.short_portfolio_ = short_portfolio
        
        return {
            'long_portfolio': long_portfolio,
            'short_portfolio': short_portfolio,
            'portfolio_results': portfolio_results,
            'full_data': df
        }
    
    def _calculate_portfolio_performance(self, df, long_portfolio, short_portfolio):
        """Calculate portfolio returns and performance metrics."""
        # Calculate value-weighted returns using market cap
        def calculate_vw_return(portfolio_df, return_col):
            if 'Market Cap (USD Billion)' in portfolio_df.columns and return_col in portfolio_df.columns:
                weights = portfolio_df['Market Cap (USD Billion)'].values
                weights = weights / weights.sum()  # Normalize to sum to 1
                returns = portfolio_df[return_col].values / 100.0  # Convert to decimal
                return float(np.dot(weights, returns))
            return 0.0
        
        # Calculate portfolio returns
        return_cols = ['1W % Change', '1M % Change']
        available_returns = [col for col in return_cols if col in df.columns]
        
        portfolio_results = {}
        
        for ret_col in available_returns:
            long_return = calculate_vw_return(long_portfolio, ret_col)
            short_return = calculate_vw_return(short_portfolio, ret_col)
            ls_return = long_return - short_return  # Long-Short spread
            
            portfolio_results[ret_col] = {
                'Long': long_return,
                'Short': short_return,
                'Long-Short': ls_return
            }
            
            print(f"\n📊 {ret_col} Portfolio Performance:")
            print(f"  Long Portfolio:   {long_return:+.2%}")
            print(f"  Short Portfolio:  {short_return:+.2%}")
            print(f"  Long-Short:       {ls_return:+.2%}")
        
        return portfolio_results
    
    def _analyze_portfolio_characteristics(self, df, long_portfolio, short_portfolio):
        """Analyze portfolio characteristics and factor exposures."""
        print("\n🔍 Portfolio Characteristics:")
        
        # Long portfolio stats
        if 'Market Cap (USD Billion)' in long_portfolio.columns:
            avg_market_cap_long = long_portfolio['Market Cap (USD Billion)'].mean()
            print(f"  Long Portfolio Avg Market Cap: ${avg_market_cap_long:.1f}B")
        
        # Sector concentration
        if 'Sector' in long_portfolio.columns:
            long_sector_dist = long_portfolio['Sector'].value_counts(normalize=True)
            print(f"  Long Portfolio Top Sectors:")
            for sector, pct in long_sector_dist.head(3).items():
                print(f"    • {sector}: {pct:.1%}")
        
        # Factor loadings of portfolios
        factor_cols = [col for col in df.columns if col.endswith('Score')]
        if factor_cols:
            long_factors = long_portfolio[factor_cols].mean()
            short_factors = short_portfolio[factor_cols].mean()
            
            print(f"\n📈 Average Factor Exposures:")
            print(f"{'Factor':<15} {'Long':<8} {'Short':<8} {'Difference':<10}")
            print("-" * 45)
            for factor in factor_cols:
                diff = long_factors[factor] - short_factors[factor]
                print(f"{factor:<15} {long_factors[factor]:>7.2f} {short_factors[factor]:>7.2f} {diff:>9.2f}")
        
        # Risk analysis
        print(f"\n⚠️  Risk Analysis:")
        if 'Debt/Equity' in df.columns:
            long_leverage = long_portfolio['Debt/Equity'].mean()
            short_leverage = short_portfolio['Debt/Equity'].mean()
            print(f"  Average Debt/Equity - Long: {long_leverage:.2f}, Short: {short_leverage:.2f}")
    
    def plot_quintile_performance(self, df, plt=None):
        """Visualize quintile performance."""
        if 'Quintile' not in df.columns:
            print("No quintile data to plot")
            return
        
        if plt is None:
            try:
                import matplotlib.pyplot as plt
            except ImportError:
                print("matplotlib not available for plotting")
                return
        
        # Calculate value-weighted returns using market cap
        def calculate_vw_return(portfolio_df, return_col):
            if 'Market Cap (USD Billion)' in portfolio_df.columns and return_col in portfolio_df.columns:
                weights = portfolio_df['Market Cap (USD Billion)'].values
                weights = weights / weights.sum()  # Normalize to sum to 1
                returns = portfolio_df[return_col].values / 100.0  # Convert to decimal
                return float(np.dot(weights, returns))
            return 0.0
        
        return_cols = ['1W % Change', '1M % Change']
        available_returns = [col for col in return_cols if col in df.columns]
        
        if not available_returns:
            print("No return data to plot")
            return
        
        plt.figure(figsize=(14, 6))
        
        for i, ret_col in enumerate(available_returns):
            plt.subplot(1, len(available_returns), i+1)
            
            quintile_returns = []
            quintile_labels = []
            
            for q in ['Q1_Worst', 'Q2_Bad', 'Q3_Neutral', 'Q4_Good', 'Q5_Best']:
                q_stocks = df[df['Quintile'] == q]
                if len(q_stocks) > 0:
                    q_return = calculate_vw_return(q_stocks, ret_col)
                    quintile_returns.append(q_return * 100)  # Convert to percentage
                    quintile_labels.append(q.replace('_', '\n'))
            
            colors = ['red', 'orange', 'yellow', 'lightgreen', 'darkgreen']
            bars = plt.bar(quintile_labels, quintile_returns, color=colors, alpha=0.7)
            
            plt.title(f'{ret_col} by Factor Quintile')
            plt.ylabel('Return (%)')
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars, quintile_returns):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1 if value >= 0 else bar.get_height() - 0.3, 
                        f'{value:.1f}%', ha='center', va='bottom' if value >= 0 else 'top')
        
        plt.tight_layout()
        plt.show()
    
    def get_top_picks(self, n=5):
        """Get top stock picks from long portfolio."""
        if self.long_portfolio_ is None:
            return "No portfolio constructed. Run construct_portfolio() first."
        
        if 'CombinedScore' in self.long_portfolio_.columns:
            top_picks = self.long_portfolio_.nlargest(n, 'CombinedScore')
            return top_picks[['Ticker', 'Company Name', 'Sector', 'CombinedScore']]
        else:
            return "No combined scores available"
    
    def get_performance_summary(self):
        """Get summary of portfolio performance."""
        if self.portfolio_results_ is None:
            return "No portfolio results available. Run construct_portfolio() first."
        
        summary = []
        for period, results in self.portfolio_results_.items():
            summary.append({
                'Period': period,
                'Long_Return': results['Long'],
                'Short_Return': results['Short'],
                'Long_Short_Alpha': results['Long-Short']
            })
        
        return pd.DataFrame(summary)