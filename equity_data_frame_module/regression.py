"""
Cross-sectional regression analysis for factor models.
"""

import pandas as pd
import numpy as np

class CrossSectionalRegressor:
    """Performs cross-sectional factor regression using matrix operations."""
    
    def __init__(self):
        self.results_ = None
        self.r_squared_ = None
        self.coefficients_ = None
    
    def fit(self, df, return_col='1M % Change', factor_cols=None):
        """
        Perform cross-sectional factor regression using matrix operations.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input DataFrame with returns and factor scores
        return_col : str
            Column name for returns (default: '1M % Change')
        factor_cols : list or None
            List of factor columns. If None, uses all columns ending with 'Score'
            
        Returns:
        --------
        tuple : (results_df, r_squared, coefficients)
        """
        print("📊 Cross-Sectional Factor Regression")
        print("=" * 50)
        
        # Handle missing return column
        if return_col not in df.columns:
            print(f"⚠ Return column '{return_col}' not found. Using synthetic returns.")
            synthetic_returns = self._create_synthetic_returns(df)
            df = df.copy()
            df[return_col] = synthetic_returns
        
        # Prepare data
        y = (df[return_col].values / 100.0).reshape(-1, 1)  # Convert to decimal returns
        
        # Determine factor columns
        if factor_cols is None:
            factor_cols = [col for col in df.columns if col.endswith('Score')]
        
        available_factors = [col for col in factor_cols if col in df.columns]
        
        if not available_factors:
            raise ValueError("No factor columns found")
        
        X_factors = df[available_factors].values
        X = np.column_stack([np.ones(len(df)), X_factors])  # Add intercept
        
        n, p = X.shape
        print(f"📏 Design matrix: {n} observations × {p} parameters")
        
        # OLS estimation: β̂ = (X'X)^(-1) X'y
        try:
            XtX = X.T @ X
            XtX_inv = np.linalg.inv(XtX)
            Xty = X.T @ y
            beta_hat = XtX_inv @ Xty
            
            # Model fit statistics
            y_hat = X @ beta_hat
            residuals = y - y_hat
            
            # R-squared
            ss_tot = np.sum((y - y.mean()) ** 2)
            ss_res = np.sum(residuals ** 2)
            r_squared = 1.0 - ss_res / ss_tot
            
            # Standard errors
            mse = ss_res / (n - p)
            var_beta = mse * np.diag(XtX_inv)
            se_beta = np.sqrt(var_beta)
            
            # t-statistics
            t_stats = beta_hat.flatten() / se_beta
            
            # Results DataFrame
            param_names = ['Intercept'] + available_factors
            results_df = pd.DataFrame({
                'Factor': param_names,
                'Coefficient': beta_hat.flatten(),
                'Std_Error': se_beta,
                't_Statistic': t_stats,
                'Significant': np.abs(t_stats) > 1.96  # 5% significance level
            })
            
            print("\n📈 Regression Results:")
            print(results_df.round(4))
            print(f"\n📊 Model Statistics:")
            print(f"R² = {r_squared:.4f}")
            print(f"Adjusted R² = {1 - (ss_res/(n-p))/(ss_tot/(n-1)):.4f}")
            print(f"MSE = {float(mse):.6f}")
            
            # Store results
            self.results_ = results_df
            self.r_squared_ = r_squared
            self.coefficients_ = beta_hat
            
            return results_df, r_squared, beta_hat
            
        except np.linalg.LinAlgError:
            print("❌ Singular matrix - cannot invert X'X")
            return None, None, None
    
    def _create_synthetic_returns(self, df):
        """Create synthetic returns based on factor scores for demonstration."""
        factor_cols = [col for col in df.columns if col.endswith('Score')]
        if len(factor_cols) >= 3:
            synthetic_returns = (0.5 * df[factor_cols[0]] + 
                               0.3 * df[factor_cols[1]] + 
                               0.2 * df[factor_cols[2]] + 
                               np.random.normal(0, 2, len(df)))
        else:
            synthetic_returns = np.random.normal(0, 5, len(df))
        
        return synthetic_returns
    
    def plot_results(self, plt=None):
        """Plot regression results if matplotlib is available."""
        if self.results_ is None:
            print("No regression results to plot. Run fit() first.")
            return
        
        if plt is None:
            try:
                import matplotlib.pyplot as plt
            except ImportError:
                print("matplotlib not available for plotting")
                return
        
        # Create plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Factor coefficients
        factors = self.results_['Factor'][1:]  # Exclude intercept
        coefs = self.results_['Coefficient'][1:]
        colors = ['green' if c > 0 else 'red' for c in coefs]
        
        ax1.barh(factors, coefs, color=colors, alpha=0.7)
        ax1.set_xlabel('Factor Loading')
        ax1.set_title('Cross-Sectional Factor Loadings')
        ax1.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        
        # Factor significance
        significance = self.results_['Significant'][1:]
        sig_colors = ['darkgreen' if sig else 'lightgray' for sig in significance]
        
        ax2.barh(factors, np.abs(self.results_['t_Statistic'][1:]), color=sig_colors, alpha=0.7)
        ax2.set_xlabel('|t-statistic|')
        ax2.set_title('Statistical Significance')
        ax2.axvline(x=1.96, color='red', linestyle='--', label='5% significance')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
    
    def get_insights(self):
        """Get key insights from regression results."""
        if self.results_ is None:
            return "No regression results available. Run fit() first."
        
        insights = []
        for i, row in self.results_.iterrows():
            if i > 0 and row['Significant']:  # Skip intercept
                direction = "positively" if row['Coefficient'] > 0 else "negatively"
                insights.append(f"• {row['Factor']} factor is {direction} related to returns (t={row['t_Statistic']:.2f})")
        
        return insights if insights else ["No significant factors found"]