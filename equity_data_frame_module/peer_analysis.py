"""
Peer relative analysis for sector and country comparisons.
"""
import plotly.graph_objects as go

import pandas as pd
import numpy as np

class PeerAnalyzer:
    """Analyzes stocks relative to sector and country peers."""
    
    def __init__(self):
        self.peer_groups = ['Sector', 'Country']
        self.valuation_metrics = ['PE Ratio (TTM)', 'Price/Sales (TTM)', 'EV/EBITDA (TTM)', 'Price/Book (TTM)']
        self.profitability_metrics = ['ROE (%)', 'ROA (%)', 'Net Margin (%)']
    
    def has_peer_analysis(self, df):
        """Check if peer analysis has already been run on this dataframe."""
        peer_columns = [col for col in df.columns if col.startswith('PeerMedian_') or col.endswith('_vs_peers_pct')]
        return len(peer_columns) > 0
    
    def analyze(self, df):
        """
        Analyze stocks relative to sector and country peers.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input DataFrame with stock data
            
        Returns:
        --------
        pandas.DataFrame : DataFrame with peer relative metrics added
        """
        print("🔍 Peer Relative Valuation Analysis")
        print("=" * 50)
        
        # Remove existing peer columns to avoid conflicts
        peer_cols_to_remove = [col for col in df.columns if col.startswith('PeerMedian_') or col.endswith('_vs_peers_pct')]
        if peer_cols_to_remove:
            df = df.drop(columns=peer_cols_to_remove)
            print(f"🧹 Removed {len(peer_cols_to_remove)} existing peer columns")
        
        # Calculate peer medians
        available_val = [col for col in self.valuation_metrics if col in df.columns]
        available_prof = [col for col in self.profitability_metrics if col in df.columns]
        
        all_metrics = available_val + available_prof
        
        if not all_metrics or not all(col in df.columns for col in self.peer_groups):
            print("⚠ Insufficient data for peer analysis")
            return df
        
        # Calculate peer medians
        peer_medians = df.groupby(self.peer_groups)[all_metrics].median().reset_index()
        
        # Rename columns to avoid conflicts
        for metric in all_metrics:
            peer_medians[f'PeerMedian_{metric}'] = peer_medians[metric]
        
        # Drop original columns from peer_medians to avoid conflicts
        peer_medians = peer_medians.drop(columns=all_metrics)
        
        # Merge back to original dataframe
        df_with_peers = df.merge(peer_medians, on=self.peer_groups, how='left')
        
        # Calculate relative metrics (% premium/discount to peers)
        for metric in all_metrics:
            peer_col = f'PeerMedian_{metric}'
            rel_col = f'{metric}_vs_peers_pct'
            if peer_col in df_with_peers.columns:
                df_with_peers[rel_col] = 100 * (df_with_peers[metric] - df_with_peers[peer_col]) / df_with_peers[peer_col]
        
        print(f"✅ Calculated peer relatives for {len(all_metrics)} metrics")
        print(f"📊 Peer group combinations: {len(peer_medians)} unique groups")
        
        # Find most undervalued stocks (cheapest vs peers)
        val_rel_cols = [f'{col}_vs_peers_pct' for col in available_val if f'{col}_vs_peers_pct' in df_with_peers.columns]
        if val_rel_cols:
            df_with_peers['Valuation_vs_Peers'] = df_with_peers[val_rel_cols].mean(axis=1)
            
            self._print_top_picks(df_with_peers, val_rel_cols)
        
        return df_with_peers
    
    def _print_top_picks(self, df_with_peers, val_rel_cols):
        """Print top undervalued and overvalued stocks."""
        print("\n🏆 Most Undervalued vs Peers (Top 3):")
        undervalued = df_with_peers.nsmallest(3, 'Valuation_vs_Peers')[
            ['Ticker', 'Company Name', 'Sector', 'Country', 'Valuation_vs_Peers'] + 
            (val_rel_cols[:3] if len(val_rel_cols) >= 3 else val_rel_cols)
        ]
        print(undervalued.round(2))
        
        print("\n🔥 Most Overvalued vs Peers (Top 3):")
        overvalued = df_with_peers.nlargest(3, 'Valuation_vs_Peers')[
            ['Ticker', 'Company Name', 'Sector', 'Country', 'Valuation_vs_Peers'] + 
            (val_rel_cols[:3] if len(val_rel_cols) >= 3 else val_rel_cols)
        ]
        print(overvalued.round(2))
    
    def plot_sector_dispersion(self, df):
        """Return a Plotly boxplot of valuation dispersion by sector."""
        if "Valuation_vs_Peers" not in df.columns or "Sector" not in df.columns:
            print("Required columns Sector and Valuation_vs_Peers not found")
            return go.Figure()

        # drop missing values for plotting
        df_plot = df[["Sector", "Valuation_vs_Peers"]].dropna()
        if df_plot.empty:
            print("No peer valuation data to plot")
            return go.Figure()

        fig = go.Figure()

        # one box per sector
        for sector, group in df_plot.groupby("Sector"):
            fig.add_trace(
                go.Box(
                    y=group["Valuation_vs_Peers"],
                    name=str(sector),
                    boxmean="sd",
                    hovertemplate=(
                        "Sector: %{name}<br>"
                        "Valuation vs peers: %{y:.2f}%<extra></extra>"
                    ),
                )
            )

        # horizontal line at zero valuation premium or discount
        fig.add_shape(
            type="line",
            x0=-0.5,
            x1=len(df_plot["Sector"].unique()) - 0.5,
            xref="x",
            y0=0,
            y1=0,
            yref="y",
            line=dict(color="red", width=2, dash="dash"),
        )

        fig.update_layout(
            title=(
                "Valuation vs peers by sector<br>"
                "(Negative indicates undervalued, positive indicates overvalued)"
            ),
            yaxis_title="Average valuation premium or discount (percent)",
            xaxis_title="Sector",
            xaxis=dict(tickangle=45),
            margin=dict(l=60, r=20, t=80, b=80),
        )

        return fig

    
    def get_sector_summary(self, df):
        """Get sector summary statistics."""
        if 'Valuation_vs_Peers' not in df.columns:
            print("No peer valuation data available")
            return None
        
        # Summary statistics by sector
        factor_cols = [col for col in df.columns if col.endswith('Score')]
        agg_dict = {
            'Valuation_vs_Peers': ['count', 'mean', 'median', 'std']
        }
        
        # Add factor scores if available
        for factor in factor_cols[:2]:  # Limit to first 2 factors to avoid clutter
            if factor in df.columns:
                agg_dict[factor] = 'mean'
        
        sector_summary = df.groupby('Sector').agg(agg_dict).round(2)
        
        print("\n📈 Sector Summary Statistics:")
        print(sector_summary)
        
        return sector_summary