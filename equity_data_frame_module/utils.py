"""
Utility functions for Bloomberg analysis package.
"""

import warnings
warnings.filterwarnings('ignore')



import numpy as np
import plotly.graph_objects as go
from common.metrics import nmise_mape


# Global flags for optional dependencies
HAS_MATPLOTLIB = False
HAS_SEABORN = False
HAS_SKLEARN = False

def check_dependencies():
    """Check and import optional dependencies."""
    global HAS_MATPLOTLIB, HAS_SEABORN, HAS_SKLEARN
    
    try:
        import matplotlib.pyplot as plt
        HAS_MATPLOTLIB = True
        print("✓ matplotlib available")
    except ImportError:
        print("⚠ matplotlib not available - will skip visualizations")
    
    try:
        import seaborn as sns
        HAS_SEABORN = True
        print("✓ seaborn available")
    except ImportError:
        print("⚠ seaborn not available - will use basic plotting")
    
    try:
        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA
        HAS_SKLEARN = True
        print("✓ scikit-learn available")
    except ImportError:
        print("⚠ scikit-learn not available - will use fallback methods")
    
    return {
        'matplotlib': HAS_MATPLOTLIB,
        'seaborn': HAS_SEABORN,
        'sklearn': HAS_SKLEARN
    }

def setup_plotting():
    """Set up plotting configuration if matplotlib is available."""
    if HAS_MATPLOTLIB:
        import matplotlib.pyplot as plt
        plt.style.use('default')
        
        if HAS_SEABORN:
            import seaborn as sns
            sns.set_palette("husl")
        
        print("✓ Plotting environment configured")
    else:
        print("⚠ matplotlib not available - plotting disabled")

def get_dependencies():
    """Get current dependency status."""
    return {
        'matplotlib': HAS_MATPLOTLIB,
        'seaborn': HAS_SEABORN,
        'sklearn': HAS_SKLEARN
    }

def plot_series(series_dict, x=None, title="Series comparison"):
    """
    Quick Plotly line chart for multiple 1D numpy series.
    
    Parameters
    ----------
    series_dict : dict[str, np.ndarray]
        Keys are labels, values are 1D numpy arrays (y-values).
    x : np.ndarray or None
        Optional common x-axis. If None, uses np.arange(len(y)) per series.
        If provided, must match the length of each y-array.
    title : str
        Plot title.
    """
    fig = go.Figure()
    
    for label, y in series_dict.items():
        y = np.asarray(y)
        
        if x is None:
            x_vals = np.arange(len(y))
        else:
            x_vals = np.asarray(x)
            if len(x_vals) != len(y):
                raise ValueError(f"x and y for '{label}' must have same length")
        
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=y,
                mode="lines",
                name=label
            )
        )
    
    fig.update_layout(
        title=title,
        xaxis_title="Index" if x is None else "x",
        yaxis_title="Value",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
    )
    
    fig.show()
    #return fig

import numpy as np
