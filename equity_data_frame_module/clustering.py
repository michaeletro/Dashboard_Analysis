from typing import Dict, List, Optional, Any
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)


@dataclass
class ClusterQualityMetrics:
    silhouette: float
    calinski_harabasz: float
    davies_bouldin: float
    inertia: float


class FactorClusterer:
    """
    Performs clustering analysis in factor space using K means with interactive visualization.

    This class provides clustering analysis of equity factor scores with
    scikit learn integration, plotly visualizations, cluster quality metrics,
    and interpretable cluster summaries.
    """

    def __init__(
        self,
        n_clusters: int = 4,
        random_state: int = 42,
        verbose: bool = True,
    ) -> None:
        """
        Initialize the FactorClusterer.

        Parameters
        ----------
        n_clusters : int, default 4
            Number of clusters to form
        random_state : int, default 42
            Random state for reproducible results
        verbose : bool, default True
            If True, print human readable summaries
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.verbose = verbose

        self.cluster_labels_: Optional[np.ndarray] = None
        self.cluster_summary_: Optional[pd.DataFrame] = None
        self.cluster_std_: Optional[pd.DataFrame] = None
        self.global_means_: Optional[pd.Series] = None
        self.global_stds_: Optional[pd.Series] = None
        self.cluster_metrics_: Optional[ClusterQualityMetrics] = None
        self.factor_cols_: Optional[List[str]] = None

        self.has_sklearn = self._check_sklearn()
        self.has_plotly = self._check_plotly()

    def _check_sklearn(self) -> bool:
        """Check if scikit learn is available."""
        try:
            import sklearn  # noqa F401
            return True
        except ImportError:
            return False

    def _check_plotly(self) -> bool:
        """Check if plotly is available."""
        try:
            import plotly  # noqa F401
            return True
        except ImportError:
            return False

    def fit_predict(
        self,
        df: pd.DataFrame,
        factor_cols: Optional[List[str]] = None,
    ) -> np.ndarray:
        """
        Perform clustering analysis in factor space.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame with factor scores and stock information
        factor_cols : list of str, optional
            Factor columns to use. If None, uses columns that end with "Score".

        Returns
        -------
        np.ndarray
            Cluster labels from 0 to n_clusters minus 1
        """
        if self.verbose:
            print("Factor space clustering analysis")
            print("=" * 50)

        # Decide factor columns
        if factor_cols is None:
            factor_cols = [c for c in df.columns if c.endswith("Score")]
        available_factors = [
            col for col in factor_cols if col in df.columns and not df[col].isna().all()
        ]
        self.factor_cols_ = available_factors

        if len(available_factors) < 2:
            if self.verbose:
                print("Insufficient factors for clustering")
            self.cluster_labels_ = np.zeros(len(df), dtype=int)
            return self.cluster_labels_

        # Prepare factor matrix
        F = df[available_factors].astype(float).fillna(0.0).values

        # Standardize
        F_means = np.mean(F, axis=0)
        F_stds = np.std(F, axis=0)
        F_scaled = (F - F_means) / (F_stds + 1e-8)

        # Perform clustering
        if self.has_sklearn:
            cluster_labels, inertia = self._sklearn_clustering(F_scaled)
        else:
            if self.verbose:
                print("sklearn not available, using random clustering")
            cluster_labels = self._simple_clustering(F_scaled)
            inertia = float("nan")

        self.cluster_labels_ = cluster_labels

        # Build summaries and interpretations
        df_temp = df.copy()
        df_temp["FactorCluster"] = cluster_labels

        # Per cluster means and standard deviations
        self.cluster_summary_ = df_temp.groupby("FactorCluster")[available_factors].mean()
        self.cluster_std_ = df_temp.groupby("FactorCluster")[available_factors].std()
        cluster_counts = df_temp["FactorCluster"].value_counts().sort_index()

        # Global stats for relative z scores
        self.global_means_ = df[available_factors].mean()
        self.global_stds_ = df[available_factors].std()

        # Quality metrics
        if self.has_sklearn:
            self.cluster_metrics_ = self._compute_quality_metrics(F_scaled, cluster_labels, inertia)

        if self.verbose:
            self._print_cluster_overview(cluster_counts)
            self._print_quality_metrics()
            self._interpret_clusters(cluster_counts)
            self._show_cluster_examples(df_temp, available_factors)

        return cluster_labels

    def _sklearn_clustering(self, F_scaled: np.ndarray) -> tuple[np.ndarray, float]:
        """Cluster using KMeans and return labels and inertia."""
        kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=20,
        )
        labels = kmeans.fit_predict(F_scaled)
        inertia = float(kmeans.inertia_)
        return labels, inertia

    def _simple_clustering(self, F_scaled: np.ndarray) -> np.ndarray:
        """Random clustering fallback when sklearn is not available."""
        rng = np.random.default_rng(self.random_state)
        return rng.integers(0, self.n_clusters, size=len(F_scaled), dtype=int)

    def _compute_quality_metrics(
        self,
        F_scaled: np.ndarray,
        labels: np.ndarray,
        inertia: float,
    ) -> ClusterQualityMetrics:
        """Compute standard internal cluster quality metrics."""
        if len(np.unique(labels)) < 2:
            # Degenerate case
            return ClusterQualityMetrics(
                silhouette=float("nan"),
                calinski_harabasz=float("nan"),
                davies_bouldin=float("nan"),
                inertia=inertia,
            )

        sil = float(silhouette_score(F_scaled, labels))
        ch = float(calinski_harabasz_score(F_scaled, labels))
        db = float(davies_bouldin_score(F_scaled, labels))

        return ClusterQualityMetrics(
            silhouette=sil,
            calinski_harabasz=ch,
            davies_bouldin=db,
            inertia=inertia,
        )

    def _print_cluster_overview(self, cluster_counts: pd.Series) -> None:
        """Print sizes and centroids."""
        print()
        print(f"Cluster analysis (k = {self.n_clusters})")
        print("Cluster sizes:", dict(cluster_counts))
        print()
        print("Cluster centroids (factor means):")
        print(self.cluster_summary_.round(3))

    def _print_quality_metrics(self) -> None:
        """Print internal cluster quality metrics."""
        if self.cluster_metrics_ is None:
            return

        m = self.cluster_metrics_
        print()
        print("Cluster quality metrics:")
        print(f"  Silhouette         {m.silhouette:.3f}  (near 1 good, near 0 weak)")
        print(f"  Calinski Harabasz  {m.calinski_harabasz:.1f}  (higher is better)")
        print(f"  Davies Bouldin     {m.davies_bouldin:.3f}  (near 0 is better)")
        print(f"  Inertia            {m.inertia:.1f}  (lower is better for fixed k)")

    def _interpret_clusters(self, cluster_counts: pd.Series) -> None:
        """
        Interpret clusters using factor z scores vs global mean.

        This produces short textual descriptions like
        "High value, high quality, small cap".
        """
        if (
            self.cluster_summary_ is None
            or self.global_means_ is None
            or self.global_stds_ is None
        ):
            return

        print()
        print("Cluster interpretations:")
        for cid in range(self.n_clusters):
            if cid not in self.cluster_summary_.index:
                continue

            centroid = self.cluster_summary_.loc[cid]
            z = (centroid - self.global_means_) / (self.global_stds_ + 1e-8)

            phrases: List[str] = []

            for factor, z_val in z.items():
                if abs(z_val) < 0.4:
                    continue  # close to average, not very informative

                name = factor.replace("Score", "")
                direction = "high" if z_val > 0 else "low"

                # Map common factor names to nicer labels
                if "Value" in factor:
                    label = "value"
                elif "Quality" in factor:
                    label = "quality"
                elif "Momentum" in factor:
                    label = "momentum"
                elif "Growth" in factor:
                    label = "growth"
                elif "Size" in factor:
                    label = "size"
                else:
                    label = name

                # For size, flip logic so it reads small or large
                if "Size" in factor:
                    label_dir = "small cap" if z_val < 0 else "large cap"
                    phrases.append(label_dir)
                else:
                    phrases.append(f"{direction} {label}")

            if not phrases:
                desc = "near average on most factors"
            else:
                desc = ", ".join(phrases)

            n = cluster_counts.get(cid, 0)
            print(f"  Cluster {cid}: {desc}  ({n} names)")

    def _show_cluster_examples(
        self,
        df_temp: pd.DataFrame,
        available_factors: List[str],
    ) -> None:
        """Show example names from each cluster."""
        for cid in range(self.n_clusters):
            cluster_stocks = df_temp[df_temp["FactorCluster"] == cid]
            if cluster_stocks.empty:
                continue

            if "Market Cap (USD Billion)" in df_temp.columns:
                examples = cluster_stocks.nlargest(3, "Market Cap (USD Billion)")
            else:
                examples = cluster_stocks.head(3)

            display_cols = ["Ticker", "Company Name", "Sector"] + available_factors[:3]
            display_cols = [c for c in display_cols if c in examples.columns]

            print()
            print(f"Cluster {cid} examples:")
            print(examples[display_cols].round(2))

    # plotting methods (plot_clusters_interactive, plot_cluster_distributions,
    # plot_clusters_3d) can stay almost identical to your current versions and
    # will use self.cluster_labels_ and self.factor_cols_ as before

    def get_cluster_report(self) -> Dict[str, Any]:
        """
        Return a structured report of the clustering.

        Returns
        -------
        dict
            Contains labels, centroids, standard deviations, quality metrics,
            and factor names.
        """
        return {
            "n_clusters": self.n_clusters,
            "labels": self.cluster_labels_,
            "factor_cols": self.factor_cols_,
            "cluster_means": self.cluster_summary_,
            "cluster_stds": self.cluster_std_,
            "global_means": self.global_means_,
            "global_stds": self.global_stds_,
            "metrics": self.cluster_metrics_,
        }
    def plot_clusters_3d(
        self,
        df: pd.DataFrame,
        show_plot: bool = True,
    ) -> Optional[Any]:
        """
        Create an interactive 3D PCA scatter plot with spheres around each cluster.
        """
        if self.cluster_labels_ is None:
            if self.verbose:
                print("No clustering results to plot. Run fit_predict() first.")
            return None

        if not self.has_sklearn:
            if self.verbose:
                print("scikit learn not available for PCA transformation.")
            return None

        if not self.has_plotly:
            if self.verbose:
                print("plotly not available for interactive plotting.")
            return None

        if self.factor_cols_ is not None and len(self.factor_cols_) >= 3:
            factor_cols = self.factor_cols_
        else:
            factor_cols = [c for c in df.columns if c.endswith("Score")]
            factor_cols = [c for c in factor_cols if not df[c].isna().all()]

        if len(factor_cols) < 3:
            if self.verbose:
                print("Insufficient factors for 3D PCA plotting (need at least 3).")
            return None

        try:
            import plotly.express as px
            import plotly.graph_objects as go

            factor_data = df[factor_cols].astype(float).fillna(0.0).values
            means = np.mean(factor_data, axis=0)
            stds = np.std(factor_data, axis=0)
            factor_scaled = (factor_data - means) / (stds + 1e-8)

            from sklearn.decomposition import PCA
            pca = PCA(n_components=3)
            factor_pca = pca.fit_transform(factor_scaled)

            plot_df = pd.DataFrame(
                {
                    "PC1": factor_pca[:, 0],
                    "PC2": factor_pca[:, 1],
                    "PC3": factor_pca[:, 2],
                    "Cluster": [f"Cluster {i}" for i in self.cluster_labels_],
                }
            )

            if "Ticker" in df.columns:
                plot_df["Ticker"] = df["Ticker"].values
            if "Company Name" in df.columns:
                plot_df["Company"] = df["Company Name"].values
            if "Sector" in df.columns:
                plot_df["Sector"] = df["Sector"].values

            hover_cols = [c for c in ["Ticker", "Company", "Sector"] if c in plot_df.columns]

            colors = px.colors.qualitative.Set1
            cluster_labels_unique = sorted(plot_df["Cluster"].unique())
            color_map = {
                cl: colors[i % len(colors)]
                for i, cl in enumerate(cluster_labels_unique)
            }

            fig = px.scatter_3d(
                plot_df,
                x="PC1",
                y="PC2",
                z="PC3",
                color="Cluster",
                hover_data=hover_cols,
                title=(
                    "3D Factor Space Clusters PCA projection<br>"
                    f"PC1: {pca.explained_variance_ratio_[0]:.1%}, "
                    f"PC2: {pca.explained_variance_ratio_[1]:.1%}, "
                    f"PC3: {pca.explained_variance_ratio_[2]:.1%} variance"
                ),
                labels={
                    "PC1": f"PC1 ({pca.explained_variance_ratio_[0]:.1%})",
                    "PC2": f"PC2 ({pca.explained_variance_ratio_[1]:.1%})",
                    "PC3": f"PC3 ({pca.explained_variance_ratio_[2]:.1%})",
                },
                color_discrete_map=color_map,
            )

            for cid in range(self.n_clusters):
                mask = self.cluster_labels_ == cid
                if not np.any(mask):
                    continue

                points = factor_pca[mask]
                centroid = points.mean(axis=0)

                distances = np.linalg.norm(points - centroid, axis=1)
                radius = float(np.std(distances))
                if radius == 0.0:
                    radius = float(np.mean(distances) + 1e-3)

                u = np.linspace(0.0, 2.0 * np.pi, 20)
                v = np.linspace(0.0, np.pi, 20)
                x_sphere = centroid[0] + radius * np.outer(np.cos(u), np.sin(v))
                y_sphere = centroid[1] + radius * np.outer(np.sin(u), np.sin(v))
                z_sphere = centroid[2] + radius * np.outer(np.ones_like(u), np.cos(v))

                cluster_name = f"Cluster {cid}"
                color = color_map.get(cluster_name, colors[cid % len(colors)])

                fig.add_trace(
                    go.Surface(
                        x=x_sphere,
                        y=y_sphere,
                        z=z_sphere,
                        opacity=0.18,
                        showscale=False,
                        colorscale=[[0.0, color], [1.0, color]],
                        name=f"{cluster_name} region",
                        hovertemplate=(
                            f"<b>{cluster_name} region</b><br>"
                            f"Radius: {radius:.3f}<extra></extra>"
                        ),
                    )
                )

                fig.add_trace(
                    go.Scatter3d(
                        x=[centroid[0]],
                        y=[centroid[1]],
                        z=[centroid[2]],
                        mode="markers+text",
                        marker=dict(
                            size=9,
                            color=color,
                            symbol="diamond",
                            line=dict(width=2, color="black"),
                        ),
                        text=[cluster_name],
                        textposition="top center",
                        name=f"{cluster_name} center",
                        showlegend=False,
                        hovertemplate=(
                            f"<b>{cluster_name} centroid</b><br>"
                            f"PC1: {centroid[0]:.3f}<br>"
                            f"PC2: {centroid[1]:.3f}<br>"
                            f"PC3: {centroid[2]:.3f}<extra></extra>"
                        ),
                    )
                )

            fig.update_layout(
                margin=dict(l=0, r=0, b=0, t=40),
                scene=dict(
                    bgcolor="black",
                    xaxis=dict(showgrid=True, gridcolor="lightgray", title_font_size=12),
                    yaxis=dict(showgrid=True, gridcolor="lightgray", title_font_size=12),
                    zaxis=dict(showgrid=True, gridcolor="lightgray", title_font_size=12),
                ),
                paper_bgcolor="black",
                font=dict(size=12),
                title_font_size=16,
                
            )

            if show_plot:
                fig.show()

            return fig

        except Exception as exc:
            if self.verbose:
                print(f"Error creating 3D interactive plot: {exc}")
            return None
