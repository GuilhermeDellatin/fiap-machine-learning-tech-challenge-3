import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score


class KMeansKEvaluator:
    """
    Evaluation of clustering metrics (Inertia, Silhouette, ARI Stability)
    for different values of K.
    """

    def __init__(
            self,
            k_min: int = 2,
            k_max: int = 10,
            base_seed: int = 42,
            n_init: int = 10,
            stability_seeds: list[int] = None,
            silhouette_sample_size: int = 10000
    ):
        self.k_min = k_min
        self.k_max = k_max
        self.base_seed = base_seed
        self.n_init = n_init
        self.stability_seeds = stability_seeds or [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 42]
        self.silhouette_sample_size = silhouette_sample_size

    def _compute_silhouette(self, X: np.ndarray, labels: np.ndarray) -> float:
        """Calculates silhouette score with optional sampling."""
        sample_size = self.silhouette_sample_size

        if sample_size is not None and X.shape[0] > sample_size:
            return float(silhouette_score(
                X, labels,
                sample_size=sample_size,
                random_state=self.base_seed
            ))
        return float(silhouette_score(X, labels))

    def evaluate(self, X: np.ndarray) -> pd.DataFrame:
        """
        Evaluates all values of K and returns a DataFrame with metrics.

        Args:
            X: Scaled data (numpy array)

        Returns:
            DataFrame with columns: k, inertia, silhouette, mean_ari_vs_base, silhouette_std_across_seeds
        """
        k_range = range(self.k_min, self.k_max + 1)
        rows = []

        for k in k_range:
            # Modelo base
            km_base = KMeans(n_clusters=k, random_state=self.base_seed, n_init=self.n_init)
            labels_base = km_base.fit_predict(X)

            # Silhouette baseline
            sil_base = self._compute_silhouette(X, labels_base)

            # Estabilidade: ARI e Silhouette em múltiplas seeds
            aris = []
            sils = [sil_base]

            for s in self.stability_seeds:
                if s == self.base_seed:
                    continue

                km_s = KMeans(n_clusters=k, random_state=s, n_init=self.n_init)
                labels_s = km_s.fit_predict(X)

                aris.append(float(adjusted_rand_score(labels_base, labels_s)))
                sils.append(self._compute_silhouette(X, labels_s))

            rows.append({
                "k": k,
                "inertia": float(km_base.inertia_),
                "silhouette": sil_base,
                "mean_ari_vs_base": float(np.mean(aris)) if aris else np.nan,
                "silhouette_std_across_seeds": float(np.std(sils)) if sils else np.nan
            })

        df_results = pd.DataFrame(rows)
        df_results["inertia_reduction_%"] = (df_results["inertia"].shift(1) - df_results["inertia"]) / df_results[
            "inertia"].shift(1) * 100

        return df_results
