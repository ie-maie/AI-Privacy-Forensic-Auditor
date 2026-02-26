from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")
sns.set_theme(style="whitegrid")

try:
    from sklearn.cluster import DBSCAN, KMeans
    from sklearn.compose import ColumnTransformer
    from sklearn.decomposition import PCA
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import silhouette_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False


def _load_clean_df(context: Dict[str, Any]) -> pd.DataFrame | None:
    if "clean_df" in context:
        return context["clean_df"].copy()

    parts: List[pd.DataFrame] = []
    for split in ("train", "val", "test"):
        path = Path(f"data/processed/{split}.csv")
        if not path.exists():
            return None
        parts.append(pd.read_csv(path))
    return pd.concat(parts, axis=0, ignore_index=True)


def _ensure_output_dirs() -> Dict[str, Path]:
    fig_dir = Path("outputs/figures")
    table_dir = Path("outputs/tables")
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    return {"figures": fig_dir, "tables": table_dir}


def _build_preprocessor(numeric_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
    numeric_pipe = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ]
    )


def _safe_silhouette(X, labels) -> float | None:
    unique = set(labels)
    if len(unique) < 2:
        return None
    if len(unique) == 2 and (-1 in unique):
        return None
    valid_mask = labels != -1
    if valid_mask.sum() < 2:
        return None
    try:
        return float(silhouette_score(X[valid_mask], labels[valid_mask]))
    except Exception:
        return None


def _plot_cluster_scatter(embedding_df: pd.DataFrame, label_col: str, out_path: Path, title: str) -> None:
    if embedding_df.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        data=embedding_df,
        x="pc1",
        y="pc2",
        hue=label_col,
        palette="tab10",
        ax=ax,
        s=40,
        alpha=0.85,
    )
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(title=label_col, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def run(config: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    if not SKLEARN_AVAILABLE:
        return {
            "step": "clustering",
            "status": "skipped",
            "message": (
                "scikit-learn is required for clustering. "
                "Install dependencies in your venv: pip install -r requirements.txt"
            ),
        }

    df = _load_clean_df(context)
    if df is None:
        return {
            "step": "clustering",
            "status": "skipped",
            "message": "No cleaned dataset available. Run preprocessing first.",
        }

    schema_cfg = config.get("schema", {})
    cluster_cfg = config.get("clustering", {})
    numeric_cols = [c for c in schema_cfg.get("numeric_features", []) if c in df.columns]
    categorical_cols = [c for c in schema_cfg.get("categorical_features", []) if c in df.columns]
    feature_cols = numeric_cols + categorical_cols
    if not feature_cols:
        return {
            "step": "clustering",
            "status": "error",
            "message": "No configured feature columns available for clustering.",
        }

    work_df = df[feature_cols].copy()
    preprocessor = _build_preprocessor(numeric_cols=numeric_cols, categorical_cols=categorical_cols)
    X_proc = preprocessor.fit_transform(work_df)

    kmeans_k = int(cluster_cfg.get("kmeans_k", 3))
    dbscan_eps = float(cluster_cfg.get("dbscan_eps", 0.8))
    dbscan_min_samples = int(cluster_cfg.get("dbscan_min_samples", 15))

    kmeans = KMeans(n_clusters=kmeans_k, random_state=42, n_init=20)
    kmeans_labels = kmeans.fit_predict(X_proc)

    dbscan = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
    dbscan_labels = dbscan.fit_predict(X_proc)

    kmeans_sil = _safe_silhouette(X_proc, kmeans_labels)
    dbscan_sil = _safe_silhouette(X_proc, dbscan_labels)

    summary_df = pd.DataFrame(
        [
            {
                "algorithm": "kmeans",
                "n_clusters": int(len(set(kmeans_labels))),
                "n_noise": 0,
                "silhouette": kmeans_sil,
            },
            {
                "algorithm": "dbscan",
                "n_clusters": int(len(set(dbscan_labels)) - (1 if -1 in set(dbscan_labels) else 0)),
                "n_noise": int((dbscan_labels == -1).sum()),
                "silhouette": dbscan_sil,
            },
        ]
    )

    per_row_df = pd.DataFrame(
        {
            "kmeans_cluster": kmeans_labels,
            "dbscan_cluster": dbscan_labels,
        }
    )

    dirs = _ensure_output_dirs()
    summary_path = dirs["tables"] / "clustering_summary.csv"
    labels_path = dirs["tables"] / "clustering_assignments.csv"
    summary_df.to_csv(summary_path, index=False)
    per_row_df.to_csv(labels_path, index=False)

    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X_proc)
    embed_df = pd.DataFrame({"pc1": X_2d[:, 0], "pc2": X_2d[:, 1]})
    embed_df["kmeans_cluster"] = kmeans_labels.astype(str)
    embed_df["dbscan_cluster"] = dbscan_labels.astype(str)

    kmeans_fig_path = dirs["figures"] / "clustering_kmeans_pca.png"
    dbscan_fig_path = dirs["figures"] / "clustering_dbscan_pca.png"
    _plot_cluster_scatter(embed_df, "kmeans_cluster", kmeans_fig_path, "K-Means clusters (PCA view)")
    _plot_cluster_scatter(embed_df, "dbscan_cluster", dbscan_fig_path, "DBSCAN clusters (PCA view)")

    return {
        "step": "clustering",
        "status": "ok",
        "message": "Clustering completed (K-Means + DBSCAN) with metrics and plots saved.",
        "summary_path": str(summary_path),
        "labels_path": str(labels_path),
        "figures": [str(kmeans_fig_path), str(dbscan_fig_path)],
    }
