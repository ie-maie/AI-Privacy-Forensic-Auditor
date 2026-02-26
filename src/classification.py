from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

try:
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False


MODEL_REGISTRY = {
    "logistic_regression": lambda seed: LogisticRegression(max_iter=1200, class_weight="balanced", random_state=seed),
    "random_forest": lambda seed: RandomForestClassifier(
        n_estimators=250,
        max_depth=None,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=seed,
    ),
}


def _load_splits(context: Dict[str, Any]) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    train_df = context.get("train_df")
    test_df = context.get("test_df")
    if isinstance(train_df, pd.DataFrame) and isinstance(test_df, pd.DataFrame) and not train_df.empty and not test_df.empty:
        return train_df.copy(), test_df.copy()

    train_path = Path("data/processed/train.csv")
    test_path = Path("data/processed/test.csv")
    if train_path.exists() and test_path.exists():
        return pd.read_csv(train_path), pd.read_csv(test_path)

    clean_df = context.get("clean_df")
    if isinstance(clean_df, pd.DataFrame) and not clean_df.empty:
        return clean_df.copy(), None

    return None, None


def _ensure_output_dirs() -> Dict[str, Path]:
    table_dir = Path("outputs/tables")
    table_dir.mkdir(parents=True, exist_ok=True)
    return {"tables": table_dir}


def _target_to_binary(series: pd.Series) -> pd.Series:
    mapping = {"<=50K": 0, ">50K": 1}
    mapped = series.astype(str).str.strip().map(mapping)
    if mapped.isna().all():
        return pd.Series(pd.factorize(series)[0], index=series.index)
    return mapped.fillna(0).astype(int)


def _build_pipeline(numeric_cols: list[str], categorical_cols: list[str], model_name: str, seed: int) -> Pipeline:
    numeric_pipe = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ]
    )
    model = MODEL_REGISTRY[model_name](seed)
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def run(config: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    if not SKLEARN_AVAILABLE:
        return {
            "step": "classification",
            "status": "skipped",
            "message": (
                "scikit-learn is required for classification. "
                "Install dependencies in your venv: pip install -r requirements.txt"
            ),
        }

    train_df, test_df = _load_splits(context)
    if train_df is None:
        return {
            "step": "classification",
            "status": "skipped",
            "message": "No cleaned dataset available. Run preprocessing first.",
        }

    schema_cfg = config.get("schema", {})
    cls_cfg = config.get("classification", {})

    target_col = schema_cfg.get("target", "income")
    if target_col not in train_df.columns:
        return {
            "step": "classification",
            "status": "error",
            "message": f"Target column '{target_col}' not found.",
        }

    numeric_cols = [c for c in schema_cfg.get("numeric_features", []) if c in train_df.columns]
    categorical_cols = [c for c in schema_cfg.get("categorical_features", []) if c in train_df.columns]
    feature_cols = numeric_cols + categorical_cols
    if not feature_cols:
        return {
            "step": "classification",
            "status": "error",
            "message": "No configured feature columns available for classification.",
        }

    configured_models = [
        m for m in cls_cfg.get("baseline_models", ["logistic_regression", "random_forest"]) if m in MODEL_REGISTRY
    ]
    if not configured_models:
        return {
            "step": "classification",
            "status": "skipped",
            "message": "No supported baseline models configured.",
        }

    if test_df is None or test_df.empty:
        X = train_df[feature_cols]
        y = _target_to_binary(train_df[target_col])
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=int(config.get("split", {}).get("random_state", 42)),
            stratify=y,
        )
    else:
        test_df = test_df.copy()
        test_df = test_df[test_df[target_col].notna()]
        X_train = train_df[feature_cols]
        y_train = _target_to_binary(train_df[target_col])
        X_test = test_df[feature_cols]
        y_test = _target_to_binary(test_df[target_col])

    rows: list[Dict[str, Any]] = []
    for model_name in configured_models:
        pipeline = _build_pipeline(
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
            model_name=model_name,
            seed=int(config.get("split", {}).get("random_state", 42)),
        )
        pipeline.fit(X_train, y_train)
        pred = pipeline.predict(X_test)
        prob = pipeline.predict_proba(X_test)[:, 1]

        rows.append(
            {
                "model": model_name,
                "accuracy": float(accuracy_score(y_test, pred)),
                "precision": float(precision_score(y_test, pred, zero_division=0)),
                "recall": float(recall_score(y_test, pred, zero_division=0)),
                "f1": float(f1_score(y_test, pred, zero_division=0)),
                "roc_auc": float(roc_auc_score(y_test, prob)),
            }
        )

    optimize_for = str(cls_cfg.get("optimize_for", "f1")).lower()
    sort_metric = optimize_for if optimize_for in {"accuracy", "precision", "recall", "f1", "roc_auc"} else "f1"
    metrics_df = pd.DataFrame(rows).sort_values(sort_metric, ascending=False).reset_index(drop=True)
    best_model = metrics_df.iloc[0]["model"] if not metrics_df.empty else None

    dirs = _ensure_output_dirs()
    metrics_path = dirs["tables"] / "classification_baselines.csv"
    metrics_df.to_csv(metrics_path, index=False)

    return {
        "step": "classification",
        "status": "ok",
        "message": f"Classification baselines completed. Best model by {sort_metric}: {best_model}.",
        "metrics_path": str(metrics_path),
        "best_model": best_model,
        "optimized_metric": sort_metric,
    }
