from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score


def run(config: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    if "train_df" not in context or "test_df" not in context:
        return {
            "step": "defense_dp",
            "status": "skipped",
            "message": "Preprocessing outputs missing.",
        }

    schema = config.get("schema", {})
    target_col = schema.get("target", "income")
    numeric_cols = schema.get("numeric_features", [])

    train_df = context["train_df"]
    test_df = context["test_df"]

    X_train = train_df[numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    y_train = (train_df[target_col] == ">50K").astype(int)

    X_test = test_df[numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    y_test = (test_df[target_col] == ">50K").astype(int)

    epsilons = [0.1, 0.5, 1.0, 2.0, 5.0]
    rows = []

    for eps in epsilons:
        noise = np.random.laplace(0, 1 / eps, X_train.shape)
        X_dp = X_train + noise

        model = LogisticRegression(max_iter=1000)
        model.fit(X_dp, y_train)

        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]

        rows.append(
            {
                "epsilon": eps,
                "accuracy": accuracy_score(y_test, preds),
                "attack_auc_proxy": roc_auc_score(y_test, probs),
            }
        )

    df = pd.DataFrame(rows)

    table_dir = Path("outputs/tables")
    fig_dir = Path("outputs/figures")
    table_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    table_path = table_dir / "defense_dp_results.csv"
    fig_path = fig_dir / "privacy_utility_curve.png"

    df.to_csv(table_path, index=False)

    plt.figure(figsize=(7, 5))
    plt.plot(df["epsilon"], df["accuracy"], marker="o", label="Utility (Accuracy)")
    plt.plot(df["epsilon"], df["attack_auc_proxy"], marker="o", label="Privacy Risk")
    plt.xlabel("Epsilon (ε)")
    plt.ylabel("Score")
    plt.title("Privacy–Utility Tradeoff")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=140)
    plt.close()

    return {
        "step": "defense_dp",
        "status": "ok",
        "results_path": str(table_path),
        "figure_path": str(fig_path),
    }
