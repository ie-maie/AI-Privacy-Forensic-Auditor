from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple

import pandas as pd

Itemset = frozenset[str]


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
    table_dir = Path("outputs/tables")
    table_dir.mkdir(parents=True, exist_ok=True)
    return {"tables": table_dir}


def _discretize_numeric(df: pd.DataFrame, numeric_cols: Iterable[str]) -> pd.DataFrame:
    result = pd.DataFrame(index=df.index)
    for col in numeric_cols:
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        non_null = series.dropna()
        if non_null.empty:
            continue

        # Quantile edges can be duplicated on sparse/heavily-zeroed columns
        # (e.g., capital_gain/capital_loss), which makes pd.cut fail with
        # "Bin edges must be unique". Build bins dynamically from unique edges.
        q_edges = non_null.quantile([0.0, 0.33, 0.66, 1.0]).tolist()
        unique_edges: list[float] = []
        for edge in q_edges:
            edge_f = float(edge)
            if not unique_edges or edge_f > unique_edges[-1]:
                unique_edges.append(edge_f)

        if len(unique_edges) <= 1:
            result[col] = pd.Series(["mid"] * len(series), index=series.index, dtype="string")
            result[col] = result[col].where(series.notna(), other="MISSING")
            continue

        label_count = len(unique_edges) - 1
        if label_count == 1:
            labels = ["mid"]
        elif label_count == 2:
            labels = ["low", "high"]
        else:
            labels = ["low", "mid", "high"][:label_count]

        bucketed = pd.cut(
            series,
            bins=unique_edges,
            labels=labels,
            include_lowest=True,
            duplicates="drop",
        )
        result[col] = bucketed.astype("string").fillna("MISSING")
    return result


def _prepare_transactions(
    df: pd.DataFrame,
    categorical_cols: Iterable[str],
    numeric_cols: Iterable[str],
    max_categories_per_col: int = 12,
) -> List[Set[str]]:
    tx_df = pd.DataFrame(index=df.index)

    for col in categorical_cols:
        if col not in df.columns:
            continue
        values = df[col].astype("string").fillna("MISSING")
        top_values = values.value_counts().head(max_categories_per_col).index
        values = values.where(values.isin(top_values), other="OTHER")
        tx_df[col] = values

    binned_numeric = _discretize_numeric(df, numeric_cols)
    for col in binned_numeric.columns:
        tx_df[col] = binned_numeric[col]

    if tx_df.empty:
        return []

    for col in tx_df.columns:
        tx_df[col] = tx_df[col].astype(str).map(lambda v: f"{col}={v}")

    return [set(row.dropna().tolist()) for _, row in tx_df.iterrows()]


def _support_count(transactions: List[Set[str]], itemset: Itemset) -> int:
    return sum(1 for tx in transactions if itemset.issubset(tx))


def _generate_candidates(prev_level: List[Itemset], k: int) -> List[Itemset]:
    prev_sorted = sorted(prev_level, key=lambda s: tuple(sorted(s)))
    prev_set = set(prev_level)
    candidates: set[Itemset] = set()

    for i in range(len(prev_sorted)):
        for j in range(i + 1, len(prev_sorted)):
            union = prev_sorted[i] | prev_sorted[j]
            if len(union) != k:
                continue
            # Apriori pruning: all (k-1)-subsets must be frequent
            if all(frozenset(union - {item}) in prev_set for item in union):
                candidates.add(union)

    return sorted(candidates, key=lambda s: tuple(sorted(s)))


def _mine_frequent_itemsets(
    transactions: List[Set[str]],
    min_support: float,
    max_len: int = 3,
) -> Dict[Itemset, float]:
    n_tx = len(transactions)
    if n_tx == 0:
        return {}

    min_count = max(1, int(min_support * n_tx))
    all_supports: Dict[Itemset, float] = {}

    item_counter: Counter[str] = Counter()
    for tx in transactions:
        item_counter.update(tx)

    current_level = [frozenset([item]) for item, count in item_counter.items() if count >= min_count]
    for itemset in current_level:
        all_supports[itemset] = item_counter[next(iter(itemset))] / n_tx

    k = 2
    while current_level and k <= max_len:
        candidates = _generate_candidates(current_level, k)
        next_level: List[Itemset] = []

        for candidate in candidates:
            count = _support_count(transactions, candidate)
            if count >= min_count:
                next_level.append(candidate)
                all_supports[candidate] = count / n_tx

        current_level = next_level
        k += 1

    return all_supports


def _generate_rules(
    supports: Dict[Itemset, float],
    min_confidence: float,
    min_lift: float,
) -> pd.DataFrame:
    rows: list[Dict[str, Any]] = []

    for itemset, support_itemset in supports.items():
        if len(itemset) < 2:
            continue

        items = sorted(itemset)
        for r in range(1, len(items)):
            # simple subset generation for small k
            from itertools import combinations

            for antecedent_tuple in combinations(items, r):
                antecedent = frozenset(antecedent_tuple)
                consequent = itemset - antecedent
                ant_support = supports.get(antecedent)
                cons_support = supports.get(consequent)
                if not ant_support or not cons_support:
                    continue

                confidence = support_itemset / ant_support
                lift = confidence / cons_support
                if confidence < min_confidence or lift < min_lift:
                    continue

                rows.append(
                    {
                        "antecedent": " & ".join(sorted(antecedent)),
                        "consequent": " & ".join(sorted(consequent)),
                        "support": support_itemset,
                        "confidence": confidence,
                        "lift": lift,
                        "antecedent_support": ant_support,
                        "consequent_support": cons_support,
                    }
                )

    if not rows:
        return pd.DataFrame(
            columns=[
                "antecedent",
                "consequent",
                "support",
                "confidence",
                "lift",
                "antecedent_support",
                "consequent_support",
            ]
        )

    return pd.DataFrame(rows).drop_duplicates().sort_values(
        ["lift", "confidence", "support"], ascending=False
    ).reset_index(drop=True)


def run(config: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    df = _load_clean_df(context)
    if df is None:
        return {
            "step": "association_rules",
            "status": "skipped",
            "message": "No cleaned dataset available. Run preprocessing first.",
        }

    schema_cfg = config.get("schema", {})
    assoc_cfg = config.get("association_rules", {})

    categorical_cols = [c for c in schema_cfg.get("categorical_features", []) if c in df.columns]
    numeric_cols = [c for c in schema_cfg.get("numeric_features", []) if c in df.columns]
    if not categorical_cols and not numeric_cols:
        return {
            "step": "association_rules",
            "status": "error",
            "message": "No eligible columns found for association-rule mining.",
        }

    min_support = float(assoc_cfg.get("min_support", 0.05))
    min_confidence = float(assoc_cfg.get("min_confidence", 0.6))
    min_lift = float(assoc_cfg.get("min_lift", 1.1))
    max_len = int(assoc_cfg.get("max_itemset_len", 3))

    transactions = _prepare_transactions(
        df=df,
        categorical_cols=categorical_cols,
        numeric_cols=numeric_cols,
    )
    supports = _mine_frequent_itemsets(transactions, min_support=min_support, max_len=max_len)
    rules_df = _generate_rules(supports, min_confidence=min_confidence, min_lift=min_lift)

    dirs = _ensure_output_dirs()
    rules_path = dirs["tables"] / "association_rules.csv"
    top_rules_path = dirs["tables"] / "association_rules_top10.csv"
    rules_df.to_csv(rules_path, index=False)
    rules_df.head(10).to_csv(top_rules_path, index=False)

    return {
        "step": "association_rules",
        "status": "ok",
        "message": f"Association-rule mining complete. Generated {len(rules_df)} rules.",
        "rules_path": str(rules_path),
        "top_rules_path": str(top_rules_path),
        "n_rules": int(len(rules_df)),
        "n_transactions": len(transactions),
    }
