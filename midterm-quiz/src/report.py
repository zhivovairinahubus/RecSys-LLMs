"""Build aggregate report tables from the validation log and cached pairs."""

from __future__ import annotations

import time

import pandas as pd

try:
    from .data import (
        BASE_DIR,
        DATA_DIR,
        HISTORY_END_SEC,
        HISTORY_PARTS,
        load_pairs,
    )
    from .methods import (
        _build_ppr_graph,
        _load_show_content,
        _window_context,
        als_predictions,
        assoc_predictions,
        content_predictions,
        ppr_predictions,
    )
    from .recommend import (
        POPULARITY_N,
        recommend_personal_frequency,
        recommend_popularity,
    )
except ImportError:
    from data import (
        BASE_DIR,
        DATA_DIR,
        HISTORY_END_SEC,
        HISTORY_PARTS,
        load_pairs,
    )
    from methods import (
        _build_ppr_graph,
        _load_show_content,
        _window_context,
        als_predictions,
        assoc_predictions,
        content_predictions,
        ppr_predictions,
    )
    from recommend import (
        POPULARITY_N,
        recommend_personal_frequency,
        recommend_popularity,
    )

RESULTS_DIR = BASE_DIR / "results"
VALIDATION_PATH = RESULTS_DIR / "validation.csv"
SUMMARY_PATH = RESULTS_DIR / "method_summary.csv"
WEEKLY_PATH = RESULTS_DIR / "weekly_coverage.csv"
HISTORY_WEEKS = (11, 20)
TARGET_WEEKS = (21, 30)
SECONDARY_WEEKS = (11, 18)
SECONDARY_TARGET = (19, 30)
PPR_MAX_ITER = 200
PPR_TOL = 1e-8
KAGGLE = {
    "popularity": (0.33047, 0.32096),
    "personal_frequency": (0.40568, 0.38897),
    "als": (0.33960, 0.32260),
    "ppr": (0.43574, 0.42335),
    "hybrid": (0.43574, 0.42335),
    "slots": (0.40497, 0.39745),
    "final": (0.44048, 0.42858),
}
SUBMISSION = {
    "popularity": "01_popularity.csv",
    "personal_frequency": "05_personal_frequency.csv",
    "als": "04_als.csv",
    "ppr": "02_final_ppr_decay_beta.csv",
    "hybrid": "02_final_ppr_decay_beta.csv",
    "slots": "07_slots_pure.csv",
    "final": "06_final_ppr_slots.csv",
}
METHOD_CONFIG = {
    "popularity": "N=2",
    "personal_frequency": "screen>=0.5, fallback_N=2",
    "assoc": "thr=0.8, sup=0.01, conf=0.2, score=confidence_lift",
    "content": "thr=0.8, profile=frequency, feat=idf",
    "als": "hist=pool, thr=0.8, f=128, reg=10, conf_alpha=0.5, seeds=3",
    "ppr": "hist=all, thr=0.5, restart=0.3, decay=0.9, beta=0.5",
    "hybrid": "ppr=1.0, assoc=0, content=0, als=0",
    "slots": "slots:channel_dow_hour,smooth:5,pure",
    "final": "slots:channel_dow_hour,smooth:5,ppr+slots0.5",
}


def _primary_contexts() -> tuple[dict[str, object], dict[str, object], pd.DataFrame, pd.DataFrame]:
    pairs = load_pairs()
    history = load_pairs(cutoff_sec=HISTORY_END_SEC[HISTORY_WEEKS])
    submission = pd.read_csv(DATA_DIR / "submission.csv")
    submission_users = {int(user_id) for user_id in submission["user_id"]}
    primary = _window_context(
        pairs,
        submission_users,
        HISTORY_WEEKS,
        TARGET_WEEKS,
        history_scope="all",
        min_screen_time=0.5,
        history_pairs=history,
    )
    primary_pool = _window_context(
        pairs,
        submission_users,
        HISTORY_WEEKS,
        TARGET_WEEKS,
        history_scope="pool",
        min_screen_time=0.8,
        history_pairs=history,
    )
    content, _ = _load_show_content(
        schedule_parts=HISTORY_PARTS + ("31-42",)
    )
    return primary, primary_pool, content, pairs


def _best_primary_rows(validation: pd.DataFrame) -> dict[str, pd.Series]:
    selected: dict[str, pd.Series] = {}
    for method in ("popularity", "personal_frequency", "assoc", "content", "als"):
        rows = validation[
            (validation["method"].eq(method))
            & (validation["window"].eq("11-20 -> 21-30"))
        ]
        if not rows.empty:
            selected[method] = rows.loc[rows["map5"].idxmax()]
    ppr_rows = validation[
        (validation["method"].eq("ppr_final"))
        & (validation["window"].eq("11-20 -> 21-30"))
    ]
    if not ppr_rows.empty:
        selected["ppr"] = ppr_rows.loc[ppr_rows["map5"].idxmax()]
    hybrid_rows = validation[
        (validation["method"].eq("hybrid"))
        & (validation["window"].eq("11-20 -> 21-30"))
    ]
    if not hybrid_rows.empty:
        selected["hybrid"] = hybrid_rows.loc[hybrid_rows["map5"].idxmax()]
    slots_rows = validation[
        (validation["method"].eq("slots"))
        & (validation["window"].eq("11-20 -> 21-30"))
        & (validation["params"].eq("slots:channel_dow_hour,smooth:5,pure"))
    ]
    if not slots_rows.empty:
        selected["slots"] = slots_rows.loc[slots_rows["map5"].idxmax()]
    final_rows = validation[
        (validation["method"].eq("slots"))
        & (validation["window"].eq("11-20 -> 21-30"))
        & (validation["params"].eq("slots:channel_dow_hour,smooth:5,ppr+slots0.5"))
    ]
    if not final_rows.empty:
        selected["final"] = final_rows.loc[final_rows["map5"].idxmax()]
    return selected


def _secondary_row(
    validation: pd.DataFrame, primary_row: pd.Series
) -> pd.Series | None:
    rows = validation[
        (validation["method"].eq(primary_row["method"]))
        & (validation["window"].eq("11-18 -> 19-30"))
        & (validation["params"].eq(primary_row["params"]))
    ]
    if rows.empty:
        return None
    return rows.iloc[0]


def _timings(
    primary: dict[str, object],
    primary_pool: dict[str, object],
    content: pd.DataFrame,
) -> dict[str, dict[str, float]]:
    users = list(primary["eligible"].keys())
    timings: dict[str, dict[str, float]] = {}

    started = time.perf_counter()
    recommend_popularity(
        primary["history_pairs"],
        primary["pool"],
        HISTORY_WEEKS,
        POPULARITY_N,
        users,
    )
    timings["popularity"] = {
        "train_seconds": 0.0,
        "list_seconds": time.perf_counter() - started,
    }

    started = time.perf_counter()
    recommend_personal_frequency(
        primary["history_pairs"],
        primary["pool"],
        HISTORY_WEEKS,
        0.5,
        primary["fallback_ranking"],
        users,
    )
    timings["personal_frequency"] = {
        "train_seconds": 0.0,
        "list_seconds": time.perf_counter() - started,
    }

    _, assoc_train, assoc_list = assoc_predictions(
        primary_pool["interactions"],
        primary_pool["pool"],
        primary_pool["fallback_ranking"],
        users,
        min_support=0.01,
        min_confidence=0.2,
        score_mode="confidence_lift",
    )
    timings["assoc"] = {
        "train_seconds": assoc_train,
        "list_seconds": assoc_list,
    }

    _, content_train, content_list = content_predictions(
        primary_pool["interactions"],
        primary_pool["pool"],
        primary_pool["fallback_ranking"],
        users,
        content,
        profile_weight="frequency",
        feature_weight="idf",
    )
    timings["content"] = {
        "train_seconds": content_train,
        "list_seconds": content_list,
    }

    als_train = 0.0
    als_list = 0.0
    for seed in (42, 123, 2026):
        _, train_seconds, list_seconds = als_predictions(
            primary_pool["interactions"],
            primary_pool["pool"],
            primary_pool["fallback_ranking"],
            users,
            factors=128,
            regularization=10.0,
            confidence_alpha=0.5,
            seed=seed,
        )
        als_train += train_seconds
        als_list += list_seconds
    timings["als"] = {"train_seconds": als_train, "list_seconds": als_list}

    graph_started = time.perf_counter()
    graph = _build_ppr_graph(
        primary["interactions"],
        "frequency",
        decay=0.9,
        reference_week=HISTORY_WEEKS[1],
    )
    graph_seconds = time.perf_counter() - graph_started
    ppr_started = time.perf_counter()
    ppr_predictions(
        graph,
        primary["pool"],
        primary["fallback_ranking"],
        users,
        restart_prob=0.3,
        airing_rate=primary["airing_rate"],
        beta=0.5,
        max_iter=PPR_MAX_ITER,
        tol=PPR_TOL,
        return_stats=True,
    )
    ppr_list = time.perf_counter() - ppr_started
    timings["ppr"] = {
        "train_seconds": graph_seconds,
        "list_seconds": ppr_list,
    }
    timings["hybrid"] = {"train_seconds": 0.0, "list_seconds": ppr_list}
    timings["slots"] = {"train_seconds": 0.0, "list_seconds": ppr_list}
    timings["final"] = timings["ppr"]
    return timings


def build_method_summary(validation: pd.DataFrame) -> pd.DataFrame:
    primary, primary_pool, content, _ = _primary_contexts()
    selected = _best_primary_rows(validation)
    timings = _timings(primary, primary_pool, content)
    rows = []
    for method in ("popularity", "personal_frequency", "als", "ppr", "assoc", "content", "hybrid", "slots", "final"):
        if method not in selected:
            continue
        row = selected[method]
        secondary = _secondary_row(validation, row)
        timing = timings[method]
        kaggle_public, kaggle_private = KAGGLE.get(method, (None, None))
        rows.append(
            {
                "method": method,
                "params": METHOD_CONFIG[method],
                "map5_primary": float(row["map5"]),
                "map5_secondary": float(secondary["map5"]) if secondary is not None else None,
                "coverage_users_primary": int(row["coverage_users"]),
                "coverage_share_primary": float(row["coverage_share"]),
                "repeat_share_primary": float(row["repeat_share"]),
                "distinct_shows_primary": int(row["distinct_shows"]),
                "coverage_users_secondary": int(secondary["coverage_users"]) if secondary is not None else None,
                "coverage_share_secondary": float(secondary["coverage_share"]) if secondary is not None else None,
                "repeat_share_secondary": float(secondary["repeat_share"]) if secondary is not None else None,
                "distinct_shows_secondary": int(secondary["distinct_shows"]) if secondary is not None else None,
                "train_seconds": timing["train_seconds"],
                "list_seconds": timing["list_seconds"],
                "kaggle_public": kaggle_public,
                "kaggle_private": kaggle_private,
                "submission_file": SUBMISSION.get(method, ""),
            }
        )
    return pd.DataFrame(rows)


def build_weekly_coverage(pairs: pd.DataFrame) -> pd.DataFrame:
    qualified = pairs.loc[pairs["screen_time"].ge(0.8)]
    grouped = pairs.groupby("air_iso_week", observed=True)
    weekly = pd.DataFrame(
        {
            "active_users": grouped["user_id"].nunique(),
            "watch_hours": grouped["overlap_sec"].sum() / 3600.0,
            "pairs_screen_ge_0_8": qualified.groupby("air_iso_week", observed=True).size(),
            "airings_screen_ge_0_8": qualified.groupby("air_iso_week", observed=True)["schedule_row_id"].nunique(),
        }
    )
    weekly = weekly.reindex(range(11, 31), fill_value=0)
    weekly.index.name = "iso_week"
    weekly = weekly.reset_index()
    weekly["iso_year"] = 2020
    weekly["week_start"] = [
        pd.Timestamp.fromisocalendar(2020, int(week), 1).date()
        for week in weekly["iso_week"]
    ]
    weekly["watch_hours"] = weekly["watch_hours"].astype(float).round(3)
    columns = [
        "iso_year",
        "iso_week",
        "week_start",
        "active_users",
        "watch_hours",
        "airings_screen_ge_0_8",
        "pairs_screen_ge_0_8",
    ]
    return weekly[columns]


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    validation = pd.read_csv(VALIDATION_PATH)
    summary = build_method_summary(validation)
    summary.to_csv(SUMMARY_PATH, index=False)
    weekly = build_weekly_coverage(load_pairs())
    weekly.to_csv(WEEKLY_PATH, index=False)

    print(f"wrote: {SUMMARY_PATH}")
    print(summary.to_string(index=False))
    print(f"\nwrote: {WEEKLY_PATH}")
    print(weekly.to_string(index=False))


if __name__ == "__main__":
    main()
