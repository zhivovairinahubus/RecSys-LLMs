"""Kaggle-style truth construction and MAP@5 evaluation."""

from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .data import BASE_DIR, DATA_DIR, load_pairs
except ImportError:
    from data import BASE_DIR, DATA_DIR, load_pairs

MIN_SCREEN_TIME = 0.8
TOP_K = 5
TIE_BREAK = (
    "n_airings desc, total_screen_time desc, last_air_start_sec desc, "
    "tv_show_id asc"
)
TIE_BREAK_RULES = {
    "main": (
        ["n_airings", "total_screen_time", "last_air_start_sec", "tv_show_id"],
        [False, False, False, True],
    ),
    "screen_time": (
        ["n_airings", "total_screen_time", "tv_show_id"],
        [False, False, True],
    ),
    "recent": (
        ["n_airings", "last_air_start_sec", "tv_show_id"],
        [False, False, True],
    ),
    "show_id": (
        ["n_airings", "tv_show_id"],
        [False, True],
    ),
}

RESULTS_DIR = BASE_DIR / "results"
VALIDATION_LOG = RESULTS_DIR / "validation.csv"
VALIDATION_COLUMNS = [
    "method",
    "params",
    "window",
    "n_users",
    "map5",
    "coverage_users",
    "coverage_share",
    "repeat_share",
    "distinct_shows",
    "seconds",
]


def build_truth(
    pairs: pd.DataFrame,
    week_start: int,
    week_end: int,
    *,
    iso_year: int | None = None,
    min_screen_time: float = MIN_SCREEN_TIME,
    top_k: int = TOP_K,
    tie_break: str = "main",
) -> pd.DataFrame:
    """Build top-k show truth for an ISO-week window.

    The window is applied to the ISO week of the airing start. Ranking is by
    the number of qualifying airings (screen_time >= min_screen_time) per show.
    Tie-breaking is configurable via ``tie_break`` for sensitivity checks.
    """
    if tie_break not in TIE_BREAK_RULES:
        raise ValueError(f"unknown tie-break rule: {tie_break}")
    window = pairs.loc[pairs["screen_time"].ge(min_screen_time)]
    if iso_year is not None:
        window = window.loc[window["air_iso_year"].eq(iso_year)]
    window = window.loc[window["air_iso_week"].between(week_start, week_end)]

    columns = [
        "user_id",
        "tv_show_id",
        "n_airings",
        "total_screen_time",
        "last_air_start_sec",
        "rank",
    ]
    if window.empty:
        return pd.DataFrame(columns=columns)

    grouped = (
        window.groupby(["user_id", "tv_show_id"], sort=False, observed=True)
        .agg(
            n_airings=("schedule_row_id", "size"),
            total_screen_time=("screen_time", "sum"),
            last_air_start_sec=("air_start_sec", "max"),
        )
        .reset_index()
    )

    sort_columns, sort_ascending = TIE_BREAK_RULES[tie_break]
    grouped = grouped.sort_values(
        ["user_id", *sort_columns],
        ascending=[True, *sort_ascending],
        kind="mergesort",
    )
    grouped = grouped.groupby("user_id", sort=False).head(top_k).copy()
    grouped["rank"] = grouped.groupby("user_id", sort=False).cumcount() + 1

    return grouped[columns].reset_index(drop=True)


def truth_to_lists(truth: pd.DataFrame) -> dict[int, list[int]]:
    if truth.empty:
        return {}
    ordered = truth.sort_values(["user_id", "rank"])
    return {
        int(user_id): [int(show_id) for show_id in group["tv_show_id"]]
        for user_id, group in ordered.groupby("user_id", sort=False)
    }


def average_precision_at_k(
    predicted: list[int] | tuple[int, ...], relevant: set[int] | list[int], k: int = 5
) -> float:
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0

    hits = 0
    score = 0.0
    for position, item in enumerate(list(predicted)[:k], start=1):
        if item in relevant_set:
            hits += 1
            score += hits / position

    return score / min(k, len(relevant_set))


def mean_average_precision_at_k(
    predictions: dict[int, list[int]],
    truths: dict[int, list[int]],
    k: int = 5,
) -> float:
    if not truths:
        return 0.0
    scores = [
        average_precision_at_k(predictions.get(user_id, []), relevant, k)
        for user_id, relevant in truths.items()
    ]
    return float(np.mean(scores))


def average_precision_per_user(
    predictions: dict[int, list[int]],
    truths: dict[int, list[int]],
    k: int = 5,
) -> np.ndarray:
    return np.asarray(
        [
            average_precision_at_k(predictions.get(user_id, []), relevant, k)
            for user_id, relevant in truths.items()
        ],
        dtype=np.float64,
    )


def paired_bootstrap_difference(
    predictions_a: dict[int, list[int]],
    predictions_b: dict[int, list[int]],
    truths: dict[int, list[int]],
    *,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict[str, float]:
    """Paired user bootstrap for MAP@5 difference A - B."""
    user_ids = list(truths)
    if not user_ids:
        return {"mean_diff": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    ap_a = average_precision_per_user(predictions_a, truths)
    ap_b = average_precision_per_user(predictions_b, truths)
    observed = float(np.mean(ap_a - ap_b))
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot, dtype=np.float64)
    n_users = len(user_ids)
    for index in range(n_boot):
        sample = rng.integers(0, n_users, size=n_users)
        draws[index] = float(np.mean((ap_a - ap_b)[sample]))
    return {
        "mean_diff": observed,
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
    }


def tie_break_sensitivity(
    predictions: dict[int, list[int]],
    pairs: pd.DataFrame,
    week_start: int,
    week_end: int,
    *,
    iso_year: int = 2020,
    user_ids: set[int] | None = None,
) -> list[dict[str, float | int | str]]:
    """MAP@5 over users with >=1 truth show under several tie-breaks."""
    rows: list[dict[str, float | int | str]] = []
    for rule in TIE_BREAK_RULES:
        truth = build_truth(
            pairs,
            week_start,
            week_end,
            iso_year=iso_year,
            tie_break=rule,
        )
        truth_lists = truth_to_lists(truth)
        if user_ids is not None:
            truth_lists = {
                user_id: relevant
                for user_id, relevant in truth_lists.items()
                if user_id in user_ids
            }
        if not truth_lists:
            rows.append(
                {
                    "rule": rule,
                    "n_users": 0,
                    "map5_all": 0.0,
                    "map5_exact5": 0.0,
                }
            )
            continue
        exact_five = {
            user_id: relevant
            for user_id, relevant in truth_lists.items()
            if len(relevant) == TOP_K
        }
        rows.append(
            {
                "rule": rule,
                "n_users": len(truth_lists),
                "map5_all": mean_average_precision_at_k(
                    predictions, truth_lists
                ),
                "map5_exact5": mean_average_precision_at_k(
                    predictions, exact_five
                )
                if exact_five
                else 0.0,
            }
        )
    return rows


def evaluate_recommendations(
    predictions: dict[int, list[int]],
    truths: dict[int, list[int]],
    history_pairs: pd.DataFrame,
    history_weeks: tuple[int, int],
    *,
    k: int = TOP_K,
    min_screen_time: float = MIN_SCREEN_TIME,
    seconds: float | None = None,
) -> dict[str, float | int]:
    """Compute MAP@5 and business metrics for a set of recommendation lists."""
    started = time.perf_counter()
    users = list(truths)
    n_users = len(users)

    if n_users == 0:
        elapsed = time.perf_counter() - started
        return {
            "n_users": 0,
            "map5": 0.0,
            "coverage_users": 0,
            "coverage_share": 0.0,
            "repeat_slots": 0,
            "repeat_share": 0.0,
            "distinct_shows": 0,
            "seconds": float(seconds) if seconds is not None else elapsed,
            "metrics_seconds": elapsed,
        }

    history = history_pairs.loc[
        history_pairs["air_iso_week"].between(history_weeks[0], history_weeks[1])
        & history_pairs["screen_time"].ge(min_screen_time)
    ]
    watched_by_user = {
        int(user_id): {int(show_id) for show_id in group["tv_show_id"]}
        for user_id, group in history.groupby("user_id", sort=False)
    }

    coverage_users = 0
    repeat_slots = 0
    distinct_shows: set[int] = set()

    for user_id, relevant in truths.items():
        predicted = [int(show_id) for show_id in predictions.get(user_id, [])[:k]]
        relevant_set = set(relevant)
        watched = watched_by_user.get(int(user_id), set())

        if relevant_set.intersection(predicted):
            coverage_users += 1
        repeat_slots += sum(1 for show_id in predicted if show_id in watched)
        distinct_shows.update(predicted)

    metrics_seconds = time.perf_counter() - started
    return {
        "n_users": n_users,
        "map5": mean_average_precision_at_k(predictions, truths, k),
        "coverage_users": coverage_users,
        "coverage_share": coverage_users / n_users,
        "repeat_slots": repeat_slots,
        "repeat_share": repeat_slots / (n_users * k),
        "distinct_shows": len(distinct_shows),
        "seconds": float(seconds) if seconds is not None else metrics_seconds,
        "metrics_seconds": metrics_seconds,
    }


def append_validation_result(row: dict[str, object]) -> Path:
    """Append one method run to the validation log, creating its header."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    write_header = not VALIDATION_LOG.exists() or VALIDATION_LOG.stat().st_size == 0

    with VALIDATION_LOG.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=VALIDATION_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow({column: row.get(column, "") for column in VALIDATION_COLUMNS})

    return VALIDATION_LOG


def _self_test() -> None:
    relevant = {"A", "B", "C"}
    x_score = average_precision_at_k(["A", "B", "D", "C", "E"], relevant)
    y_score = average_precision_at_k(["D", "A", "E", "B", "C"], relevant)

    assert np.isclose(x_score, 0.9166666666666666)
    assert np.isclose(y_score, 0.5333333333333333)

    print(f"self-test X: {x_score:.4f}")
    print(f"self-test Y: {y_score:.4f}")


def _submission_truth_counts(
    pairs: pd.DataFrame, submission_users: set[int], week_start: int, week_end: int
) -> tuple[int, int, int, int]:
    truth = build_truth(pairs, week_start, week_end, iso_year=2020)
    sizes = {
        int(user_id): int(size)
        for user_id, size in truth.groupby("user_id", sort=False).size().items()
    }

    truth_users = len(sizes)
    with_any_truth = sum(1 for user_id in submission_users if sizes.get(user_id, 0) > 0)
    with_exactly_five = sum(
        1 for user_id in submission_users if sizes.get(user_id, 0) == 5
    )
    with_at_least_five = sum(
        1 for user_id in submission_users if sizes.get(user_id, 0) >= 5
    )
    return truth_users, with_any_truth, with_exactly_five, with_at_least_five


def main() -> None:
    _self_test()

    pairs = load_pairs()
    share = float((pairs["screen_time"] >= 0.8).mean())
    print(f"user-airing pairs: {len(pairs):,}")
    print(
        "screen_time >= 0.8: "
        f"{share:.4%} ({int((pairs['screen_time'] >= 0.8).sum()):,} pairs)"
    )
    print(f"tie-break: {TIE_BREAK}")

    submission = pd.read_csv(DATA_DIR / "submission.csv")
    submission_users = {int(user_id) for user_id in submission["user_id"]}

    for week_start, week_end in [(21, 30), (19, 30)]:
        truth_users, with_any, exactly_five, at_least_five = _submission_truth_counts(
            pairs, submission_users, week_start, week_end
        )
        print(f"window {week_start}-{week_end}:")
        print(f"  truth users: {truth_users:,}")
        print(f"  submission users with truth: {with_any:,}")
        print(f"  submission users with exactly 5 shows: {exactly_five:,}")
        print(f"  submission users with at least 5 shows: {at_least_five:,}")


if __name__ == "__main__":
    main()
