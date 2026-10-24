"""Candidate pools, popularity baseline, and local validation."""

from __future__ import annotations

import time
from collections.abc import Iterable, Sequence

import pandas as pd

try:
    from .data import DATA_DIR, HISTORY_PARTS, load_pairs, load_schedule
    from .evaluate import (
        append_validation_result,
        build_truth,
        evaluate_recommendations,
        truth_to_lists,
    )
except ImportError:
    from data import DATA_DIR, HISTORY_PARTS, load_pairs, load_schedule
    from evaluate import (
        append_validation_result,
        build_truth,
        evaluate_recommendations,
        truth_to_lists,
    )

YEAR = 2020
K = 5
MIN_SCREEN_TIME = 0.8
POPULARITY_N = 2
POPULARITY_TIE_BREAK = (
    "n_users desc, n_airings desc, total_screen_time desc, tv_show_id asc"
)
PERSONAL_TIE_BREAK = (
    "n_airings desc, total_screen_time desc, last_air_start_sec desc, "
    "tv_show_id asc"
)


def build_candidate_pool(
    target_weeks: tuple[int, int],
    schedule_parts: Iterable[str],
    *,
    iso_year: int = YEAR,
) -> list[int]:
    """Return tv_show_id values with non-zero id airing in the target weeks."""
    schedule = load_schedule(parts=schedule_parts, drop_unlabeled=True)
    target = schedule.loc[
        schedule["air_iso_year"].eq(iso_year)
        & schedule["air_iso_week"].between(target_weeks[0], target_weeks[1])
    ]
    return sorted(int(show_id) for show_id in target["tv_show_id"].unique())


def effective_history_weeks(
    history_weeks: tuple[int, int], n_weeks: int
) -> tuple[int, int, int]:
    start, end = history_weeks
    effective = min(n_weeks, end - start + 1)
    return end - effective + 1, end, effective


def popularity_ranking(
    pairs: pd.DataFrame,
    pool: Sequence[int],
    history_weeks: tuple[int, int],
    n_weeks: int,
    *,
    min_screen_time: float = MIN_SCREEN_TIME,
) -> pd.DataFrame:
    """Rank pool shows by distinct users with a qualifying airing in the window."""
    start, end, _ = effective_history_weeks(history_weeks, n_weeks)
    history = pairs.loc[
        pairs["air_iso_week"].between(start, end)
        & pairs["screen_time"].ge(min_screen_time)
        & pairs["tv_show_id"].isin(pool)
    ]

    counts = (
        history.groupby("tv_show_id", sort=False, observed=True)
        .agg(
            n_users=("user_id", "nunique"),
            n_airings=("schedule_row_id", "size"),
            total_screen_time=("screen_time", "sum"),
        )
        .reset_index()
    )

    ranking = pd.DataFrame({"tv_show_id": list(pool)}).merge(
        counts, on="tv_show_id", how="left", validate="one_to_one"
    )
    ranking[["n_users", "n_airings"]] = (
        ranking[["n_users", "n_airings"]].fillna(0).astype("int64")
    )
    ranking["total_screen_time"] = ranking["total_screen_time"].fillna(0.0)

    return ranking.sort_values(
        ["n_users", "n_airings", "total_screen_time", "tv_show_id"],
        ascending=[False, False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)


def fill_recommendations(
    primary: Iterable[int], fallback: Iterable[int], k: int = K
) -> list[int]:
    """Return k distinct non-zero ids, taking missing slots from fallback."""
    result: list[int] = []
    seen: set[int] = set()
    for source in (primary, fallback):
        for show_id in source:
            show_id = int(show_id)
            if show_id == 0 or show_id in seen:
                continue
            result.append(show_id)
            seen.add(show_id)
            if len(result) == k:
                return result
    raise ValueError(f"cannot build {k} unique non-zero recommendations")


def popularity_recommendations(
    pairs: pd.DataFrame,
    pool: Sequence[int],
    history_weeks: tuple[int, int],
    n_weeks: int,
    *,
    k: int = K,
) -> list[int]:
    ranking = popularity_ranking(pairs, pool, history_weeks, n_weeks)
    ranked_ids = [int(show_id) for show_id in ranking["tv_show_id"]]
    return fill_recommendations(ranked_ids, ranked_ids, k=k)


def recommend_popularity(
    pairs: pd.DataFrame,
    pool: Sequence[int],
    history_weeks: tuple[int, int],
    n_weeks: int,
    user_ids: Iterable[int],
    *,
    k: int = K,
) -> dict[int, list[int]]:
    top = popularity_recommendations(
        pairs, pool, history_weeks, n_weeks, k=k
    )
    return {int(user_id): top.copy() for user_id in user_ids}


def personal_frequency_ranking(
    pairs: pd.DataFrame,
    pool: Sequence[int],
    history_weeks: tuple[int, int],
    threshold: float,
) -> dict[int, list[int]]:
    """Rank each user's pool shows by qualifying airings in the history window."""
    history = pairs.loc[
        pairs["air_iso_week"].between(history_weeks[0], history_weeks[1])
        & pairs["screen_time"].ge(threshold)
        & pairs["tv_show_id"].isin(pool)
    ]
    if history.empty:
        return {}

    grouped = (
        history.groupby(["user_id", "tv_show_id"], sort=False, observed=True)
        .agg(
            n_airings=("schedule_row_id", "size"),
            total_screen_time=("screen_time", "sum"),
            last_air_start_sec=("air_start_sec", "max"),
        )
        .reset_index()
    )
    grouped = grouped.sort_values(
        [
            "user_id",
            "n_airings",
            "total_screen_time",
            "last_air_start_sec",
            "tv_show_id",
        ],
        ascending=[True, False, False, False, True],
        kind="mergesort",
    )
    return {
        int(user_id): [int(show_id) for show_id in group["tv_show_id"]]
        for user_id, group in grouped.groupby("user_id", sort=False)
    }


def recommend_personal_frequency(
    pairs: pd.DataFrame,
    pool: Sequence[int],
    history_weeks: tuple[int, int],
    threshold: float,
    fallback_ranking: pd.DataFrame,
    user_ids: Iterable[int],
    *,
    k: int = K,
) -> dict[int, list[int]]:
    personal = personal_frequency_ranking(pairs, pool, history_weeks, threshold)
    fallback = [int(show_id) for show_id in fallback_ranking["tv_show_id"]]
    return {
        int(user_id): fill_recommendations(
            personal.get(int(user_id), []), fallback, k=k
        )
        for user_id in user_ids
    }


def cold_start_counts(
    pairs: pd.DataFrame,
    history_weeks: tuple[int, int],
    submission_users: set[int],
) -> tuple[int, int]:
    history = pairs.loc[
        pairs["air_iso_week"].between(history_weeks[0], history_weeks[1])
    ]
    users_with_pairs = {int(user_id) for user_id in history["user_id"].unique()}
    qualifying = history.loc[history["screen_time"].ge(MIN_SCREEN_TIME)]
    users_with_qualifying = {
        int(user_id) for user_id in qualifying["user_id"].unique()
    }
    return (
        len(submission_users - users_with_pairs),
        len(submission_users - users_with_qualifying),
    )


def _evaluate_and_log(
    method: str,
    params: str,
    window: str,
    predictions: dict[int, list[int]],
    eligible: dict[int, list[int]],
    pairs: pd.DataFrame,
    history_weeks: tuple[int, int],
    generation_seconds: float,
) -> dict[str, float | int]:
    metrics = evaluate_recommendations(
        predictions, eligible, pairs, history_weeks
    )
    metrics["seconds"] = generation_seconds + float(metrics["metrics_seconds"])
    append_validation_result(
        {
            "method": method,
            "params": params,
            "window": window,
            "n_users": metrics["n_users"],
            "map5": metrics["map5"],
            "coverage_users": metrics["coverage_users"],
            "coverage_share": metrics["coverage_share"],
            "repeat_share": metrics["repeat_share"],
            "distinct_shows": metrics["distinct_shows"],
            "seconds": metrics["seconds"],
        }
    )
    return metrics


def _print_metrics_row(
    method: str, params: str, metrics: dict[str, float | int]
) -> None:
    print(
        f"  {method:<18} {params:<32} "
        f"{float(metrics['map5']):.6f}  "
        f"{int(metrics['coverage_users']):>5}/{int(metrics['n_users']):<5} "
        f"{float(metrics['repeat_share']):>7.4f}  "
        f"{int(metrics['distinct_shows']):>6}  "
        f"{float(metrics['seconds']):>7.3f}"
    )


def _validation_table(
    pairs: pd.DataFrame,
    submission_users: set[int],
    history_weeks: tuple[int, int],
    target_weeks: tuple[int, int],
    schedule_parts: Iterable[str],
    n_values: Sequence[int],
    thresholds: Sequence[float],
) -> list[dict[str, object]]:
    pool = build_candidate_pool(target_weeks, schedule_parts)
    truth = build_truth(pairs, target_weeks[0], target_weeks[1], iso_year=YEAR)
    truth_lists = truth_to_lists(truth)
    eligible = {
        user_id: relevant
        for user_id, relevant in truth_lists.items()
        if user_id in submission_users and len(relevant) == K
    }

    window = (
        f"{history_weeks[0]}-{history_weeks[1]} -> "
        f"{target_weeks[0]}-{target_weeks[1]}"
    )
    print(f"window: {window}")
    print(f"  candidate pool: {len(pool):,} shows")
    print(f"  eligible submission users with exactly {K} truth shows: {len(eligible):,}")
    print(f"  popularity tie-break: {POPULARITY_TIE_BREAK}")
    print(f"  personal-frequency tie-break: {PERSONAL_TIE_BREAK}")
    print(
        f"  {'method':<18} {'params':<32} {'MAP@5':>8}  "
        f"{'coverage':>12} {'repeat':>7}  {'shows':>6}  {'sec':>7}"
    )

    results: list[dict[str, object]] = []

    for n_weeks in n_values:
        effective_start, effective_end, effective = effective_history_weeks(
            history_weeks, n_weeks
        )
        params = (
            f"N={n_weeks} (weeks {effective_start}-{effective_end}, "
            f"{effective}w)"
        )
        started = time.perf_counter()
        predictions = recommend_popularity(
            pairs, pool, history_weeks, n_weeks, eligible.keys()
        )
        generation_seconds = time.perf_counter() - started
        metrics = _evaluate_and_log(
            "popularity",
            params,
            window,
            predictions,
            eligible,
            pairs,
            history_weeks,
            generation_seconds,
        )
        _print_metrics_row("popularity", params, metrics)
        results.append(
            {"method": "popularity", "params": params, **metrics}
        )

    fallback_ranking = popularity_ranking(
        pairs, pool, history_weeks, POPULARITY_N
    )
    for threshold in thresholds:
        params = f"screen>={threshold}, fallback_N={POPULARITY_N}"
        started = time.perf_counter()
        predictions = recommend_personal_frequency(
            pairs,
            pool,
            history_weeks,
            threshold,
            fallback_ranking,
            eligible.keys(),
        )
        generation_seconds = time.perf_counter() - started
        metrics = _evaluate_and_log(
            "personal_frequency",
            params,
            window,
            predictions,
            eligible,
            pairs,
            history_weeks,
            generation_seconds,
        )
        _print_metrics_row("personal_frequency", params, metrics)
        results.append(
            {"method": "personal_frequency", "params": params, **metrics}
        )

    return results


def main() -> None:
    started = time.perf_counter()
    pairs = load_pairs()
    submission = pd.read_csv(DATA_DIR / "submission.csv")
    submission_users = {int(user_id) for user_id in submission["user_id"]}

    n_values = (2, 4, 6, 10)
    thresholds = (0.8, 0.5)
    primary = _validation_table(
        pairs,
        submission_users,
        (11, 20),
        (21, 30),
        HISTORY_PARTS,
        n_values,
        thresholds,
    )
    print()
    _validation_table(
        pairs,
        submission_users,
        (11, 18),
        (19, 30),
        HISTORY_PARTS,
        n_values,
        thresholds,
    )

    popularity_results = [
        row for row in primary if row["method"] == "popularity"
    ]
    best = max(popularity_results, key=lambda row: float(row["map5"]))
    print()
    print(
        f"best popularity N on 11-20 -> 21-30: {best['params']} "
        f"(MAP@5={float(best['map5']):.6f})"
    )

    print("cold start in submission.csv:")
    for label, history_weeks in [
        ("history 11-20", (11, 20)),
        ("history 11-18", (11, 18)),
        ("history 11-30 (test)", (11, 30)),
    ]:
        no_pairs, no_qualifying = cold_start_counts(
            pairs, history_weeks, submission_users
        )
        print(
            f"  {label}: no labeled pairs={no_pairs:,}, "
            f"no screen_time>=0.8 pairs={no_qualifying:,}"
        )

    print("validation log: midterm-quiz/results/validation.csv")
    print(f"elapsed: {time.perf_counter() - started:.2f}s")


if __name__ == "__main__":
    main()
