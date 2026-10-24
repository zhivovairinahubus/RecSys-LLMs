"""Course recommendation methods for sweet.tv.

Covers parwise association rules, content-based filtering (CBF),
implicit ALS, and Personalized PageRank (PPR) on a user–show bipartite
graph with a user-specific restart distribution (personalized restart from the
user's history). PPR uses iterative power-like propagation; report trains and
lists time per run."""

from __future__ import annotations

import re
import time
from collections.abc import Iterable, Sequence
from datetime import datetime

import numpy as np
import pandas as pd
import scipy.sparse as sp
from implicit.als import AlternatingLeastSquares

try:
    from .data import (
        DATA_DIR,
        HISTORY_END_SEC,
        HISTORY_PARTS,
        SCHEDULE_PATHS,
        load_pairs,
        load_schedule,
    )
    from .evaluate import (
        append_validation_result,
        average_precision_per_user,
        build_truth,
        evaluate_recommendations,
        paired_bootstrap_difference,
        tie_break_sensitivity,
        truth_to_lists,
    )
    from .recommend import (
        K,
        MIN_SCREEN_TIME,
        POPULARITY_N,
        YEAR,
        build_candidate_pool,
        fill_recommendations,
        popularity_ranking,
        recommend_personal_frequency,
        recommend_popularity,
    )
except ImportError:
    from data import (
        DATA_DIR,
        HISTORY_END_SEC,
        HISTORY_PARTS,
        SCHEDULE_PATHS,
        load_pairs,
        load_schedule,
    )
    from evaluate import (
        append_validation_result,
        average_precision_per_user,
        build_truth,
        evaluate_recommendations,
        paired_bootstrap_difference,
        tie_break_sensitivity,
        truth_to_lists,
    )
    from recommend import (
        K,
        MIN_SCREEN_TIME,
        POPULARITY_N,
        YEAR,
        build_candidate_pool,
        fill_recommendations,
        popularity_ranking,
        recommend_personal_frequency,
        recommend_popularity,
    )

import argparse
from pathlib import Path

SEED = 42
ALS_SEEDS = (42, 123, 2026)
ALS_ITERATIONS = 20
ALS_FACTORS = (64, 128)
ALS_REGULARIZATION = (1.0, 3.0, 10.0, 30.0, 100.0)
ALS_CONFIDENCE_ALPHA = (0.25, 0.5, 1.0, 2.0, 5.0)
PPR_RESTART = (0.15, 0.3, 0.5)
PPR_WEIGHT_MODES = ("binary", "frequency")
PPR_MAX_ITER = 30
PPR_TOL = 1e-8
HISTORY_SCOPES = ("pool", "all")
INTERACTION_SCREEN_TIMES = (0.8, 0.5)
ASSOC_SUPPORT = (0.005, 0.01, 0.02, 0.03, 0.05)
ASSOC_CONFIDENCE = (0.1, 0.2)
ASSOC_SCORE_MODES = ("confidence", "confidence_lift")
PPR_DECAY = (1.0, 0.9, 0.8)
PPR_BETA = (0.0, 0.25, 0.5, 1.0)
RRF_K = 60
RRF_WEIGHTS = tuple(
    (assoc_weight, content_weight, als_weight)
    for assoc_weight in (0.0, 0.1, 0.25, 0.5)
    for content_weight in (0.0, 0.1, 0.25, 0.5)
    for als_weight in (0.0, 0.1, 0.25, 0.5)
)
CONTENT_PROFILE_WEIGHTS = ("frequency", "binary")
CONTENT_FEATURE_WEIGHTS = ("binary", "idf")
CONTENT_COLUMNS = (
    "tv_show_id",
    "tv_show_category",
    "tv_show_genre_1",
    "tv_show_genre_2",
    "tv_show_genre_3",
    "year_of_production",
    "director",
    "actors",
)
CONTENT_ATTRIBUTES = CONTENT_COLUMNS[1:]
SLOT_SMOOTHINGS = (0, 1, 5)
SLOT_WEIGHTS = (0.1, 0.2, 0.3, 0.4, 0.5)
SLOT_DEFS = (
    "channel_dow_hour",
    "channel_hour",
    "channel_weekpart_hour",
    "channel_dow_2hour",
)
SWITCH_HISTORY_LIMITS = (1, 2, 3, 5)
SWITCH_COMPONENT_WEIGHTS = (0.25, 0.5, 1.0)
SWITCH_PPR_RESTART = 0.3
SWITCH_PPR_DECAY = 0.9
SWITCH_PPR_BETA = 0.5


def _window_context(
    pairs: pd.DataFrame,
    submission_users: set[int],
    history_weeks: tuple[int, int],
    target_weeks: tuple[int, int],
    *,
    history_scope: str = "pool",
    min_screen_time: float = MIN_SCREEN_TIME,
    history_pairs: pd.DataFrame | None = None,
) -> dict[str, object]:
    if history_pairs is None:
        history_pairs = pairs
    pool = build_candidate_pool(target_weeks, HISTORY_PARTS)
    truth = build_truth(
        pairs, target_weeks[0], target_weeks[1], iso_year=YEAR
    )
    truth_lists = truth_to_lists(truth)
    truth_users_all = {
        user_id: relevant
        for user_id, relevant in truth_lists.items()
        if user_id in submission_users
    }
    eligible = {
        user_id: relevant
        for user_id, relevant in truth_users_all.items()
        if len(relevant) == K
    }
    fallback_ranking = popularity_ranking(
        history_pairs, pool, history_weeks, POPULARITY_N
    )
    interaction_pool = pool if history_scope == "pool" else None
    interactions = _history_interactions(
        history_pairs,
        interaction_pool,
        history_weeks,
        min_screen_time=min_screen_time,
    )
    airing_rate = _airing_rate(
        load_schedule(parts=HISTORY_PARTS),
        history_weeks,
        target_weeks,
        pool,
    )
    return {
        "pool": pool,
        "eligible": eligible,
        "truth_users_all": truth_users_all,
        "fallback_ranking": fallback_ranking,
        "interactions": interactions,
        "airing_rate": airing_rate,
        "history_weeks": history_weeks,
        "target_weeks": target_weeks,
        "history_scope": history_scope,
        "min_screen_time": min_screen_time,
        "history_pairs": history_pairs,
        "window": (
            f"{history_weeks[0]}-{history_weeks[1]} -> "
            f"{target_weeks[0]}-{target_weeks[1]}"
        ),
    }


def _airing_rate(
    schedule: pd.DataFrame,
    history_weeks: tuple[int, int],
    target_weeks: tuple[int, int],
    pool: Sequence[int],
) -> pd.Series:
    """Target airings per week divided by history airings per week, per show."""
    history_span = history_weeks[1] - history_weeks[0] + 1
    target_span = target_weeks[1] - target_weeks[0] + 1
    schedule = schedule.loc[schedule["tv_show_id"].isin(pool)]
    in_history = schedule.loc[
        schedule["air_iso_week"].between(history_weeks[0], history_weeks[1])
    ]
    in_target = schedule.loc[
        schedule["air_iso_week"].between(target_weeks[0], target_weeks[1])
    ]
    history_counts = (
        in_history.groupby("tv_show_id", observed=True)["schedule_row_id"]
        .nunique()
        .div(history_span)
    )
    target_counts = (
        in_target.groupby("tv_show_id", observed=True)["schedule_row_id"]
        .nunique()
        .div(target_span)
    )
    rate = target_counts.reindex(list(pool), fill_value=0.0)
    history_rate = history_counts.reindex(list(pool), fill_value=0.0)
    ratio = rate.div(history_rate.where(history_rate > 0), fill_value=1.0)
    ratio = ratio.replace([np.inf, -np.inf], 1.0).fillna(1.0)
    return ratio.astype(np.float64)


def _history_interactions(
    pairs: pd.DataFrame,
    pool: Sequence[int] | None,
    history_weeks: tuple[int, int],
    *,
    min_screen_time: float = MIN_SCREEN_TIME,
) -> pd.DataFrame:
    mask = pairs["air_iso_week"].between(
        history_weeks[0], history_weeks[1]
    ) & pairs["screen_time"].ge(min_screen_time)
    if pool is not None:
        mask &= pairs["tv_show_id"].isin(pool)
    history = pairs.loc[mask]
    if history.empty:
        return pd.DataFrame(
            columns=[
                "user_id",
                "tv_show_id",
                "n_airings",
                "total_screen_time",
                "last_air_iso_week",
            ]
        )

    return (
        history.groupby(["user_id", "tv_show_id"], sort=False, observed=True)
        .agg(
            n_airings=("schedule_row_id", "size"),
            total_screen_time=("screen_time", "sum"),
            last_air_iso_week=("air_iso_week", "max"),
        )
        .reset_index()
    )


def _build_als_matrix(
    interactions: pd.DataFrame,
) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray]:
    user_codes, user_uniques = pd.factorize(interactions["user_id"], sort=True)
    item_codes, item_uniques = pd.factorize(
        interactions["tv_show_id"], sort=True
    )
    matrix = sp.csr_matrix(
        (
            interactions["n_airings"].to_numpy(dtype=np.float32),
            (user_codes, item_codes),
        ),
        shape=(len(user_uniques), len(item_uniques)),
        dtype=np.float32,
    )
    return matrix, np.asarray(user_uniques), np.asarray(item_uniques)


def als_predictions(
    interactions: pd.DataFrame,
    pool: Sequence[int],
    fallback_ranking: pd.DataFrame,
    user_ids: Iterable[int],
    *,
    factors: int,
    regularization: float,
    confidence_alpha: float,
    seed: int = SEED,
    return_stats: bool = False,
) -> (
    tuple[dict[int, list[int]], float, float]
    | tuple[dict[int, list[int]], float, float, dict[str, int]]
):
    """Fit ALS with confidence = 1 + confidence_alpha * n_airings."""
    user_ids = [int(user_id) for user_id in user_ids]
    fallback = [int(show_id) for show_id in fallback_ranking["tv_show_id"]]
    if interactions.empty:
        predictions = {
            user_id: fill_recommendations([], fallback) for user_id in user_ids
        }
        if return_stats:
            return predictions, 0.0, 0.0, {"fallback_users": len(user_ids)}
        return predictions, 0.0, 0.0

    matrix, user_uniques, item_uniques = _build_als_matrix(interactions)
    matrix.data = 1.0 + confidence_alpha * matrix.data

    model = AlternatingLeastSquares(
        factors=factors,
        regularization=regularization,
        alpha=1.0,
        iterations=ALS_ITERATIONS,
        random_state=seed,
        use_gpu=False,
        calculate_training_loss=False,
    )
    started = time.perf_counter()
    model.fit(matrix, show_progress=False)
    train_seconds = time.perf_counter() - started

    started = time.perf_counter()
    user_to_row = {int(user_id): row for row, user_id in enumerate(user_uniques)}
    known_users = [user_id for user_id in user_ids if user_id in user_to_row]
    predictions: dict[int, list[int]] = {}
    fallback_users = 0

    pool_mask = np.isin(
        item_uniques, np.asarray(list(pool), dtype=item_uniques.dtype)
    )
    pool_items = item_uniques[pool_mask]
    if known_users and len(pool_items):
        rows = np.fromiter(
            (user_to_row[user_id] for user_id in known_users),
            dtype=np.int64,
            count=len(known_users),
        )
        scores = model.user_factors[rows] @ model.item_factors[pool_mask].T
        order = np.argsort(-scores, axis=1, kind="stable")
        for user_id, row_order, row_scores in zip(known_users, order, scores):
            ranked_ids = pool_items[row_order]
            ranked_scores = row_scores[row_order]
            primary = ranked_ids[ranked_scores > 0][:K].tolist()
            if len(primary) < K:
                fallback_users += 1
            predictions[user_id] = fill_recommendations(primary, fallback)

    for user_id in user_ids:
        if user_id not in user_to_row:
            fallback_users += 1
        predictions.setdefault(user_id, fill_recommendations([], fallback))

    list_seconds = time.perf_counter() - started
    if return_stats:
        return (
            predictions,
            train_seconds,
            list_seconds,
            {"fallback_users": fallback_users},
        )
    return predictions, train_seconds, list_seconds


def _build_ppr_graph(
    interactions: pd.DataFrame,
    weight_mode: str,
    *,
    decay: float = 1.0,
    reference_week: int | None = None,
) -> dict[str, object]:
    user_codes, user_uniques = pd.factorize(interactions["user_id"], sort=True)
    show_codes, show_uniques = pd.factorize(
        interactions["tv_show_id"], sort=True
    )
    n_users = len(user_uniques)
    n_shows = len(show_uniques)
    n_nodes = n_users + n_shows

    if weight_mode == "binary":
        weights = np.ones(len(interactions), dtype=np.float64)
    elif weight_mode == "frequency":
        weights = interactions["n_airings"].to_numpy(dtype=np.float64)
    else:
        raise ValueError(f"unknown weight mode: {weight_mode}")

    if decay != 1.0 and reference_week is not None:
        age = reference_week - interactions["last_air_iso_week"].to_numpy(
            dtype=np.float64
        )
        decay_factor = np.power(decay, age)
        weights = weights * decay_factor

    user_totals = np.bincount(user_codes, weights=weights, minlength=n_users)
    show_totals = np.bincount(show_codes, weights=weights, minlength=n_shows)

    rows = np.concatenate([user_codes, n_users + show_codes])
    cols = np.concatenate([n_users + show_codes, user_codes])
    data = np.concatenate(
        [
            weights / user_totals[user_codes],
            weights / show_totals[show_codes],
        ]
    )
    transition = sp.csr_matrix(
        (data, (rows, cols)), shape=(n_nodes, n_nodes), dtype=np.float64
    )

    restarts: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    work = pd.DataFrame(
        {
            "user_id": interactions["user_id"].to_numpy(),
            "show_code": show_codes,
            "weight": weights,
        }
    )
    for user_id, group in work.groupby("user_id", sort=False):
        restart_weights = group["weight"].to_numpy(dtype=np.float64)
        restart_weights = restart_weights / restart_weights.sum()
        restarts[int(user_id)] = (
            n_users + group["show_code"].to_numpy(dtype=np.int64),
            restart_weights,
        )

    return {
        "transition_t": transition.T.tocsr(),
        "n_nodes": n_nodes,
        "n_users": n_users,
        "show_ids": np.asarray(show_uniques),
        "restarts": restarts,
    }


def ppr_predictions(
    graph: dict[str, object],
    pool: Sequence[int],
    fallback_ranking: pd.DataFrame,
    user_ids: Iterable[int],
    *,
    restart_prob: float,
    airing_rate: pd.Series | None = None,
    beta: float = 0.0,
    max_iter: int = PPR_MAX_ITER,
    tol: float = PPR_TOL,
    return_stats: bool = False,
) -> dict[int, list[int]] | tuple[dict[int, list[int]], dict[str, float]]:
    fallback = [int(show_id) for show_id in fallback_ranking["tv_show_id"]]
    transition_t = graph["transition_t"]
    n_nodes = int(graph["n_nodes"])
    n_users = int(graph["n_users"])
    show_ids = graph["show_ids"]
    restarts = graph["restarts"]
    pool_mask = np.isin(show_ids, np.asarray(list(pool), dtype=show_ids.dtype))
    pool_show_ids = show_ids[pool_mask]
    if beta != 0.0 and airing_rate is not None:
        rate = airing_rate.reindex(pool_show_ids).to_numpy(dtype=np.float64)
        multiplier = np.power(np.clip(rate, 1e-6, None), beta)
    else:
        multiplier = np.ones(len(pool_show_ids), dtype=np.float64)

    predictions: dict[int, list[int]] = {}
    iteration_counts: list[int] = []
    residuals: list[float] = []
    fallback_users = 0
    for user_id in user_ids:
        user_id = int(user_id)
        restart_data = restarts.get(user_id)
        if restart_data is None:
            fallback_users += 1
            predictions[user_id] = fill_recommendations([], fallback)
            continue

        restart_nodes, restart_weights = restart_data
        restart = np.zeros(n_nodes, dtype=np.float64)
        restart[restart_nodes] = restart_weights
        scores = restart.copy()
        residual = np.inf
        iterations = 0
        for iterations in range(1, max_iter + 1):
            updated = restart_prob * restart + (1.0 - restart_prob) * (
                transition_t @ scores
            )
            residual = float(np.abs(updated - scores).sum())
            scores = updated
            if residual < tol:
                break
        iteration_counts.append(iterations)
        residuals.append(residual)

        show_scores = scores[n_users:][pool_mask] * multiplier
        order = np.argsort(-show_scores, kind="stable")
        primary = pool_show_ids[order][show_scores[order] > 0][:K].tolist()
        if len(primary) < K:
            fallback_users += 1
        predictions[user_id] = fill_recommendations(primary, fallback)

    if not return_stats:
        return predictions
    stats = {
        "mean_iterations": float(np.mean(iteration_counts)) if iteration_counts else 0.0,
        "max_iterations": float(max(iteration_counts)) if iteration_counts else 0.0,
        "max_residual": float(max(residuals)) if residuals else 0.0,
        "fallback_users": fallback_users,
    }
    return predictions, stats



def _binary_history_matrix(
    interactions: pd.DataFrame,
) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray]:
    user_codes, user_uniques = pd.factorize(interactions["user_id"], sort=True)
    item_codes, item_uniques = pd.factorize(
        interactions["tv_show_id"], sort=True
    )
    matrix = sp.csr_matrix(
        (
            np.ones(len(interactions), dtype=np.float64),
            (user_codes, item_codes),
        ),
        shape=(len(user_uniques), len(item_uniques)),
        dtype=np.float64,
    )
    return matrix, np.asarray(user_uniques), np.asarray(item_uniques)


def _association_rule_matrix(
    interactions: pd.DataFrame,
    pool: Sequence[int],
    min_support: float,
    min_confidence: float,
    score_mode: str,
) -> tuple[sp.csr_matrix | None, sp.csr_matrix | None, np.ndarray | None, np.ndarray]:
    pool_ids = np.asarray(sorted(int(show_id) for show_id in pool), dtype=np.int64)
    if interactions.empty:
        return None, None, None, pool_ids

    matrix, user_uniques, item_ids = _binary_history_matrix(interactions)
    n_users = matrix.shape[0]
    item_counts = np.asarray(matrix.sum(axis=0)).ravel()
    pool_item_codes = np.flatnonzero(np.isin(item_ids, pool_ids))
    if len(pool_item_codes) == 0:
        return (
            sp.csr_matrix((len(item_ids), len(pool_ids))),
            matrix,
            user_uniques,
            pool_ids,
        )

    pool_counts = item_counts[pool_item_codes]
    pool_consequent_cols = np.searchsorted(
        pool_ids, item_ids[pool_item_codes]
    )
    cooccurrence = (matrix.T @ matrix).tocsr()
    cooccurrence_pool = cooccurrence[:, pool_item_codes].tocsr()
    min_count = min_support * n_users
    frequent_items = np.flatnonzero(item_counts >= min_count)

    antecedents: list[np.ndarray] = []
    consequents: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    for antecedent in frequent_items:
        row = cooccurrence_pool.getrow(antecedent)
        if row.nnz == 0:
            continue
        local_consequents = row.indices
        counts = row.data
        support_a = item_counts[antecedent]
        confidence = counts / support_a
        lift = confidence * n_users / pool_counts[local_consequents]
        keep = (
            (pool_item_codes[local_consequents] != antecedent)
            & (counts >= min_count)
            & (confidence >= min_confidence)
            & (lift > 1.0)
        )
        if not keep.any():
            continue
        local_consequents = local_consequents[keep]
        confidence = confidence[keep]
        lift = lift[keep]
        antecedents.append(
            np.full(len(local_consequents), antecedent, dtype=np.int64)
        )
        consequents.append(pool_consequent_cols[local_consequents])
        if score_mode == "confidence":
            weights.append(confidence)
        elif score_mode == "confidence_lift":
            weights.append(confidence * lift)
        else:
            raise ValueError(f"unknown score mode: {score_mode}")

    if antecedents:
        rule_matrix = sp.csr_matrix(
            (
                np.concatenate(weights),
                (
                    np.concatenate(antecedents),
                    np.concatenate(consequents),
                ),
            ),
            shape=(len(item_ids), len(pool_ids)),
            dtype=np.float64,
        )
    else:
        rule_matrix = sp.csr_matrix(
            (len(item_ids), len(pool_ids)), dtype=np.float64
        )
    return rule_matrix, matrix, user_uniques, pool_ids


def assoc_predictions(
    interactions: pd.DataFrame,
    pool: Sequence[int],
    fallback_ranking: pd.DataFrame,
    user_ids: Iterable[int],
    *,
    min_support: float,
    min_confidence: float,
    score_mode: str,
) -> tuple[dict[int, list[int]], float, float]:
    user_ids = [int(user_id) for user_id in user_ids]
    fallback = [int(show_id) for show_id in fallback_ranking["tv_show_id"]]
    if interactions.empty:
        return (
            {
                user_id: fill_recommendations([], fallback)
                for user_id in user_ids
            },
            0.0,
            0.0,
        )

    started = time.perf_counter()
    rule_matrix, matrix, user_uniques, pool_ids = _association_rule_matrix(
        interactions,
        pool,
        min_support,
        min_confidence,
        score_mode,
    )
    train_seconds = time.perf_counter() - started

    started = time.perf_counter()
    predictions: dict[int, list[int]] = {}
    if (
        rule_matrix is not None
        and matrix is not None
        and user_uniques is not None
        and rule_matrix.shape[1] > 0
    ):
        user_to_row = {
            int(user_id): row for row, user_id in enumerate(user_uniques)
        }
        known_users = [
            user_id for user_id in user_ids if user_id in user_to_row
        ]
        if known_users:
            rows = np.fromiter(
                (user_to_row[user_id] for user_id in known_users),
                dtype=np.int64,
                count=len(known_users),
            )
            scores = (matrix[rows] @ rule_matrix).toarray()
            order = np.argsort(-scores, axis=1, kind="stable")
            for user_id, row_order, row_scores in zip(
                known_users, order, scores
            ):
                ranked_ids = pool_ids[row_order]
                ranked_scores = row_scores[row_order]
                primary = ranked_ids[ranked_scores > 0][:K].tolist()
                predictions[user_id] = fill_recommendations(
                    primary, fallback
                )

    for user_id in user_ids:
        predictions.setdefault(user_id, fill_recommendations([], fallback))

    list_seconds = time.perf_counter() - started
    return predictions, train_seconds, list_seconds


def _load_show_content(
    schedule_parts: Iterable[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, tuple[int, int]]]:
    """Load content attributes from the schedule that defines the pool."""
    if schedule_parts is None:
        schedule_parts = HISTORY_PARTS
    frames = []
    for part in schedule_parts:
        frame = pd.read_csv(
            SCHEDULE_PATHS[part],
            usecols=list(CONTENT_COLUMNS),
            dtype={
                "tv_show_id": "int64",
                "tv_show_category": "string",
                "tv_show_genre_1": "string",
                "tv_show_genre_2": "string",
                "tv_show_genre_3": "string",
                "year_of_production": "string",
                "director": "string",
                "actors": "string",
            },
            low_memory=False,
        )
        frames.append(frame)
    schedule = pd.concat(frames, ignore_index=True)
    schedule = schedule.loc[schedule["tv_show_id"].ne(0)]
    for column in CONTENT_ATTRIBUTES:
        schedule[column] = (
            schedule[column]
            .astype("string")
            .str.strip()
            .replace({"": pd.NA, "nan": pd.NA})
        )

    consistency: dict[str, tuple[int, int]] = {}
    for column in CONTENT_ATTRIBUTES:
        unique = (
            schedule.dropna(subset=[column])
            .groupby("tv_show_id")[column]
            .nunique()
        )
        consistency[column] = (len(unique), int((unique > 1).sum()))

    show_ids = np.sort(schedule["tv_show_id"].unique())
    content = pd.DataFrame({"tv_show_id": show_ids})
    for column in CONTENT_ATTRIBUTES:
        values = schedule[["tv_show_id", column]].dropna()
        counts = (
            values.value_counts(["tv_show_id", column], sort=False)
            .reset_index(name="n")
            .sort_values(
                ["tv_show_id", "n", column],
                ascending=[True, False, True],
                kind="mergesort",
            )
        )
        canonical = counts.drop_duplicates("tv_show_id")[
            ["tv_show_id", column]
        ]
        content = content.merge(canonical, on="tv_show_id", how="left")

    years = content["year_of_production"].str.extract(r"(\d{4})", expand=False)
    years = pd.to_numeric(years, errors="coerce")
    content["year_bin"] = (
        pd.cut(
            years,
            bins=[-np.inf, 1989, 1999, 2009, 2019, np.inf],
            labels=["<=1989", "1990s", "2000s", "2010s", "2020s"],
        )
        .astype("string")
        .fillna("unknown")
    )
    return content, consistency


def _content_feature_matrix(
    content: pd.DataFrame, feature_weight: str
) -> tuple[sp.csr_matrix, np.ndarray]:
    row_tokens: list[list[str]] = []
    for row in content.itertuples(index=False):
        tokens: list[str] = []
        if not pd.isna(row.tv_show_category):
            tokens.append(f"cat:{row.tv_show_category}")
        for attribute in ("tv_show_genre_1", "tv_show_genre_2", "tv_show_genre_3"):
            value = getattr(row, attribute)
            if not pd.isna(value):
                tokens.append(f"genre:{value}")
        if not pd.isna(row.year_bin):
            tokens.append(f"year:{row.year_bin}")
        for attribute, prefix in (("director", "dir"), ("actors", "actor")):
            value = getattr(row, attribute)
            if pd.isna(value):
                continue
            for name in re.split(r"[,;]", str(value)):
                name = name.strip()
                if name:
                    tokens.append(f"{prefix}:{name}")
        row_tokens.append(list(dict.fromkeys(tokens)))

    flat_tokens = [token for tokens in row_tokens for token in tokens]
    show_ids = content["tv_show_id"].to_numpy()
    if not flat_tokens:
        return sp.csr_matrix((len(content), 0)), show_ids

    codes, uniques = pd.factorize(
        np.asarray(flat_tokens, dtype=object), sort=True
    )
    row_codes = np.repeat(
        np.arange(len(row_tokens)),
        [len(tokens) for tokens in row_tokens],
    )
    matrix = sp.csr_matrix(
        (
            np.ones(len(flat_tokens), dtype=np.float64),
            (row_codes, codes),
        ),
        shape=(len(content), len(uniques)),
        dtype=np.float64,
    )
    matrix.sum_duplicates()
    matrix.data[:] = 1.0

    if feature_weight == "idf":
        document_frequency = np.asarray((matrix > 0).sum(axis=0)).ravel()
        idf = (
            np.log((1.0 + len(content)) / (1.0 + document_frequency)) + 1.0
        )
        matrix = matrix.multiply(idf[None, :]).tocsr()
    elif feature_weight != "binary":
        raise ValueError(f"unknown feature weight: {feature_weight}")

    norms = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel())
    inverse = np.zeros_like(norms)
    nonzero = norms > 0
    inverse[nonzero] = 1.0 / norms[nonzero]
    matrix = (sp.diags(inverse) @ matrix).tocsr()
    return matrix, show_ids


def content_predictions(
    interactions: pd.DataFrame,
    pool: Sequence[int],
    fallback_ranking: pd.DataFrame,
    user_ids: Iterable[int],
    content: pd.DataFrame,
    *,
    profile_weight: str,
    feature_weight: str,
) -> tuple[dict[int, list[int]], float, float]:
    user_ids = [int(user_id) for user_id in user_ids]
    fallback = [int(show_id) for show_id in fallback_ranking["tv_show_id"]]

    started = time.perf_counter()
    feature_matrix, content_show_ids = _content_feature_matrix(
        content, feature_weight
    )
    show_index = pd.Series(
        np.arange(len(content_show_ids)), index=pd.Index(content_show_ids)
    )
    user_codes, user_uniques = pd.factorize(
        interactions["user_id"], sort=True
    )
    show_codes = show_index.reindex(interactions["tv_show_id"]).to_numpy()
    valid = ~pd.isna(show_codes)

    if profile_weight == "frequency":
        values = interactions["n_airings"].to_numpy(dtype=np.float64)
    elif profile_weight == "binary":
        values = np.ones(len(interactions), dtype=np.float64)
    else:
        raise ValueError(f"unknown profile weight: {profile_weight}")

    user_item = sp.csr_matrix(
        (
            values[valid],
            (
                user_codes[valid],
                show_codes[valid].astype(np.int64),
            ),
        ),
        shape=(len(user_uniques), len(content_show_ids)),
        dtype=np.float64,
    )
    profiles = (user_item @ feature_matrix).tocsr()
    norms = np.sqrt(np.asarray(profiles.multiply(profiles).sum(axis=1)).ravel())
    inverse = np.zeros_like(norms)
    nonzero = norms > 0
    inverse[nonzero] = 1.0 / norms[nonzero]
    profiles = (sp.diags(inverse) @ profiles).tocsr()
    train_seconds = time.perf_counter() - started

    started = time.perf_counter()
    predictions: dict[int, list[int]] = {}
    pool_ids = np.asarray(pool, dtype=np.int64)
    pool_codes = show_index.reindex(pool_ids).to_numpy()
    valid_pool = ~pd.isna(pool_codes)
    pool_valid_ids = pool_ids[valid_pool]
    candidates = feature_matrix[
        pool_codes[valid_pool].astype(np.int64)
    ]

    if (
        len(pool_valid_ids) > 0
        and feature_matrix.shape[1] > 0
        and len(user_uniques) > 0
    ):
        user_to_row = {
            int(user_id): row for row, user_id in enumerate(user_uniques)
        }
        known_users = [
            user_id for user_id in user_ids if user_id in user_to_row
        ]
        if known_users:
            rows = np.fromiter(
                (user_to_row[user_id] for user_id in known_users),
                dtype=np.int64,
                count=len(known_users),
            )
            scores = (profiles[rows] @ candidates.T).toarray()
            order = np.argsort(-scores, axis=1, kind="stable")
            for user_id, row_order, row_scores in zip(
                known_users, order, scores
            ):
                ranked_ids = pool_valid_ids[row_order]
                ranked_scores = row_scores[row_order]
                primary = ranked_ids[ranked_scores > 0][:K].tolist()
                predictions[user_id] = fill_recommendations(
                    primary, fallback
                )

    for user_id in user_ids:
        predictions.setdefault(user_id, fill_recommendations([], fallback))

    list_seconds = time.perf_counter() - started
    return predictions, train_seconds, list_seconds

def _log_run(
    method: str,
    params: str,
    context: dict[str, object],
    predictions: dict[int, list[int]],
    pairs: pd.DataFrame,
    history_weeks: tuple[int, int],
    generation_seconds: float,
) -> dict[str, float | int]:
    metrics = evaluate_recommendations(
        predictions,
        context["eligible"],
        pairs,
        history_weeks,
    )
    metrics["seconds"] = generation_seconds + float(metrics["metrics_seconds"])
    append_validation_result(
        {
            "method": method,
            "params": params,
            "window": context["window"],
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


def _print_row(
    method: str,
    params: str,
    metrics: dict[str, float | int],
    train_seconds: float,
    list_seconds: float,
) -> None:
    print(
        f"  {method:<6} {params:<64} "
        f"{float(metrics['map5']):.6f}  "
        f"{int(metrics['coverage_users']):>5}/{int(metrics['n_users']):<5} "
        f"{float(metrics['repeat_share']):>7.4f}  "
        f"{int(metrics['distinct_shows']):>6}  "
        f"{train_seconds:>7.3f}  {list_seconds:>7.3f}  "
        f"{float(metrics['seconds']):>7.3f}"
    )


def _print_header(context: dict[str, object]) -> None:
    print(f"window: {context['window']}")
    print(
        f"  history scope: {context['history_scope']}, "
        f"interaction screen_time >= {float(context['min_screen_time']):g}"
    )
    print(f"  candidate pool: {len(context['pool']):,} shows")
    print(
        "  eligible submission users with exactly "
        f"{K} truth shows: {len(context['eligible']):,}"
    )
    print(f"  interactions: {len(context['interactions']):,}")
    print(
        f"  {'method':<6} {'params':<64} {'MAP@5':>8}  "
        f"{'coverage':>12} {'repeat':>7}  {'shows':>6}  "
        f"{'train':>7}  {'list':>7}  {'total':>7}"
    )


def _als_grid(
    context: dict[str, object],
    pairs: pd.DataFrame,
    history_weeks: tuple[int, int],
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for factors in ALS_FACTORS:
        for regularization in ALS_REGULARIZATION:
            for confidence_alpha in ALS_CONFIDENCE_ALPHA:
                seed_maps: dict[int, dict[int, list[int]]] = {}
                train_total = 0.0
                list_total = 0.0
                maps: list[float] = []
                for seed in ALS_SEEDS:
                    predictions, train_seconds, list_seconds = als_predictions(
                        context["interactions"],
                        context["pool"],
                        context["fallback_ranking"],
                        context["eligible"].keys(),
                        factors=factors,
                        regularization=regularization,
                        confidence_alpha=confidence_alpha,
                        seed=seed,
                    )
                    seed_maps[seed] = predictions
                    train_total += train_seconds
                    list_total += list_seconds
                    metrics = evaluate_recommendations(
                        predictions,
                        context["eligible"],
                        pairs,
                        history_weeks,
                    )
                    maps.append(float(metrics["map5"]))

                metrics = evaluate_recommendations(
                    seed_maps[ALS_SEEDS[0]],
                    context["eligible"],
                    pairs,
                    history_weeks,
                )
                metrics["map5"] = float(np.mean(maps))
                metrics["seconds"] = (
                    train_total + list_total + float(metrics["metrics_seconds"])
                )
                params = (
                    f"hist={context['history_scope']}, "
                    f"thr={float(context['min_screen_time']):g}, "
                    f"f={factors}, reg={regularization:g}, "
                    f"conf_alpha={confidence_alpha:g}, "
                    f"seeds={len(ALS_SEEDS)}"
                )
                append_validation_result(
                    {
                        "method": "als",
                        "params": params,
                        "window": context["window"],
                        "n_users": metrics["n_users"],
                        "map5": metrics["map5"],
                        "coverage_users": metrics["coverage_users"],
                        "coverage_share": metrics["coverage_share"],
                        "repeat_share": metrics["repeat_share"],
                        "distinct_shows": metrics["distinct_shows"],
                        "seconds": metrics["seconds"],
                    }
                )
                print(
                    f"  als    {params:<64} "
                    f"{float(np.mean(maps)):.6f}±{float(np.std(maps)):.6f}  "
                    f"{int(metrics['coverage_users']):>5}/{int(metrics['n_users']):<5} "
                    f"{float(metrics['repeat_share']):>7.4f}  "
                    f"{int(metrics['distinct_shows']):>6}  "
                    f"{train_total:>7.3f}  {list_total:>7.3f}  "
                    f"{metrics['seconds']-float(metrics['metrics_seconds']):>7.3f}"
                )
                results.append(
                    {
                        "method": "als",
                        "params": params,
                        **metrics,
                        "map5_mean": float(np.mean(maps)),
                        "map5_std": float(np.std(maps)),
                        "train_seconds": train_total,
                        "list_seconds": list_total,
                        "history_scope": context["history_scope"],
                        "min_screen_time": context["min_screen_time"],
                        "kwargs": {
                            "factors": factors,
                            "regularization": regularization,
                            "confidence_alpha": confidence_alpha,
                            "seed": ALS_SEEDS[0],
                        },
                        "predictions": seed_maps[ALS_SEEDS[0]],
                    }
                )
    return results


def _assoc_grid(
    context: dict[str, object], pairs: pd.DataFrame, history_weeks: tuple[int, int]
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for min_support in ASSOC_SUPPORT:
        for min_confidence in ASSOC_CONFIDENCE:
            for score_mode in ASSOC_SCORE_MODES:
                params = (
                    f"thr={float(context['min_screen_time']):g}, "
                    f"sup={min_support:g}, conf={min_confidence:g}, "
                    f"score={score_mode}"
                )
                predictions, train_seconds, list_seconds = assoc_predictions(
                    context["interactions"],
                    context["pool"],
                    context["fallback_ranking"],
                    context["eligible"].keys(),
                    min_support=min_support,
                    min_confidence=min_confidence,
                    score_mode=score_mode,
                )
                metrics = _log_run(
                    "assoc",
                    params,
                    context,
                    predictions,
                    pairs,
                    history_weeks,
                    train_seconds + list_seconds,
                )
                _print_row(
                    "assoc", params, metrics, train_seconds, list_seconds
                )
                results.append(
                    {
                        "method": "assoc",
                        "params": params,
                        **metrics,
                        "train_seconds": train_seconds,
                        "list_seconds": list_seconds,
                        "history_scope": context["history_scope"],
                        "min_screen_time": context["min_screen_time"],
                        "kwargs": {
                            "min_support": min_support,
                            "min_confidence": min_confidence,
                            "score_mode": score_mode,
                        },
                    }
                )
    return results


def _content_grid(
    context: dict[str, object],
    pairs: pd.DataFrame,
    history_weeks: tuple[int, int],
    content: pd.DataFrame,
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for profile_weight in CONTENT_PROFILE_WEIGHTS:
        for feature_weight in CONTENT_FEATURE_WEIGHTS:
            params = (
                f"thr={float(context['min_screen_time']):g}, "
                f"profile={profile_weight}, feat={feature_weight}"
            )
            predictions, train_seconds, list_seconds = content_predictions(
                context["interactions"],
                context["pool"],
                context["fallback_ranking"],
                context["eligible"].keys(),
                content,
                profile_weight=profile_weight,
                feature_weight=feature_weight,
            )
            metrics = _log_run(
                "content",
                params,
                context,
                predictions,
                pairs,
                history_weeks,
                train_seconds + list_seconds,
            )
            _print_row(
                "content", params, metrics, train_seconds, list_seconds
            )
            results.append(
                {
                    "method": "content",
                    "params": params,
                    **metrics,
                    "train_seconds": train_seconds,
                    "list_seconds": list_seconds,
                    "history_scope": context["history_scope"],
                    "min_screen_time": context["min_screen_time"],
                    "kwargs": {
                        "profile_weight": profile_weight,
                        "feature_weight": feature_weight,
                    },
                }
            )
    return results


def _baseline_grid(
    context: dict[str, object], pairs: pd.DataFrame, history_weeks: tuple[int, int]
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    history_pairs = context["history_pairs"]
    users = context["eligible"].keys()
    for n_weeks in (2, 4, 6, 10):
        params = f"N={n_weeks}"
        started = time.perf_counter()
        predictions = recommend_popularity(
            history_pairs,
            context["pool"],
            history_weeks,
            n_weeks,
            users,
        )
        generation_seconds = time.perf_counter() - started
        metrics = _log_run(
            "popularity",
            params,
            context,
            predictions,
            pairs,
            history_weeks,
            generation_seconds,
        )
        _print_row("pop", params, metrics, 0.0, generation_seconds)
        results.append(
            {
                "method": "popularity",
                "params": params,
                **metrics,
                "history_scope": "all",
                "min_screen_time": MIN_SCREEN_TIME,
                "kwargs": {"n_weeks": n_weeks},
            }
        )

    for threshold in (0.8, 0.5):
        params = f"screen>={threshold:g}, fallback_N={POPULARITY_N}"
        started = time.perf_counter()
        predictions = recommend_personal_frequency(
            history_pairs,
            context["pool"],
            history_weeks,
            threshold,
            context["fallback_ranking"],
            users,
        )
        generation_seconds = time.perf_counter() - started
        metrics = _log_run(
            "personal_frequency",
            params,
            context,
            predictions,
            pairs,
            history_weeks,
            generation_seconds,
        )
        _print_row("pers", params, metrics, 0.0, generation_seconds)
        results.append(
            {
                "method": "personal_frequency",
                "params": params,
                **metrics,
                "history_scope": "all",
                "min_screen_time": threshold,
                "kwargs": {
                    "threshold": threshold,
                    "fallback_ranking": context["fallback_ranking"],
                },
            }
        )
    return results


def _ppr_decay_grid(
    context: dict[str, object], pairs: pd.DataFrame, history_weeks: tuple[int, int]
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    reference_week = int(context["history_weeks"][1])
    for weight_mode in PPR_WEIGHT_MODES:
        for decay in PPR_DECAY:
            graph_started = time.perf_counter()
            graph = _build_ppr_graph(
                context["interactions"],
                weight_mode,
                decay=decay,
                reference_week=reference_week,
            )
            graph_seconds = time.perf_counter() - graph_started

            for restart_prob in PPR_RESTART:
                params = (
                    f"hist={context['history_scope']}, "
                    f"thr={float(context['min_screen_time']):g}, "
                    f"restart={restart_prob:g}, weights={weight_mode}, "
                    f"decay={decay:g}"
                )
                started = time.perf_counter()
                predictions, ppr_stats = ppr_predictions(
                    graph,
                    context["pool"],
                    context["fallback_ranking"],
                    context["eligible"].keys(),
                    restart_prob=restart_prob,
                    return_stats=True,
                )
                list_seconds = time.perf_counter() - started
                metrics = _log_run(
                    "ppr_decay",
                    params,
                    context,
                    predictions,
                    pairs,
                    history_weeks,
                    graph_seconds + list_seconds,
                )
                print(
                    f"  ppr    {params:<64} "
                    f"{float(metrics['map5']):.6f}  "
                    f"{int(metrics['coverage_users']):>5}/{int(metrics['n_users']):<5} "
                    f"{float(metrics['repeat_share']):>7.4f}  "
                    f"{int(metrics['distinct_shows']):>6}  "
                    f"{graph_seconds:>7.3f}  {list_seconds:>7.3f}  "
                    f"{float(metrics['seconds']):>7.3f}  "
                    f"iter={ppr_stats['mean_iterations']:.1f}/"
                    f"{ppr_stats['max_iterations']:.0f} "
                    f"res={ppr_stats['max_residual']:.1e}"
                )
                results.append(
                    {
                        "method": "ppr_decay",
                        "params": params,
                        **metrics,
                        "train_seconds": graph_seconds,
                        "list_seconds": list_seconds,
                        "history_scope": context["history_scope"],
                        "min_screen_time": context["min_screen_time"],
                        "kwargs": {
                            "restart_prob": restart_prob,
                            "weight_mode": weight_mode,
                            "decay": decay,
                        },
                        "ppr_stats": ppr_stats,
                    }
                )
    return results


def _ppr_beta_grid(
    context: dict[str, object], pairs: pd.DataFrame, history_weeks: tuple[int, int]
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    reference_week = int(context["history_weeks"][1])
    graph_started = time.perf_counter()
    graph = _build_ppr_graph(
        context["interactions"],
        "frequency",
        decay=1.0,
        reference_week=reference_week,
    )
    graph_build_seconds = time.perf_counter() - graph_started
    for beta in PPR_BETA:
        params = (
            f"hist={context['history_scope']}, "
            f"thr={float(context['min_screen_time']):g}, "
            f"restart=0.3, weights=frequency, beta={beta:g}"
        )
        started = time.perf_counter()
        predictions, ppr_stats = ppr_predictions(
            graph,
            context["pool"],
            context["fallback_ranking"],
            context["eligible"].keys(),
            restart_prob=0.3,
            airing_rate=context["airing_rate"],
            beta=beta,
            return_stats=True,
        )
        list_seconds = time.perf_counter() - started
        metrics = _log_run(
            "ppr_beta",
            params,
            context,
            predictions,
            pairs,
            history_weeks,
            graph_build_seconds + list_seconds,
        )
        print(
            f"  ppr    {params:<64} "
            f"{float(metrics['map5']):.6f}  "
            f"{int(metrics['coverage_users']):>5}/{int(metrics['n_users']):<5} "
            f"{float(metrics['repeat_share']):>7.4f}  "
            f"{int(metrics['distinct_shows']):>6}  "
            f"{graph_build_seconds:>7.3f}  {list_seconds:>7.3f}  "
            f"{float(metrics['seconds']):>7.3f}  "
            f"iter={ppr_stats['mean_iterations']:.1f}/"
            f"{ppr_stats['max_iterations']:.0f} "
            f"res={ppr_stats['max_residual']:.1e}"
        )
        results.append(
            {
                "method": "ppr_beta",
                "params": params,
                **metrics,
                "train_seconds": graph_build_seconds,
                "list_seconds": list_seconds,
                "history_scope": context["history_scope"],
                "min_screen_time": context["min_screen_time"],
                "kwargs": {
                    "restart_prob": 0.3,
                    "weight_mode": "frequency",
                    "beta": beta,
                },
                "ppr_stats": ppr_stats,
            }
        )
    return results


def _ppr_combined_grid(
    context: dict[str, object], pairs: pd.DataFrame, history_weeks: tuple[int, int]
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    reference_week = int(context["history_weeks"][1])
    for decay in PPR_DECAY:
        graph_started = time.perf_counter()
        graph = _build_ppr_graph(
            context["interactions"],
            "frequency",
            decay=decay,
            reference_week=reference_week,
        )
        graph_seconds = time.perf_counter() - graph_started
        for restart_prob in PPR_RESTART:
            for beta in PPR_BETA:
                params = (
                    f"hist={context['history_scope']}, "
                    f"thr={float(context['min_screen_time']):g}, "
                    f"restart={restart_prob:g}, decay={decay:g}, "
                    f"beta={beta:g}, weights=frequency"
                )
                started = time.perf_counter()
                predictions, ppr_stats = ppr_predictions(
                    graph,
                    context["pool"],
                    context["fallback_ranking"],
                    context["eligible"].keys(),
                    restart_prob=restart_prob,
                    airing_rate=context["airing_rate"],
                    beta=beta,
                    return_stats=True,
                )
                list_seconds = time.perf_counter() - started
                metrics = _log_run(
                    "ppr_final",
                    params,
                    context,
                    predictions,
                    pairs,
                    history_weeks,
                    graph_seconds + list_seconds,
                )
                print(
                    f"  pprfin {params:<64} "
                    f"{float(metrics['map5']):.6f}  "
                    f"{int(metrics['coverage_users']):>5}/{int(metrics['n_users']):<5} "
                    f"{float(metrics['repeat_share']):>7.4f}  "
                    f"{int(metrics['distinct_shows']):>6}  "
                    f"{graph_seconds:>7.3f}  {list_seconds:>7.3f}  "
                    f"{float(metrics['seconds']):>7.3f}  "
                    f"iter={ppr_stats['mean_iterations']:.1f}/"
                    f"{ppr_stats['max_iterations']:.0f} "
                    f"res={ppr_stats['max_residual']:.1e}"
                )
                results.append(
                    {
                        "method": "ppr_final",
                        "params": params,
                        **metrics,
                        "train_seconds": graph_seconds,
                        "list_seconds": list_seconds,
                        "history_scope": context["history_scope"],
                        "min_screen_time": context["min_screen_time"],
                        "kwargs": {
                            "restart_prob": restart_prob,
                            "weight_mode": "frequency",
                            "decay": decay,
                            "beta": beta,
                        },
                        "ppr_stats": ppr_stats,
                    }
                )
    return results


def _best_result(results: list[dict[str, object]]) -> dict[str, object]:
    return max(results, key=lambda row: float(row["map5"]))


def component_predictions(
    best: dict[str, object],
    context: dict[str, object],
    *,
    user_ids: Iterable[int] | None = None,
) -> tuple[dict[int, list[int]], float, float]:
    """Regenerate one component's recommendations for a given validation set."""
    method = str(best["method"])
    kwargs = best["kwargs"]
    users = list(context["eligible"] if user_ids is None else user_ids)
    if method == "als":
        return als_predictions(
            context["interactions"],
            context["pool"],
            context["fallback_ranking"],
            users,
            **kwargs,
        )
    if method == "assoc":
        return assoc_predictions(
            context["interactions"],
            context["pool"],
            context["fallback_ranking"],
            users,
            **kwargs,
        )
    if method == "content":
        return content_predictions(
            context["interactions"],
            context["pool"],
            context["fallback_ranking"],
            users,
            best["content"],
            **kwargs,
        )
    if method == "popularity":
        started = time.perf_counter()
        predictions = recommend_popularity(
            context["history_pairs"],
            context["pool"],
            context["history_weeks"],
            kwargs["n_weeks"],
            users,
        )
        elapsed = time.perf_counter() - started
        return predictions, 0.0, elapsed
    if method == "personal_frequency":
        started = time.perf_counter()
        predictions = recommend_personal_frequency(
            context["history_pairs"],
            context["pool"],
            context["history_weeks"],
            kwargs["threshold"],
            context["fallback_ranking"],
            users,
        )
        elapsed = time.perf_counter() - started
        return predictions, 0.0, elapsed

    reference_week = int(context["history_weeks"][1])
    graph_started = time.perf_counter()
    graph = _build_ppr_graph(
        context["interactions"],
        kwargs["weight_mode"],
        decay=kwargs.get("decay", 1.0),
        reference_week=reference_week,
    )
    train_seconds = time.perf_counter() - graph_started
    started = time.perf_counter()
    predictions = ppr_predictions(
        graph,
        context["pool"],
        context["fallback_ranking"],
        users,
        restart_prob=kwargs["restart_prob"],
        airing_rate=context["airing_rate"],
        beta=kwargs.get("beta", 0.0),
    )
    list_seconds = time.perf_counter() - started
    return predictions, train_seconds, list_seconds


def rrf_predictions(
    rankings: Sequence[tuple[float, dict[int, list[int]]]],
    pool: Sequence[int],
    fallback_ranking: pd.DataFrame,
    user_ids: Iterable[int],
    *,
    rrf_k: int = RRF_K,
) -> dict[int, list[int]]:
    """Fuse weighted ranked lists with reciprocal rank fusion."""
    pool_set = set(int(show_id) for show_id in pool)
    fallback = [int(show_id) for show_id in fallback_ranking["tv_show_id"]]
    predictions: dict[int, list[int]] = {}
    for user_id in user_ids:
        user_id = int(user_id)
        scores: dict[int, float] = {}
        for weight, ranking in rankings:
            for position, show_id in enumerate(ranking.get(user_id, []), start=1):
                show_id = int(show_id)
                if show_id not in pool_set:
                    continue
                scores[show_id] = scores.get(show_id, 0.0) + weight / (
                    rrf_k + position
                )
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        primary = [show_id for show_id, _ in ranked][:K]
        predictions[user_id] = fill_recommendations(primary, fallback)
    return predictions


def _rerun_best(
    best: dict[str, object],
    context: dict[str, object],
    pairs: pd.DataFrame,
    history_weeks: tuple[int, int],
) -> None:
    method = str(best["method"])
    predictions, train_seconds, list_seconds = component_predictions(
        best, context
    )
    metrics = _log_run(
        method,
        str(best["params"]),
        context,
        predictions,
        pairs,
        history_weeks,
        train_seconds + list_seconds,
    )
    _print_row(method, str(best["params"]), metrics, train_seconds, list_seconds)
    best["secondary_map5"] = metrics["map5"]
    best["secondary_params"] = best["params"]


def _context_for_row(
    row: dict[str, object] | None,
    primary: dict[str, object],
    primary_pool: dict[str, object],
    secondary: dict[str, object] | None = None,
    secondary_pool: dict[str, object] | None = None,
) -> dict[str, object] | None:
    if row is None:
        return None
    is_pool = row.get("history_scope") == "pool"
    if secondary is not None and secondary_pool is not None:
        return secondary_pool if is_pool else secondary
    return primary_pool if is_pool else primary




def _get_slot_keys(air_start_sec: np.ndarray, channel_id: np.ndarray, slot_def: str) -> np.ndarray:
    """Vectorized slot key generation as strings."""
    dts = pd.to_datetime(air_start_sec, unit="s")
    ch = channel_id.astype(str)
    if slot_def == "channel_dow_hour":
        return ch + "_" + dts.day_of_week.astype(str) + "_" + dts.hour.astype(str)
    if slot_def == "channel_hour":
        return ch + "_" + dts.hour.astype(str)
    if slot_def == "channel_weekpart_hour":
        wp = (dts.day_of_week >= 5).astype(int).astype(str)
        return ch + "_" + wp + "_" + dts.hour.astype(str)
    if slot_def == "channel_dow_2hour":
        return ch + "_" + dts.day_of_week.astype(str) + "_" + (dts.hour // 2).astype(str)
    raise ValueError(f"unknown slot_def: {slot_def}")


def _slot_score_matrix(
    pairs: pd.DataFrame,
    history_weeks: tuple[int, int],
    target_weeks: tuple[int, int],
    target_parts: Iterable[str],
    pool: Sequence[int],
    user_ids: Iterable[int],
    slot_def: str,
    smoothing: float,
) -> tuple[sp.csr_matrix, list[int], list[int]]:
    """Compute User-Show slot scores using sparse matrix multiplication."""
    user_ids = sorted([int(u) for u in user_ids])
    pool = sorted([int(s) for s in pool])
    user2idx = {u: i for i, u in enumerate(user_ids)}
    show2idx = {s: i for i, s in enumerate(pool)}

    # 1. N_sched(s) from history schedule
    # We need schedule for the history window
    # Assuming load_schedule with parts containing history window
    hist_parts = []
    if history_weeks[0] <= 20: hist_parts.append("11-20")
    if history_weeks[1] >= 21: hist_parts.append("21-30")
    # Actually just use parts from history_weeks
    sched_hist_full = load_schedule(parts=tuple(hist_parts))
    sched_hist = sched_hist_full.loc[sched_hist_full["air_iso_week"].between(history_weeks[0], history_weeks[1])]
    
    sched_hist_slots = _get_slot_keys(
        sched_hist["air_start_sec"].to_numpy(),
        sched_hist["channel_id"].to_numpy(),
        slot_def
    )
    n_sched = pd.Series(sched_hist_slots).value_counts().to_dict()

    # 2. N_user(u, s) from pairs
    # Use context-provided history pairs filtered to screen_time >= 0.8
    hist_pairs = pairs.loc[
        pairs["user_id"].isin(user2idx) &
        pairs["air_iso_week"].between(history_weeks[0], history_weeks[1]) &
        (pairs["screen_time"] >= 0.8)
    ].copy()
    hist_pairs["slot"] = _get_slot_keys(
        hist_pairs["air_start_sec"].to_numpy(),
        hist_pairs["channel_id"].to_numpy(),
        slot_def
    )
    user_slot_counts = hist_pairs.groupby(["user_id", "slot"]).size().reset_index(name="n_user")
    
    # 3. Target schedule slot-show matrix S
    sched_target_full = load_schedule(parts=target_parts)
    sched_target = sched_target_full.loc[
        sched_target_full["air_iso_week"].between(target_weeks[0], target_weeks[1]) &
        sched_target_full["tv_show_id"].isin(show2idx)
    ].copy()
    sched_target["slot"] = _get_slot_keys(
        sched_target["air_start_sec"].to_numpy(),
        sched_target["channel_id"].to_numpy(),
        slot_def
    )
    slot_show_counts = sched_target.groupby(["slot", "tv_show_id"]).size().reset_index(name="n_target")

    # Mapping slots to indices
    all_slots = sorted(list(set(n_sched.keys()) | set(slot_show_counts["slot"].unique())))
    slot2idx = {s: i for i, s in enumerate(all_slots)}
    
    # Build User-Slot matrix U
    user_slot_counts["p_val"] = user_slot_counts["n_user"] / (user_slot_counts["slot"].map(n_sched).fillna(0) + smoothing)
    u_rows = user_slot_counts["user_id"].map(user2idx).to_numpy()
    u_cols = user_slot_counts["slot"].map(slot2idx).to_numpy()
    u_data = user_slot_counts["p_val"].to_numpy()
    U = sp.csr_matrix((u_data, (u_rows, u_cols)), shape=(len(user_ids), len(all_slots)))

    # Build Slot-Show matrix S
    s_rows = slot_show_counts["slot"].map(slot2idx).to_numpy()
    s_cols = slot_show_counts["tv_show_id"].map(show2idx).to_numpy()
    s_data = slot_show_counts["n_target"].to_numpy()
    S = sp.csr_matrix((s_data, (s_rows, s_cols)), shape=(len(all_slots), len(pool)))

    # 4. Multiply
    R = U @ S
    return R, user_ids, pool



def _blend_ppr_slots_optimized(
    ppr_matrix: sp.csr_matrix,
    slot_matrix: sp.csr_matrix,
    pool: list[int],
    fallback_ranking: pd.DataFrame,
    user_ids: list[int],
    w: float,
) -> dict[int, list[int]]:
    """Blend PPR and Slot scores using matrices and produce top-K."""
    # Normalize PPR
    ppr_max = np.asarray(ppr_matrix.max(axis=1).todense()).ravel()
    ppr_max[ppr_max == 0] = 1.0
    ppr_norm = sp.diags(1.0 / ppr_max) @ ppr_matrix
    
    # Normalize Slot
    slot_max = np.asarray(slot_matrix.max(axis=1).todense()).ravel()
    slot_max[slot_max == 0] = 1.0
    slot_norm = sp.diags(1.0 / slot_max) @ slot_matrix
    
    # Ensure shapes match just in case
    if ppr_norm.shape != slot_norm.shape:
        print(f"Shapes mismatch: {ppr_norm.shape} vs {slot_norm.shape}")
    
    # Blend
    blended = ppr_norm + w * slot_norm
    
    # Extract top-K
    fallback = [int(x) for x in fallback_ranking["tv_show_id"]]
    predictions = {}
    for i, user_id in enumerate(user_ids):
        row = blended.getrow(i)
        if row.nnz > 0:
            # Sort indices by data desc
            order = np.argsort(-row.data)[:K]
            idx = row.indices[order]
            primary = [pool[j] for j in idx]
            predictions[user_id] = fill_recommendations(primary, fallback)
        else:
            predictions[user_id] = fill_recommendations([], fallback)
    return predictions


def _ppr_matrix_for_context(context: dict[str, object]) -> tuple[sp.csr_matrix, list[int], list[int]]:
    """Compute PPR scores for all eligible users as a matrix."""
    users = sorted(list(context["eligible"].keys()))
    pool = sorted(list(context["pool"]))
    show2idx = {s: i for i, s in enumerate(pool)}
    
    graph = _build_ppr_graph(
        context["interactions"],
        "frequency",
        decay=SWITCH_PPR_DECAY,
        reference_week=int(context["history_weeks"][1]),
    )
    n_nodes = int(graph["n_nodes"])
    n_users_node = int(graph["n_users"])
    show_ids = graph["show_ids"]
    restarts = graph["restarts"]
    pool_mask = np.isin(show_ids, np.asarray(pool, dtype=show_ids.dtype))
    pool_show_ids = show_ids[pool_mask]
    pool_show_cols = np.array([show2idx[int(s)] for s in pool_show_ids])
    transition_t = graph["transition_t"]
    
    # Beta adjustment
    airing_rate = context["airing_rate"]
    multiplier = np.power(np.clip(airing_rate.reindex(pool_show_ids).to_numpy(dtype=np.float64), 1e-6, None), SWITCH_PPR_BETA)
    
    rows, cols, data = [], [], []
    for i, user_id in enumerate(users):
        restart_data = restarts.get(user_id)
        if restart_data is None: continue
        restart_nodes, restart_weights = restart_data
        restart = np.zeros(n_nodes, dtype=np.float64)
        restart[restart_nodes] = restart_weights
        scores = restart.copy()
        for _ in range(1, 55): # Fixed iterations
            scores = SWITCH_PPR_RESTART * restart + (1.0 - SWITCH_PPR_RESTART) * (transition_t @ scores)
        
        show_scores = scores[n_users_node:][pool_mask] * multiplier
        pos = np.where(show_scores > 0)[0]
        if len(pos) > 0:
            rows.extend([i] * len(pos))
            cols.extend(pool_show_cols[pos])
            data.extend(show_scores[pos])
            
    return sp.csr_matrix((data, (rows, cols)), shape=(len(users), len(pool))), users, pool


def _slots_experiment(
    primary: dict[str, object],
    primary_pool: dict[str, object],
    secondary: dict[str, object],
    secondary_pool: dict[str, object],
    pairs: pd.DataFrame,
) -> None:
    # 1. PPR matrices
    print("slots: computing PPR matrices...")
    ppr_mat_p, users_p, pool_p = _ppr_matrix_for_context(primary)
    ppr_mat_s, users_s, pool_s = _ppr_matrix_for_context(secondary)
    
    # Baseline MAP@5 (pure PPR)
    ppr_pred_p = _blend_ppr_slots_optimized(ppr_mat_p, ppr_mat_p, pool_p, primary["fallback_ranking"], users_p, 0.0)
    baseline_p = float(np.mean(average_precision_per_user(ppr_pred_p, primary["eligible"])))
    print(f"slots: primary PPR baseline MAP@5={baseline_p:.6f}")
    
    results = []
    # Grid search on primary
    for slot_def in SLOT_DEFS:
        for smoothing in SLOT_SMOOTHINGS:
            slot_mat_p, _, _ = _slot_score_matrix(
                primary["history_pairs"], primary["history_weeks"], primary["target_weeks"], 
                HISTORY_PARTS, pool_p, users_p, slot_def, smoothing
            )
            for w in SLOT_WEIGHTS:
                params = f"slots:{slot_def},smooth:{smoothing},ppr+slots{w:g}"
                started = time.perf_counter()
                predictions = _blend_ppr_slots_optimized(ppr_mat_p, slot_mat_p, pool_p, primary["fallback_ranking"], users_p, w)
                elapsed = time.perf_counter() - started
                metrics = _log_run("slots", params, primary, predictions, pairs, primary["history_weeks"], elapsed)
                _print_row("slots", params, metrics, 0.0, elapsed)
                results.append({
                    "method": "slots", "params": params, "map5": metrics["map5"],
                    "slot_def": slot_def, "smoothing": smoothing, "w": w,
                    "predictions": predictions
                })
    
    # Log pure slot models for best config (channel_dow_hour, smooth:5)
    print()
    print("evaluating pure slots (channel_dow_hour, smooth=5)...")
    pure_slot_mat_p, _, _ = _slot_score_matrix(
        pairs, primary["history_weeks"], primary["target_weeks"],
        HISTORY_PARTS, pool_p, users_p, "channel_dow_hour", 5.0
    )
    pure_pred_p = _blend_ppr_slots_optimized(
        sp.csr_matrix(pure_slot_mat_p.shape), pure_slot_mat_p, pool_p, primary["fallback_ranking"], users_p, 1.0
    )
    pure_metrics_p = _log_run("slots", "slots:channel_dow_hour,smooth:5,pure", primary, pure_pred_p, pairs, primary["history_weeks"], 0.0)
    _print_row("slots", "slots:channel_dow_hour,smooth:5,pure", pure_metrics_p, 0.0, 0.0)

    pure_slot_mat_s, _, _ = _slot_score_matrix(
        pairs, secondary["history_weeks"], secondary["target_weeks"],
        HISTORY_PARTS, pool_s, users_s, "channel_dow_hour", 5.0
    )
    pure_pred_s = _blend_ppr_slots_optimized(
        sp.csr_matrix(pure_slot_mat_s.shape), pure_slot_mat_s, pool_s, secondary["fallback_ranking"], users_s, 1.0
    )
    pure_metrics_s = _log_run("slots", "slots:channel_dow_hour,smooth:5,pure", secondary, pure_pred_s, pairs, secondary["history_weeks"], 0.0)
    _print_row("slots", "slots:channel_dow_hour,smooth:5,pure", pure_metrics_s, 0.0, 0.0)

    best = max(results, key=lambda x: x["map5"])
    best_pred_p = best["predictions"]
    boot_p = paired_bootstrap_difference(best_pred_p, ppr_pred_p, primary["eligible"])
    
    print()
    print(f"best slots: {best['params']}")
    print(f"  primary MAP@5={float(best['map5']):.6f} (final {baseline_p:.6f}, diff {float(best['map5'])-baseline_p:+.6f})")
    print(f"  bootstrap diff vs final: {boot_p['mean_diff']:+.6f} [{boot_p['ci_low']:+.6f}, {boot_p['ci_high']:+.6f}]")

    # Secondary check
    print()
    print("secondary recheck (11-18 -> 19-30):")
    ppr_pred_s = _blend_ppr_slots_optimized(ppr_mat_s, ppr_mat_s, pool_s, secondary["fallback_ranking"], users_s, 0.0)
    baseline_s = float(np.mean(average_precision_per_user(ppr_pred_s, secondary["eligible"])))
    
    slot_mat_s, _, _ = _slot_score_matrix(
        secondary["history_pairs"], secondary["history_weeks"], secondary["target_weeks"],
        HISTORY_PARTS, pool_s, users_s, best["slot_def"], best["smoothing"]
    )
    started = time.perf_counter()
    pred_s = _blend_ppr_slots_optimized(ppr_mat_s, slot_mat_s, pool_s, secondary["fallback_ranking"], users_s, best["w"])
    elapsed_s = time.perf_counter() - started
    metrics_s = _log_run("slots", best["params"], secondary, pred_s, pairs, secondary["history_weeks"], elapsed_s)
    boot_s = paired_bootstrap_difference(pred_s, ppr_pred_s, secondary["eligible"])
    
    print(f"  MAP@5={float(metrics_s['map5']):.6f} (final {baseline_s:.6f}, diff {float(metrics_s['map5'])-baseline_s:+.6f})")
    print(f"  bootstrap diff vs final: {boot_s['mean_diff']:+.6f} [{boot_s['ci_low']:+.6f}, {boot_s['ci_high']:+.6f}]")

    global _SLOTS_BEST
    _SLOTS_BEST = {
        "slot_def": best["slot_def"], "smoothing": best["smoothing"], "w": best["w"], "params": best["params"],
        "pred_primary": best_pred_p, "pred_secondary": pred_s,
        "ppr_pred_primary": ppr_pred_p, "ppr_pred_secondary": ppr_pred_s
    }
def _normalize_max(predictions: dict[int, list[int]], scores: dict[int, dict[int, float]]) -> dict[int, dict[int, float]]:
    norm: dict[int, dict[int, float]] = {}
    for user_id, pred in predictions.items():
        user_scores = scores.get(user_id, {})
        if not user_scores:
            norm[user_id] = {}
            continue
        m = max(user_scores.values()) if user_scores else 1.0
        if m == 0:
            norm[user_id] = dict(user_scores)
        else:
            norm[user_id] = {k: float(v) / float(m) for k, v in user_scores.items()}
    return norm


def _blend_scores(ppr_scores: dict[int, dict[int, float]], slot_scores: dict[int, dict[int, float]], w: float) -> dict[int, list[int]]:
    pool = None  # not needed
    # need pool/fallback; caller handles
    raise NotImplementedError




def _history_show_counts(context: dict[str, object]) -> dict[int, int]:
    """Unique shows with screen_time >= 0.5 in the history window."""
    history = context["history_pairs"].loc[
        context["history_pairs"]["air_iso_week"].between(
            context["history_weeks"][0], context["history_weeks"][1]
        )
        & context["history_pairs"]["screen_time"].ge(0.5)
    ]
    return {
        int(user_id): int(count)
        for user_id, count in history.groupby("user_id", observed=True)[
            "tv_show_id"
        ]
        .nunique()
        .items()
    }


def _switch_components(
    context: dict[str, object], content: pd.DataFrame
) -> tuple[dict[int, list[int]], dict[int, list[int]], dict[int, list[int]], float]:
    """Build final PPR, association rules, and CBF lists once for switching."""
    started = time.perf_counter()
    users = list(context["eligible"])
    graph = _build_ppr_graph(
        context["interactions"],
        "frequency",
        decay=SWITCH_PPR_DECAY,
        reference_week=int(context["history_weeks"][1]),
    )
    final = ppr_predictions(
        graph,
        context["pool"],
        context["fallback_ranking"],
        users,
        restart_prob=SWITCH_PPR_RESTART,
        airing_rate=context["airing_rate"],
        beta=SWITCH_PPR_BETA,
    )
    rules, _, _ = assoc_predictions(
        context["interactions"],
        context["pool"],
        context["fallback_ranking"],
        users,
        min_support=0.01,
        min_confidence=0.2,
        score_mode="confidence_lift",
    )
    cbf, _, _ = content_predictions(
        context["interactions"],
        context["pool"],
        context["fallback_ranking"],
        users,
        content,
        profile_weight="frequency",
        feature_weight="idf",
    )
    return final, rules, cbf, time.perf_counter() - started


def _apply_switch(
    final: dict[int, list[int]],
    rules: dict[int, list[int]],
    cbf: dict[int, list[int]],
    history_counts: dict[int, int],
    context: dict[str, object],
    *,
    max_history: int,
    rules_weight: float,
    cbf_weight: float,
) -> tuple[dict[int, list[int]], int]:
    """Use rank fusion only for users with a short history, keep PPR otherwise."""
    short_users = [
        int(user_id)
        for user_id in context["eligible"]
        if history_counts.get(int(user_id), 0) <= max_history
    ]
    predictions = dict(final)
    if not short_users:
        return predictions, 0
    components: list[tuple[float, dict[int, list[int]]]] = [(1.0, final)]
    if rules_weight > 0:
        components.append((rules_weight, rules))
    if cbf_weight > 0:
        components.append((cbf_weight, cbf))
    fused = rrf_predictions(
        components,
        context["pool"],
        context["fallback_ranking"],
        short_users,
    )
    predictions.update(fused)
    return predictions, len(short_users)


def _switch_configs() -> Iterable[tuple[int, str, float, float]]:
    for max_history in SWITCH_HISTORY_LIMITS:
        for weight in SWITCH_COMPONENT_WEIGHTS:
            yield max_history, "rules", weight, 0.0
        for weight in SWITCH_COMPONENT_WEIGHTS:
            yield max_history, "cbf", 0.0, weight
        for rules_weight in SWITCH_COMPONENT_WEIGHTS:
            for cbf_weight in SWITCH_COMPONENT_WEIGHTS:
                yield max_history, "both", rules_weight, cbf_weight


def _switch_params(
    max_history: int, source: str, rules_weight: float, cbf_weight: float
) -> str:
    return (
        f"hist_shows<={max_history}, ppr=1, "
        f"assoc={rules_weight:g}, cbf={cbf_weight:g}, "
        f"source={source}"
    )


def _switch_experiment(
    primary: dict[str, object],
    primary_pool: dict[str, object],
    secondary: dict[str, object],
    secondary_pool: dict[str, object],
    content: pd.DataFrame,
    pairs: pd.DataFrame,
) -> None:
    final, rules, cbf, component_seconds = _switch_components(
        primary, content
    )
    history_counts = _history_show_counts(primary)
    baseline = float(
        np.mean(
            average_precision_per_user(final, primary["eligible"])
        )
    )
    print(
        f"switch components: {component_seconds:.2f}s; "
        f"final baseline MAP@5={baseline:.6f}"
    )
    results: list[dict[str, object]] = []
    for max_history, source, rules_weight, cbf_weight in _switch_configs():
        params = _switch_params(
            max_history, source, rules_weight, cbf_weight
        )
        started = time.perf_counter()
        predictions, affected = _apply_switch(
            final,
            rules,
            cbf,
            history_counts,
            primary,
            max_history=max_history,
            rules_weight=rules_weight,
            cbf_weight=cbf_weight,
        )
        metrics = _log_run(
            "switch",
            params,
            primary,
            predictions,
            pairs,
            (11, 20),
            time.perf_counter() - started,
        )
        _print_row(
            "switch", params, metrics, 0.0, time.perf_counter() - started
        )
        results.append(
            {
                "method": "switch",
                "params": params,
                **metrics,
                "max_history": max_history,
                "source": source,
                "rules_weight": rules_weight,
                "cbf_weight": cbf_weight,
                "affected": affected,
                "predictions": predictions,
            }
        )

    best = _best_result(results)
    best_predictions = best["predictions"]
    changed = sum(
        1
        for user_id in primary["eligible"]
        if best_predictions[int(user_id)] != final[int(user_id)]
    )
    boot = paired_bootstrap_difference(
        best_predictions, final, primary["eligible"]
    )
    print()
    print(f"best switch: {best['params']}")
    print(
        f"  primary MAP@5={float(best['map5']):.6f} "
        f"(final {baseline:.6f}, diff {float(best['map5']) - baseline:+.6f})"
    )
    print(
        f"  affected users (eligible): {best['affected']:,}/"
        f"{len(primary['eligible']):,}; changed lists: {changed:,}"
    )
    print(
        f"  bootstrap diff vs final: {boot['mean_diff']:+.6f} "
        f"[{boot['ci_low']:+.6f}, {boot['ci_high']:+.6f}]"
    )

    print()
    print("secondary recheck (11-18 -> 19-30):")
    final_s, rules_s, cbf_s, component_seconds_s = _switch_components(
        secondary, content
    )
    history_counts_s = _history_show_counts(secondary)
    baseline_s = float(
        np.mean(
            average_precision_per_user(final_s, secondary["eligible"])
        )
    )
    predictions_s, affected_s = _apply_switch(
        final_s,
        rules_s,
        cbf_s,
        history_counts_s,
        secondary,
        max_history=int(best["max_history"]),
        rules_weight=float(best["rules_weight"]),
        cbf_weight=float(best["cbf_weight"]),
    )
    metrics_s = _log_run(
        "switch",
        str(best["params"]),
        secondary,
        predictions_s,
        pairs,
        (11, 18),
        component_seconds_s,
    )
    changed_s = sum(
        1
        for user_id in secondary["eligible"]
        if predictions_s[int(user_id)] != final_s[int(user_id)]
    )
    boot_s = paired_bootstrap_difference(
        predictions_s, final_s, secondary["eligible"]
    )
    print(
        f"  MAP@5={float(metrics_s['map5']):.6f} "
        f"(final {baseline_s:.6f}, diff {float(metrics_s['map5']) - baseline_s:+.6f})"
    )
    print(
        f"  affected users (eligible): {affected_s:,}/"
        f"{len(secondary['eligible']):,}; changed lists: {changed_s:,}"
    )
    print(
        f"  bootstrap diff vs final: {boot_s['mean_diff']:+.6f} "
        f"[{boot_s['ci_low']:+.6f}, {boot_s['ci_high']:+.6f}]"
    )


def _hybrid_predictions(
    best_ppr: dict[str, object],
    best_assoc: dict[str, object],
    best_content: dict[str, object],
    best_als: dict[str, object],
    hybrid_context: dict[str, object],
    primary: dict[str, object],
    primary_pool: dict[str, object],
    weights: tuple[float, float, float],
    *,
    user_ids: Iterable[int] | None = None,
) -> dict[int, list[int]]:
    components = []
    for weight, best in (
        (1.0, best_ppr),
        (weights[0], best_assoc),
        (weights[1], best_content),
        (weights[2], best_als),
    ):
        context = _context_for_row(best, primary, primary_pool)
        predictions, _, _ = component_predictions(
            best, context, user_ids=user_ids
        )
        components.append((weight, predictions))
    users = list(hybrid_context["eligible"] if user_ids is None else user_ids)
    return rrf_predictions(
        components,
        hybrid_context["pool"],
        hybrid_context["fallback_ranking"],
        users,
    )


def _sensitivity_report(
    primary: dict[str, object],
    primary_pool: dict[str, object],
    components: dict[str, dict[str, object]],
    hybrid_weights: tuple[float, float, float],
    pairs: pd.DataFrame,
) -> None:
    users = list(primary["truth_users_all"])
    predictions: dict[str, dict[int, list[int]]] = {}
    for label, best in components.items():
        context = _context_for_row(best, primary, primary_pool)
        predictions[label], _, _ = component_predictions(
            best, context, user_ids=users
        )
    predictions["hybrid"] = _hybrid_predictions(
        components["ppr"],
        components["assoc"],
        components["content"],
        components["als"],
        primary_pool,
        primary,
        primary_pool,
        hybrid_weights,
        user_ids=users,
    )

    print()
    print("tie-break sensitivity (MAP@5 over all users with >=1 truth show):")
    rules = None
    table: dict[str, dict[str, float]] = {}
    for label, recs in predictions.items():
        rows = tie_break_sensitivity(
            recs,
            pairs,
            21,
            30,
            user_ids=set(primary["truth_users_all"]),
        )
        if rules is None:
            rules = [str(row["rule"]) for row in rows]
        table[label] = {
            str(row["rule"]): float(row["map5_all"]) for row in rows
        }
    print(f"  {'method':<18} " + " ".join(f"{rule:>12}" for rule in rules))
    for label, values in table.items():
        print(
            f"  {label:<18} "
            + " ".join(f"{values[rule]:>12.6f}" for rule in rules)
        )
    truth = primary["truth_users_all"]
    print(f"  users with >=1 truth show: {len(truth):,}")
    print("paired user bootstrap (1000 draws, 95% CI, main tie-break):")
    for label_a, label_b in (
        ("ppr", "personal_frequency"),
        ("personal_frequency", "popularity"),
        ("hybrid", "popularity"),
        ("hybrid", "ppr"),
        ("hybrid", "assoc"),
        ("hybrid", "content"),
        ("hybrid", "als"),
    ):
        stats = paired_bootstrap_difference(
            predictions[label_a], predictions[label_b], truth
        )
        print(
            f"  {label_a:<18} - {label_b:<18} "
            f"{stats['mean_diff']:+.6f} "
            f"[{stats['ci_low']:+.6f}, {stats['ci_high']:+.6f}]"
        )


def _final_table(entries: list[tuple[str, dict[str, object] | None]]) -> None:
    print()
    print("final table (best variant per method):")
    print(
        f"  {'method':<18} {'primary MAP':>11} {'secondary MAP':>13} "
        f"{'coverage':>10} {'params'}"
    )
    for label, row in entries:
        if row is None:
            continue
        secondary = row.get("secondary_map5")
        secondary_text = (
            f"{float(secondary):.6f}" if secondary is not None else "-"
        )
        coverage = (
            f"{int(row['coverage_users'])}/{int(row['n_users'])}"
        )
        params = str(row["params"])
        if len(params) > 86:
            params = params[:83] + "..."
        print(
            f"  {label:<18} {float(row['map5']):>11.6f} {secondary_text:>13} "
            f"{coverage:>10} {params}"
        )


def _reset_validation_log() -> None:
    path = Path(__file__).resolve().parents[1] / "results" / "validation.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)




_SLOTS_BEST = None




def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--grid",
        choices=["als", "ppr", "assoc", "content", "switch", "slots", "final", "all"],
        default="all",
    )
    parser.add_argument("--reset-validation", action="store_true", default=False)
    args = parser.parse_args(argv)
    run_all = args.grid in ("all", "final")

    started = time.perf_counter()
    pairs = load_pairs()
    primary_history_pairs = load_pairs(
        cutoff_sec=HISTORY_END_SEC[(11, 20)]
    )
    secondary_history_pairs = load_pairs(
        cutoff_sec=HISTORY_END_SEC[(11, 18)]
    )
    submission = pd.read_csv(DATA_DIR / "submission.csv")
    submission_users = {int(user_id) for user_id in submission["user_id"]}

    content_started = time.perf_counter()
    # CBF attributes from schedule defining the pool (same for validation/test)
    content, consistency = _load_show_content(
        schedule_parts=HISTORY_PARTS + ("31-42",)
    )
    content_seconds = time.perf_counter() - content_started
    print(
        f"content attributes for {len(content):,} shows "
        f"loaded in {content_seconds:.3f}s"
    )
    print("  consistency (shows with attribute, inconsistent across airings):")
    for column, (n_shows, n_inconsistent) in consistency.items():
        print(f"    {column:<20} {n_shows:>6}  {n_inconsistent:>4}")
    print()

    if args.reset_validation or run_all:
        _reset_validation_log()

    primary = _window_context(
        pairs,
        submission_users,
        (11, 20),
        (21, 30),
        history_scope="all",
        min_screen_time=0.5,
        history_pairs=primary_history_pairs,
    )
    primary_pool = _window_context(
        pairs,
        submission_users,
        (11, 20),
        (21, 30),
        history_scope="pool",
        min_screen_time=0.8,
        history_pairs=primary_history_pairs,
    )

    if args.grid == "switch":
        secondary = _window_context(
            pairs,
            submission_users,
            (11, 18),
            (19, 30),
            history_scope="all",
            min_screen_time=0.5,
            history_pairs=secondary_history_pairs,
        )
        secondary_pool = _window_context(
            pairs,
            submission_users,
            (11, 18),
            (19, 30),
            history_scope="pool",
            min_screen_time=0.8,
            history_pairs=secondary_history_pairs,
        )
        _switch_experiment(
            primary,
            primary_pool,
            secondary,
            secondary_pool,
            content,
            pairs,
        )
        print(f"\nswitch grid elapsed: {time.perf_counter() - started:.2f}s")
        return

    if args.grid == "slots":
        secondary = _window_context(
            pairs,
            submission_users,
            (11, 18),
            (19, 30),
            history_scope="all",
            min_screen_time=0.5,
            history_pairs=secondary_history_pairs,
        )
        secondary_pool = _window_context(
            pairs,
            submission_users,
            (11, 18),
            (19, 30),
            history_scope="pool",
            min_screen_time=0.8,
            history_pairs=secondary_history_pairs,
        )
        _slots_experiment(
            primary,
            primary_pool,
            secondary,
            secondary_pool,
            pairs,
        )
        print(f"\nslots grid elapsed: {time.perf_counter() - started:.2f}s")
        return

    primary_results: list[dict[str, object]] = []
    if args.grid in ("assoc", "all", "final"):
        _print_header(primary)
        primary_results.extend(_assoc_grid(primary, pairs, (11, 20)))
        print()
        _print_header(primary_pool)
        primary_results.extend(_assoc_grid(primary_pool, pairs, (11, 20)))
        print()
    if args.grid in ("ppr", "all", "final"):
        if args.grid == "ppr":
            _print_header(primary)
            primary_results.extend(_ppr_decay_grid(primary, pairs, (11, 20)))
            print()
            _print_header(primary_pool)
            primary_results.extend(
                _ppr_beta_grid(primary_pool, pairs, (11, 20))
            )
            print()
        _print_header(primary)
        primary_results.extend(_ppr_combined_grid(primary, pairs, (11, 20)))
        print()
        _print_header(primary_pool)
        primary_results.extend(
            _ppr_combined_grid(primary_pool, pairs, (11, 20))
        )
        print()

    baseline_rows: list[dict[str, object]] = []
    if run_all:
        baseline_rows = _baseline_grid(primary, pairs, (11, 20))
        primary_results.extend(baseline_rows)
        print()

    if not run_all:
        if args.grid == "content":
            _content_grid(primary, pairs, (11, 20), content)
            _content_grid(primary_pool, pairs, (11, 20), content)
        if args.grid == "als":
            _als_grid(primary_pool, pairs, (11, 20))
        print(f"\n{args.grid} grid elapsed: {time.perf_counter() - started:.2f}s")
        return

    assoc_rows = [
        row for row in primary_results if row["method"] == "assoc"
    ]
    best_assoc = _best_result(assoc_rows) if assoc_rows else None

    ppr_rows = [
        row
        for row in primary_results
        if row["method"] in ("ppr_decay", "ppr_beta", "ppr_final")
    ]
    ppr_plain_rows = [
        row
        for row in ppr_rows
        if row["method"] == "ppr_decay"
        or (
            row["method"] == "ppr_final"
            and float(row["kwargs"].get("beta", 0.0)) == 0.0
        )
    ]
    ppr_plain = _best_result(ppr_plain_rows) if ppr_plain_rows else None
    best_ppr = _best_result(ppr_rows) if ppr_rows else None
    best_popularity = _best_result(
        [r for r in baseline_rows if r["method"] == "popularity"]
    ) if baseline_rows else None
    best_personal = _best_result(
        [r for r in baseline_rows if r["method"] == "personal_frequency"]
    ) if baseline_rows else None

    print("best on 11-20 -> 21-30:")
    if best_assoc is not None:
        print(
            f"  assoc:     {best_assoc['params']} "
            f"(MAP@5={float(best_assoc['map5']):.6f})"
        )
    if ppr_plain is not None:
        print(
            f"  ppr plain: {ppr_plain['params']} "
            f"(MAP@5={float(ppr_plain['map5']):.6f})"
        )
    if best_ppr is not None:
        print(
            f"  ppr best:  {best_ppr['params']} "
            f"(MAP@5={float(best_ppr['map5']):.6f})"
        )

    if best_assoc is not None:
        best_assoc["content"] = content
    if best_ppr is not None:
        best_ppr["content"] = content
    best_content = None
    best_als = None
    if args.grid in ("content", "all", "final"):
        content_rows = _content_grid(primary, pairs, (11, 20), content)
        content_rows += _content_grid(primary_pool, pairs, (11, 20), content)
        best_content = _best_result(content_rows)
        best_content["content"] = content
    if args.grid in ("als", "all", "final"):
        best_als = _best_result(
            [
                row
                for row in _als_grid(primary_pool, pairs, (11, 20))
            ]
        )
        best_als["content"] = content

    print()
    print("components on 11-20 -> 21-30:")
    for best in (best_ppr, best_assoc, best_content, best_als):
        if best is None:
            continue
        params = str(best["params"])
        if len(params) > 60:
            params = params[:57] + "..."
        print(
            f"  {str(best['method']):<8} {params:<60} "
            f"{float(best['map5']):.6f}"
        )

    component_maps = {
        "ppr": (best_ppr, component_predictions(best_ppr, _context_for_row(best_ppr, primary, primary_pool))),
        "assoc": (best_assoc, component_predictions(best_assoc, _context_for_row(best_assoc, primary, primary_pool))),
        "content": (best_content, component_predictions(best_content, _context_for_row(best_content, primary, primary_pool))),
        "als": (best_als, component_predictions(best_als, _context_for_row(best_als, primary, primary_pool))),
    }
    component_predictions_map = {
        label: result[1][0] for label, result in component_maps.items()
    }

    print()
    print("rank fusion (PPR with assoc/content/als) on 11-20 -> 21-30:")
    hybrid_results: list[dict[str, object]] = []
    for weight in RRF_WEIGHTS:
        params = (
            f"ppr=1.0, assoc={weight[0]:g}, "
            f"content={weight[1]:g}, als={weight[2]:g}"
        )
        started_rrf = time.perf_counter()
        predictions = rrf_predictions(
            [
                (1.0, component_predictions_map["ppr"]),
                (weight[0], component_predictions_map["assoc"]),
                (weight[1], component_predictions_map["content"]),
                (weight[2], component_predictions_map["als"]),
            ],
            primary_pool["pool"],
            primary_pool["fallback_ranking"],
            primary_pool["eligible"],
        )
        rrf_seconds = time.perf_counter() - started_rrf
        metrics = _log_run(
            "hybrid",
            params,
            primary_pool,
            predictions,
            pairs,
            (11, 20),
            rrf_seconds,
        )
        hybrid_results.append(
            {
                "method": "hybrid",
                "params": params,
                **metrics,
                "train_seconds": 0.0,
                "list_seconds": rrf_seconds,
                "history_scope": "pool",
                "min_screen_time": 0.8,
                "weights": weight,
            }
        )

    best_hybrid = _best_result(hybrid_results)
    print()
    print(
        f"  best hybrid: {best_hybrid['params']} "
        f"(MAP@5={float(best_hybrid['map5']):.6f})"
    )
    print("  component comparison:")
    for label, (best, _) in component_maps.items():
        delta = float(best_hybrid["map5"]) - float(best["map5"])
        print(
            f"    hybrid - {label:<8} {delta:+.6f} "
            f"(hybrid {float(best_hybrid['map5']):.6f}, "
            f"{label} {float(best['map5']):.6f})"
        )

    print()
    print("recheck best on 11-18 -> 19-30:")
    secondary = _window_context(
        pairs, submission_users, (11, 18), (19, 30),
        history_scope="all", min_screen_time=0.5,
        history_pairs=secondary_history_pairs,
    )
    secondary_pool = _window_context(
        pairs, submission_users, (11, 18), (19, 30),
        history_scope="pool", min_screen_time=0.8,
        history_pairs=secondary_history_pairs,
    )
    _print_header(secondary)
    _print_header(secondary_pool)
    for best in (
        best_popularity,
        best_personal,
        best_ppr,
        None if ppr_plain is best_ppr else ppr_plain,
        best_assoc,
        best_content,
        best_als,
    ):
        if best is None:
            continue
        context = _context_for_row(
            best, primary, primary_pool, secondary, secondary_pool
        )
        _rerun_best(best, context, pairs, (11, 18))

    hybrid_context = secondary_pool
    predictions = _hybrid_predictions(
        best_ppr,
        best_assoc,
        best_content,
        best_als,
        hybrid_context,
        secondary,
        secondary_pool,
        best_hybrid["weights"],
    )
    metrics = _log_run(
        "hybrid",
        str(best_hybrid["params"]),
        hybrid_context,
        predictions,
        pairs,
        (11, 18),
        0.0,
    )
    _print_row("hybrid", str(best_hybrid["params"]), metrics, 0.0, 0.0)
    best_hybrid["secondary_map5"] = metrics["map5"]
    print("  secondary component comparison:")
    for label, best in (
        ("ppr", best_ppr),
        ("assoc", best_assoc),
        ("content", best_content),
        ("als", best_als),
    ):
        delta = float(best_hybrid["secondary_map5"]) - float(
            best["secondary_map5"]
        )
        print(
            f"    hybrid - {label:<8} {delta:+.6f} "
            f"(hybrid {float(best_hybrid['secondary_map5']):.6f}, "
            f"{label} {float(best['secondary_map5']):.6f})"
        )

    _sensitivity_report(
        primary,
        primary_pool,
        {
            "popularity": best_popularity,
            "personal_frequency": best_personal,
            "ppr": best_ppr,
            "assoc": best_assoc,
            "content": best_content,
            "als": best_als,
        },
        best_hybrid["weights"],
        pairs,
    )

    _final_table(
        [
            ("popularity", best_popularity),
            ("personal_frequency", best_personal),
            ("assoc", best_assoc),
            ("content", best_content),
            ("als", best_als),
            ("ppr", best_ppr),
            ("hybrid (final)", best_hybrid),
        ]
    )
    print(f"\nfull run elapsed: {time.perf_counter() - started:.2f}s")


if __name__ == "__main__":
    main()
