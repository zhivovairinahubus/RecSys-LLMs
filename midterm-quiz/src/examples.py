"""Build qualitative examples and a manual screen-time check for the report."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import scipy.sparse as sp

try:
    from .data import (
        BASE_DIR,
        DATA_DIR,
        HISTORY_END_SEC,
        HISTORY_PARTS,
        SCHEDULE_PATHS,
        VIEWS_PATH,
        load_pairs,
    )
    from .evaluate import average_precision_at_k
    from .methods import (
        _build_ppr_graph,
        _load_show_content,
        _window_context,
        _ppr_matrix_for_context,
        _slot_score_matrix,
        _blend_ppr_slots_optimized,
        als_predictions,
        assoc_predictions,
        content_predictions,
        ppr_predictions,
        rrf_predictions,
    )
    from .recommend import (
        K,
        POPULARITY_N,
        personal_frequency_ranking,
        recommend_personal_frequency,
        recommend_popularity,
    )
except ImportError:
    from data import (
        BASE_DIR,
        DATA_DIR,
        HISTORY_END_SEC,
        HISTORY_PARTS,
        SCHEDULE_PATHS,
        VIEWS_PATH,
        load_pairs,
    )
    from evaluate import average_precision_at_k
    from methods import (
        _build_ppr_graph,
        _load_show_content,
        _window_context,
        _ppr_matrix_for_context,
        _slot_score_matrix,
        _blend_ppr_slots_optimized,
        als_predictions,
        assoc_predictions,
        content_predictions,
        ppr_predictions,
        rrf_predictions,
    )
    from recommend import (
        K,
        POPULARITY_N,
        personal_frequency_ranking,
        recommend_personal_frequency,
        recommend_popularity,
    )

RESULTS_DIR = BASE_DIR / "results"
EXAMPLES_PATH = RESULTS_DIR / "examples.csv"
HAND_CHECK_PATH = RESULTS_DIR / "hand_check.txt"
HISTORY_WEEKS = (11, 20)
TARGET_WEEKS = (21, 30)
ACTIVE_USER = 3193355660422450548
LIGHT_USER = 3799694907521863650
PPR_RESTART = 0.3
PPR_DECAY = 0.9
PPR_BETA = 0.5
PPR_MAX_ITER = 200
PPR_TOL = 1e-8
SENTINEL_FALLBACK = pd.DataFrame({"tv_show_id": [-1, -2, -3, -4, -5]})
METHOD_ORDER = (
    "popularity",
    "personal_frequency",
    "als",
    "ppr",
    "rules",
    "cbf",
    "final",
    "final_slots",
    "slots_pure",
)
METHOD_PARAMS = {
    "popularity": "N=2",
    "personal_frequency": "screen>=0.5, fallback_N=2",
    "als": "hist=pool, thr=0.8, f=128, reg=10, conf_alpha=0.5, seed=42",
    "ppr": "hist=all, thr=0.5, restart=0.3, decay=0.9, beta=0.5",
    "rules": "thr=0.8, sup=0.01, conf=0.2, score=confidence_lift",
    "cbf": "thr=0.8, profile=frequency, feat=idf",
    "final": "ppr=1.0, assoc=0, content=0, als=0",
    "final_slots": "slots:channel_dow_hour,smooth:5,ppr+slots0.5",
    "slots_pure": "slots:channel_dow_hour,smooth:5,pure",
}


def _load_metadata() -> tuple[dict[int, str], dict[int, str]]:
    frames = []
    for part in HISTORY_PARTS + ("31-42",):
        frame = pd.read_csv(
            SCHEDULE_PATHS[part],
            usecols=["tv_show_id", "tv_show_title"],
            dtype={"tv_show_id": "int64", "tv_show_title": "string"},
            low_memory=False,
        )
        frames.append(frame)
    schedule = pd.concat(frames, ignore_index=True)
    schedule = schedule.loc[schedule["tv_show_id"].ne(0)]
    schedule["tv_show_title"] = (
        schedule["tv_show_title"].astype("string").str.strip().replace({"": pd.NA})
    )
    title_counts = (
        schedule.dropna(subset=["tv_show_title"])
        .value_counts(["tv_show_id", "tv_show_title"], sort=False)
        .reset_index(name="n")
        .sort_values(
            ["tv_show_id", "n", "tv_show_title"],
            ascending=[True, False, True],
            kind="mergesort",
        )
        .drop_duplicates("tv_show_id")
    )
    titles = {
        int(row.tv_show_id): str(row.tv_show_title)
        for row in title_counts.itertuples(index=False)
    }
    content, _ = _load_show_content(
        schedule_parts=HISTORY_PARTS + ("31-42",)
    )
    categories = {
        int(row.tv_show_id): str(row.tv_show_category)
        if pd.notna(row.tv_show_category)
        else "—"
        for row in content.itertuples(index=False)
    }
    return titles, categories


def _meta_text(show_id: int, titles: dict[int, str], categories: dict[int, str]) -> tuple[str, str]:
    return titles.get(int(show_id), f"show {int(show_id)}"), categories.get(
        int(show_id), "—"
    )


def _history_frame(
    history_pairs: pd.DataFrame, user_id: int
) -> pd.DataFrame:
    history = history_pairs.loc[
        history_pairs["user_id"].eq(user_id)
        & history_pairs["air_iso_week"].between(HISTORY_WEEKS[0], HISTORY_WEEKS[1])
        & history_pairs["screen_time"].ge(0.5)
    ]
    if history.empty:
        return history
    return (
        history.groupby("tv_show_id", observed=True)
        .agg(
            n_airings=("schedule_row_id", "size"),
            total_screen_time=("screen_time", "sum"),
            last_air_start_sec=("air_start_sec", "max"),
        )
        .reset_index()
        .sort_values(
            ["n_airings", "total_screen_time", "last_air_start_sec", "tv_show_id"],
            ascending=[False, False, False, True],
            kind="mergesort",
        )
    )


def _select_users(primary: dict[str, object]) -> tuple[list[int], dict[int, int]]:
    interactions = primary["interactions"]
    counts = (
        interactions.groupby("user_id", observed=True)["tv_show_id"]
        .nunique()
        .to_dict()
    )
    eligible_counts = pd.Series(
        {
            int(user_id): int(counts.get(int(user_id), 0))
            for user_id in primary["eligible"]
        }
    )
    active_rank = eligible_counts.rank(pct=True, method="max").loc[ACTIVE_USER]
    light_rank = eligible_counts.rank(pct=True, method="max").loc[LIGHT_USER]
    print(
        f"active user {ACTIVE_USER}: {int(eligible_counts[ACTIVE_USER])} history shows "
        f"(pct={active_rank:.3f})"
    )
    print(
        f"light user  {LIGHT_USER}: {int(eligible_counts[LIGHT_USER])} history shows "
        f"(pct={light_rank:.3f})"
    )
    return [ACTIVE_USER, LIGHT_USER], {
        int(user_id): int(count) for user_id, count in eligible_counts.items()
    }


def _generate_methods(
    primary: dict[str, object],
    primary_pool: dict[str, object],
    content: pd.DataFrame,
    user_ids: list[int],
) -> tuple[dict[str, dict[int, list[int]]], dict[str, dict[int, int]]]:
    predictions: dict[str, dict[int, list[int]]] = {}
    fallback: dict[str, dict[int, int]] = {}

    predictions["popularity"] = recommend_popularity(
        primary["history_pairs"],
        primary["pool"],
        HISTORY_WEEKS,
        POPULARITY_N,
        user_ids,
    )
    fallback["popularity"] = {int(user_id): 0 for user_id in user_ids}

    personal_primary = personal_frequency_ranking(
        primary["history_pairs"],
        primary["pool"],
        HISTORY_WEEKS,
        0.5,
    )
    predictions["personal_frequency"] = recommend_personal_frequency(
        primary["history_pairs"],
        primary["pool"],
        HISTORY_WEEKS,
        0.5,
        primary["fallback_ranking"],
        user_ids,
    )
    fallback["personal_frequency"] = {
        int(user_id): max(0, K - len(personal_primary.get(int(user_id), [])[:K]))
        for user_id in user_ids
    }

    als_pred, _, _, als_stats = als_predictions(
        primary_pool["interactions"],
        primary_pool["pool"],
        primary_pool["fallback_ranking"],
        user_ids,
        factors=128,
        regularization=10.0,
        confidence_alpha=0.5,
        seed=42,
        return_stats=True,
    )
    als_sentinel, _, _ = als_predictions(
        primary_pool["interactions"],
        primary_pool["pool"],
        SENTINEL_FALLBACK,
        user_ids,
        factors=128,
        regularization=10.0,
        confidence_alpha=0.5,
        seed=42,
    )
    predictions["als"] = als_pred
    fallback["als"] = {
        int(user_id): sum(1 for show_id in als_sentinel[int(user_id)] if show_id < 0)
        for user_id in user_ids
    }

    graph_started = time.perf_counter()
    graph = _build_ppr_graph(
        primary["interactions"],
        "frequency",
        decay=PPR_DECAY,
        reference_week=HISTORY_WEEKS[1],
    )
    ppr_pred, ppr_stats = ppr_predictions(
        graph,
        primary["pool"],
        primary["fallback_ranking"],
        user_ids,
        restart_prob=PPR_RESTART,
        airing_rate=primary["airing_rate"],
        beta=PPR_BETA,
        max_iter=PPR_MAX_ITER,
        tol=PPR_TOL,
        return_stats=True,
    )
    ppr_sentinel = ppr_predictions(
        graph,
        primary["pool"],
        SENTINEL_FALLBACK,
        user_ids,
        restart_prob=PPR_RESTART,
        airing_rate=primary["airing_rate"],
        beta=PPR_BETA,
        max_iter=PPR_MAX_ITER,
        tol=PPR_TOL,
    )
    predictions["ppr"] = ppr_pred
    fallback["ppr"] = {
        int(user_id): sum(1 for show_id in ppr_sentinel[int(user_id)] if show_id < 0)
        for user_id in user_ids
    }

    assoc_pred, _, _ = assoc_predictions(
        primary_pool["interactions"],
        primary_pool["pool"],
        primary_pool["fallback_ranking"],
        user_ids,
        min_support=0.01,
        min_confidence=0.2,
        score_mode="confidence_lift",
    )
    assoc_sentinel, _, _ = assoc_predictions(
        primary_pool["interactions"],
        primary_pool["pool"],
        SENTINEL_FALLBACK,
        user_ids,
        min_support=0.01,
        min_confidence=0.2,
        score_mode="confidence_lift",
    )
    predictions["rules"] = assoc_pred
    fallback["rules"] = {
        int(user_id): sum(1 for show_id in assoc_sentinel[int(user_id)] if show_id < 0)
        for user_id in user_ids
    }

    cbf_pred, _, _ = content_predictions(
        primary_pool["interactions"],
        primary_pool["pool"],
        primary_pool["fallback_ranking"],
        user_ids,
        content,
        profile_weight="frequency",
        feature_weight="idf",
    )
    cbf_sentinel, _, _ = content_predictions(
        primary_pool["interactions"],
        primary_pool["pool"],
        SENTINEL_FALLBACK,
        user_ids,
        content,
        profile_weight="frequency",
        feature_weight="idf",
    )
    predictions["cbf"] = cbf_pred
    fallback["cbf"] = {
        int(user_id): sum(1 for show_id in cbf_sentinel[int(user_id)] if show_id < 0)
        for user_id in user_ids
    }

    predictions["final"] = rrf_predictions(
        [(1.0, ppr_pred)],
        primary["pool"],
        primary["fallback_ranking"],
        user_ids,
    )
    fallback["final"] = dict(fallback["ppr"])

    # Final slots
    # Need PPR matrix specifically for these 2 users
    ppr_mat, _, _ = _ppr_matrix_for_context({**primary, "eligible": {u: 5 for u in user_ids}})
    
    slot_mat, _, _ = _slot_score_matrix(
        primary["history_pairs"],
        HISTORY_WEEKS,
        TARGET_WEEKS,
        ("21-30",),
        primary["pool"],
        user_ids,
        "channel_dow_hour",
        5.0,
    )
    predictions["final_slots"] = _blend_ppr_slots_optimized(
        ppr_mat, slot_mat, primary["pool"], primary["fallback_ranking"], user_ids, 0.5
    )
    fallback["final_slots"] = {int(u): 0 for u in user_ids}
    
    # Pure slots: use zero matrix for PPR
    ppr_mat_zero = sp.csr_matrix(slot_mat.shape)
    predictions["slots_pure"] = _blend_ppr_slots_optimized(
        ppr_mat_zero, slot_mat, primary["pool"], primary["fallback_ranking"], user_ids, 1.0
    )
    fallback["slots_pure"] = {int(u): 0 for u in user_ids}
    
    print(
        f"PPR iterations: mean={ppr_stats['mean_iterations']:.1f}, "
        f"max={ppr_stats['max_iterations']:.0f}, "
        f"residual={ppr_stats['max_residual']:.3e}, "
        f"users_with_fallback={ppr_stats['fallback_users']}"
    )
    print(f"PPR graph built in {time.perf_counter() - graph_started:.2f}s")
    return predictions, fallback


def build_examples(
    primary: dict[str, object],
    predictions: dict[str, dict[int, list[int]]],
    fallback: dict[str, dict[int, int]],
    history_sizes: dict[int, int],
    titles: dict[int, str],
    categories: dict[int, str],
    user_ids: list[int],
) -> pd.DataFrame:
    rows = []
    for user_id in user_ids:
        truth = [int(show_id) for show_id in primary["eligible"][user_id]]
        history = _history_frame(primary["history_pairs"], user_id)
        history_ids = [int(show_id) for show_id in history["tv_show_id"]]
        history_top = history.head(5)
        history_top_ids = [int(show_id) for show_id in history_top["tv_show_id"]]
        truth_titles = [
            _meta_text(show_id, titles, categories)[0] for show_id in truth
        ]
        truth_categories = [
            _meta_text(show_id, titles, categories)[1] for show_id in truth
        ]
        history_titles = [
            _meta_text(show_id, titles, categories)[0] for show_id in history_top_ids
        ]
        history_categories = [
            _meta_text(show_id, titles, categories)[1] for show_id in history_top_ids
        ]
        user_kind = "active" if user_id == ACTIVE_USER else "light"
        for method in METHOD_ORDER:
            method_predictions = predictions[method][user_id]
            method_fallback = fallback[method][user_id]
            primary_count = max(0, K - method_fallback)
            ap5 = average_precision_at_k(method_predictions, truth, K)
            for rank, show_id in enumerate(method_predictions[:K], start=1):
                title, category = _meta_text(show_id, titles, categories)
                row = {
                    "user_id": user_id,
                    "user_kind": user_kind,
                    "history_size": history_sizes[user_id],
                    "method": method,
                    "method_params": METHOD_PARAMS[method],
                    "rank": rank,
                    "tv_show_id": int(show_id),
                    "title": title,
                    "category": category,
                    "in_history": int(show_id) in set(history_ids),
                    "in_truth": int(show_id) in set(truth),
                    "truth_rank": truth.index(int(show_id)) + 1
                    if int(show_id) in truth
                    else "",
                    "primary_count": primary_count,
                    "fallback_count": method_fallback,
                    "ap5": ap5,
                    "truth_ids": " ".join(str(show_id) for show_id in truth),
                    "truth_titles": " | ".join(truth_titles),
                    "truth_categories": " | ".join(truth_categories),
                    "history_top_ids": " ".join(
                        str(show_id) for show_id in history_top_ids
                    ),
                    "history_top_titles": " | ".join(history_titles),
                    "history_top_categories": " | ".join(history_categories),
                }
                rows.append(row)
    return pd.DataFrame(rows)


def _load_schedule_rows() -> pd.DataFrame:
    frames = []
    for part in HISTORY_PARTS:
        frame = pd.read_csv(
            SCHEDULE_PATHS[part],
            usecols=[
                "channel_id",
                "channel_title",
                "start_time",
                "duration",
                "tv_show_title",
                "tv_show_id",
                "tv_show_category",
            ],
            dtype={
                "channel_id": "int64",
                "duration": "int64",
                "tv_show_id": "int64",
                "channel_title": "string",
                "tv_show_title": "string",
                "tv_show_category": "string",
            },
            low_memory=False,
        )
        frames.append(frame)
    schedule = pd.concat(frames, ignore_index=True)
    schedule = schedule.loc[schedule["tv_show_id"].ne(0)]
    schedule["start_time"] = pd.to_datetime(
        schedule["start_time"], format="%d.%m.%Y %H:%M:%S"
    )
    schedule["air_start_sec"] = schedule["start_time"].to_numpy(
        dtype="datetime64[s]"
    ).astype("int64")
    schedule["channel_title"] = schedule["channel_title"].fillna("—")
    schedule["tv_show_title"] = schedule["tv_show_title"].fillna("—")
    schedule["tv_show_category"] = schedule["tv_show_category"].fillna("—")
    return schedule


def _load_views() -> pd.DataFrame:
    views = pd.read_csv(
        VIEWS_PATH,
        usecols=["user_id", "vsetv_id", "start_time", "stop_time", "duraton"],
        dtype={
            "user_id": "uint64",
            "vsetv_id": "int64",
            "start_time": "string",
            "stop_time": "string",
            "duraton": "int64",
        },
    )
    views = views.rename(columns={"vsetv_id": "channel_id", "duraton": "duration"})
    views["start_time"] = pd.to_datetime(
        views["start_time"], format="%Y-%m-%d %H:%M:%S"
    )
    views["stop_time"] = pd.to_datetime(
        views["stop_time"], format="%Y-%m-%d %H:%M:%S"
    )
    views["view_start_sec"] = views["start_time"].to_numpy(
        dtype="datetime64[s]"
    ).astype("int64")
    views["view_end_sec"] = views["stop_time"].to_numpy(
        dtype="datetime64[s]"
    ).astype("int64")
    return views


def _hand_check(
    user_id: int,
    primary: dict[str, object],
    final_predictions: dict[int, list[int]],
    titles: dict[int, str],
    categories: dict[int, str],
) -> str:
    pairs = load_pairs()
    candidates = pairs.loc[
        pairs["user_id"].eq(user_id)
        & pairs["n_views"].eq(1)
        & pairs["screen_time"].ge(0.8)
    ].sort_values(["screen_time", "overlap_sec"], ascending=False)
    if candidates.empty:
        candidates = pairs.loc[pairs["user_id"].eq(user_id)].sort_values(
            ["screen_time", "overlap_sec"], ascending=False
        )
    cache_row = candidates.iloc[0]

    views = _load_views()
    user_views = views.loc[views["user_id"].eq(user_id)].copy()
    air_start = int(cache_row["air_start_sec"])
    air_end = air_start + int(cache_row["duration"])
    user_views["overlap"] = (
        np.minimum(user_views["view_end_sec"], air_end)
        - np.maximum(user_views["view_start_sec"], air_start)
    ).clip(lower=0)
    view_row = user_views.sort_values(
        ["overlap", "view_start_sec"], ascending=[False, True]
    ).iloc[0]

    schedule = _load_schedule_rows()
    airing = schedule.loc[
        schedule["channel_id"].eq(int(cache_row["channel_id"]))
        & schedule["air_start_sec"].eq(air_start)
        & schedule["duration"].eq(int(cache_row["duration"]))
        & schedule["tv_show_id"].eq(int(cache_row["tv_show_id"]))
    ]
    if airing.empty:
        airing_text = (
            f"channel_id={int(cache_row['channel_id'])}, "
            f"show_id={int(cache_row['tv_show_id'])}, "
            f"air_start_sec={air_start}, duration={int(cache_row['duration'])}s"
        )
        show_title = titles.get(int(cache_row["tv_show_id"]), "—")
        category = categories.get(int(cache_row["tv_show_id"]), "—")
    else:
        airing_row = airing.iloc[0]
        airing_text = (
            f"channel={airing_row['channel_title']}, "
            f"show={airing_row['tv_show_title']} "
            f"(id={int(airing_row['tv_show_id'])}, category={airing_row['tv_show_category']}), "
            f"start={airing_row['start_time']:%d.%m.%Y %H:%M:%S}, "
            f"duration={int(airing_row['duration'])}s"
        )
        show_title = str(airing_row["tv_show_title"])
        category = str(airing_row["tv_show_category"])

    truth = [int(show_id) for show_id in primary["eligible"][user_id]]
    predicted = final_predictions[user_id]
    hits = []
    misses = []
    for position, show_id in enumerate(predicted, start=1):
        title, _ = _meta_text(show_id, titles, categories)
        if show_id in truth:
            hits.append(
                f"rank {position}: {show_id} — {title} (truth rank {truth.index(show_id) + 1})"
            )
        else:
            misses.append(f"rank {position}: {show_id} — {title}")
    ap5 = average_precision_at_k(predicted, truth)
    raw_row_number = int(view_row.name) + 2
    lines = [
        "Manual screen-time check",
        f"user_id: {user_id}",
        f"history shows (screen_time >= 0.5): {len(_history_frame(primary['history_pairs'], user_id))}",
        "",
        f"raw view row: {raw_row_number}",
        f"  channel={view_row['channel_id']}, start={view_row['start_time']:%Y-%m-%d %H:%M:%S}, "
        f"stop={view_row['stop_time']:%Y-%m-%d %H:%M:%S}, duration={int(view_row['duration'])}s",
        "",
        "intersected airing:",
        f"  {airing_text}",
        f"  overlap_sec={int(cache_row['overlap_sec'])}, "
        f"screen_time={float(cache_row['screen_time']):.6f}",
        "",
        f"final AP@5: {ap5:.6f}",
        "final hit positions:",
    ]
    lines.extend([f"  {hit}" for hit in hits] if hits else ["  none"])
    lines.append("final misses:")
    lines.extend([f"  {miss}" for miss in misses] if misses else ["  none"])
    lines.append(
        "truth: "
        + " | ".join(
            f"{show_id} — {_meta_text(show_id, titles, categories)[0]}" for show_id in truth
        )
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pairs = load_pairs()
    history_pairs = load_pairs(cutoff_sec=HISTORY_END_SEC[HISTORY_WEEKS])
    submission = pd.read_csv(DATA_DIR / "submission.csv")
    submission_users = {int(user_id) for user_id in submission["user_id"]}
    primary = _window_context(
        pairs,
        submission_users,
        HISTORY_WEEKS,
        TARGET_WEEKS,
        history_scope="all",
        min_screen_time=0.5,
        history_pairs=history_pairs,
    )
    primary_pool = _window_context(
        pairs,
        submission_users,
        HISTORY_WEEKS,
        TARGET_WEEKS,
        history_scope="pool",
        min_screen_time=0.8,
        history_pairs=history_pairs,
    )
    content, _ = _load_show_content(
        schedule_parts=HISTORY_PARTS + ("31-42",)
    )
    titles, categories = _load_metadata()
    user_ids, history_sizes = _select_users(primary)
    predictions, fallback = _generate_methods(
        primary, primary_pool, content, user_ids
    )
    examples = build_examples(
        primary,
        predictions,
        fallback,
        history_sizes,
        titles,
        categories,
        user_ids,
    )
    examples.to_csv(EXAMPLES_PATH, index=False)
    hand_check = _hand_check(
        ACTIVE_USER,
        primary,
        predictions["final"],
        titles,
        categories,
    )
    HAND_CHECK_PATH.write_text(hand_check, encoding="utf-8")
    print(f"wrote: {EXAMPLES_PATH} ({len(examples):,} rows)")
    print(f"wrote: {HAND_CHECK_PATH}")
    print("fallback slots by method:")
    for method in METHOD_ORDER:
        print(
            f"  {method:<18} "
            + ", ".join(
                f"{user_id}:{fallback[method][user_id]}"
                for user_id in user_ids
            )
        )
    print("\nhand check:")
    print(hand_check)


if __name__ == "__main__":
    main()
