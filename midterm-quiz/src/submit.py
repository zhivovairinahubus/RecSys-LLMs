"""Build and validate test submissions for selected recommenders."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

try:
    from .data import (
        BASE_DIR,
        DATA_DIR,
        HISTORY_END_SEC,
        HISTORY_PARTS,
        build_pairs,
        load_pairs,
        load_schedule,
    )
    from .methods import (
        PPR_TOL,
        _airing_rate,
        _build_ppr_graph,
        _history_interactions,
        _ppr_matrix_for_context,
        _slot_score_matrix,
        _blend_ppr_slots_optimized,
        als_predictions,
        ppr_predictions,
    )
    from .recommend import (
        K,
        POPULARITY_N,
        YEAR,
        build_candidate_pool,
        personal_frequency_ranking,
        popularity_ranking,
        recommend_personal_frequency,
        recommend_popularity,
    )
except ImportError:
    from data import (
        BASE_DIR,
        DATA_DIR,
        HISTORY_END_SEC,
        HISTORY_PARTS,
        build_pairs,
        load_pairs,
        load_schedule,
    )
    from methods import (
        PPR_TOL,
        _airing_rate,
        _build_ppr_graph,
        _history_interactions,
        _ppr_matrix_for_context,
        _slot_score_matrix,
        _blend_ppr_slots_optimized,
        als_predictions,
        ppr_predictions,
    )
    from recommend import (
        K,
        POPULARITY_N,
        YEAR,
        build_candidate_pool,
        personal_frequency_ranking,
        popularity_ranking,
        recommend_personal_frequency,
        recommend_popularity,
    )

TEMPLATE_PATH = DATA_DIR / "submission.csv"
SUBMISSIONS_DIR = BASE_DIR / "submissions"
HISTORY_WEEKS = (11, 30)
TARGET_WEEKS = (31, 42)
TARGET_PARTS = ("31-42",)
HISTORY_CUTOFF = HISTORY_END_SEC[HISTORY_WEEKS]
DEFAULT_MAX_ITER = 30
CONVERGED_MAX_ITER = 200
PPR_RESTART = 0.3
PPR_DECAY = 0.9
PPR_BETA = 0.5
PPR_HISTORY_SCOPE = "all"
PPR_MIN_SCREEN_TIME = 0.5
PLAIN_PPR_DECAY = 0.9
PLAIN_PPR_HISTORY_SCOPE = "all"
PLAIN_PPR_MIN_SCREEN_TIME = 0.5
DEFAULT_HISTORY_SCOPE = "pool"
DEFAULT_MIN_SCREEN_TIME = 0.8
ALS_FACTORS = 128
ALS_REGULARIZATION = 10.0
ALS_CONFIDENCE_ALPHA = 0.5
ALS_SEED = 42
PERSONAL_THRESHOLD = 0.5
METHOD_OUTPUTS = {
    "popularity": SUBMISSIONS_DIR / "01_popularity.csv",
    "final_ppr": SUBMISSIONS_DIR / "02_final_ppr_decay_beta.csv",
    "ppr_plain": SUBMISSIONS_DIR / "03_ppr_plain.csv",
    "als": SUBMISSIONS_DIR / "04_als.csv",
    "personal_frequency": SUBMISSIONS_DIR / "05_personal_frequency.csv",
    "slots": SUBMISSIONS_DIR / "06_final_ppr_slots.csv",
    "slots_pure": SUBMISSIONS_DIR / "07_slots_pure.csv",
}
METHOD_NAMES = {
    "popularity": "popularity",
    "final_ppr": "PPR all/0.5 restart=0.3, decay=0.9, beta=0.5",
    "ppr_plain": "PPR all/0.5 restart=0.3, decay=0.9, no beta",
    "als": "ALS f=128, reg=10, alpha=0.5, seed=42",
    "personal_frequency": "personal frequency >=0.5",
    "slots": "PPR + Slots (channel_dow_hour, smooth=5, w=0.5)",
    "slots_pure": "Pure Slots (channel_dow_hour, smooth=5)",
}


def load_history_pairs(use_cache: bool) -> pd.DataFrame:
    if use_cache:
        return load_pairs(parts=HISTORY_PARTS, cutoff_sec=HISTORY_CUTOFF)
    return build_pairs(parts=HISTORY_PARTS, cutoff_sec=HISTORY_CUTOFF)


def prepare_context(
    use_cache: bool,
    *,
    history_scope: str,
    min_screen_time: float,
) -> dict[str, object]:
    history_pairs = load_history_pairs(use_cache=use_cache)
    pool = build_candidate_pool(TARGET_WEEKS, TARGET_PARTS, iso_year=YEAR)
    schedule = load_schedule(parts=HISTORY_PARTS + TARGET_PARTS)
    fallback_ranking = popularity_ranking(
        history_pairs, pool, HISTORY_WEEKS, POPULARITY_N
    )
    interaction_pool = pool if history_scope == "pool" else None
    interactions = _history_interactions(
        history_pairs,
        interaction_pool,
        HISTORY_WEEKS,
        min_screen_time=min_screen_time,
    )
    airing_rate = _airing_rate(
        schedule, HISTORY_WEEKS, TARGET_WEEKS, pool
    )
    return {
        "history_pairs": history_pairs,
        "pool": pool,
        "fallback_ranking": fallback_ranking,
        "interactions": interactions,
        "airing_rate": airing_rate,
        "history_scope": history_scope,
        "min_screen_time": min_screen_time,
    }


def _ppr_config(method: str) -> tuple[str, float, float, float, float]:
    if method == "final_ppr":
        return (
            PPR_HISTORY_SCOPE,
            PPR_MIN_SCREEN_TIME,
            PPR_RESTART,
            PPR_DECAY,
            PPR_BETA,
        )
    return (
        PLAIN_PPR_HISTORY_SCOPE,
        PLAIN_PPR_MIN_SCREEN_TIME,
        PPR_RESTART,
        PLAIN_PPR_DECAY,
        0.0,
    )


def _ppr(
    context: dict[str, object],
    user_ids: list[int],
    *,
    restart_prob: float,
    decay: float,
    beta: float,
    max_iter: int,
) -> tuple[dict[int, list[int]], dict[str, float | int], float]:
    started = time.perf_counter()
    graph = _build_ppr_graph(
        context["interactions"],
        "frequency",
        decay=decay,
        reference_week=HISTORY_WEEKS[1],
    )
    predictions, stats = ppr_predictions(
        graph,
        context["pool"],
        context["fallback_ranking"],
        user_ids,
        restart_prob=restart_prob,
        airing_rate=context["airing_rate"],
        beta=beta,
        max_iter=max_iter,
        tol=PPR_TOL,
        return_stats=True,
    )
    return predictions, stats, time.perf_counter() - started


def _generate(
    method: str,
    context: dict[str, object],
    user_ids: list[int],
    *,
    max_iter: int,
    check_convergence: bool,
) -> tuple[dict[int, list[int]], dict[str, float | int], float]:
    started = time.perf_counter()
    if method == "popularity":
        predictions = recommend_popularity(
            context["history_pairs"],
            context["pool"],
            HISTORY_WEEKS,
            POPULARITY_N,
            user_ids,
        )
        return predictions, {"fallback_users": 0}, time.perf_counter() - started

    if method == "personal_frequency":
        personal = personal_frequency_ranking(
            context["history_pairs"],
            context["pool"],
            HISTORY_WEEKS,
            PERSONAL_THRESHOLD,
        )
        fallback_users = sum(
            1 for user_id in user_ids if len(personal.get(user_id, [])) < K
        )
        predictions = recommend_personal_frequency(
            context["history_pairs"],
            context["pool"],
            HISTORY_WEEKS,
            PERSONAL_THRESHOLD,
            context["fallback_ranking"],
            user_ids,
        )
        return (
            predictions,
            {"fallback_users": fallback_users},
            time.perf_counter() - started,
        )

    if method in ("slots", "slots_pure"):
        context["history_weeks"] = HISTORY_WEEKS
        context["target_weeks"] = TARGET_WEEKS
        context["eligible"] = {u: 5 for u in user_ids}
        ppr_mat, users, pool = _ppr_matrix_for_context(context)
        slot_mat, _, _ = _slot_score_matrix(
            context["history_pairs"],
            HISTORY_WEEKS,
            TARGET_WEEKS,
            TARGET_PARTS,
            pool,
            users,
            "channel_dow_hour",
            5.0,
        )
        w = 0.5 if method == "slots" else 1.0
        if method == "slots_pure":
            ppr_mat = sp.csr_matrix(ppr_mat.shape)
        predictions = _blend_ppr_slots_optimized(
            ppr_mat, slot_mat, pool, context["fallback_ranking"], users, w
        )
        return predictions, {"slot_def": "channel_dow_hour", "smoothing": 5, "w": w}, time.perf_counter() - started

    if method == "als":
        predictions, _, _, stats = als_predictions(
            context["interactions"],
            context["pool"],
            context["fallback_ranking"],
            user_ids,
            factors=ALS_FACTORS,
            regularization=ALS_REGULARIZATION,
            confidence_alpha=ALS_CONFIDENCE_ALPHA,
            seed=ALS_SEED,
            return_stats=True,
        )
        return predictions, stats, time.perf_counter() - started

    if method == "ppr_plain":
        predictions, stats, elapsed = _ppr(
            context,
            user_ids,
            restart_prob=PPR_RESTART,
            decay=PLAIN_PPR_DECAY,
            beta=0.0,
            max_iter=max(max_iter, CONVERGED_MAX_ITER),
        )
        return predictions, stats, elapsed

    predictions, stats, elapsed = _ppr(
        context,
        user_ids,
        restart_prob=PPR_RESTART,
        decay=PPR_DECAY,
        beta=PPR_BETA,
        max_iter=max_iter,
    )
    if check_convergence and float(stats["max_residual"]) > PPR_TOL:
        converged, converged_stats, converged_elapsed = _ppr(
            context,
            user_ids,
            restart_prob=PPR_RESTART,
            decay=PPR_DECAY,
            beta=PPR_BETA,
            max_iter=CONVERGED_MAX_ITER,
        )
        changed = sum(
            1
            for user_id in user_ids
            if predictions.get(user_id) != converged.get(user_id)
        )
        original_iterations = stats["max_iterations"]
        original_residual = stats["max_residual"]
        print(
            "convergence check: "
            f"{int(original_iterations)} iter, "
            f"residual={float(original_residual):.3e}; "
            f"max_iter={CONVERGED_MAX_ITER} -> "
            f"{int(converged_stats['max_iterations'])} iter, "
            f"residual={float(converged_stats['max_residual']):.3e}; "
            f"lists changed for {changed:,}/{len(user_ids):,} users"
        )
        if changed:
            predictions = converged
            stats = converged_stats
        stats = {
            **stats,
            "original_iterations": original_iterations,
            "original_residual": original_residual,
            "convergence_changed_users": changed,
            "converged_iterations": converged_stats["max_iterations"],
            "converged_residual": converged_stats["max_residual"],
        }
        elapsed += converged_elapsed
    return predictions, stats, elapsed


def build_submission(
    method: str,
    output: Path,
    *,
    use_cache: bool,
    max_iter: int,
    check_convergence: bool,
) -> dict[str, object]:
    started = time.perf_counter()
    template = pd.read_csv(TEMPLATE_PATH)
    user_ids = [int(user_id) for user_id in template["user_id"]]
    if method in ("final_ppr", "ppr_plain", "slots", "slots_pure"):
        config_method = "final_ppr" if method in ("slots", "slots_pure") else method
        history_scope, min_screen_time, _, _, _ = _ppr_config(config_method)
    else:
        history_scope = DEFAULT_HISTORY_SCOPE
        min_screen_time = DEFAULT_MIN_SCREEN_TIME
    context = prepare_context(
        use_cache=use_cache,
        history_scope=history_scope,
        min_screen_time=min_screen_time,
    )
    data_seconds = time.perf_counter() - started

    predictions, stats, generation_seconds = _generate(
        method,
        context,
        user_ids,
        max_iter=max_iter,
        check_convergence=check_convergence,
    )
    submission = pd.DataFrame(
        {
            "user_id": user_ids,
            "tv_show_id": [
                " ".join(str(show_id) for show_id in predictions[user_id])
                for user_id in user_ids
            ],
        }
    )
    submission["user_id"] = submission["user_id"].astype("uint64")
    output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output, index=False)
    checks = validate_submission(output, template, context["pool"])
    total_seconds = time.perf_counter() - started
    return {
        "template": template,
        "submission": submission,
        "pool": context["pool"],
        "history_scope": context["history_scope"],
        "min_screen_time": context["min_screen_time"],
        "checks": checks,
        "data_seconds": data_seconds,
        "generation_seconds": generation_seconds,
        "total_seconds": total_seconds,
        "stats": stats,
    }


def validate_submission(
    path: Path, template: pd.DataFrame, pool: list[int]
) -> dict[str, bool]:
    with path.open("r", encoding="utf-8") as file:
        header = file.readline().strip()

    generated = pd.read_csv(
        path, dtype={"user_id": "uint64", "tv_show_id": "string"}
    )

    expected_header = "user_id,tv_show_id"
    header_ok = header == expected_header
    rows_ok = len(generated) == len(template)
    users_ok = np.array_equal(
        generated["user_id"].to_numpy(dtype="uint64"),
        template["user_id"].to_numpy(dtype="uint64"),
    )

    pool_set = set(pool)
    rows_valid = True
    for value in generated["tv_show_id"]:
        parts = value.split()
        if len(parts) != K or len(set(parts)) != K:
            rows_valid = False
            break
        show_ids = [int(part) for part in parts]
        if any(show_id == 0 or show_id not in pool_set for show_id in show_ids):
            rows_valid = False
            break

    checks = {
        "header": header_ok,
        "row_count": rows_ok,
        "user_id_order": users_ok,
        "rows_have_5_distinct_pool_ids": rows_valid,
    }
    if not all(checks.values()):
        raise AssertionError(f"submission validation failed: {checks}")
    return checks


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        choices=["popularity", "final_ppr", "ppr_plain", "als", "personal_frequency", "slots", "slots_pure"],
        default="final_ppr",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--no-cache", action="store_true", default=False)
    parser.add_argument("--max-iter", type=int, default=DEFAULT_MAX_ITER)
    parser.add_argument(
        "--no-convergence-check", action="store_true", default=False
    )
    args = parser.parse_args(argv)

    output = args.output or METHOD_OUTPUTS[args.method]
    result = build_submission(
        args.method,
        output,
        use_cache=not args.no_cache,
        max_iter=args.max_iter,
        check_convergence=not args.no_convergence_check,
    )
    checks = result["checks"]
    stats = result["stats"]
    template = result["template"]

    print(f"method: {args.method} ({METHOD_NAMES[args.method]})")
    print(
        f"history: {HISTORY_WEEKS[0]}-{HISTORY_WEEKS[1]}, "
        f"target: {TARGET_WEEKS[0]}-{TARGET_WEEKS[1]}"
    )
    print(
        f"candidate pool: {len(result['pool']):,} shows; "
        f"users: {len(template):,}"
    )
    print(
        f"base: hist={result['history_scope']}, "
        f"thr={float(result['min_screen_time']):g}"
    )
    print(f"wrote: {output}")
    print(f"fallback users: {int(stats.get('fallback_users', 0)):,}")
    if "max_iterations" in stats:
        print(
            f"PPR iterations: mean={float(stats['mean_iterations']):.1f}, "
            f"max={int(stats['max_iterations'])}; "
            f"max residual={float(stats['max_residual']):.3e}"
        )
        if "original_iterations" in stats:
            print(
                f"PPR at original max_iter: "
                f"max={int(stats['original_iterations'])}, "
                f"residual={float(stats['original_residual']):.3e}; "
                f"at max_iter={CONVERGED_MAX_ITER}: "
                f"max={int(stats['converged_iterations'])}, "
                f"residual={float(stats['converged_residual']):.3e}; "
                f"changed_users={int(stats['convergence_changed_users'])}"
            )
    print("validation:")
    print(f"  header == 'user_id,tv_show_id': {checks['header']}")
    print(f"  rows == {len(template):,}: {checks['row_count']}")
    print(f"  user_id same order: {checks['user_id_order']}")
    print(
        "  5 distinct non-zero pool ids per row: "
        f"{checks['rows_have_5_distinct_pool_ids']}"
    )
    print(
        f"timing: data={result['data_seconds']:.2f}s, "
        f"generate={result['generation_seconds']:.2f}s, "
        f"total={result['total_seconds']:.2f}s "
        f"({'cache' if not args.no_cache else 'raw'})"
    )


if __name__ == "__main__":
    main()
