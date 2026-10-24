"""Load sweet.tv views and schedules, then build user-airing screen-time pairs."""

from __future__ import annotations

import time
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = BASE_DIR / "cache"

VIEWS_PATH = DATA_DIR / "dataset11-30.csv"
SCHEDULE_PATHS = {
    "11-20": DATA_DIR / "export_arh_11-20-final.csv",
    "21-30": DATA_DIR / "export_arh_21-30-final.csv",
    "31-42": DATA_DIR / "export_arh_31-42-final.csv",
}
HISTORY_PARTS = ("11-20", "21-30")
CACHE_VERSION = 3
# Exclusive wall-clock end of history for each analysis window (ISO 2020).
HISTORY_END_SEC = {
    (11, 18): int(pd.Timestamp("2020-05-04").timestamp()),
    (11, 20): int(pd.Timestamp("2020-05-18").timestamp()),
    (11, 30): int(pd.Timestamp("2020-07-27").timestamp()),
}


def _to_epoch_seconds(values: pd.Series) -> np.ndarray:
    """Convert datetimes to int64 seconds without assuming a nanosecond dtype."""
    return values.to_numpy(dtype="datetime64[s]").astype("int64")


def _cache_path(
    drop_unlabeled: bool, parts: Iterable[str], cutoff_sec: int | None = None
) -> Path:
    parts = tuple(parts)
    key = "history" if parts == HISTORY_PARTS else "_".join(parts)
    kind = "labeled" if drop_unlabeled else "all"
    suffix = "" if cutoff_sec is None else f"_cut{cutoff_sec}"
    return CACHE_DIR / f"view_airing_pairs_{key}_{kind}_v{CACHE_VERSION}{suffix}.parquet"


def load_views(path: Path = VIEWS_PATH) -> pd.DataFrame:
    views = pd.read_csv(
        path,
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
    views["view_start_sec"] = _to_epoch_seconds(views["start_time"])
    # stop_time is wall-clock local time; the 117 views crossing the spring DST
    # night have stop_time - start_time = duraton + 3600, so use stop_time for
    # the intersection with the (wall-clock) schedule instead of start+duration.
    views["view_end_sec"] = _to_epoch_seconds(views["stop_time"])
    iso = views["start_time"].dt.isocalendar()
    views["view_iso_year"] = iso["year"].astype("int16")
    views["view_iso_week"] = iso["week"].astype("int8")
    return views[
        [
            "user_id",
            "channel_id",
            "view_start_sec",
            "view_end_sec",
            "view_iso_year",
            "view_iso_week",
        ]
    ]


def load_schedule(
    parts: Iterable[str] | None = None, drop_unlabeled: bool = True
) -> pd.DataFrame:
    if parts is None:
        parts = HISTORY_PARTS
    frames = []
    for part in parts:
        frame = pd.read_csv(
            SCHEDULE_PATHS[part],
            usecols=["channel_id", "start_time", "duration", "tv_show_id"],
            dtype={
                "channel_id": "int64",
                "start_time": "string",
                "duration": "int64",
                "tv_show_id": "int64",
            },
        )
        frame["schedule_part"] = part
        frames.append(frame)

    schedule = pd.concat(frames, ignore_index=True)
    schedule["start_time"] = pd.to_datetime(
        schedule["start_time"], format="%d.%m.%Y %H:%M:%S"
    )
    schedule["air_start_sec"] = _to_epoch_seconds(schedule["start_time"])
    schedule["air_end_sec"] = schedule["air_start_sec"] + schedule["duration"]
    schedule["schedule_row_id"] = np.arange(len(schedule), dtype="int64")
    iso = schedule["start_time"].dt.isocalendar()
    schedule["air_iso_year"] = iso["year"].astype("int16")
    schedule["air_iso_week"] = iso["week"].astype("int8")

    if drop_unlabeled:
        schedule = schedule.loc[schedule["tv_show_id"].ne(0)]
    schedule = schedule.loc[schedule["duration"].gt(0)]

    return schedule[
        [
            "schedule_row_id",
            "channel_id",
            "air_start_sec",
            "air_end_sec",
            "duration",
            "tv_show_id",
            "air_iso_year",
            "air_iso_week",
            "schedule_part",
        ]
    ].reset_index(drop=True)


def _union_length(lo: np.ndarray, hi: np.ndarray) -> int:
    """Length of the union of sorted [lo, hi) intervals."""
    order = np.argsort(lo, kind="stable")
    lo = lo[order]
    hi = hi[order]
    total = 0
    current_lo = lo[0]
    current_hi = hi[0]
    for index in range(1, len(lo)):
        if lo[index] > current_hi:
            total += current_hi - current_lo
            current_lo = lo[index]
            current_hi = hi[index]
        elif hi[index] > current_hi:
            current_hi = hi[index]
    total += current_hi - current_lo
    return int(total)


def _merge_overlaps(raw_pairs: pd.DataFrame) -> pd.DataFrame:
    """Union clipped view intervals per (user, airing); keep max last view."""
    grouped = raw_pairs.groupby(
        ["user_id", "schedule_row_id"], sort=False, observed=True
    )
    n_views = grouped.size().rename("n_views")
    last_view = grouped["last_view_start_sec"].max()

    multi = n_views[n_views > 1]
    if multi.empty:
        overlap = grouped["overlap_sec"].sum()
    else:
        single = grouped["overlap_sec"].sum()
        keys = multi.index
        multi_rows = raw_pairs.set_index(
            ["user_id", "schedule_row_id"]
        ).loc[keys]
        merged = [
            _union_length(
                part["clip_lo"].to_numpy(dtype="int64"),
                part["clip_hi"].to_numpy(dtype="int64"),
            )
            for _, part in multi_rows.groupby(level=[0, 1], sort=False)
        ]
        multi_frame = pd.DataFrame(
            {"overlap_sec": merged},
            index=multi.index,
        )
        single_frame = single.to_frame("overlap_sec").drop(
            index=multi_frame.index, errors="ignore"
        )
        overlap = pd.concat([single_frame, multi_frame])["overlap_sec"]

    result = pd.concat([overlap, n_views, last_view], axis=1).reset_index()
    return result[
        [
            "user_id",
            "schedule_row_id",
            "overlap_sec",
            "n_views",
            "last_view_start_sec",
        ]
    ]


def _interval_join(
    views: pd.DataFrame,
    schedule: pd.DataFrame,
    cutoff_sec: int | None = None,
) -> pd.DataFrame:
    user_values = views["user_id"].to_numpy(dtype="uint64")
    view_start = views["view_start_sec"].to_numpy(dtype="int64")
    view_end = views["view_end_sec"].to_numpy(dtype="int64")
    if cutoff_sec is not None:
        # Drop the part of a view that happens after the history end so that
        # target-window playtime cannot leak into the history features.
        view_start = np.minimum(view_start, cutoff_sec)
        view_end = np.minimum(view_end, cutoff_sec)
        keep = view_end > view_start
        views = views.loc[keep]
        user_values = user_values[keep]
        view_start = view_start[keep]
        view_end = view_end[keep]
    view_groups = views.groupby("channel_id", sort=False).indices

    schedule_groups = {
        int(channel_id): group.sort_values("air_start_sec")
        for channel_id, group in schedule.groupby("channel_id", sort=False)
    }

    users: list[np.ndarray] = []
    row_ids: list[np.ndarray] = []
    overlaps: list[np.ndarray] = []
    starts: list[np.ndarray] = []
    clip_lo: list[np.ndarray] = []
    clip_hi: list[np.ndarray] = []

    for channel_id, view_index in view_groups.items():
        group = schedule_groups.get(int(channel_id))
        if group is None or group.empty:
            continue

        air_start = group["air_start_sec"].to_numpy(dtype="int64")
        air_end = group["air_end_sec"].to_numpy(dtype="int64")
        schedule_row_id = group["schedule_row_id"].to_numpy(dtype="int64")

        starts_for_channel = view_start[view_index]
        ends_for_channel = view_end[view_index]

        lo = np.searchsorted(
            np.maximum.accumulate(air_end), starts_for_channel, side="right"
        )
        hi = np.searchsorted(air_start, ends_for_channel, side="left") - 1
        count = np.maximum(hi - lo + 1, 0)
        if count.size == 0 or count.max() == 0:
            continue

        for offset in range(int(count.max())):
            selected = np.flatnonzero(count > offset)
            if selected.size == 0:
                continue

            air_index = lo[selected] + offset
            valid = air_index < air_start.size
            selected = selected[valid]
            air_index = air_index[valid]
            if selected.size == 0:
                continue

            lo_clip = np.maximum(
                starts_for_channel[selected], air_start[air_index]
            )
            hi_clip = np.minimum(
                ends_for_channel[selected], air_end[air_index]
            )
            overlap = hi_clip - lo_clip
            positive = overlap > 0
            if not positive.any():
                continue

            selected = selected[positive]
            air_index = air_index[positive]
            users.append(user_values[view_index][selected])
            row_ids.append(schedule_row_id[air_index])
            overlaps.append(overlap[positive].astype("int64"))
            starts.append(starts_for_channel[selected])
            clip_lo.append(lo_clip[positive])
            clip_hi.append(hi_clip[positive])

    if not users:
        return pd.DataFrame(
            columns=[
                "user_id",
                "schedule_row_id",
                "tv_show_id",
                "channel_id",
                "air_start_sec",
                "duration",
                "air_iso_year",
                "air_iso_week",
                "overlap_sec",
                "n_views",
                "last_view_start_sec",
                "screen_time",
            ]
        )

    raw_pairs = pd.DataFrame(
        {
            "user_id": np.concatenate(users),
            "schedule_row_id": np.concatenate(row_ids),
            "overlap_sec": np.concatenate(overlaps),
            "clip_lo": np.concatenate(clip_lo),
            "clip_hi": np.concatenate(clip_hi),
            "last_view_start_sec": np.concatenate(starts),
        }
    )

    # Merge the clipped intervals per (user, airing) so that overlapping views of
    # the same airing count their union, not their sum (screen_time stays <= 1).
    aggregated = _merge_overlaps(raw_pairs)

    metadata = schedule[
        [
            "schedule_row_id",
            "tv_show_id",
            "channel_id",
            "air_start_sec",
            "duration",
            "air_iso_year",
            "air_iso_week",
        ]
    ]
    pairs = aggregated.merge(
        metadata, on="schedule_row_id", how="left", validate="many_to_one"
    )
    pairs["screen_time"] = pairs["overlap_sec"] / pairs["duration"]
    pairs["n_views"] = pairs["n_views"].astype("int32")

    return pairs[
        [
            "user_id",
            "schedule_row_id",
            "tv_show_id",
            "channel_id",
            "air_start_sec",
            "duration",
            "air_iso_year",
            "air_iso_week",
            "overlap_sec",
            "n_views",
            "last_view_start_sec",
            "screen_time",
        ]
    ]


def build_pairs(
    parts: Iterable[str] | None = None,
    drop_unlabeled: bool = True,
    cutoff_sec: int | None = None,
) -> pd.DataFrame:
    views = load_views()
    schedule = load_schedule(parts=parts, drop_unlabeled=drop_unlabeled)
    return _interval_join(views, schedule, cutoff_sec=cutoff_sec)


def load_pairs(
    parts: Iterable[str] | None = None,
    drop_unlabeled: bool = True,
    force_rebuild: bool = False,
    cutoff_sec: int | None = None,
) -> pd.DataFrame:
    if parts is None:
        parts = HISTORY_PARTS
    parts = tuple(parts)
    path = _cache_path(drop_unlabeled, parts, cutoff_sec=cutoff_sec)

    if path.exists() and not force_rebuild:
        return pd.read_parquet(path)

    pairs = build_pairs(parts=parts, drop_unlabeled=drop_unlabeled, cutoff_sec=cutoff_sec)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    pairs.to_parquet(path, index=False)
    return pairs


def main() -> None:
    cache_path = _cache_path(True, HISTORY_PARTS)
    was_cached = cache_path.exists()
    started = time.perf_counter()
    pairs = load_pairs()
    elapsed = time.perf_counter() - started

    share = float((pairs["screen_time"] >= 0.8).mean())
    print(f"cache: {cache_path}")
    print(f"mode: {'loaded from cache' if was_cached else 'built and cached'}")
    print(f"user-airing pairs: {len(pairs):,}")
    print(f"users: {pairs['user_id'].nunique():,}")
    print(
        "screen_time >= 0.8: "
        f"{share:.4%} ({int((pairs['screen_time'] >= 0.8).sum()):,} pairs)"
    )
    print(f"elapsed: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
