# sweet.tv Recommendation System

This project implements and evaluates recommendation methods for the sweet.tv midterm task. The goal is to predict five TV shows a user will watch in the next window given their viewing history.

## Problem

- History window: weeks `11–20`. Target window: weeks `21–30` (validation) or `31–42` (test).
- For each user we need top-5 show recommendations (`tv_show_id`).
- Truth is built from user–airing pairs with `screen_time >= 0.8` in the target window, ranked by qualifying airings (top-5 per user). For the official validation we use users with exactly 5 truth shows.
- Metric: `MAP@5`. Evaluation uses `MAP@5` over eligible users (with exactly 5 truth shows in the target window).

## Data and preprocessing

Files in `data/` (unchanged):

- `dataset11-30.csv` — user views: `user_id`, `vsetv_id`, `start_time`, `stop_time`, `duraton`
- `export_arh_11-20-final.csv`, `export_arh_21-30-final.csv`, `export_arh_31-42-final.csv` — schedules (channel, start time, duration, title, `tv_show_id`, category, genres, year, director, actors)
- `submission.csv` — test template

**Definition of screen time**: For each (user, view interval) intersected with each airing interval, `screen_time = (length of union of intersections of user views with the airing) / (duration of the airing)`. Intersections of the same user with the same airing are unioned (so `screen_time <= 1`). Views crossing the DST spring night are handled using wall-clock end time. Only history events before the window cutoff are used as features to avoid leakage. Cached pairs are stored in `cache/view_airing_pairs_history_labeled_v3*.parquet` (cache version `v3`).

Different thresholds are used by different components:
- History for some signals and examples: `screen_time >= 0.5`
- Candidate pool / ALS / association rules / CBF: `screen_time >= 0.8`

## Methods

| Method | Description | Primary MAP@5 (11–20→21–30) | Secondary MAP@5 (11–18→19–30) |
|---|---|---:|---:|
| `popularity` | Most frequent shows in history, `N=2` weeks | 0.290052 | 0.308346 |
| `personal_frequency` | Per-user top shows by qualifying airings, `screen>=0.5`, fallback to popularity | 0.373125 | 0.381660 |
| `als` | Implicit ALS on pool interactions (`hist=pool, thr=0.8`), `factors=128, reg=10, alpha=0.5`, seeds `42,123,2026` (averaged) | 0.353809 | 0.375304 |
| `assoc` | Association rules (antecedent→consequent), scored by `confidence × lift`, `thr=0.8, sup=0.01, conf=0.2` | 0.297093 | 0.321543 |
| `content` | Content-based: profile from categories/genres/year/director/actors, TF profile weighted by frequency, features IDF, cosine similarity | 0.231638 | 0.239091 |
| `ppr` | Personalized PageRank on user–show bipartite graph (all history, `screen>=0.5`, frequency weights, decay `0.9`, restart `0.3`, `beta=0.5` for airing rate); submission 02 | 0.397968 | 0.414237 |
| `slots` | Slot-based CBF alone: per-user completion rate in (channel, weekday, hour) slots, summed over the programme's target airings (`channel_dow_hour`, smoothing `5`); submission 07 | 0.380983 | 0.381657 |
| **`final`** | **PPR (row above) normalized per user + 0.5 × slot score normalized per user**; submission 06 | **0.408481** | **0.418310** |

## Final results

### Validation summary (`results/method_summary.csv`)
| method | params | map5_primary | map5_secondary | coverage_users_primary | coverage_share_primary | repeat_share_primary | distinct_shows_primary | coverage_users_secondary | coverage_share_secondary | repeat_share_secondary | distinct_shows_secondary | train_seconds | list_seconds | kaggle_public | kaggle_private | submission_file |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| popularity | N=2 | 0.290052 | 0.308346 | 920 | 0.800000 | 0.672000 | 5 | 985 | 0.821518 | 0.624187 | 5 | 0.000000 | 0.026084 | 0.33047 | 0.32096 | 01_popularity.csv |
| personal_frequency | screen>=0.5, fallback_N=2 | 0.373125 | 0.381660 | 1044 | 0.907826 | 0.957217 | 329 | 1088 | 0.907423 | 0.937615 | 361 | 0.000000 | 0.185569 | 0.40568 | 0.38897 | 05_personal_frequency.csv |
| als | hist=pool, thr=0.8, f=128, reg=10, conf_alpha=0.5, seeds=3 | 0.353809 | 0.375304 | 1032 | 0.897391 | 0.970087 | 153 | 1087 | 0.906589 | 0.952460 | 164 | 5.630028 | 0.332118 | 0.33960 | 0.32260 | 04_als.csv |
| ppr | hist=all, thr=0.5, restart=0.3, decay=0.9, beta=0.5 | 0.397968 | 0.414237 | 1062 | 0.923478 | 0.871478 | 259 | 1114 | 0.929108 | 0.837865 | 291 | 0.150328 | 22.292256 | 0.43574 | 0.42335 | 02_final_ppr_decay_beta.csv |
| assoc | thr=0.8, sup=0.01, conf=0.2, score=confidence_lift | 0.297093 | 0.321543 | 1012 | 0.880000 | 0.709913 | 47 | 1080 | 0.900751 | 0.659883 | 45 | 0.041102 | 0.077167 | — | — | — |
| content | thr=0.8, profile=frequency, feat=idf | 0.231638 | 0.239091 | 980 | 0.852174 | 0.573739 | 262 | 1016 | 0.847373 | 0.549625 | 299 | 0.109911 | 0.485973 | — | — | — |
| hybrid | ppr=1.0, assoc=0, content=0, als=0 | 0.397968 | 0.414237 | 1062 | 0.923478 | 0.871478 | 259 | 1114 | 0.929108 | 0.837865 | 291 | 0.000000 | 22.292256 | 0.43574 | 0.42335 | 02_final_ppr_decay_beta.csv |
| slots | slots:channel_dow_hour,smooth:5,pure | 0.380983 | 0.381657 | 1046 | 0.909565 | 0.877913 | 227 | 1094 | 0.912427 | 0.850876 | 282 | 0.000000 | 22.292256 | 0.40497 | 0.39745 | 07_slots_pure.csv |
| final | slots:channel_dow_hour,smooth:5,ppr+slots0.5 | 0.408481 | 0.418310 | 1063 | 0.924348 | 0.904348 | 226 | 1112 | 0.927440 | 0.872727 | 258 | 0.150328 | 22.292256 | 0.44048 | 0.42858 | 06_final_ppr_slots.csv |

`hybrid` = reciprocal rank fusion of PPR with ALS, rules and CBF; the weights selected on validation are 1/0/0/0, so its lists equal PPR (submission 02).

### Kaggle leaderboard
| Submission | Method | Public | Private |
|---|---|---:|---:|
| `01_popularity.csv` | popularity | 0.33047 | 0.32096 |
| `05_personal_frequency.csv` | personal_frequency | 0.40568 | 0.38897 |
| `04_als.csv` | ALS | 0.33960 | 0.32260 |
| `03_ppr_plain.csv` | PPR (no beta) | 0.43033 | 0.42026 |
| `02_final_ppr_decay_beta.csv` | PPR (decay + beta) | 0.43574 | 0.42335 |
| **`06_final_ppr_slots.csv`** | **PPR + Slots** | **0.44048** | **0.42858** |
| `07_slots_pure.csv` | Pure Slots | 0.40497 | 0.39745 |

**Final submission file:** `submissions/06_final_ppr_slots.csv` (SHA256: `f98da5dddf559d174873715ef9f3b9294bf90c58511a64b15942de92de4d826c`)

## Qualitative examples

`results/examples.csv` contains top-5 recommendations of every method for two users (with column `ap5` = AP@5 of that method for the user):

- Active user `3193355660422450548` (112 shows in history). Truth: `[2400508, 6700482, 5900256, 2400467, 240081]`. Final (`final_slots`) AP@5 `0.710000` (hits at ranks 1, 2, 4, 5); PPR AP@5 `0.643333`.
- Light user `3799694907521863650` (2 shows in history). Final and PPR AP@5 `0.000000`; rules `0.200000`, CBF `0.040000`, slots alone `0.040000`.

`results/hand_check.txt` contains a manual inspection of one view for the active user: raw row, intersected airing, `overlap_sec=7800`, `screen_time=1.000000`, and final hit positions.

## Sensitivity

`results/sensitivity.txt` (also in `results/run_all.log`) reports MAP@5 over all 1,238 users with at least one truth show under four tie-break rules, and paired user bootstrap intervals (1,000 draws, 95%) for differences between methods. `results/slots.log` reports the bootstrap of the final strategy against PPR on both windows.

## Notes on result files

- In `results/examples.csv`, method `final` is the RRF hybrid, whose lists equal PPR (submission 02); the final strategy is `final_slots` (submission 06). `results/hand_check.txt` shows the PPR list of the active user (AP@5 0.643333); the final list of the same user has AP@5 0.71.
- `results/validation.csv` keeps the slot rows of the first slot implementation (pure rows with MAP@5 0.343-0.352, replaced); the current code gives the pure slot rows 0.380983 / 0.381657.


## Structure
```text
midterm-quiz/
├── README.md
├── requirements.txt
├── data/           # input data (unchanged)
├── cache/          # pair caches
├── results/        # validation.csv, method_summary.csv, weekly_coverage.csv, examples.csv,
│                   # hand_check.txt, run_all.log, sensitivity.txt, switch.log,
│                   # report.log, examples.log, data.log, submissions.log
└── src/            # code
    ├── data.py
    ├── evaluate.py
    ├── recommend.py
    ├── methods.py
    ├── submit.py
    ├── report.py
    └── examples.py
submissions/        # 01_popularity.csv, 02_final_ppr_decay_beta.csv, 03_ppr_plain.csv,
                    # 04_als.csv, 05_personal_frequency.csv, 06_final_ppr_slots.csv,
                    # 07_slots_pure.csv
```

## Environment

Tested with Python 3.12.14:
```
pandas==3.1.0
numpy==2.5.4
scipy==1.18.1
implicit==0.7.3
pyarrow==26.0.0
threadpoolctl==3.7.0
```

In the environment used, `pip` was not present; install dependencies with `ensurepip` if needed:
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m ensurepip --upgrade
python -m pip install -r midterm-quiz/requirements.txt
```
Below commands use `~/.venvs/mq/bin/python`; substitute your interpreter.

## Reproduction (exact commands and approximate times)

All outputs are logged in `midterm-quiz/results/` (`run_all.log`, `sensitivity.txt`, `switch.log`, `data.log`, `report.log`, `examples.log`, `submissions.log`).

From repo root:

```bash
# 1) Build/load pair cache (data.log)
cd midterm-quiz
/usr/bin/time -p ~/.venvs/mq/bin/python src/data.py
# ~0.3s real (cache load)

# 2) Full grid + final selection + sensitivity/boot + table (run_all.log)
/usr/bin/time -p ~/.venvs/mq/bin/python src/methods.py --grid all
# ~1040s real

# 3) Switch experiment for short history (switch.log)
/usr/bin/time -p ~/.venvs/mq/bin/python src/methods.py --grid switch
# ~54s real

# 4) Slots experiment grid (slots.log)
/usr/bin/time -p ~/.venvs/mq/bin/python src/methods.py --grid slots
# ~136s real

# 5) Build summary and weekly coverage (report.log)
/usr/bin/time -p ~/.venvs/mq/bin/python src/report.py
# ~41s real

# 6) Build examples and hand check (examples.log)
/usr/bin/time -p ~/.venvs/mq/bin/python src/examples.py
# ~26s real

# 7) Generate test submissions (submissions.log)
/usr/bin/time -p ~/.venvs/mq/bin/python src/submit.py --method popularity
/usr/bin/time -p ~/.venvs/mq/bin/python src/submit.py --method final_ppr
/usr/bin/time -p ~/.venvs/mq/bin/python src/submit.py --method ppr_plain
/usr/bin/time -p ~/.venvs/mq/bin/python src/submit.py --method als
/usr/bin/time -p ~/.venvs/mq/bin/python src/submit.py --method personal_frequency
/usr/bin/time -p ~/.venvs/mq/bin/python src/submit.py --method slots
/usr/bin/time -p ~/.venvs/mq/bin/python src/submit.py --method slots_pure
```

Final submission is `submissions/06_final_ppr_slots.csv`. `data.py`, `report.py`, `examples.py`, `submit.py` use cached pairs where possible (`--no-cache` forces rebuild).
