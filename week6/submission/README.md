# HW6 — Personalized PageRank movie recommendations (submission)

Graph-based recommendation on MovieLens 100K, comparing **popularity**, **global
PageRank**, **personalized PageRank (PPR)** and **matrix factorization (MF)** on a
shared evaluation cohort. The core propagation is implemented in `week6/graph.js`
and driven by `week6/script.js`; this folder holds the machine-readable exports.

## Files in this folder

| File | What it is |
| --- | --- |
| `verification.json` | Self-check export from the page (`serializeVerification()`), 12 checks. |
| `results.csv` | Retrieval metrics per method/alpha (`serializeResults()`). |
| `recommendations.csv` | Top-10 lists for users 1 and 2 across methods (`serializeRecommendations()`). |
| `README.md` | This file. |

## How to run

The app must be served over HTTP from the `week6/` directory (it fetches
`u.data`, `u.item` and `mf_checkpoint.json`; opening `index.html` as a `file://`
URL is blocked by the browser). From the repository root:

```sh
cd week6
python3 -m http.server 8000
# open http://127.0.0.1:8000/index.html
```

Press the buttons in this exact order (the order matters):

1. **Build graph** — builds the bipartite graph (the data is loaded and split
   chronologically during page load, before this button).
2. **Run tests** — the 12 self-checks (all PASS).
3. **Run alpha sweep (validation)** — PPR Recall@10 over α ∈ {0.50, 0.70, 0.85, 0.95}; best is **0.50**.
4. **Set the α slider to 0.50.**
5. **Evaluate Recall@10 — all four methods** — the test-split comparison.
   *Evaluate reads α from the slider, so step 4 must come first; the sweep in step
   3 only picks the value, it does not apply it.*
6. **Run two-user demo (alpha / seed)** — the one-factor demonstration.
7. **Recommend Top-10** with each method (ppr, mf, popularity, pagerank) for the two users.
8. **Export verification / results / recommendations** to regenerate the three files here.

## Versions actually used

| Component | Version |
| --- | --- |
| TensorFlow.js | 4.22.0 (CDN referenced in `week6/index.html`: `@tensorflow/tfjs@4.22.0/dist/tf.min.js`) |
| TensorFlow.js backend | `webgl` (recorded in `verification.json`) |
| Google Chrome | 155.0.8059.85 (headless, via `playwright-core`) |
| Node.js | v22.23.3 |
| OpenCode | 1.18.30 |
| playwright-core (week6-checks only) | 1.64.1 |

## Where MF comes from

MF is **not** retrained for HW6. The page loads the HW5 model from
`week6/mf_checkpoint.json` and scores with `μ + p·q` (unclamped), exactly as the
page's `mfScores`.

| Field | Value |
| --- | --- |
| Source | HW5 `week5-checks/results/factors_final.json`, git commit `72c4a02` ("HW5: matrix factorization in TF.js and check scripts") |
| Selected checkpoint | epoch 3 (best validation epoch) |
| Factors | 16 |
| Global mean | 3.5806026199776437 |
| Shapes | users [944, 16], movies [1683, 16] |
| Validation RMSE (clamped to [1,5]) | 1.0160939234475768 |
| Test RMSE (clamped to [1,5]) | 1.0582837991811909 |
| `mf_checkpoint.json` SHA-1 | `02659335cfeb964e0b33823f868b2793509e3800` |

If `mf_checkpoint.json` cannot be fetched, `hw6EnsureMf` falls back to training
the built-in reference in the browser (`hw6TrainMf`); that path is not used for
the submitted numbers.

## What was implemented in the code

- `week6/graph.js`: `rowNormalize`, `buildGraph` helpers wired, and the efficient
  batched `pageRank` / `personalizedPageRank` / `personalizedPageRankBatch` by
  power iteration. Propagation is done with TensorFlow.js `tf.matMul` over the
  bipartite adjacency (`R` and `Rᵀ`) with vector degree normalization — no dense
  square `Q` and no transpose loops. Same dangling policy
  (`p_next = α(Qᵀp + d·v) + (1−α)v`), stopping rule (L1 < 1e-6 or 500 iterations)
  and `converged=false` on hitting the cap.
- PPR restart convention: on the MovieLens path the first argument is a **raw
  user id**, translated with `GRAPH.nodeOfUser`; a `null` first argument is the
  uniform restart (what `pageRank` delegates with). `personalizedPageRankBatch`
  always takes **node indices**. An unknown user id throws a clear error.
- `week6/script.js`: implemented `movieScores`, `recommendTopK`, `recallAtK`, plus
  the page-side plumbing (batched PPR cohort priming, PPR score cache) and the MF
  checkpoint loading path.
- Tensor cache is built once per graph and reused (4 live graph tensors, disposed
  on rebuild; MF adds 2 more variables once loaded, created inside `tf.tidy`);
  column vectors are built directly as 2-D tensors to avoid leaked 1-D
  intermediates.

## Framework defects left in place (deliberately) and why

- **Dataset sparsity (week6/script.js, dataset summary).** The starter computes
  `1 − 100000/((numUsers+1)·(MAX_MOVIE_ID+1))` = 93.71% using 944×1683 cells; the
  true observed matrix is 943 users × 1682 movies → 93.6953%. This is
  scaffolding display text, not part of the graded TODO(hw6) surface, so it is
  left untouched and reported here.
- **`u.item` is decoded as UTF-8 (week6/data.js), which corrupts 9 titles.** The
  file is latin1-encoded, but `data.js:47` reads it with `await
  moviesResponse.text()`, and the Fetch standard always decodes `Response.text()`
  as UTF-8 regardless of any `charset` in the HTTP header (GitHub Pages serves
  `u.item` as `application/octet-stream` anyway, so no charset is honoured). Each
  accented byte therefore decodes to `U+FFFD` (`\uFFFD`). Measured in the page
  (`probe.js`, `movies.filter(m => m.title.includes("\uFFFD"))`): **9 of 1682**
  titles are corrupted, and they are the only titles carrying non-ASCII chars:

  | movie id | title as decoded in the page |
  | --- | --- |
  | 543 | `Misérables, Les (1995)` → `Mis\uFFFDrables, Les (1995)` |
  | 1005 | `Double vie de Véronique, La …` → `Double vie de V\uFFFDronique, La …` |
  | 1104 | `C'est arrivé près de chez vous (1992)` → `C'est arriv\uFFFD pr\uFFFDs …` |
  | 1233 | `Nénette et Boni (1996)` → `N\uFFFDnette et Boni (1996)` |
  | 1252 | `Contempt (Mépris, Le) (1963)` → `Contempt (M\uFFFDpris, Le) (1963)` |
  | 1322 | `Metisse (Café au Lait) (1993)` → `Metisse (Caf\uFFFD au Lait) (1993)` |
  | 1366 | `JLG/JLG - autoportrait de décembre (1994)` → `… d\uFFFDcembre (1994)` |
  | 1623 | `Cérémonie, La (1995)` → `C\uFFFDr\uFFFDmonie, La (1995)` |
  | 1633 | `Á káldum klaka (Cold Fever) (1994)` → `\uFFFD k\uFFFDldum klaka …` |

  `data.js` is untouched scaffolding, so the page's titles stay corrupted there.
  This does **not** affect the graph, rankings or metrics (ids and ratings are
  ASCII), and none of these 9 movies appears in the submitted
  `recommendations.csv` for users 1 and 2 (verified: 0 of 100 rows). Only the
  display titles of those 9 rows would differ; the underlying IDs, scores and
  popularity match the independent reference.
- **Self-check #10 inspects popularity only.** It verifies candidate-pool /
  exclusion invariants through the popularity ranking and does not exercise the
  graph-based rankings for that particular invariant; the graph rankings are
  covered by the other checks and by the independent cross-check. Left as-is.
- **Evaluate / alpha-sweep ordering.** The alpha sweep reports the best α for
  validation but does not write it to the slider; Evaluate reads the slider. The
  UI therefore requires step 4 (set the slider) before step 5 (Evaluate). This is
  an ordering constraint of the provided controls, documented above rather than
  changed.

Two scaffolding behaviours were **corrected** because they produced wrong
numbers rather than merely awkward UI:

- **Fallback MF collapsed to the global mean.** The starter's in-browser training
  penalty was applied to a single summed gradient instead of averaged over the
  batch rows, so the model under-fit and validation RMSE stayed at ~1.1830
  (predicting the mean) for every epoch. `hw6TrainMf` now applies the HW5
  batch-row L2 penalty; the fallback reaches validation RMSE ≈ 0.982 (epoch 3).
  This path is not used for the submitted numbers (the HW5 checkpoint is).
- **Adapter node shift.** The adapter used to pass a node index to
  `personalizedPageRank` where the MovieLens convention expects a raw user id,
  shifting the restart by one node. The adapter now calls
  `personalizedPageRank(rawUserId, alpha)` first (and lets `graph.js` translate via
  `GRAPH.nodeOfUser`), falling back to the node-index form only for non-MovieLens
  graphs.

No self-check is failing and no run reported nonconvergence at the selected
settings.

## Reproducing the checks: `week6-checks/`

A separate, self-contained suite (excluded from the HW app). It relies on
`playwright-core` and a local Chrome.

```sh
cd week6-checks
npm install
node reference.js   # independent pure-JS float64 reference -> results/reference.txt(.json)
node users.js       # business comparison over the test cohort -> results/users.txt
node probe.js       # headless page probe + cross-check -> results/probe_final.txt,
                    #   and writes week6/submission/{verification.json,results.csv,recommendations.csv}
```

`reference.js` and `users.js` are pure JavaScript (no TensorFlow.js) so the page
numbers can be compared against a genuine independent implementation on the same
split and graph.

### Cross-check summary (page vs independent reference)

- Recall@10 on test — popularity 0.076023, pagerank 0.079693, ppr 0.112189: all MATCH.
- Top-10 PPR users 1 and 2: ids MATCH, max |Δscore| 1.8e-9 / 2.4e-10.
- Selected alpha on validation: 0.50 (both).

### `probe.js` measurements (defence notes)

Recorded with Chrome 155.0.8059.85, TF.js 4.22.0, backend `webgl`.

Timings (ms): page load → dataset/split ready 117; Build graph 31; Run tests
1052; alpha sweep (4 alphas) 17324; MF load cold 88; Evaluate comparison 4874.

Tensor count (`tf.memory().numTensors`): 4 after the two `pageRank(0.85)` calls, 4
after Run tests, 4 after a *second* Run tests, 4 after the alpha sweep, 6 after
Evaluate (MF loaded: 4 graph-cache + 2 MF variables), 6 after a repeated sweep +
comparison. Nothing grows on repetition; the MF variables are created inside
`tf.tidy` so the initial-value tensors are released.

Convergence (page `pageRank` / PPR user 1), all `converged=true`:

| α | pageRank it / residual | PPR(user 1) it / residual |
| --- | --- | --- |
| 0.50 | 19 / 5.49e-7 | 22 / 5.05e-7 |
| 0.70 | 36 / 7.57e-7 | 42 / 8.05e-7 |
| 0.85 | 78 / 9.54e-7 | 91 / 9.24e-7 |
| 0.95 | 243 / 9.66e-7 | 291 / 9.74e-7 |

`tf.*` API audit (source references vs the loaded runtime): `graph.js` uses 6
distinct members (`concat, matMul, min, tensor1d, tensor2d, tidy`), `script.js`
uses 17 (`Variable, clipByValue, gather, getBackend, losses, matMul, mean, mul,
nextFrame, ready, setBackend, sum, tensor1d, tensor2d, tidy, train, variable`);
every one exists in tfjs 4.22.0 — none missing.
