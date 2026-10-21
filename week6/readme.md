# Graph-Based Recommendations with Personalized PageRank

Course: LLM4Rec, HSE University
Instructor: Seungmin Jin
Task type: Individual assignment
Runtime: JavaScript and TensorFlow.js in a web browser

## Status and scope

This document defines the assignment. It does not claim that a new starter application, automated tests, or reference results on MovieLens have already been implemented.

Reuse the MovieLens 100K files and evaluation conventions from the previous matrix-factorization assignment. Do not overwrite the original `week5/` files. The directory and deadline for the new submission will be announced by the instructor.

GCN, GAT, temporal attention, and purchase-probability modeling are outside the required implementation.

## 1. Learning goals

- Construct a user-movie bipartite graph from positive training interactions.
- Explain row-normalized transitions, probability propagation, restart distributions, and stationary distributions.
- Implement PageRank and Personalized PageRank (PPR) with TensorFlow.js tensor operations.
- Turn movie-node scores into eligible Top-10 recommendations.
- Compare popularity, global PageRank, PPR, and your previous MF recommender under a shared retrieval protocol.
- Make an evidence-based deployment recommendation rather than assuming that a newer or more personalized algorithm is better.

## 2. Dataset and continuity with the previous assignment

Use `week5/u.data` and `week5/u.item`, and keep the previous per-user chronological 80%/10%/10% split. Do not overwrite the original `week5/` files. Treat missing ratings as unobserved, not as zero ratings.

Report the counts actually loaded by your application and all subsequent exclusions. The split is chronological within each user, not globally time-consistent; disclose that limitation and note that MovieLens records rating times, not viewing sessions.

## 3. Required implementation

### 3.1 Data preparation

Report users, movies, ratings, sparsity, positive training interactions, zero-degree users and movies, and the split record counts. The graph's node universe is the set of user IDs and movie IDs observed in the training split; map IDs to contiguous indices explicitly and never add a padding node to the graph or the recommendation pool.

```text
R[u, i] = 1 if user u rated movie i >= 4.0 in TRAINING
R[u, i] = 0 otherwise
```

A zero means no positive training edge; it is not a negative preference. Connect users and movies in both directions; there are no user-user or movie-movie edges.

### 3.2 Transition convention

```text
A = [[0, R],
     [R^T, 0]]
```

Normalize each nonzero row to obtain link-following probabilities. Notation: `alpha` is the probability of following a link, `1 - alpha` is the probability of restart, `v` is the normalized restart distribution, `p` is the column vector of node probabilities.

```text
p_next = alpha * P^T * p + (1 - alpha) * v
```

The transpose collects incoming probability for each destination; it does not reverse the graph's movement rules.

### 3.3 Zero-degree nodes

Do not divide by zero or silently discard probability mass. Implement the equivalent update without materializing dense dangling rows:

```text
p_next = alpha * (Q^T * p + d * v) + (1 - alpha) * v
```

where `Q` has normalized rows for positive-degree nodes and zero rows otherwise, and `d` is the current mass on zero-degree nodes. Report zero-degree counts and the policy; it must be the same rule in global PageRank and PPR.

### 3.4 Global PageRank and PPR

Global PageRank: `v` is uniform over all graph nodes. PPR for user `u`: `v` is one-hot at that user's node. Initialize `p` to `v`. Stop when the L1 difference between consecutive distributions is below `1e-6`, or after 500 iterations. Report the iteration count, final residual, and whether convergence was reached; an iteration cap is not proof of convergence. Do not force a user's final score to equal the restart probability.

### 3.5 TensorFlow.js requirements

Use TensorFlow.js for the core batched propagation. JavaScript loops are acceptable for parsing, ID mapping, formatting, and small fixtures; they must not replace the main propagation with per-edge loops. You may avoid the full square adjacency matrix by implementing its two off-diagonal blocks with `R` and `R^T`, vector degree normalization, and `tf.matMul`. Await `tf.ready()`, report `tf.getBackend()`, manage tensors with `tf.tidy()` and dispose long-lived tensors, read results asynchronously where practical, never put an async callback inside `tf.tidy()`, and make sure repeated runs do not accumulate tensors. Record runtime and backend without comparing timings across devices. PPR is an iterative calculation, not neural-network training; no optimizer is required. Write code, comments and JSDoc in English and document inputs, outputs, indexing, assumptions and failure cases.

## 4. Verification before MovieLens experiments

### 4.1 Reference graph

Node order A, B, C, D; directed edges `A -> B`, `A -> C`, `B -> C`, `C -> A`, `D -> C`. A divides its link-following probability equally between B and C; B, C and D each have one outgoing link; there are no dangling nodes.

For alpha = 0.85 and uniform restart:

```text
p = [0.3725268513, 0.1958239118, 0.3941492369, 0.0375000000]
ranking: C > A > B > D
```

For alpha = 0.85 and restart only at A:

```text
p = [0.4522328999, 0.1921989825, 0.3555681176, 0.0000000000]
ranking: A > C > B > D
```

These are numerical reference values for this specific test graph, not expected MovieLens outcomes.

### 4.2 Additional tests

Verify and report: normalized rows sum to 1; every restart distribution is nonnegative and sums to 1; iterations preserve total probability to numerical tolerance; stationary scores are nonnegative; one extra update produces approximately the same distribution; a separate zero-degree-node fixture preserves mass; graph direction and matrix orientation match; recommendation filtering removes the correct movies. Do not claim to have run checks you did not run.

## 5. From scores to Top-10 recommendations

Use all movies appearing in the training split as the candidate pool. For final TEST recommendations: extract movie-node scores only; exclude all movies the user rated in training or validation regardless of value; sort descending by score with ties broken by ascending movie ID; return up to 10; display IDs, titles, genres and scores. Report if fewer than 10 eligible candidates exist, and whether zero-score movies had to be included; do not claim that tie-broken zero-score candidates have demonstrated relevance. For VALIDATION recommendations exclude training-rated movies only. PPR scores are long-run graph-walk probabilities: not purchase probabilities, not predicted ratings, and not comparable in scale to MF scores.

## 6. Methods and experimental controls

Compare four methods: (1) Popularity — count distinct users with a positive training rating per movie; (2) Global PageRank with uniform restart; (3) PPR with restart at the target user; (4) MF — reuse your previous implementation and selected model. Do not alter the MF scoring definition to resemble PPR; state the model and checkpoint used.

### 6.1 Alpha selection

For PPR compare `alpha` in `{0.50, 0.70, 0.85, 0.95}` on validation Recall@10 and select one shared alpha for the full evaluation population; do not use test outcomes. Break exact ties by the smaller alpha. Use the same alpha for the primary global PageRank comparison and disclose this control; this is not a claim that global PageRank was independently optimized. MF keeps the previous validation-RMSE checkpoint rule; disclose that MF and PPR use different validation objectives.

### 6.2 One-factor-at-a-time demonstration

Choose two users from the common test cohort and keep their IDs fixed. Show three PPR settings: baseline (user A, alpha = 0.85), change seed only (user B, alpha = 0.85), change alpha only (user A, alpha = 0.50). Compare both changed settings with the baseline and never attribute a difference to one factor after changing both. Do not assume larger alpha always improves novelty, or that the restart seed must be the highest-scoring node.

## 7. Retrieval evaluation and popularity evidence

### 7.1 Shared evaluation cohort

Use the same candidate rules and the same users for all four methods. A user is evaluable when the user appears in training with at least one positive training edge, the selected MF implementation can score the required training-movie candidates, and at least one relevant validation/test movie remains eligible under the appropriate filtering rules. Apply the shared intersection for the primary comparison, disclose any additional restriction, and report how many users are excluded. This is a warm-user evaluation; it does not establish cold-start performance.

### 7.2 Recall@10

A relevant movie has rating >= 4.0 in the evaluation split and must belong to the training-movie candidate pool and pass that split's previously-rated exclusion rule.

```text
Recall@10 = relevant eligible movies retrieved / all relevant eligible movies
```

Report macro-average Recall@10 over the shared cohort, the cohort size, and exclusion counts; separately count relevant movies absent from the training candidate pool and users without eligible relevant movies. Do not compute rating RMSE for PPR.

### 7.3 Recommendation popularity

```text
popularity[i] = number of distinct users with training rating >= 4.0 for movie i
```

For each reported Top-10 list include each movie's popularity and the list's mean popularity, and for the two selected users report the number of common movies in their Top-10 lists within each method (descriptive evidence, not proof of better personalization). Do not treat unknown test movies as dislikes or popularity as platform-wide impressions.

## 8. Business & Algorithmic Analysis - exactly ONE question

### Should the platform keep MF, replace it with PPR, or use both?

Recommend one deployment strategy — keep MF as the main recommender, replace it with PPR, or use the two methods in different recommendation sections — supported by Recall@10 on the common cohort, the two users' Top-10 lists with titles and genres, their training histories, the training-set popularity of the recommended movies, and the observed effects of changing the restart user and alpha separately.

Do not assume a different list is better personalization, that less popular movies are automatically better discoveries, or that PPR scores are purchase probabilities. If you propose separate sections, name each section's purpose and explain why your results support the assignment. State what the offline results cannot establish about current-session intent, viewing completion, retention or revenue, and propose one online experiment before rollout specifying the intervention, primary outcome, at least one guardrail, and how user-level assignment avoids contamination. No extra business essay is required.

## 9. Deliverables

Submit: the JavaScript application and its source files with TensorFlow.js used in the core propagation; `README.md` with exact run instructions and dependency versions actually used; `verification.json` or an equivalent machine-readable test-result export; `results.csv` with retrieval metrics, common-cohort size, exclusions and selected settings; `recommendations.csv` with user ID, method, alpha where applicable, rank, movie ID, title, genres, score and training popularity; a course-format `report.pdf` covering implementation evidence and the single business question; the unmodified AI-session export required by the course, if that requirement remains in force. Preserve the original dataset and previous assignment files, and state any incomplete component, failed test, nonconverged run or unavailable prior MF output.

Confirm the current grading guide, submission directory, deadline and AI-export requirement with the instructor before publishing the final assignment. Do not invent an official numerical rubric or a deadline.

## 10. Evidence and citation integrity

Cite the original dataset and verify all paper titles, authors, years and URLs. Do not fabricate runtime measurements, experiment results, tests or online business effects. Make sure tables match the application's actual exports, and separate measured observations, proposed explanations, and proposed deployment actions.

Dataset provenance: MovieLens 100K, GroupLens, University of Minnesota. The previous assignment provides the citation: F. Maxwell Harper and Joseph A. Konstan, "The MovieLens Datasets: History and Context," ACM Transactions on Interactive Intelligent Systems, 2015. Verify the bibliographic record before including it in the submitted report.

## 11. Final checklist

- [ ] Same MovieLens files and split as the previous assignment.
- [ ] Graph built from positive training ratings only.
- [ ] All observed training/validation ratings used for final exclusion.
- [ ] Alpha consistently means link-following probability.
- [ ] Matrix orientation and dangling-mass handling tested.
- [ ] No unintended padding node or dataset mutation.
- [ ] TensorFlow.js propagation and memory management implemented.
- [ ] Shared evaluation cohort and candidates across methods.
- [ ] Validation used for selection; test used only for final evaluation.
- [ ] Popularity calculated from training data only.
- [ ] One setting changed per demonstration.
- [ ] PPR scores not presented as ratings or purchase probabilities.
- [ ] One evidence-based business decision, with limitations and an online experiment.
- [ ] All missing work and verification failures disclosed.

---

## Local run instructions (added for this starter)

TensorFlow.js is loaded from the CDN in `index.html`; d3 is vendored locally.

Serve the folder over HTTP (the data layer uses `fetch`, so `file://` will not work):

    cd week6
    python3 -m http.server 8000

Then open `http://localhost:8000/index.html`.
