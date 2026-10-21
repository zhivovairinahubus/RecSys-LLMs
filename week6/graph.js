/**
 * week6/graph.js - HW6 graph / Personalized PageRank layer.
 *
 * This is a plain (classic) script, NOT an ES module. `index.html` loads the
 * TensorFlow.js runtime from a CDN, then the vendored d3 v7 build, then
 * `data.js`, then this file, then `script.js`:
 *
 *   <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js"></script>
 *   <script src="vendor/d3.min.js"></script>
 *   <script src="data.js"></script>
 *   <script src="graph.js"></script>
 *   <script src="script.js"></script>
 *
 * `week6/data.js` (loaded before this file) exposes the MovieLens 100k data
 * layer as globals: `movies`, `ratings`, `numUsers`, `numMovies`,
 * `ratingMatrix`, `genreNames`, `SPLIT_EXCLUSIONS` and `splitByTimestamp()`.
 * This file does not mutate them.
 *
 * IMPLEMENTED HERE (HW6):
 *   1. `buildGraph(train)`            - bipartite graph over training positives.
 *   2. `rowNormalize(graph)`          - row-stochastic transition Q + dangling mask.
 *   3. `pageRank(alpha, opts)`        - uniform-restart PageRank.
 *   4. `personalizedPageRank(userId, alpha, opts)` - one-hot restart at one user.
 *
 * SCAFFOLDING (unchanged): the `REFERENCE_GRAPH` constant with the two
 * reference vectors and both rankings from readme.md section 4.1 (plus the
 * derived `REFERENCE_GRAPH.transition`, see below), the graph
 * accessors (`GRAPH`, `setGraph`, `getGraph`), the d3 graph view
 * (`renderGraphView`), the 4-node fixture teaching demo (`fixtureStep` /
 * `fixtureReset`), and the `#graph-summary-body` renderer.
 *
 * SCOPE NOTE: d3 is used ONLY by the graph view. The propagation functions use
 * TensorFlow.js tensor operations on `R` and its transpose for the MovieLens
 * graph (readme.md section 3.5); the fixture demo below is the only
 * plain-JavaScript propagation loop in this file.
 *
 * @module week6/graph
 */

// ---------------------------------------------------------------------------
// Typedefs (documentation only - no runtime effect)
// ---------------------------------------------------------------------------

/**
 * A single rating record as produced by `data.js` / `splitByTimestamp()`.
 *
 * @typedef {Object} Rating
 * @property {number} userId
 * @property {number} movieId
 * @property {number} rating    1..5 (MovieLens scale).
 * @property {number} timestamp Unix seconds.
 */

/**
 * The bipartite user-movie graph returned by `buildGraph`.
 *
 * @typedef {Object} Graph
 * @property {number} nUsers        Number of distinct user ids observed in TRAIN.
 * @property {number} nMovies       Number of distinct movie ids observed in TRAIN.
 * @property {number[]} userIds     User ids, ascending; index = local user index.
 * @property {number[]} movieIds    Movie ids, ascending; index = local movie index.
 * @property {Map<number,number>} userIndexOf  userId -> local user index [0, nUsers).
 * @property {Map<number,number>} movieIndexOf movieId -> local movie index [0, nMovies).
 * @property {number} nodeCount     nUsers + nMovies (no padding node).
 * @property {function(number):number} nodeOfUser    userId -> global node index.
 * @property {function(number):number} movieOfNode   global node index -> movieId.
 * @property {function(number):boolean} isUserNode   global node index -> user?
 * @property {Float64Array} degree  Bipartite degree over node index.
 * @property {number[]} zeroDegreeUsers  User ids with degree 0.
 * @property {number[]} zeroDegreeMovies Movie ids with degree 0.
 * @property {number} positiveEdges Number of positive training edges (1 entries in R).
 * @property {number[][]} R         R[u][i] = 1 iff user u rated movie i >= 4.0
 *   in TRAINING, else 0. Shape [nUsers][nMovies].
 */

/**
 * The row-normalized transition returned by `rowNormalize`.
 *
 * @typedef {Object} RowNormalized
 * @property {number[][]} Q             Row-stochastic transition; positive-degree
 *   rows sum to 1, zero-degree rows are all zero.
 * @property {boolean[]} danglingMask   true for zero-degree node indices.
 */

/**
 * A PageRank / PPR result.
 *
 * @typedef {Object} PageRankResult
 * @property {Float64Array} p     Node probabilities, node order, sums to ~1.
 * @property {number} iterations  Number of updates performed.
 * @property {number} residual    Final L1 difference between consecutive distributions.
 * @property {boolean} converged  true when residual < tol before the cap.
 */

/**
 * An explicit transition supplied to `pageRank` / `personalizedPageRank`
 * through `opts.transition`.
 *
 * Use this to propagate on a caller-provided graph instead of the global
 * `GRAPH` (the HW6 harness uses it to run the implementation on the 4-node
 * reference fixture, which has no MovieLens graph). When present it replaces
 * the global-graph lookup for that call.
 *
 * @typedef {Object} TransitionSpec
 * @property {number[][]} Q          Row-normalized transition; `Q[i][j]` is the
 *   probability of moving from node `i` to node `j`. Square (n x n). A
 *   positive-degree row sums to 1; a zero-degree row is all zero. Already
 *   normalized, so the implementation must NOT re-normalize it.
 * @property {Uint8Array} danglingMask Nonzero (1) exactly at the zero-degree
 *   node indices, indexed by position in the same node order as `Q`.
 * @property {string[]} [nodeLabels] Optional labels for display and tests only;
 *   the propagation never reads them. `REFERENCE_GRAPH.transition.nodeLabels`
 *   is `['A', 'B', 'C', 'D']`.
 */

/**
 * Options for `renderGraphView`.
 *
 * @typedef {Object} GraphViewOptions
 * @property {("fixture"|"ego")} [mode]   Which graph to draw.
 * @property {number|null} [seedUserId]   Seed user for the "ego" subgraph.
 * @property {("degree"|"score")} [colorBy] Node fill encoding.
 * @property {number} [maxNodes]          Cap on drawn nodes (default 120).
 * @property {function(number):number} [getScore] Node index -> score, used when
 *   colorBy === "score".
 */

// ---------------------------------------------------------------------------
// Reference graph (provided, read-only by convention)
//
// Exactly the readme.md section 4.1 fixture: node order A, B, C, D; directed
// edges A -> B, A -> C, B -> C, C -> A, D -> C; no dangling nodes. The two
// reference vectors and both rankings are reproduced verbatim from readme.md
// section 4.1.
// ---------------------------------------------------------------------------

/**
 * The 4-node reference graph from readme.md section 4.1, plus its reference
 * results. `nodes` assigns A, B to the left (user) column and C, D to the right
 * (movie) column so the fixture can reuse the same deterministic two-column
 * layout as the data view. Treat this object as read-only.
 */
const REFERENCE_GRAPH = {
  nodeOrder: ["A", "B", "C", "D"],
  nodes: [
    { index: 0, label: "A", kind: "user" },
    { index: 1, label: "B", kind: "user" },
    { index: 2, label: "C", kind: "movie" },
    { index: 3, label: "D", kind: "movie" }
  ],
  // [from, to] pairs in node index space (A = 0, B = 1, C = 2, D = 3).
  edges: [
    [0, 1], // A -> B
    [0, 2], // A -> C
    [1, 2], // B -> C
    [2, 0], // C -> A
    [3, 2]  // D -> C
  ],
  edgeLabels: ["A -> B", "A -> C", "B -> C", "C -> A", "D -> C"],
  hasDanglingNodes: false,
  reference: {
    // alpha = 0.85, uniform restart over all four nodes.
    uniform: {
      alpha: 0.85,
      restart: "uniform",
      p: [0.3725268513, 0.1958239118, 0.3941492369, 0.0375000000],
      ranking: [2, 0, 1, 3],
      rankingLabel: "C > A > B > D"
    },
    // alpha = 0.85, restart only at A.
    restartAtA: {
      alpha: 0.85,
      restart: "A",
      restartIndex: 0,
      p: [0.4522328999, 0.1921989825, 0.3555681176, 0.0000000000],
      ranking: [0, 2, 1, 3],
      rankingLabel: "A > C > B > D"
    }
  }
};

// PROVIDED: the fixture's row-normalized transition, DERIVED from
// `REFERENCE_GRAPH.edges` (never a hand-typed matrix). `Q` is what
// `pageRank` / `personalizedPageRank` must consume when the harness passes
// `opts.transition`; `danglingMask` is a Uint8Array, 1 at zero-degree node
// indices (all zeros here, since A/B/C/D all have an outgoing link).
//
// The load-time self-test below is deliberately NON-FATAL: it records state in
// `REFERENCE_GRAPH.transitionSelfTest` instead of throwing, so a malformed
// fixture can never break page load. It adds no global name.
(function () {
  const fixture = graphFixtureTransition(); // derives P from `edges`
  const n = REFERENCE_GRAPH.nodeOrder.length;
  const Q = fixture.P.map(function (row) { return row.slice(); });
  const danglingMask = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    if (fixture.out[i] === 0) danglingMask[i] = 1;
  }

  REFERENCE_GRAPH.transition = {
    Q: Q,
    danglingMask: danglingMask,
    nodeLabels: REFERENCE_GRAPH.nodeOrder.slice()
  };

  let maxRowSumError = 0;
  for (let i = 0; i < n; i++) {
    let sum = 0;
    for (let j = 0; j < n; j++) sum += Q[i][j];
    maxRowSumError = Math.max(maxRowSumError, Math.abs(sum - 1));
  }
  let danglingCount = 0;
  for (let i = 0; i < n; i++) {
    if (danglingMask[i]) danglingCount += 1;
  }
  REFERENCE_GRAPH.transitionSelfTest = {
    rows: n,
    cols: Q.length ? Q[0].length : 0,
    maxRowSumError: maxRowSumError,
    rowsSumToOne: maxRowSumError < 1e-12,
    danglingCount: danglingCount,
    danglingMaskAllZero: danglingCount === 0
  };
})();

// ---------------------------------------------------------------------------
// Graph-view theme (private)
//
// Canvas is #FFFFFF; the categorical pair is Okabe-Ito blue/orange (red-green
// colour-blind safe), and scores use a single-hue sequential blue ramp.
// ---------------------------------------------------------------------------

/**
 * Fixed visual parameters for the graph view. Private to graph.js.
 */
const graphTheme = {
  width: 1000,
  height: 420,
  margin: { top: 34, right: 110, bottom: 26, left: 110 },
  userColor: "#0072B2",
  movieColor: "#E69F00",
  edgeColor: "#b9c0ca",
  nodeStroke: "#ffffff",
  zeroStroke: "#1c2128",
  labelColor: "#1c2128",
  noteColor: "#5b6472",
  minRadius: 6,
  maxRadius: 22,
  rampLight: [232, 243, 252],
  rampDark: [8, 48, 96],
  tooltipId: "graph-tooltip"
};

// ---------------------------------------------------------------------------
// Module state (public: GRAPH_VIEW_STATE)
// ---------------------------------------------------------------------------

/**
 * Small module-level state for the graph view. Mutated only by the rendering
 * and fixture-demo code in this file.
 */
const GRAPH_VIEW_STATE = {
  mode: "fixture",
  seedUserId: null,
  colorBy: "degree",
  maxNodes: 120,
  lastRender: null,
  fixture: {
    iteration: 0,
    p: null,
    residual: null,
    restart: "uniform",
    alpha: 0.85
  }
};

// ---------------------------------------------------------------------------
// Graph accessors (public)
// ---------------------------------------------------------------------------

/**
 * The current graph, or null before `buildGraph` has been wired in.
 *
 * @type {Graph|null}
 */
let GRAPH = null;

/**
 * Replace the current graph and refresh the dependent UI.
 *
 * `script.js` should call this (rather than assigning `GRAPH` directly) so the
 * graph summary and the graph view re-render whenever the graph changes. A
 * `null` argument clears the graph and restores the empty states.
 *
 * @param {Graph|null} g
 * @returns {Graph|null} the stored graph.
 */
function setGraph(g) {
  GRAPH = g || null;
  try {
    graphRenderSummary(GRAPH);
  } catch (error) {
    graphSetLegendMessage("Graph summary failed: " + graphErrorMessage(error));
  }
  try {
    renderGraphView({ mode: GRAPH_VIEW_STATE.mode });
  } catch (error) {
    graphSetLegendMessage("Graph view failed: " + graphErrorMessage(error));
  }
  return GRAPH;
}

/**
 * Read the current graph without changing any state.
 *
 * @returns {Graph|null}
 */
function getGraph() {
  return GRAPH;
}

// ---------------------------------------------------------------------------
// Propagation core (implemented)
//
// buildGraph / rowNormalize / pageRank / personalizedPageRank implement the
// readme.md contracts. Keep the signatures and return shapes; the graph view and
// the fixture demo consume the same graph object.
// ---------------------------------------------------------------------------

/**
 * Build the bipartite user-movie graph from the TRAINING split.
 *
 * Rules (readme.md section 3.1):
 *  - The node universe is the set of user ids and movie ids observed in
 *    `train` ONLY. Map ids to contiguous indices explicitly; never add a
 *    padding node (no index-0 "dummy" row).
 *  - Node order is users first, then movies:
 *      user u  -> global node userIndexOf.get(u) in [0, nUsers)
 *      movie m -> global node nUsers + movieIndexOf.get(m) in [nUsers, nUsers + nMovies)
 *    The graph view relies on this exact order (it treats node < nUsers as a
 *    user node and indexes degree by the global node index).
 *  - R[u][i] = 1 iff user u rated movie i with rating >= 4.0 in TRAINING, else
 *    0. A zero is "no positive edge", not a negative opinion.
 *  - degree is the bipartite degree over BOTH node kinds.
 *
 * Assumptions: `train` is an array of {userId, movieId, rating, timestamp}
 * records; ratings >= 4.0 are positives. Failure cases: a non-array `train`
 * should be treated as empty (return an empty graph) rather than throwing.
 *
 * @param {Rating[]} train
 * @returns {Graph}
 */
function buildGraph(train) {
  if (!Array.isArray(train) || train.length === 0) {
    return {
      nUsers: 0,
      nMovies: 0,
      userIds: [],
      movieIds: [],
      userIndexOf: new Map(),
      movieIndexOf: new Map(),
      nodeCount: 0,
      nodeOfUser: function (userId) {
        const idx = this.userIndexOf.get(userId);
        return idx === undefined ? -1 : idx;
      },
      movieOfNode: function (node) {
        if (!this.isUserNode(node)) {
          const local = node - this.nUsers;
          if (local >= 0 && local < this.movieIds.length) return this.movieIds[local];
        }
        return undefined;
      },
      isUserNode: function (node) {
        return node >= 0 && node < this.nUsers;
      },
      degree: new Float64Array(0),
      zeroDegreeUsers: [],
      zeroDegreeMovies: [],
      positiveEdges: 0,
      R: []
    };
  }

  const userSet = new Set();
  const movieSet = new Set();
  for (let i = 0; i < train.length; i++) {
    const r = train[i];
    if (!r) continue;
    userSet.add(r.userId);
    movieSet.add(r.movieId);
  }

  const userIds = Array.from(userSet).sort((a, b) => {
    const na = Number(a), nb = Number(b);
    if (na < nb) return -1;
    if (na > nb) return 1;
    return 0;
  });
  const movieIds = Array.from(movieSet).sort((a, b) => {
    const na = Number(a), nb = Number(b);
    if (na < nb) return -1;
    if (na > nb) return 1;
    return 0;
  });

  const userIndexOf = new Map();
  for (let u = 0; u < userIds.length; u++) userIndexOf.set(userIds[u], u);
  const movieIndexOf = new Map();
  for (let m = 0; m < movieIds.length; m++) movieIndexOf.set(movieIds[m], m);

  const nUsers = userIds.length;
  const nMovies = movieIds.length;
  const nodeCount = nUsers + nMovies;

  const R = [];
  for (let u = 0; u < nUsers; u++) {
    R.push(new Array(nMovies).fill(0));
  }

  let positiveEdges = 0;
  for (let i = 0; i < train.length; i++) {
    const r = train[i];
    if (!r) continue;
    if (Number(r.rating) < 4.0) continue;
    const uIdx = userIndexOf.get(r.userId);
    const mIdx = movieIndexOf.get(r.movieId);
    if (uIdx === undefined || mIdx === undefined) continue;
    if (R[uIdx][mIdx] === 0) {
      R[uIdx][mIdx] = 1;
      positiveEdges++;
    }
  }

  const degree = new Float64Array(nodeCount);
  for (let u = 0; u < nUsers; u++) {
    let d = 0;
    const row = R[u];
    for (let m = 0; m < nMovies; m++) if (row[m]) d++;
    degree[u] = d;
  }
  for (let m = 0; m < nMovies; m++) {
    let d = 0;
    for (let u = 0; u < nUsers; u++) if (R[u][m]) d++;
    degree[nUsers + m] = d;
  }

  const zeroDegreeUsers = [];
  for (let u = 0; u < nUsers; u++) if (degree[u] === 0) zeroDegreeUsers.push(userIds[u]);
  const zeroDegreeMovies = [];
  for (let m = 0; m < nMovies; m++) if (degree[nUsers + m] === 0) zeroDegreeMovies.push(movieIds[m]);

  return {
    nUsers: nUsers,
    nMovies: nMovies,
    userIds: userIds,
    movieIds: movieIds,
    userIndexOf: userIndexOf,
    movieIndexOf: movieIndexOf,
    nodeCount: nodeCount,
    nodeOfUser: function (userId) {
      const idx = this.userIndexOf.get(userId);
      return idx === undefined ? -1 : idx;
    },
    movieOfNode: function (node) {
      if (!this.isUserNode(node)) {
        const local = node - this.nUsers;
        if (local >= 0 && local < this.movieIds.length) return this.movieIds[local];
      }
      return undefined;
    },
    isUserNode: function (node) {
      return node >= 0 && node < this.nUsers;
    },
    degree: degree,
    zeroDegreeUsers: zeroDegreeUsers,
    zeroDegreeMovies: zeroDegreeMovies,
    positiveEdges: positiveEdges,
    R: R
  };
}

/**
 * Row-normalize the transition matrix and record the zero-degree nodes.
 *
 * For each positive-degree node, divide its row by its degree so the
 * row sums to 1; zero-degree rows are left all-zero and marked in
 * `danglingMask`. The dangling mass is handled in the update exactly as in
 * readme.md section 3.3:
 *
 *   p_next = alpha * (Q^T * p + d * v) + (1 - alpha) * v
 *
 * where d is the probability mass currently sitting on zero-degree nodes.
 *
 * This helper returns plain JavaScript arrays so it can be inspected and tested;
 * the propagation functions lift them into tensors for the batched update. The
 * full transition is built from the two off-diagonal blocks (R row-normalized
 * by user degree and R-transpose row-normalized by movie degree) so the square
 * adjacency matrix is never materialized, using `tf.matMul` and a
 * `danglingMask`-derived correction for the dangling term. Q and danglingMask
 * are deliberately plain so tests can verify
 * row sums and the mask without a backend.
 *
 * Assumptions: `graph` comes from `buildGraph` (node order users then movies).
 * Failure cases: a null/missing graph returns { Q: [], danglingMask: [] }.
 *
 * @param {Graph} graph
 * @returns {RowNormalized}
 */
function rowNormalize(graph) {
  if (!graph || !graph.nodeCount) {
    return { Q: [], danglingMask: [] };
  }
  const n = graph.nodeCount;
  const nUsers = graph.nUsers || 0;
  const nMovies = graph.nMovies || 0;
  const R = graph.R || [];
  const degree = graph.degree || new Float64Array(n);

  const Q = [];
  for (let i = 0; i < n; i++) {
    Q.push(new Array(n).fill(0));
  }

  const danglingMask = [];
  for (let i = 0; i < n; i++) danglingMask.push(false);

  // Users -> Movies
  for (let u = 0; u < nUsers; u++) {
    const d = Number(degree[u] || 0);
    if (d <= 0) {
      danglingMask[u] = true;
      continue;
    }
    const row = R[u] || [];
    for (let m = 0; m < nMovies; m++) {
      if (row[m]) {
        Q[u][nUsers + m] = 1 / d;
      }
    }
  }

  // Movies -> Users
  for (let m = 0; m < nMovies; m++) {
    const nodeIdx = nUsers + m;
    const d = Number(degree[nodeIdx] || 0);
    if (d <= 0) {
      danglingMask[nodeIdx] = true;
      continue;
    }
    for (let u = 0; u < nUsers; u++) {
      if (R[u] && R[u][m]) {
        Q[nodeIdx][u] = 1 / d;
      }
    }
  }

  return { Q: Q, danglingMask: danglingMask };
}

/**
 * Global PageRank with a uniform restart distribution.
 *
 * Power iteration with a uniform restart vector v
 * (readme.md sections 3.2-3.4), using TensorFlow.js for the batched
 * propagation:
 *
 *   p_0 = v ;  p_next = alpha * (Q^T * p + d * v) + (1 - alpha) * v
 *
 * Initialize p to v. Stop when the L1 difference between consecutive
 * distributions is below `tol`, or after `maxIter` iterations; an iteration cap
 * is NOT proof of convergence, so `converged` must be false when the cap is
 * hit. Do not force any node's score to equal the restart probability.
 *
 * TRANSITION SOURCE - part of the contract:
 *  - `opts.transition` present (a {@link TransitionSpec}, e.g.
 *    `REFERENCE_GRAPH.transition`): propagate on THAT matrix and do not require
 *    the global `GRAPH`. `Q` is already row-normalized, so do not re-normalize
 *    it; read `danglingMask` by position (`danglingMask[i]` is 1 iff node `i` is
 *    zero-degree). This is how the HW6 harness runs your implementation on the
 *    4-node reference fixture.
 *  - `opts.transition` absent: behave exactly as for MovieLens - require the
 *    global `GRAPH` (from `buildGraph` + `rowNormalize`) and build the
 *    transition from its `R` / `degree`.
 *  - Neither present: throw a clear Error; never silently fall back to a
 *    uniform distribution.
 *
 * NODE-INDEX CONVENTION: node indices are global positions. For the MovieLens
 * graph they follow `buildGraph` (users `[0, nUsers)`, then movies
 * `[nUsers, nUsers + nMovies)`, no padding node). For a supplied
 * `opts.transition` they follow `Q`'s own order. `p[i]` is the probability of
 * node `i` in that same order.
 *
 * The zero-degree policy must be identical here and in `personalizedPageRank`.
 *
 * @param {number} alpha Link-following probability in [0, 1]; 1 - alpha restarts.
 * @param {{tol?: number, maxIter?: number, transition?: TransitionSpec}} [opts]
 *   Defaults tol = 1e-6, maxIter = 500, transition absent (use the global GRAPH).
 * @returns {PageRankResult} `{p, iterations, residual, converged}`: `p` is a
 *   Float64Array over node indices that sums to ~1; `iterations` is the number
 *   of updates performed; `residual` is the final L1 difference
 *   `sum_j |p_next[j] - p[j]|`; `converged` is true only when `residual < tol`
 *   before the iteration cap was reached.
 * @throws {Error} when neither `opts.transition` nor the global `GRAPH` exists.
 */
function pageRank(alpha, opts) {
  return personalizedPageRank(null, alpha, opts);
}

/**
 * Personalized PageRank with a one-hot restart at one user's node.
 *
 * Identical to `pageRank` except the restart vector v is one-hot at the restart
 * node. It accepts the same `opts.transition` override, the same stopping rule,
 * the same zero-degree policy and the same return shape
 * `{p, iterations, residual, converged}`.
 *
 * RESTART ARGUMENT - TWO CONVENTIONS, ONE PER PATH:
 *  - default MovieLens path (`opts.transition` absent): the first argument is a
 *    RAW USER ID (the MovieLens user id such as `1` or `943`), NOT a node index.
 *    It is translated to a node with `GRAPH.nodeOfUser(userId)`. This is what the
 *    real callers pass: the page adapter `hw6AdapterPPR` calls
 *    `personalizedPageRank(rawUserId, alpha)` with the raw id, and the page's
 *    two-user demonstration / recall path routes raw ids through
 *    `hw6UserNodeIndex`. Passing a node index here would seed
 *    the wrong node - e.g. raw user `943` maps to node `942` while the value
 *    `943` is the first movie node - so the raw id is the only safe convention.
 *    A `null` / `undefined` first argument is the uniform restart (no seeding):
 *    this is exactly `pageRank(alpha)`, which delegates here with `null`.
 *  - `opts.transition` path: the first argument is a NODE INDEX in `[0, Q.length)`
 *    over `Q`'s own node order, used directly. This is what the page's
 *    `opts.transition` self-checks pass (a fixture node index), and what a
 *    `TransitionSpec` carries (`REFERENCE_GRAPH.transition.nodeLabels` names the
 *    fixture's nodes for display).
 *
 * FAILURE CASES: on the MovieLens path an unknown / out-of-range raw user id must
 * throw a clear Error naming that id (never a silent uniform restart); on the
 * `opts.transition` path an out-of-range node index must throw naming the index;
 * and when neither `opts.transition` nor the global `GRAPH` is available, throw
 * instead of guessing.
 *
 * @param {?number} userId Raw MovieLens user id on the default path (or `null`
 *   for a uniform restart), or a node index on the `opts.transition` path
 *   (see RESTART ARGUMENT).
 * @param {number} alpha  Link-following probability in [0, 1]; 1 - alpha restarts.
 * @param {{tol?: number, maxIter?: number, transition?: TransitionSpec}} [opts]
 *   Same options as `pageRank`; defaults tol = 1e-6, maxIter = 500, transition
 *   absent (use the global GRAPH).
 * @returns {PageRankResult} `{p, iterations, residual, converged}` (see `pageRank`).
 * @throws {Error} when the restart user id / node index is unknown or out of
 *   range, or when neither `opts.transition` nor the global `GRAPH` exists.
 */
function personalizedPageRank(userId, alpha, opts) {
  const options = opts || {};
  let restartNode;
  if (options.transition) {
    // Node-index convention: the caller supplies a position in Q's own order.
    restartNode = userId;
  } else if (userId === null || userId === undefined) {
    // Uniform restart: this is exactly `pageRank(alpha)`, which delegates here.
    // A null restart is not a user lookup, so it stays the uniform distribution.
    restartNode = null;
  } else {
    // MovieLens convention: the caller supplies a raw user id; translate it.
    const graph = (typeof getGraph === "function") ? getGraph() : (typeof GRAPH !== "undefined" ? GRAPH : null);
    if (!graph) {
      throw new Error("personalizedPageRank: no active GRAPH and no opts.transition");
    }
    const node = (typeof graph.nodeOfUser === "function") ? Number(graph.nodeOfUser(userId)) : NaN;
    if (!Number.isFinite(node) || node < 0) {
      throw new Error("personalizedPageRank: unknown user id " + userId + " (not present in GRAPH.userIds)");
    }
    restartNode = node;
  }
  const batch = personalizedPageRankBatch([restartNode], alpha, options);
  const p = batch.vectors[0];
  const pOut = new Float64Array(p.length);
  for (let i = 0; i < p.length; i++) pOut[i] = p[i];
  return { p: pOut, iterations: batch.iterations[0], residual: batch.residual[0], converged: batch.converged[0] };
}

/**
 * Batched Personalized PageRank over NODE-INDEX restart seeds.
 *
 * The restart argument `restartIndices` is ALWAYS a list of global NODE INDICES
 * (not raw user ids): users `[0, nUsers)`, then movies `[nUsers, nUsers+nMovies)`
 * on the MovieLens path, or `Q`'s own order on the `opts.transition` path. Each
 * returned vector is one column of the final distribution `[n x B]`; the caller
 * that holds raw user ids is responsible for mapping them (see
 * `personalizedPageRank`, which does the raw-id -> node translation).
 *
 * @param {number[]} restartIndices Node indices, one per restart column.
 * @param {number} alpha
 * @param {{tol?: number, maxIter?: number, transition?: TransitionSpec}} [opts]
 * @returns {{vectors:Float64Array[], iterations:number[], residual:number[],
 *   converged:boolean[]}} One entry per restart index.
 * @throws {Error} when an index is out of range or no transition is available.
 */
function personalizedPageRankBatch(restartIndices, alpha, opts) {
  opts = opts || {};
  const tol = opts.tol === undefined ? 1e-6 : opts.tol;
  const maxIter = opts.maxIter === undefined ? 500 : opts.maxIter;
  const a = Number(alpha);
  if (typeof tf === 'undefined') throw new Error("TensorFlow.js not available");
  if (!Array.isArray(restartIndices)) restartIndices = [restartIndices];

  let n = 0;
  let isTransition = false;
  let QMatArr = null;
  let danglingMaskArr = null;
  let graph = null;

  if (opts.transition) {
    QMatArr = opts.transition.Q;
    danglingMaskArr = opts.transition.danglingMask || [];
    n = QMatArr ? QMatArr.length : 0;
    isTransition = true;
  } else {
    graph = (typeof getGraph === 'function') ? getGraph() : (typeof GRAPH !== 'undefined' ? GRAPH : null);
    if (!graph) throw new Error("personalizedPageRank requires either opts.transition or an active GRAPH");
    n = graph.nodeCount || 0;
    isTransition = false;
  }
  if (!n) throw new Error("personalizedPageRank: invalid transition");

  if (!isTransition) {
    graphPprEnsureCache(graph);
  }

  const B = restartIndices.length;
  for (let b = 0; b < B; b++) {
    const rid = restartIndices[b];
    if (rid === null || rid === undefined) continue;
    const seed = Number(rid);
    if (!Number.isFinite(seed) || seed < 0 || seed >= n) {
      throw new Error("personalizedPageRank: unknown or out of range restart index " + rid);
    }
  }

  // v and P as flat Float32Array, row-major [n x B]
  const vFlat = new Float32Array(n * B);
  for (let b = 0; b < B; b++) {
    const rid = restartIndices[b];
    if (rid === null || rid === undefined) {
      const uni = 1 / n;
      for (let i = 0; i < n; i++) vFlat[i * B + b] = uni;
    } else {
      const seed = Number(rid);
      vFlat[seed * B + b] = 1;
    }
  }
  const pFlat = Float32Array.from(vFlat);

  const iterations = new Int32Array(B).fill(0);
  const residualArr = new Float64Array(B).fill(Infinity);
  const convergedArr = new Array(B).fill(false);
  const frozen = new Array(B).fill(false);
  let totalFrozen = 0;

  while (totalFrozen < B) {
    const maxIterCur = Math.max.apply(null, Array.from(iterations));
    if (maxIterCur >= maxIter) break;

    let pNextData = null;
    const pNext = tf.tidy(() => {
      const PMat = tf.tensor2d(pFlat, [n, B]);
      const vMat = tf.tensor2d(vFlat, [n, B]);
      let out;
      if (isTransition) {
        out = graphPprStepBatchTransition(QMatArr, a, PMat, vMat, danglingMaskArr);
      } else {
        out = graphPprStepBatchMovieLens(graph, a, PMat, vMat);
      }
      return out;
    });
    pNextData = pNext.dataSync();
    pNext.dispose();

    for (let b = 0; b < B; b++) {
      if (frozen[b]) continue;
      let l1 = 0;
      for (let i = 0; i < n; i++) {
        const idx = i * B + b;
        const oldv = pFlat[idx];
        const newv = pNextData[idx];
        l1 += Math.abs(newv - oldv);
        pFlat[idx] = newv;
      }
      iterations[b]++;
      residualArr[b] = l1;
      if (l1 < tol) {
        convergedArr[b] = true;
        frozen[b] = true;
        totalFrozen++;
      } else if (iterations[b] >= maxIter) {
        convergedArr[b] = false;
        frozen[b] = true;
        totalFrozen++;
      }
    }
  }

  for (let b = 0; b < B; b++) {
    if (!frozen[b] && iterations[b] >= maxIter) {
      frozen[b] = true;
      if (convergedArr[b] !== true) convergedArr[b] = false;
    }
  }

  const vectors = [];
  for (let b = 0; b < B; b++) {
    const vcol = new Float64Array(n);
    for (let i = 0; i < n; i++) vcol[i] = pFlat[i * B + b];
    vectors.push(vcol);
  }
  const itArr = Array.from(iterations);
  const resArr = Array.from(residualArr);
  const convArr = Array.from(convergedArr);
  return { vectors: vectors, iterations: itArr, residual: resArr, converged: convArr };
}

// ---------------------------------------------------------------------------
// Small helpers (private, all graph-prefixed to avoid collisions)
// ---------------------------------------------------------------------------

/**
 * Readable message from an unknown thrown value.
 *
 * @param {*} error
 * @returns {string}
 */
function graphErrorMessage(error) {
  return String(error && error.message ? error.message : error);
}

/**
 * Escape text before inserting it into HTML (the local copy, so this file does
 * not depend on any helper defined by script.js).
 *
 * @param {*} text
 * @returns {string}
 */
function graphEscape(text) {
  return String(text)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

/**
 * Read alpha from the #alpha range input, falling back to 0.85.
 *
 * @returns {number}
 */
function graphReadAlpha() {
  const el = document.getElementById("alpha");
  const value = el ? Number(el.value) : Number.NaN;
  return Number.isFinite(value) ? value : 0.85;
}

/**
 * Map t in [0, 1] onto the single-hue sequential blue ramp.
 *
 * @param {number} t
 * @returns {string} an rgb(...) string.
 */
function graphRampColor(t) {
  const n = Number(t);
  const clamped = Math.max(0, Math.min(1, Number.isFinite(n) ? n : 0));
  const lo = graphTheme.rampLight;
  const hi = graphTheme.rampDark;
  const mix = (a, b, s) => Math.round(a + (b - a) * s);
  const r = mix(lo[0], hi[0], clamped);
  const g = mix(lo[1], hi[1], clamped);
  const b = mix(lo[2], hi[2], clamped);
  return "rgb(" + r + ", " + g + ", " + b + ")";
}

/**
 * Position lookup with a safe fallback for a missing node.
 *
 * @param {Map<number,{x:number,y:number}>} positions
 * @param {number} node
 * @returns {{x:number,y:number}}
 */
function graphPos(positions, node) {
  const pt = positions.get(node);
  return pt || { x: 0, y: 0 };
}

// ---------------------------------------------------------------------------
// Fixture demo helpers (private)
// ---------------------------------------------------------------------------

/**
 * Out-degree of a fixture node (number of outgoing reference edges).
 *
 * @param {number} index
 * @returns {number}
 */
function graphFixtureOutDegree(index) {
  let d = 0;
  for (const edge of REFERENCE_GRAPH.edges) {
    if (edge[0] === index) d += 1;
  }
  return d;
}

/**
 * Build the row-stochastic transition for the 4-node reference fixture.
 *
 * @returns {{P: number[][], out: number[]}} P[from][to] and the out-degrees.
 */
function graphFixtureTransition() {
  const n = REFERENCE_GRAPH.nodes.length;
  const out = new Array(n).fill(0);
  for (const edge of REFERENCE_GRAPH.edges) out[edge[0]] += 1;

  const P = [];
  for (let i = 0; i < n; i++) P.push(new Array(n).fill(0));
  for (const edge of REFERENCE_GRAPH.edges) {
    const from = edge[0];
    const to = edge[1];
    if (out[from] > 0) P[from][to] = 1 / out[from];
  }
  return { P, out };
}

/**
 * Restart distribution for the fixture: uniform, or one-hot at a named node.
 *
 * @param {string} restart "uniform" or one of A/B/C/D.
 * @returns {number[]}
 */
function graphFixtureRestartVector(restart) {
  const n = REFERENCE_GRAPH.nodes.length;
  const v = new Array(n).fill(0);
  if (!restart || restart === "uniform") {
    v.fill(1 / n);
    return v;
  }
  const index = REFERENCE_GRAPH.nodeOrder.indexOf(restart);
  if (index < 0) {
    v.fill(1 / n);
    return v;
  }
  v[index] = 1;
  return v;
}

/**
 * One plain-JavaScript propagation step on the 4-node fixture.
 *
 * The fixture has no dangling nodes, so the d * v term from readme.md
 * section 3.3 is not needed here. Returns the new distribution and the L1
 * change.
 *
 * @param {number} alpha
 * @param {string} restart
 * @param {number[]} pPrev
 * @returns {{p: number[], residual: number}}
 */
function graphFixturePropagate(alpha, restart, pPrev) {
  const transition = graphFixtureTransition();
  const P = transition.P;
  const n = REFERENCE_GRAPH.nodes.length;
  const v = graphFixtureRestartVector(restart);
  const pNext = new Array(n).fill(0);

  // p_next[j] = alpha * sum_i p[i] * P[i][j] + (1 - alpha) * v[j]
  for (let i = 0; i < n; i++) {
    const mass = pPrev[i];
    if (!mass) continue;
    for (let j = 0; j < n; j++) {
      if (P[i][j] !== 0) pNext[j] += alpha * mass * P[i][j];
    }
  }
  for (let j = 0; j < n; j++) pNext[j] += (1 - alpha) * v[j];

  let residual = 0;
  for (let j = 0; j < n; j++) residual += Math.abs(pNext[j] - pPrev[j]);
  return { p: pNext, residual };
}

// ---------------------------------------------------------------------------
// View builders (private)
// ---------------------------------------------------------------------------

/**
 * Build the drawable view for the 4-node fixture.
 *
 * @returns {{kind:string, nodes:Object[], edges:Object[], omitted:number,
 *            nodeCount:number, note:string}}
 */
function graphBuildFixtureView() {
  const fixture = GRAPH_VIEW_STATE.fixture;
  const p = fixture.p || graphFixtureRestartVector(fixture.restart || "uniform");
  const nodes = REFERENCE_GRAPH.nodes.map((nd) => ({
    node: nd.index,
    key: "fixture:" + nd.label,
    label: nd.label,
    kind: nd.kind,
    id: nd.label,
    degree: graphFixtureOutDegree(nd.index),
    score: p[nd.index]
  }));
  const edges = REFERENCE_GRAPH.edges.map((edge) => ({
    source: edge[0],
    target: edge[1],
    directed: true
  }));
  return {
    kind: "fixture",
    nodes: nodes,
    edges: edges,
    omitted: 0,
    nodeCount: nodes.length,
    note: "4-node reference fixture (A, B, C, D) - no dangling nodes"
  };
}

/**
 * Neighbours of a node in the bipartite graph (users <-> movies), or [] when
 * the graph is missing. Uses the users-then-movies node order from `buildGraph`.
 *
 * @param {Graph} graph
 * @param {number} node
 * @returns {number[]}
 */
function graphNeighbors(graph, node) {
  const result = [];
  if (!graph || !graph.R) return result;
  if (graph.isUserNode(node)) {
    const row = graph.R[node];
    if (!row) return result;
    for (let i = 0; i < row.length; i++) {
      if (row[i]) result.push(graph.nUsers + i);
    }
  } else {
    const movieLocal = node - graph.nUsers;
    for (let u = 0; u < graph.nUsers; u++) {
      const row = graph.R[u];
      if (row && row[movieLocal]) result.push(u);
    }
  }
  return result;
}

/**
 * Build the capped ego subgraph for a seed user.
 *
 * Selection: the seed user's node plus its neighbours and their neighbours (a
 * 2-hop walk) in the bipartite graph, then apply the maxNodes cap by hop layer
 * (seed, then 1-hop, then 2-hop, each sorted by ascending node index) so the
 * connected core survives. `omitted` is how many reachable nodes the cap
 * left out, so the legend can report it.
 *
 * @param {number|null} seedUserId
 * @param {function(number):number} [getScore]
 * @returns {{kind:string, nodes:Object[], edges:Object[], omitted:number,
 *            nodeCount:number, note:string}}
 */
function graphBuildEgoView(seedUserId, getScore) {
  const graph = GRAPH;
  if (!graph || !graph.nodeCount) {
    return {
      kind: "empty",
      nodes: [],
      edges: [],
      omitted: 0,
      nodeCount: 0,
      note: "No graph yet - build the graph to draw the ego subgraph."
    };
  }

  const seed = seedUserId === null || seedUserId === undefined ? null : Number(seedUserId);
  if (seed === null || !graph.userIndexOf || !graph.userIndexOf.has(seed)) {
    return {
      kind: "empty",
      nodes: [],
      edges: [],
      omitted: 0,
      nodeCount: graph.nodeCount,
      note: "Pick a seed user (click a user node) to draw its ego subgraph."
    };
  }

  const maxNodes = Math.max(1, Math.floor(GRAPH_VIEW_STATE.maxNodes) || 120);
  const startNode = graph.nodeOfUser(seed);

  // Unbounded 2-hop reachable set, kept per layer so the cap can prefer nearer
  // nodes. On the bipartite graph the seed's 1-hop neighbours are its movies and
  // the 2-hop ones are co-users; all user node indices are smaller than every
  // movie index, so capping in plain node-index order would keep only users and
  // draw an edgeless column. Layer order (seed, then 1-hop, then 2-hop) keeps a
  // connected subgraph: the seed and its movies always make the cut.
  const reachable = new Set([startNode]);
  const layers = []; // layers[d] = nodes first reached at depth d (1-based hops)
  let frontier = [startNode];
  for (let depth = 0; depth < 2; depth++) {
    const next = [];
    for (const node of frontier) {
      for (const nb of graphNeighbors(graph, node)) {
        if (!reachable.has(nb)) {
          reachable.add(nb);
          next.push(nb);
        }
      }
    }
    layers.push(next);
    frontier = next;
    if (reachable.size > 500000) break; // hard safety valve
  }

  const ordered = [startNode];
  for (const layer of layers) {
    ordered.push(...layer.slice().sort((a, b) => a - b));
  }
  const selected = new Set(ordered.slice(0, maxNodes));
  const omitted = reachable.size - selected.size;

  const nodes = [];
  for (const node of ordered) {
    if (!selected.has(node)) continue;
    const isUser = graph.isUserNode(node);
    const id = isUser ? graph.userIds[node] : graph.movieOfNode(node);
    let score = null;
    if (typeof getScore === "function") {
      const raw = Number(getScore(node));
      score = Number.isFinite(raw) ? raw : null;
    }
    nodes.push({
      node: node,
      key: "node:" + node,
      label: isUser ? "U" + id : "M" + id,
      kind: isUser ? "user" : "movie",
      id: id,
      degree: graph.degree ? graph.degree[node] : 0,
      score: score
    });
  }

  const edges = [];
  for (const a of nodes) {
    if (a.kind !== "user") continue;
    const uLocal = graph.userIndexOf.get(a.id);
    const row = graph.R ? graph.R[uLocal] : null;
    if (!row) continue;
    for (const b of nodes) {
      if (b.kind !== "movie") continue;
      const mLocal = graph.movieIndexOf.get(b.id);
      if (mLocal === undefined) continue;
      if (row[mLocal]) edges.push({ source: a.node, target: b.node, directed: false });
    }
  }

  return {
    kind: "ego",
    nodes: nodes,
    edges: edges,
    omitted: omitted,
    nodeCount: graph.nodeCount,
    note: "Ego subgraph of user " + seed + " - drawn " + nodes.length + " of " + graph.nodeCount + " nodes"
  };
}

/**
 * Deterministic two-column bipartite layout: users on the left, movies on the
 * right, evenly spaced top-to-bottom in ascending node order.
 *
 * @param {Object[]} nodes
 * @param {number} width
 * @param {number} height
 * @param {{top:number,right:number,bottom:number,left:number}} margin
 * @returns {{positions: Map<number,{x:number,y:number}>, leftX:number, rightX:number}}
 */
function graphLayoutTwoColumn(nodes, width, height, margin) {
  const positions = new Map();
  const usableWidth = width - margin.left - margin.right;
  const usableHeight = height - margin.top - margin.bottom;
  const leftX = margin.left + usableWidth * 0.2;
  const rightX = margin.left + usableWidth * 0.8;

  const place = (list, x) => {
    const n = list.length;
    list.forEach((item, i) => {
      const y = n <= 1 ? margin.top + usableHeight / 2 : margin.top + (usableHeight * i) / (n - 1);
      positions.set(item.node, { x: x, y: y });
    });
  };

  const users = nodes.filter((n) => n.kind === "user").sort((a, b) => a.node - b.node);
  const movies = nodes.filter((n) => n.kind === "movie").sort((a, b) => a.node - b.node);
  const others = nodes
    .filter((n) => n.kind !== "user" && n.kind !== "movie")
    .sort((a, b) => a.node - b.node);

  place(users, leftX);
  place(movies, rightX);
  place(others, leftX);

  return { positions: positions, leftX: leftX, rightX: rightX };
}

// ---------------------------------------------------------------------------
// d3 drawing (private)
// ---------------------------------------------------------------------------

/**
 * Draw one view (fixture or ego) into the #graph-view SVG element.
 *
 * Read-only with respect to GRAPH and any score arrays: it only reads nodes,
 * edges and score values. Zero-degree nodes are drawn as hollow rings with a
 * label so the dangling case is visible.
 *
 * @param {SVGElement} svgEl
 * @param {Object} view result of `graphBuildFixtureView` / `graphBuildEgoView`.
 * @param {{scoreAccessor: (function(number):number|null)|null, colorByScore: boolean}} opts
 * @returns {void}
 */
function graphDrawView(svgEl, view, opts) {
  const scoreAccessor = opts && typeof opts.scoreAccessor === "function" ? opts.scoreAccessor : null;
  const colorByScore = !!(opts && opts.colorByScore);
  const width = graphTheme.width;
  const height = graphTheme.height;
  const margin = graphTheme.margin;

  const svg = d3.select(svgEl);
  svg.selectAll("*").remove();
  svg.attr("viewBox", "0 0 " + width + " " + height);
  svg.attr("preserveAspectRatio", "xMidYMid meet");

  const defs = svg.append("defs");
  defs
    .append("marker")
    .attr("id", "graph-arrowhead")
    .attr("viewBox", "0 -5 10 10")
    .attr("refX", 10)
    .attr("refY", 0)
    .attr("markerWidth", 6)
    .attr("markerHeight", 6)
    .attr("orient", "auto")
    .append("path")
    .attr("d", "M0,-5L10,0L0,5")
    .attr("fill", graphTheme.edgeColor);

  if (!view || view.nodes.length === 0) {
    svg
      .append("text")
      .attr("x", width / 2)
      .attr("y", height / 2)
      .attr("text-anchor", "middle")
      .attr("font-size", 15)
      .attr("fill", graphTheme.noteColor)
      .text(view && view.note ? view.note : "Nothing to draw yet.");
    return;
  }

  const layout = graphLayoutTwoColumn(view.nodes, width, height, margin);
  const positions = layout.positions;

  svg
    .append("text")
    .attr("x", layout.leftX)
    .attr("y", margin.top - 12)
    .attr("text-anchor", "middle")
    .attr("font-size", 12)
    .attr("fill", graphTheme.noteColor)
    .text("users");
  svg
    .append("text")
    .attr("x", layout.rightX)
    .attr("y", margin.top - 12)
    .attr("text-anchor", "middle")
    .attr("font-size", 12)
    .attr("fill", graphTheme.noteColor)
    .text("movies");

  // Resolve scores and the shared scales.
  const scoreByNode = new Map();
  let maxScore = 0;
  for (const n of view.nodes) {
    let score = null;
    if (scoreAccessor) {
      const raw = Number(scoreAccessor(n.node));
      score = Number.isFinite(raw) ? raw : null;
    } else if (n.score !== null && n.score !== undefined) {
      const raw = Number(n.score);
      score = Number.isFinite(raw) ? raw : null;
    }
    scoreByNode.set(n.node, score);
    if (score !== null && score > maxScore) maxScore = score;
  }
  const maxDegree = view.nodes.reduce((m, n) => Math.max(m, Number(n.degree) || 0), 1);

  const normFor = (n) => {
    if (colorByScore) {
      const score = scoreByNode.get(n.node);
      return score !== null && maxScore > 0 ? Math.min(1, score / maxScore) : 0;
    }
    const deg = Math.max(0, Number(n.degree) || 0);
    return maxDegree > 0 ? Math.min(1, deg / maxDegree) : 0;
  };
  const radiusFor = (n) =>
    graphTheme.minRadius +
    (graphTheme.maxRadius - graphTheme.minRadius) * Math.sqrt(normFor(n));

  // Edges first so nodes sit on top.
  svg
    .append("g")
    .attr("class", "graph-edges")
    .selectAll("line")
    .data(view.edges)
    .join("line")
    .attr("x1", (d) => graphPos(positions, d.source).x)
    .attr("y1", (d) => graphPos(positions, d.source).y)
    .attr("x2", (d) => graphPos(positions, d.target).x)
    .attr("y2", (d) => graphPos(positions, d.target).y)
    .attr("stroke", graphTheme.edgeColor)
    .attr("stroke-width", 1)
    .attr("marker-end", (d) => (d.directed ? "url(#graph-arrowhead)" : null));

  const nodeSel = svg
    .append("g")
    .attr("class", "graph-nodes")
    .selectAll("g.graph-node")
    .data(view.nodes, (d) => d.key)
    .join("g")
    .attr("class", "graph-node")
    .attr("transform", (d) => {
      const pt = graphPos(positions, d.node);
      return "translate(" + pt.x + "," + pt.y + ")";
    })
    .style("cursor", "pointer");

  nodeSel.each(function eachNode(d) {
    const group = d3.select(this);
    const degree = Math.max(0, Number(d.degree) || 0);
    const isZero = degree === 0;
    const radius = radiusFor(d);
    const score = scoreByNode.get(d.node);

    let fill = d.kind === "user" ? graphTheme.userColor : graphTheme.movieColor;
    if (colorByScore) fill = graphRampColor(normFor(d));

    group
      .append("circle")
      .attr("r", radius)
      .attr("fill", isZero ? "#ffffff" : fill)
      .attr("stroke", isZero ? graphTheme.zeroStroke : graphTheme.nodeStroke)
      .attr("stroke-width", isZero ? 2.5 : 1.5);

    // Every fixture node is labelled; in the ego view only zero-degree nodes and
    // the seed user are labelled to keep a dense view readable.
    const showLabel =
      view.kind === "fixture" ||
      isZero ||
      (view.kind === "ego" && d.kind === "user" && d.id === GRAPH_VIEW_STATE.seedUserId);

    if (showLabel) {
      group
        .append("text")
        .attr("x", 0)
        .attr("y", -(radius + 4))
        .attr("text-anchor", "middle")
        .attr("font-size", 11)
        .attr("fill", graphTheme.labelColor)
        .attr("pointer-events", "none")
        .text(d.label);
    }

    group.on("mouseenter", (event) => graphShowTooltip(event, d, score));
    group.on("mousemove", (event) => graphShowTooltip(event, d, score));
    group.on("mouseleave", graphHideTooltip);
    group.on("click", () => graphHandleNodeClick(d, view));
  });
}

// ---------------------------------------------------------------------------
// Tooltip (private)
// ---------------------------------------------------------------------------

/**
 * Lazily create and return the single shared tooltip element.
 *
 * @returns {HTMLElement}
 */
function graphEnsureTooltip() {
  let el = document.getElementById(graphTheme.tooltipId);
  if (el) return el;
  el = document.createElement("div");
  el.id = graphTheme.tooltipId;
  el.setAttribute("role", "tooltip");
  el.style.position = "absolute";
  el.style.pointerEvents = "none";
  el.style.background = "#1c2128";
  el.style.color = "#ffffff";
  el.style.padding = "0.35rem 0.5rem";
  el.style.borderRadius = "6px";
  el.style.fontSize = "0.78rem";
  el.style.lineHeight = "1.35";
  el.style.zIndex = "10";
  el.style.whiteSpace = "nowrap";
  el.style.display = "none";
  document.body.appendChild(el);
  return el;
}

/**
 * Show the hover tooltip for a node: id, kind, degree and current score.
 *
 * @param {MouseEvent} event
 * @param {{id:(number|string), kind:string, node:number, degree:number}} d
 * @param {number|null} score
 * @returns {void}
 */
function graphShowTooltip(event, d, score) {
  const el = graphEnsureTooltip();
  const kind = d.kind === "user" ? "user" : "movie";
  const scoreText = score === null || score === undefined || !Number.isFinite(Number(score))
    ? "-"
    : Number(score).toFixed(6);
  el.textContent =
    kind + " " + d.id + "  |  node " + d.node + "  |  degree " + d.degree + "  |  score " + scoreText;
  el.style.display = "block";
  const left = (event && event.pageX ? event.pageX : 0) + 12;
  const top = (event && event.pageY ? event.pageY : 0) + 12;
  el.style.left = left + "px";
  el.style.top = top + "px";
}

/**
 * Hide the hover tooltip.
 *
 * @returns {void}
 */
function graphHideTooltip() {
  const el = document.getElementById(graphTheme.tooltipId);
  if (el) el.style.display = "none";
}

/**
 * Handle a click on a node: in the fixture mode it moves the restart node; in
 * the ego mode it selects a seed user. Never mutates GRAPH or any score array.
 *
 * @param {{label:string, kind:string, id:(number|string)}} d
 * @param {{kind:string}} view
 * @returns {void}
 */
function graphHandleNodeClick(d, view) {
  if (view.kind === "fixture") {
    GRAPH_VIEW_STATE.fixture.restart = d.label;
    fixtureReset();
    return;
  }
  if (view.kind === "ego" && d.kind === "user") {
    GRAPH_VIEW_STATE.seedUserId = Number(d.id);
    renderGraphView({ mode: "ego", seedUserId: Number(d.id) });
  }
}

// ---------------------------------------------------------------------------
// Legend and controls (private)
// ---------------------------------------------------------------------------

/**
 * A coloured square swatch for the legend.
 *
 * @param {string} color
 * @returns {string} HTML
 */
function graphSwatch(color) {
  return '<span class="legend-swatch" style="background:' + color + '"></span>';
}

/**
 * A hollow-ring swatch for the zero-degree legend entry.
 *
 * @returns {string} HTML
 */
function graphRingSwatch() {
  return (
    '<span class="legend-swatch" style="background:#ffffff;border:2px solid ' +
    graphTheme.zeroStroke +
    ';border-radius:50%"></span>'
  );
}

/**
 * A gradient swatch showing the single-hue sequential ramp.
 *
 * @returns {string} HTML
 */
function graphRampSwatch() {
  return (
    '<span class="legend-swatch" style="background:linear-gradient(90deg, ' +
    graphRampColor(0) +
    ", " +
    graphRampColor(1) +
    ')"></span>'
  );
}

/**
 * One legend row: swatch (may be empty) + text.
 *
 * @param {string} swatchHtml
 * @param {string} text
 * @returns {string} HTML
 */
function graphLegendItem(swatchHtml, text) {
  return (
    '<span class="legend-item">' + swatchHtml + "<span>" + graphEscape(text) + "</span></span>"
  );
}

/**
 * Replace the legend with a plain message (used for errors / unavailable d3).
 *
 * @param {string} message
 * @returns {void}
 */
function graphSetLegendMessage(message) {
  const target = document.getElementById("graph-legend");
  if (!target) return;
  target.innerHTML = '<p class="empty-state">' + graphEscape(message) + "</p>";
}

/**
 * Write one line per colour/size meaning into #graph-legend.
 *
 * @param {Object} view
 * @param {{colorByScore: boolean}} opts
 * @returns {void}
 */
function graphRenderLegend(view, opts) {
  const target = document.getElementById("graph-legend");
  if (!target) return;
  const colorByScore = !!(opts && opts.colorByScore);
  const items = [];

  if (view && view.kind === "fixture") {
    items.push(graphLegendItem(graphSwatch(graphTheme.userColor), "user node (A, B)"));
    items.push(graphLegendItem(graphSwatch(graphTheme.movieColor), "movie node (C, D)"));
    items.push(graphLegendItem(graphRampSwatch(), "fill = current probability mass (single-hue ramp)"));
    items.push(graphLegendItem(graphRingSwatch(), "hollow ring = zero-degree node"));
    items.push(graphLegendItem("", "arrow = link direction; this fixture has no dangling nodes"));
    items.push(graphLegendItem("", "radius = outgoing degree; click a node to restart there"));
    target.innerHTML = items.join("");
    return;
  }

  if (!view || view.kind === "empty") {
    const note = view && view.note ? view.note : "No graph to draw yet.";
    target.innerHTML = '<p class="empty-state">' + graphEscape(note) + "</p>";
    return;
  }

  items.push(graphLegendItem(graphSwatch(graphTheme.userColor), "user node (left column)"));
  items.push(graphLegendItem(graphSwatch(graphTheme.movieColor), "movie node (right column)"));
  if (colorByScore) {
    items.push(graphLegendItem(graphRampSwatch(), "fill = score (single-hue sequential ramp)"));
  } else {
    items.push(graphLegendItem("", "fill = node kind; switch colour-by to score to use the ramp"));
  }
  items.push(graphLegendItem(graphRingSwatch(), "hollow ring with label = zero-degree node"));
  items.push(
    graphLegendItem(
      "",
      colorByScore ? "radius scales with score" : "radius scales with degree"
    )
  );
  if (view.omitted > 0) {
    items.push(
      graphLegendItem(
        "",
        view.omitted + " node(s) omitted by the " + GRAPH_VIEW_STATE.maxNodes + "-node cap"
      )
    );
  }
  target.innerHTML = items.join("");
}

/**
 * Fill #graph-controls with the view-mode controls.
 *
 * @returns {void}
 */
function graphRenderControls() {
  const target = document.getElementById("graph-controls");
  if (!target) return;
  const state = GRAPH_VIEW_STATE;

  const modeOptions = ["fixture", "ego"]
    .map((m) => '<option value="' + m + '"' + (state.mode === m ? " selected" : "") + ">" + m + "</option>")
    .join("");
  const colorOptions = ["degree", "score"]
    .map((c) => '<option value="' + c + '"' + (state.colorBy === c ? " selected" : "") + ">" + c + "</option>")
    .join("");
  const restart = state.fixture.restart || "uniform";
  const restartOptions = ["uniform"].concat(REFERENCE_GRAPH.nodeOrder)
    .map((r) => '<option value="' + r + '"' + (restart === r ? " selected" : "") + ">" + r + "</option>")
    .join("");

  const seedText = state.seedUserId === null ? "- (click a user node)" : String(state.seedUserId);

  target.innerHTML =
    '<label class="legend-item">View <select id="graph-mode-select">' + modeOptions + "</select></label>" +
    '<label class="legend-item">Colour by <select id="graph-colorby-select"' +
    (state.mode === "fixture" ? " disabled" : "") +
    ">" + colorOptions + "</select></label>" +
    '<label class="legend-item">Fixture restart <select id="graph-restart-select"' +
    (state.mode === "fixture" ? "" : " disabled") +
    ">" + restartOptions + "</select></label>" +
    '<span class="hint">Seed user: ' + graphEscape(seedText) + "</span>";

  const modeSelect = document.getElementById("graph-mode-select");
  if (modeSelect) {
    modeSelect.addEventListener("change", () => renderGraphView({ mode: modeSelect.value }));
  }
  const colorSelect = document.getElementById("graph-colorby-select");
  if (colorSelect) {
    colorSelect.addEventListener("change", () => renderGraphView({ colorBy: colorSelect.value }));
  }
  const restartSelect = document.getElementById("graph-restart-select");
  if (restartSelect) {
    restartSelect.addEventListener("change", () => {
      GRAPH_VIEW_STATE.fixture.restart = restartSelect.value;
      fixtureReset();
    });
  }
}

// ---------------------------------------------------------------------------
// Graph summary (private)
// ---------------------------------------------------------------------------

/**
 * Numeric value of a count-or-list field (arrays report their length).
 *
 * @param {*} value
 * @returns {number}
 */
function graphCount(value) {
  if (Array.isArray(value)) return value.length;
  const n = Number(value);
  return Number.isFinite(n) ? n : 0;
}

/**
 * Render the #graph-summary-body panel: nodes, edges and zero-degree counts.
 *
 * @param {Graph|null} graph
 * @returns {void}
 */
function graphRenderSummary(graph) {
  const target = document.getElementById("graph-summary-body");
  if (!target) return;
  if (!graph) {
    target.innerHTML =
      '<p class="empty-state">No graph yet. Press "Build graph" to report nodes, edges, ' +
      "zero-degree users, zero-degree movies and positive training interactions.</p>";
    return;
  }
  const nUsers = graphCount(graph.nUsers);
  const nMovies = graphCount(graph.nMovies);
  const nodeCount = graphCount(graph.nodeCount) || nUsers + nMovies;
  const positiveEdges = graphCount(graph.positiveEdges);
  const zeroUsers = graphCount(graph.zeroDegreeUsers);
  const zeroMovies = graphCount(graph.zeroDegreeMovies);

  target.innerHTML =
    '<dl class="summary-list">' +
    "<dt>User nodes</dt><dd>" + nUsers.toLocaleString("en-US") + "</dd>" +
    "<dt>Movie nodes</dt><dd>" + nMovies.toLocaleString("en-US") + "</dd>" +
    "<dt>Nodes (no padding node)</dt><dd>" + nodeCount.toLocaleString("en-US") + "</dd>" +
    "<dt>Positive training edges</dt><dd>" + positiveEdges.toLocaleString("en-US") + "</dd>" +
    "<dt>Zero-degree users</dt><dd>" + zeroUsers.toLocaleString("en-US") + "</dd>" +
    "<dt>Zero-degree movies</dt><dd>" + zeroMovies.toLocaleString("en-US") + "</dd>" +
    "</dl>";
}

// ---------------------------------------------------------------------------
// Graph view (public: renderGraphView)
// ---------------------------------------------------------------------------

/**
 * Render the graph view into #graph-view, #graph-legend and #graph-controls.
 *
 * Merges `opts` into the module state, builds either the deterministic 4-node
 * fixture view or the capped ego subgraph of the seed user, and draws it with
 * d3 v7 in a two-column bipartite layout. The view is read-only: it never
 * modifies GRAPH or any score array.
 *
 * Degrades gracefully: with no d3, no graph, or no seed user it writes a plain
 * message instead of throwing; with GRAPH still null (graph not yet built) the
 * fixture demo still draws.
 *
 * @param {GraphViewOptions} [opts]
 * @returns {(Object|null)} a small description of the last render, or null.
 */
function renderGraphView(opts) {
  const o = opts || {};
  if (o.mode === "fixture" || o.mode === "ego") GRAPH_VIEW_STATE.mode = o.mode;
  if (o.seedUserId !== undefined) {
    GRAPH_VIEW_STATE.seedUserId = o.seedUserId === null ? null : Number(o.seedUserId);
  }
  if (o.colorBy === "degree" || o.colorBy === "score") GRAPH_VIEW_STATE.colorBy = o.colorBy;
  if (o.maxNodes !== undefined && o.maxNodes !== null && Number.isFinite(Number(o.maxNodes))) {
    GRAPH_VIEW_STATE.maxNodes = Math.max(1, Math.floor(Number(o.maxNodes)));
  }

  const svgEl = document.getElementById("graph-view");
  if (!svgEl) return null;
  if (typeof d3 === "undefined") {
    graphSetLegendMessage("d3 is not available, so the graph view cannot be drawn. Check that vendor/d3.min.js loaded.");
    return null;
  }

  const mode = GRAPH_VIEW_STATE.mode;
  const external = typeof o.getScore === "function" ? o.getScore : null;
  let view;
  let scoreAccessor = null;
  let colorByScore = false;

  if (mode === "fixture") {
    view = graphBuildFixtureView();
    scoreAccessor = (node) => {
      const p = GRAPH_VIEW_STATE.fixture.p;
      return p ? p[node] : 0;
    };
    colorByScore = true;
  } else {
    view = graphBuildEgoView(GRAPH_VIEW_STATE.seedUserId, external);
    if (external && GRAPH_VIEW_STATE.colorBy === "score") {
      scoreAccessor = external;
      colorByScore = true;
    }
  }

  try {
    graphDrawView(svgEl, view, { scoreAccessor: scoreAccessor, colorByScore: colorByScore });
  } catch (error) {
    try {
      d3.select(svgEl).selectAll("*").remove();
    } catch (ignored) {
      /* the SVG may not be an element we can clear */
    }
    graphSetLegendMessage("Graph view could not be drawn: " + graphErrorMessage(error));
  }

  graphRenderLegend(view, { colorByScore: colorByScore });
  graphRenderControls();

  let userCount = 0;
  let movieCount = 0;
  for (const node of view.nodes) {
    if (node.kind === "user") userCount++;
    else if (node.kind === "movie") movieCount++;
  }
  GRAPH_VIEW_STATE.lastRender = {
    mode: mode,
    seedUserId: GRAPH_VIEW_STATE.seedUserId,
    colorByScore: colorByScore,
    nodeCount: view.nodes.length,
    userCount: userCount,
    movieCount: movieCount,
    edgeCount: view.edges.length,
    omitted: view.omitted || 0,
    note: view.note || ""
  };
  return GRAPH_VIEW_STATE.lastRender;
}

// ---------------------------------------------------------------------------
// Fixture teaching demo (public: fixtureStep / fixtureReset)
//
// THIS DEMO EXISTS ONLY FOR THE 4-NODE FIXTURE. It is a tiny plain-JavaScript
// propagation so the animation is self-contained, and it is NOT a template for
// the MovieLens implementation. The MovieLens propagation uses TensorFlow.js
// tensor operations on R and R-transpose (readme.md section 3.5), as required by
// pageRank / personalizedPageRank.
// ---------------------------------------------------------------------------

/**
 * Update #convergence-results with the fixture iteration and its L1 residual.
 *
 * @returns {void}
 */
function graphRenderConvergence() {
  const target = document.getElementById("convergence-results");
  if (!target) return;
  const fixture = GRAPH_VIEW_STATE.fixture;
  const p = fixture.p || [];
  const values = p
    .map((value, i) => REFERENCE_GRAPH.nodeOrder[i] + " " + Number(value).toFixed(10))
    .join(", ");
  const residual = fixture.residual === null ? "-" : Number(fixture.residual).toExponential(3);

  target.innerHTML =
    "<p><strong>Fixture demo (4-node reference graph)</strong> - iteration " +
    fixture.iteration +
    ", alpha " +
    Number(fixture.alpha !== undefined ? fixture.alpha : graphReadAlpha()).toFixed(2) +
    ", restart " +
    graphEscape(fixture.restart || "uniform") +
    ".</p>" +
    '<dl class="summary-list">' +
    "<dt>L1 residual (last step)</dt><dd>" + graphEscape(residual) + "</dd>" +
    "<dt>Current p</dt><dd>" + graphEscape(values) + "</dd>" +
    "</dl>" +
    '<p class="hint">This plain-JavaScript demo exists only for the 4-node fixture and is not a ' +
    "template for the MovieLens implementation, which must use TensorFlow.js tensor operations " +
    "on R and R-transpose (readme.md section 3.5).</p>";
}

/**
 * Advance the 4-node fixture demo by one propagation step and redraw it.
 *
 * @returns {void}
 */
function fixtureStep() {
  const fixture = GRAPH_VIEW_STATE.fixture;
  const alpha = graphReadAlpha();
  if (!fixture.p) {
    fixture.p = graphFixtureRestartVector(fixture.restart || "uniform");
    fixture.iteration = 0;
  }
  const result = graphFixturePropagate(alpha, fixture.restart || "uniform", fixture.p);
  fixture.p = result.p;
  fixture.iteration += 1;
  fixture.residual = result.residual;
  fixture.alpha = alpha;
  GRAPH_VIEW_STATE.mode = "fixture";
  renderGraphView({ mode: "fixture" });
  graphRenderConvergence();
}

/**
 * Reset the fixture demo to iteration 0 (the restart distribution) and redraw.
 *
 * @returns {void}
 */
function fixtureReset() {
  const fixture = GRAPH_VIEW_STATE.fixture;
  fixture.alpha = graphReadAlpha();
  fixture.iteration = 0;
  fixture.p = graphFixtureRestartVector(fixture.restart || "uniform");
  fixture.residual = null;
  GRAPH_VIEW_STATE.mode = "fixture";
  renderGraphView({ mode: "fixture" });
  graphRenderConvergence();
}

// ---------------------------------------------------------------------------
// Bootstrap (private)
// ---------------------------------------------------------------------------

/**
 * Wire the fixture step/reset buttons and draw the fixture demo once.
 *
 * @returns {void}
 */
function graphInitGraphView() {
  const stepButton = document.getElementById("step-btn");
  if (stepButton) stepButton.addEventListener("click", fixtureStep);
  const resetButton = document.getElementById("reset-steps");
  if (resetButton) resetButton.addEventListener("click", fixtureReset);
  fixtureReset();
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", graphInitGraphView);
} else {
  graphInitGraphView();
}

// ---------------------------------------------------------------------------
// Efficient propagation
// ---------------------------------------------------------------------------

const GRAPH_PPR_CACHE = {
  RTensor: null,
  invDegU: null,
  invDegM: null,
  danglingMaskTensor: null,
  nUsers: 0,
  nMovies: 0,
  n: 0,
  forGraph: null
};

function graphPprDisposeCache() {
  try {
    if (GRAPH_PPR_CACHE.RTensor) { GRAPH_PPR_CACHE.RTensor.dispose(); GRAPH_PPR_CACHE.RTensor = null; }
    if (GRAPH_PPR_CACHE.invDegU) { GRAPH_PPR_CACHE.invDegU.dispose(); GRAPH_PPR_CACHE.invDegU = null; }
    if (GRAPH_PPR_CACHE.invDegM) { GRAPH_PPR_CACHE.invDegM.dispose(); GRAPH_PPR_CACHE.invDegM = null; }
    if (GRAPH_PPR_CACHE.danglingMaskTensor) { GRAPH_PPR_CACHE.danglingMaskTensor.dispose(); GRAPH_PPR_CACHE.danglingMaskTensor = null; }
  } catch (e) {}
  GRAPH_PPR_CACHE.nUsers = 0; GRAPH_PPR_CACHE.nMovies = 0; GRAPH_PPR_CACHE.n = 0; GRAPH_PPR_CACHE.forGraph = null;
}

function graphPprEnsureCache(graph) {
  if (typeof tf === 'undefined') return null;
  if (GRAPH_PPR_CACHE.forGraph === graph && GRAPH_PPR_CACHE.RTensor) return GRAPH_PPR_CACHE;
  graphPprDisposeCache();
  if (!graph || !graph.nUsers || !graph.nMovies || !graph.R) return null;
  const nUsers = graph.nUsers, nMovies = graph.nMovies, n = nUsers + nMovies;
  const R = graph.R;
  const RData = new Float32Array(nUsers * nMovies);
  for (let u = 0; u < nUsers; u++) {
    const row = R[u] || [];
    for (let m = 0; m < nMovies; m++) RData[u * nMovies + m] = row[m] ? 1.0 : 0.0;
  }
  const RTensor = tf.tensor2d(RData, [nUsers, nMovies]);
  const invDegUData = new Float32Array(nUsers);
  const invDegMData = new Float32Array(nMovies);
  for (let u = 0; u < nUsers; u++) {
    const d = Number(graph.degree[u] || 0);
    invDegUData[u] = d > 0 ? 1.0 / d : 0.0;
  }
  for (let m = 0; m < nMovies; m++) {
    const d = Number(graph.degree[nUsers + m] || 0);
    invDegMData[m] = d > 0 ? 1.0 / d : 0.0;
  }
  // Build the column vectors as 2-D tensors directly: `tensor1d(...).reshape(...)`
  // would leave the intermediate 1-D tensor alive (this runs outside tf.tidy).
  const invDegU = tf.tensor2d(invDegUData, [nUsers, 1]);
  const invDegM = tf.tensor2d(invDegMData, [nMovies, 1]);
  const danglingMaskData = new Float32Array(n);
  for (let i = 0; i < n; i++) danglingMaskData[i] = Number(graph.degree[i] || 0) === 0 ? 1.0 : 0.0;
  const danglingMaskTensor = tf.tensor2d(danglingMaskData, [n, 1]);
  GRAPH_PPR_CACHE.RTensor = RTensor; GRAPH_PPR_CACHE.invDegU = invDegU; GRAPH_PPR_CACHE.invDegM = invDegM;
  GRAPH_PPR_CACHE.danglingMaskTensor = danglingMaskTensor;
  GRAPH_PPR_CACHE.nUsers = nUsers; GRAPH_PPR_CACHE.nMovies = nMovies; GRAPH_PPR_CACHE.n = n; GRAPH_PPR_CACHE.forGraph = graph;
  return GRAPH_PPR_CACHE;
}

function graphPprStepBatchMovieLens(graph, alpha, PMat, vMat) {
  const cache = graphPprEnsureCache(graph);
  const n = cache.n, nUsers = cache.nUsers, nMovies = cache.nMovies;
  const B = PMat.shape[1];
  return tf.tidy(() => {
    const Pusers = PMat.slice([0, 0], [nUsers, B]);
    const Pmovies = PMat.slice([nUsers, 0], [nMovies, B]);
    const msgU2M = tf.matMul(GRAPH_PPR_CACHE.RTensor.transpose(), Pusers.mul(GRAPH_PPR_CACHE.invDegU));
    const msgM2U = tf.matMul(GRAPH_PPR_CACHE.RTensor, Pmovies.mul(GRAPH_PPR_CACHE.invDegM));
    const qp = tf.concat([msgM2U, msgU2M], 0);
    const d = PMat.mul(GRAPH_PPR_CACHE.danglingMaskTensor).sum(0).reshape([1, B]);
    return qp.add(vMat.mul(d)).mul(alpha).add(vMat.mul(1 - alpha));
  });
}

function graphPprStepBatchTransition(QMatArr, alpha, PMat, vMat, danglingMaskArr) {
  const n = QMatArr.length;
  const B = PMat.shape[1];
  return tf.tidy(() => {
    const QtData = new Float32Array(n * n);
    for (let j = 0; j < n; j++) {
      for (let i = 0; i < n; i++) QtData[j * n + i] = QMatArr[i][j];
    }
    const Qt = tf.tensor2d(QtData, [n, n]);
    const qp = tf.matMul(Qt, PMat);
    const danglingData = new Float32Array(n);
    for (let i = 0; i < n; i++) danglingData[i] = (danglingMaskArr[i]) ? 1.0 : 0.0;
    const dangling = tf.tensor1d(danglingData).reshape([n, 1]);
    const d = PMat.mul(dangling).sum(0).reshape([1, B]);
    const res = qp.add(vMat.mul(d)).mul(alpha).add(vMat.mul(1 - alpha));
    return res;
  });
}
