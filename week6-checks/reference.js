/**
 * week6-checks/reference.js — independent pure-JavaScript (float64) reference
 * for the HW6 MovieLens retrieval experiment.
 *
 * No TensorFlow.js, no page code: it re-derives everything from week6/u.data and
 * week6/u.item so the numbers can be compared against the browser implementation
 * as a genuine independent check.
 *
 * It reproduces, from the specs in week6/readme.md sections 3.1-3.5, 5, 6.1, 7:
 *   - the same chronological per-user 80/10/10 split,
 *   - the bipartite graph from positive (>= 4.0) TRAIN ratings,
 *   - row-normalized PageRank / PPR by power iteration with the same
 *     zero-degree (dangling) policy, restart and stopping rule,
 *   - the shared val/test cohorts and exclusion rules,
 *   - Recall@10 for popularity, pagerank and ppr,
 *   - the validation alpha sweep over {0.50, 0.70, 0.85, 0.95} plus the
 *     iteration count for each alpha.
 *
 * Usage: node week6-checks/reference.js
 * Writes results/reference.txt and results/reference.json.
 */

"use strict";

const fs = require("fs");
const path = require("path");

const WEEK6 = path.join(__dirname, "..", "week6");
const RESULTS = path.join(__dirname, "results");

const ALPHA_GRID = [0.50, 0.70, 0.85, 0.95];
const K = 10;
const TOL = 1e-6;
const MAX_ITER = 500;
const MIN_RATINGS = 5;
const POSITIVE = 4.0;

// ---------------------------------------------------------------------------
// Data
// ---------------------------------------------------------------------------

/** Parse u.data: "userId\tmovieId\trating\ttimestamp". */
function parseRatings(text) {
  const out = [];
  for (const line of text.split("\n")) {
    if (!line.trim()) continue;
    const f = line.split("\t");
    if (f.length < 4) continue;
    out.push({
      userId: parseInt(f[0], 10),
      movieId: parseInt(f[1], 10),
      rating: parseFloat(f[2]),
      timestamp: parseInt(f[3], 10),
    });
  }
  return out;
}

/** Parse u.item: "id|title|..."; latin1 so the title bytes decode correctly. */
function parseMovies(text) {
  const byId = new Map();
  for (const line of text.split("\n")) {
    if (!line.trim()) continue;
    const fields = line.split("|");
    const id = parseInt(fields[0], 10);
    if (!Number.isFinite(id)) continue;
    byId.set(id, fields[1] || "Movie #" + id);
  }
  return byId;
}

/** Split each user's ratings by ascending timestamp into 80/10/10. */
function splitByTimestamp(ratings) {
  const byUser = new Map();
  for (const r of ratings) {
    if (!byUser.has(r.userId)) byUser.set(r.userId, []);
    byUser.get(r.userId).push(r);
  }
  const train = [];
  const val = [];
  const test = [];
  for (const [, list] of byUser) {
    const n = list.length;
    if (n < MIN_RATINGS) continue;
    const sorted = list
      .slice()
      .sort((a, b) => a.timestamp - b.timestamp || a.movieId - b.movieId);
    const nTrain = Math.floor(0.8 * n);
    const nVal = Math.floor(0.1 * n);
    for (let i = 0; i < n; i++) {
      if (i < nTrain) train.push(sorted[i]);
      else if (i < nTrain + nVal) val.push(sorted[i]);
      else test.push(sorted[i]);
    }
  }
  return { train, val, test };
}

// ---------------------------------------------------------------------------
// Graph
// ---------------------------------------------------------------------------

/**
 * Build the bipartite graph and the derived indices, mirroring readme 3.1:
 * users occupy nodes [0, nUsers), movies [nUsers, nUsers + nMovies).
 */
function buildGraph(train) {
  const userSet = new Set();
  const movieSet = new Set();
  for (const r of train) {
    userSet.add(r.userId);
    movieSet.add(r.movieId);
  }
  const userIds = Array.from(userSet).sort((a, b) => a - b);
  const movieIds = Array.from(movieSet).sort((a, b) => a - b);
  const nUsers = userIds.length;
  const nMovies = movieIds.length;
  const userIndexOf = new Map();
  userIds.forEach((u, i) => userIndexOf.set(u, i));
  const movieIndexOf = new Map();
  movieIds.forEach((m, i) => movieIndexOf.set(m, i));

  // adjacency lists over node indices (only positive train edges)
  const adj = Array.from({ length: nUsers + nMovies }, () => []);
  for (const r of train) {
    if (r.rating < POSITIVE) continue;
    const u = userIndexOf.get(r.userId);
    const m = movieIndexOf.get(r.movieId);
    if (u === undefined || m === undefined) continue;
    // user -> movie, movie -> user (each edge is undirected-ish in direction:
    // probability flows both ways along a bipartite edge)
    adj[u].push(nUsers + m);
    adj[nUsers + m].push(u);
  }

  const n = nUsers + nMovies;
  const degree = new Float64Array(n);
  const danglingMask = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    degree[i] = adj[i].length;
    if (degree[i] === 0) danglingMask[i] = 1;
  }

  return { userIds, movieIds, nUsers, nMovies, n, userIndexOf, movieIndexOf, adj, degree, danglingMask };
}

// ---------------------------------------------------------------------------
// PageRank / PPR (power iteration)
// ---------------------------------------------------------------------------

/**
 * Power iteration with the readme 3.3 dangling policy:
 *   p_next = alpha * (Q^T p + d * v) + (1 - alpha) * v
 * where d = sum of p over dangling nodes and v is the restart distribution.
 *
 * @param {Object} graph
 * @param {number} alpha
 * @param {Float64Array|null} restart One-hot or uniform restart; null => uniform.
 * @returns {{p:Float64Array, iterations:number, residual:number, converged:boolean}}
 */
function propagate(graph, alpha, restart) {
  const n = graph.n;
  const v = restart || uniform(n);
  let p = Float64Array.from(v);
  let iterations = 0;
  let residual = Infinity;
  let converged = false;
  while (iterations < MAX_ITER) {
    let danglingMass = 0;
    for (let i = 0; i < n; i++) if (graph.danglingMask[i]) danglingMass += p[i];
    const next = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const neighbors = graph.adj[i];
      if (neighbors.length === 0) continue; // its mass went to danglingMass
      const share = (alpha * p[i]) / neighbors.length;
      for (const j of neighbors) next[j] += share;
    }
    for (let i = 0; i < n; i++) {
      next[i] += alpha * danglingMass * v[i] + (1 - alpha) * v[i];
    }
    let l1 = 0;
    for (let i = 0; i < n; i++) l1 += Math.abs(next[i] - p[i]);
    p = next;
    iterations++;
    residual = l1;
    if (l1 < TOL) {
      converged = true;
      break;
    }
  }
  return { p, iterations, residual, converged };
}

function uniform(n) {
  const v = new Float64Array(n);
  v.fill(1 / n);
  return v;
}

function oneHot(n, index) {
  const v = new Float64Array(n);
  v[index] = 1;
  return v;
}

/** Movie-node scores in graph.movieIds order. */
function movieScores(p, graph) {
  const out = new Float64Array(graph.nMovies);
  for (let k = 0; k < graph.nMovies; k++) out[k] = p[graph.nUsers + k];
  return out;
}

// ---------------------------------------------------------------------------
// Split indices (mirror of the page's hw6BuildSplitIndices)
// ---------------------------------------------------------------------------

function buildSplitIndices(split) {
  const names = ["train", "val", "test"];
  const candidateSet = new Set();
  const trainUserSet = new Set();
  const popularity = new Map(); // movieId -> distinct positive users
  const positiveUsers = new Set();
  const pairsBySplit = { train: new Map(), val: new Map(), test: new Map() };
  const gradesBySplit = { train: new Map(), val: new Map(), test: new Map() };

  for (const name of names) {
    for (const row of split[name]) {
      let pairSet = pairsBySplit[name].get(row.userId);
      if (!pairSet) {
        pairSet = new Set();
        pairsBySplit[name].set(row.userId, pairSet);
      }
      pairSet.add(row.movieId);
      let gradeMap = gradesBySplit[name].get(row.userId);
      if (!gradeMap) {
        gradeMap = new Map();
        gradesBySplit[name].set(row.userId, gradeMap);
      }
      gradeMap.set(row.movieId, row.rating);
    }
  }

  for (const row of split.train) {
    candidateSet.add(row.movieId);
    trainUserSet.add(row.userId);
    if (row.rating >= POSITIVE) {
      let users = popularity.get(row.movieId);
      if (!users) {
        users = new Set();
        popularity.set(row.movieId, users);
      }
      users.add(row.userId);
      positiveUsers.add(row.userId);
    }
  }

  for (const [movieId, users] of popularity) popularity.set(movieId, users.size);

  return {
    candidateIds: Array.from(candidateSet).sort((a, b) => a - b),
    candidateSet,
    trainUserIds: Array.from(trainUserSet).sort((a, b) => a - b),
    popularity,
    positiveUsers,
    pairsBySplit,
    gradesBySplit,
  };
}

function exclusionSet(idx, splitName, userId) {
  const out = new Set();
  const add = (name) => {
    const set = idx.pairsBySplit[name].get(userId);
    if (set) set.forEach((m) => out.add(m));
  };
  add("train");
  if (splitName === "test") add("val");
  return out;
}

function sharedCohort(idx, splitName) {
  const name = splitName === "val" ? "val" : "test";
  const grades = idx.gradesBySplit[name];
  const cohort = [];
  for (const userId of idx.trainUserIds) {
    if (!idx.positiveUsers.has(userId)) continue;
    const gradeMap = grades.get(userId);
    if (!gradeMap) continue;
    const excluded = exclusionSet(idx, name, userId);
    let relevant = 0;
    for (const [movieId, rating] of gradeMap) {
      if (rating >= POSITIVE && idx.candidateSet.has(movieId) && !excluded.has(movieId)) relevant++;
    }
    if (relevant > 0) cohort.push(userId);
  }
  return cohort;
}

function cohortExclusions(idx, splitName) {
  const name = splitName === "val" ? "val" : "test";
  const grades = idx.gradesBySplit[name];
  let excludedUnseen = 0;
  let excludedNoEligible = 0;
  for (const userId of idx.trainUserIds) {
    const gradeMap = grades.get(userId);
    if (!gradeMap) continue;
    const excluded = exclusionSet(idx, name, userId);
    let relevant = 0;
    for (const [movieId, rating] of gradeMap) {
      if (rating < POSITIVE) continue;
      if (!idx.candidateSet.has(movieId)) {
        excludedUnseen++;
        continue;
      }
      if (excluded.has(movieId)) continue;
      relevant++;
    }
    if (relevant === 0) excludedNoEligible++;
  }
  return { excludedUnseen, excludedNoEligible };
}

// ---------------------------------------------------------------------------
// Recommendation + recall
// ---------------------------------------------------------------------------

/**
 * Rank the candidate pool for one method/user/split, mirroring script.js
 * `recommendTopK`: filter exclusions, sort score desc / id asc, slice to K.
 */
function recommendTopK(scoresById, idx, userId, splitName, K) {
  const excluded = exclusionSet(idx, splitName, userId);
  const rows = [];
  for (const movieId of idx.candidateIds) {
    if (excluded.has(movieId)) continue;
    const score = scoresById.has(movieId) ? scoresById.get(movieId) : Number.NEGATIVE_INFINITY;
    if (!Number.isFinite(score)) continue;
    rows.push({ movieId, score, popularity: idx.popularity.get(movieId) || 0 });
  }
  rows.sort((a, b) => (b.score - a.score) || (a.movieId - b.movieId));
  return rows.slice(0, K);
}

/**
 * Build a movieId -> score map for a method.
 * @param {"popularity"|"pagerank"|"ppr"} method
 */
function scoresForMethod(method, idx, graph, alpha, userId) {
  const map = new Map();
  if (method === "popularity") {
    for (const movieId of idx.candidateIds) map.set(movieId, idx.popularity.get(movieId) || 0);
    return { map, iterations: null };
  }
  let p;
  let iterations = null;
  if (method === "pagerank") {
    const r = propagate(graph, alpha, null);
    p = r.p;
    iterations = r.iterations;
  } else {
    const node = graph.nUsers ? nodeOfUser(graph, userId) : -1;
    if (node < 0) return { map, iterations: null };
    const r = propagate(graph, alpha, oneHot(graph.n, node));
    p = r.p;
    iterations = r.iterations;
  }
  const perMovie = movieScores(p, graph);
  for (let k = 0; k < graph.nMovies; k++) map.set(graph.movieIds[k], perMovie[k]);
  return { map, iterations };
}

function nodeOfUser(graph, userId) {
  const idx = graph.userIndexOf.get(userId);
  return idx === undefined ? -1 : idx;
}

/** Macro-average Recall@K over the shared cohort. */
function recallAtK(method, idx, graph, alpha, splitName, K, pprCache) {
  const cohort = sharedCohort(idx, splitName);
  const grades = idx.gradesBySplit[splitName];
  let sum = 0;
  let users = 0;
  const perUser = [];
  let aggIterations = null;
  for (const userId of cohort) {
    const gradeMap = grades.get(userId);
    const excluded = exclusionSet(idx, splitName, userId);
    const relevantSet = new Set();
    if (gradeMap) {
      for (const [movieId, rating] of gradeMap) {
        if (rating >= POSITIVE && idx.candidateSet.has(movieId) && !excluded.has(movieId)) relevantSet.add(movieId);
      }
    }
    if (relevantSet.size === 0) continue;
    let map;
    let iterations = null;
    if (pprCache && method === "ppr" && pprCache.has(userId)) {
      map = pprCache.get(userId);
    } else {
      const r = scoresForMethod(method, idx, graph, alpha, userId);
      map = r.map;
      iterations = r.iterations;
      if (pprCache && method === "ppr") pprCache.set(userId, map);
    }
    if (aggIterations === null && iterations !== null) aggIterations = iterations;
    const recs = recommendTopK(map, idx, userId, splitName, K);
    let hit = 0;
    for (const rec of recs) if (relevantSet.has(rec.movieId)) hit++;
    sum += hit / relevantSet.size;
    users++;
    perUser.push({ userId, relevant: relevantSet.size, hit });
  }
  return {
    recall: users > 0 ? sum / users : 0,
    cohortSize: cohort.length,
    evaluated: users,
    iterations: aggIterations,
    perUser,
  };
}

/** Top-10 for a method/user, with title + popularity (for the JSON dump). */
function topTen(method, idx, graph, movies, alpha, splitName, userId, K, pprCache) {
  let map;
  if (pprCache && method === "ppr" && pprCache.has(userId)) {
    map = pprCache.get(userId);
  } else {
    const r = scoresForMethod(method, idx, graph, alpha, userId);
    map = r.map;
    if (pprCache && method === "ppr") pprCache.set(userId, map);
  }
  const recs = recommendTopK(map, idx, userId, splitName, K);
  return recs.map((rec) => ({
    movieId: rec.movieId,
    title: movies.get(rec.movieId) || "Movie #" + rec.movieId,
    score: rec.score,
    popularity: idx.popularity.get(rec.movieId) || 0,
  }));
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

function main() {
  fs.mkdirSync(RESULTS, { recursive: true });
  const ratings = parseRatings(fs.readFileSync(path.join(WEEK6, "u.data"), "utf8"));
  const movies = parseMovies(fs.readFileSync(path.join(WEEK6, "u.item"), "latin1"));
  const split = splitByTimestamp(ratings);
  const idx = buildSplitIndices(split);
  const graph = buildGraph(split.train);

  // sanity: any movie in the positive pool must be a train movie (it is, by construction)
  const out = [];
  const json = { split: {}, graph: {}, cohorts: {}, alphaSweep: [], recall: {}, top10: {} };

  out.push(`train / val / test = ${split.train.length} / ${split.val.length} / ${split.test.length}`);
  out.push(`graph: ${graph.nUsers} users + ${graph.nMovies} movies = ${graph.n} nodes`);
  let zeroU = 0, zeroM = 0, posEdges = 0;
  for (let i = 0; i < graph.nUsers; i++) if (graph.degree[i] === 0) zeroU++;
  for (let i = graph.nUsers; i < graph.n; i++) if (graph.degree[i] === 0) zeroM++;
  for (let i = 0; i < graph.n; i++) posEdges += graph.degree[i];
  posEdges = posEdges / 2;
  out.push(`zero-degree users / movies = ${zeroU} / ${zeroM}; positive edges = ${posEdges}`);
  json.split = { train: split.train.length, val: split.val.length, test: split.test.length };
  json.graph = { users: graph.nUsers, movies: graph.nMovies, nodes: graph.n, zeroDegreeUsers: zeroU, zeroDegreeMovies: zeroM, positiveEdges: posEdges };

  // Cohorts
  const valCohort = sharedCohort(idx, "val");
  const testCohort = sharedCohort(idx, "test");
  const valEx = cohortExclusions(idx, "val");
  const testEx = cohortExclusions(idx, "test");
  out.push(`cohort val = ${valCohort.length} (excludedUnseen ${valEx.excludedUnseen}, excludedNoEligible ${valEx.excludedNoEligible})`);
  out.push(`cohort test = ${testCohort.length} (excludedUnseen ${testEx.excludedUnseen}, excludedNoEligible ${testEx.excludedNoEligible})`);
  json.cohorts = {
    val: { size: valCohort.length, ...valEx },
    test: { size: testCohort.length, ...testEx },
  };

  // ---- Train mean, mean-prediction RMSE, unseen movies, relevance counts ----
  out.push("");
  out.push("=== Train mean and mean-prediction RMSE ===");
  let trainSum = 0;
  for (const r of split.train) trainSum += r.rating;
  const trainMean = split.train.length > 0 ? trainSum / split.train.length : 0;
  const rmseMean = (rows) => {
    let s = 0;
    for (const r of rows) {
      const d = r.rating - trainMean;
      s += d * d;
    }
    return rows.length > 0 ? Math.sqrt(s / rows.length) : 0;
  };
  const valRmseMean = rmseMean(split.val);
  const testRmseMean = rmseMean(split.test);
  out.push(`train mean rating mu = ${trainSum}/${split.train.length} = ${trainMean} (fraction)`);
  out.push(`train mean rating mu (10 dp) = ${trainMean.toFixed(10)}`);
  out.push(`RMSE predicting mu on val  = ${valRmseMean.toFixed(6)}`);
  out.push(`RMSE predicting mu on test = ${testRmseMean.toFixed(6)}`);
  json.trainMean = { mu: trainMean, count: split.train.length, rmseVal: valRmseMean, rmseTest: testRmseMean };

  out.push("");
  out.push("=== Movies in u.item with no train rating ===");
  const trainRatedMovies = new Set();
  for (const r of split.train) trainRatedMovies.add(r.movieId);
  let unseenMovies = 0;
  for (const movieId of movies.keys()) if (!trainRatedMovies.has(movieId)) unseenMovies++;
  out.push(`u.item movies = ${movies.size}; with no train rating = ${unseenMovies}; with >= 1 train rating (candidate pool) = ${idx.candidateIds.length}`);
  json.unseenMovies = { total: movies.size, noTrainRating: unseenMovies, candidatePool: idx.candidateIds.length };

  out.push("");
  out.push("=== Relevance in the TEST cohort (rating >= 4, in pool, not rated train/val) ===");
  let relevantPairs = 0;
  let relevantOutsidePool = 0;
  for (const userId of testCohort) {
    const gradeMap = idx.gradesBySplit.test.get(userId);
    if (!gradeMap) continue;
    const excluded = exclusionSet(idx, "test", userId);
    for (const [movieId, rating] of gradeMap) {
      if (rating < POSITIVE) continue;
      if (!idx.candidateSet.has(movieId)) { relevantOutsidePool++; continue; }
      if (excluded.has(movieId)) continue;
      relevantPairs++;
    }
  }
  out.push(`relevant pairs in the test cohort = ${relevantPairs}`);
  out.push(`relevant test ratings outside the pool = ${relevantOutsidePool}`);
  json.relevance = { testCohortPairs: relevantPairs, testOutsidePool: relevantOutsidePool };

  // ---- PPR @ 0.50 over the whole test cohort: Top-10 quality signals --------
  out.push("");
  out.push("=== PPR @ alpha=0.50, whole test cohort Top-10 quality ===");
  {
    const pprCache = new Map();
    let minTopScore = Infinity;
    let listsWithZeroFill = 0;
    let usersFewerThan10 = 0;
    let evaluated = 0;
    for (const userId of testCohort) {
      const excluded = exclusionSet(idx, "test", userId);
      let eligible = 0;
      for (const movieId of idx.candidateIds) {
        if (excluded.has(movieId)) continue;
        eligible++;
      }
      if (eligible < K) usersFewerThan10++;
      const r = scoresForMethod("ppr", idx, graph, 0.50, userId);
      pprCache.set(userId, r.map);
      const recs = recommendTopK(r.map, idx, userId, "test", K);
      evaluated++;
      let hasZero = false;
      for (const rec of recs) {
        if (rec.score < minTopScore) minTopScore = rec.score;
        if (!(rec.score > 0)) hasZero = true;
      }
      if (hasZero) listsWithZeroFill++;
    }
    out.push(`cohort users evaluated = ${evaluated}`);
    out.push(`minimum Top-10 PPR score = ${minTopScore.toExponential(6)} (${minTopScore.toFixed(10)})`);
    out.push(`lists that included a zero-score movie = ${listsWithZeroFill}`);
    out.push(`users with fewer than ${K} eligible candidates = ${usersFewerThan10}`);
    json.ppr050 = {
      cohortUsers: evaluated,
      minTop10Score: minTopScore,
      listsWithZeroFill: listsWithZeroFill,
      usersFewerThanK: usersFewerThan10,
    };
  }

  // Alpha sweep on validation (PPR), iteration counts per alpha
  out.push("");
  out.push("=== Alpha sweep (validation Recall@10, PPR) ===");
  let best = null;
  let bestRecall = -Infinity;
  for (const alpha of ALPHA_GRID) {
    const pprCache = new Map();
    const r = recallAtK("ppr", idx, graph, alpha, "val", K, pprCache);
    out.push(
      `alpha ${alpha.toFixed(2)}: recall=${r.recall.toFixed(10)} iterations=${r.iterations} cohort=${r.cohortSize}`,
    );
    json.alphaSweep.push({ alpha, recall: r.recall, iterations: r.iterations, cohortSize: r.cohortSize });
    if (r.recall > bestRecall) {
      bestRecall = r.recall;
      best = alpha;
    }
  }
  out.push(`best alpha (ties -> smaller) = ${best}`);
  json.bestAlpha = best;

  // Primary comparison on test at the selected alpha
  out.push("");
  out.push(`=== Test comparison at alpha=${best} ===`);
  json.recall.test = {};
  for (const method of ["popularity", "pagerank", "ppr"]) {
    const pprCache = method === "ppr" ? new Map() : null;
    const r = recallAtK(method, idx, graph, best, "test", K, pprCache);
    out.push(`${method}: recall=${r.recall.toFixed(10)} cohort=${r.cohortSize} iterations=${r.iterations}`);
    json.recall.test[method] = { recall: r.recall, cohortSize: r.cohortSize, iterations: r.iterations };
  }

  // Top-10 for users 1 and 2 (the two smallest test-cohort users, which match
  // the page's two-user demo ordering) for each method at the selected alpha
  out.push("");
  out.push("=== Top-10 users 1 and 2 (test split) ===");
  json.top10 = {};
  for (const n of [1, 2]) {
    const userId = testCohort[n - 1];
    json.top10["user" + n] = { userId, methods: {} };
    for (const method of ["popularity", "pagerank", "ppr"]) {
      const pprCache = method === "ppr" ? new Map() : null;
      const recs = topTen(method, idx, graph, movies, best, "test", userId, K, pprCache);
      json.top10["user" + n].methods[method] = recs;
      out.push(`user ${userId} / ${method}:`);
      for (const rec of recs) {
        out.push(`  ${String(rec.movieId).padStart(4)}  ${rec.score.toFixed(10)}  pop=${rec.popularity}  ${rec.title}`);
      }
    }
  }

  // Also expose a PPR vector fingerprint for user 1's node at 0.85 so probe.js
  // can compare the page's raw vector.
  const ppr85 = propagate(graph, 0.85, oneHot(graph.n, nodeOfUser(graph, 1)));
  json.pprUser1Node085 = {
    node: nodeOfUser(graph, 1),
    iterations: ppr85.iterations,
    residual: ppr85.residual,
    converged: ppr85.converged,
    sum: Array.from(ppr85.p).reduce((a, b) => a + b, 0),
    topMovieNodes: movieScores(ppr85.p, graph)
      ? topIndices(movieScores(ppr85.p, graph), 10).map((k) => ({ movieId: graph.movieIds[k], score: movieScores(ppr85.p, graph)[k] }))
      : [],
  };

  const txt = out.join("\n") + "\n";
  fs.writeFileSync(path.join(RESULTS, "reference.txt"), txt);
  fs.writeFileSync(path.join(RESULTS, "reference.json"), JSON.stringify(json, null, 2));
  console.log(txt);
  console.log("wrote results/reference.txt and results/reference.json");
}

function topIndices(arr, k) {
  const idxs = Array.from(arr.keys());
  idxs.sort((a, b) => (arr[b] - arr[a]) || (a - b));
  return idxs.slice(0, k);
}

main();
