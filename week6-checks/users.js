/**
 * week6-checks/users.js — business-level comparison of the four HW6 methods
 * (popularity, pagerank, ppr, mf) on the shared test cohort, pure JavaScript,
 * no TensorFlow.js and no page code.
 *
 * Same split and graph as reference.js (chronological per-user 80/10/10, positive
 * train edges >= 4.0, row-normalized power iteration with the same dangling
 * policy / stopping rule). PPR runs at the alpha selected on validation
 * (ALPHA = 0.50). MF is the HW5 model loaded from week6/mf_checkpoint.json and
 * scored as mu + p . q (unclamped), exactly as the page's `mfScores`.
 *
 * Answers the business question "keep MF, replace it with PPR, or show both?":
 *   - Recall@10;
 *   - conversion coverage: users with at least one test 4-5 item in their Top-10;
 *   - catalog coverage: distinct movies over all Top-10 lists together;
 *   - mean list popularity (distinct train users with rating >= 4) and the share
 *     of slots on long-tail movies (fewer than 5 train ratings);
 *   - mean Top-10 overlap between users for ppr-vs-popularity and ppr-vs-mf;
 *   - MF restricted to movies with >= N train ratings, N = 5, 20, 50;
 *   - user cards 1 and 2 (train profile, genre mix, Top-10 ppr + mf with genres,
 *     popularity and whether the user later rated the movie 4-5 in test);
 *   - one-factor demo: user 1 @ 0.85 -> user 2 @ 0.85 -> user 1 @ 0.50.
 *
 * Usage: node week6-checks/users.js
 * Writes results/users.txt.
 */

"use strict";

const fs = require("fs");
const path = require("path");

const WEEK6 = path.join(__dirname, "..", "week6");
const RESULTS = path.join(__dirname, "results");

const ALPHA = 0.50;
const K = 10;
const TOL = 1e-6;
const MAX_ITER = 500;
const MIN_RATINGS = 5;
const POSITIVE = 4.0;

const GENRE_NAMES = [
  "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy",
  "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
  "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
];

// ---------------------------------------------------------------------------
// Data
// ---------------------------------------------------------------------------

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

/** Parse u.item: id | title | release | video | url | 19 genre flags. */
function parseMovies(text) {
  const byId = new Map();
  for (const line of text.split("\n")) {
    if (!line.trim()) continue;
    const fields = line.split("|");
    const id = parseInt(fields[0], 10);
    if (!Number.isFinite(id)) continue;
    const genres = [];
    for (let g = 0; g < GENRE_NAMES.length; g++) {
      if (fields[5 + g] === "1") genres.push(GENRE_NAMES[g]);
    }
    byId.set(id, { title: fields[1] || "Movie #" + id, genres });
  }
  return byId;
}

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
    const sorted = list.slice().sort((a, b) => a.timestamp - b.timestamp || a.movieId - b.movieId);
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
// Graph (identical construction to reference.js)
// ---------------------------------------------------------------------------

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

  const adj = Array.from({ length: nUsers + nMovies }, () => []);
  for (const r of train) {
    if (r.rating < POSITIVE) continue;
    const u = userIndexOf.get(r.userId);
    const m = movieIndexOf.get(r.movieId);
    if (u === undefined || m === undefined) continue;
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
// Power iteration (identical to reference.js)
// ---------------------------------------------------------------------------

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
      if (neighbors.length === 0) continue;
      const share = (alpha * p[i]) / neighbors.length;
      for (const j of neighbors) next[j] += share;
    }
    for (let i = 0; i < n; i++) next[i] += alpha * danglingMass * v[i] + (1 - alpha) * v[i];
    let l1 = 0;
    for (let i = 0; i < n; i++) l1 += Math.abs(next[i] - p[i]);
    p = next;
    iterations++;
    residual = l1;
    if (l1 < TOL) { converged = true; break; }
  }
  return { p, iterations, residual, converged };
}

function uniform(n) { const v = new Float64Array(n); v.fill(1 / n); return v; }
function oneHot(n, index) { const v = new Float64Array(n); v[index] = 1; return v; }

/** Full node-distribution -> movieId -> score map over all graph movies. */
function movieScoreMap(p, graph) {
  const map = new Map();
  for (let k = 0; k < graph.nMovies; k++) map.set(graph.movieIds[k], p[graph.nUsers + k]);
  return map;
}

// ---------------------------------------------------------------------------
// Split indices (identical to reference.js)
// ---------------------------------------------------------------------------

function buildSplitIndices(split) {
  const names = ["train", "val", "test"];
  const candidateSet = new Set();
  const trainUserSet = new Set();
  const popularity = new Map();
  const positiveUsers = new Set();
  const pairsBySplit = { train: new Map(), val: new Map(), test: new Map() };
  const gradesBySplit = { train: new Map(), val: new Map(), test: new Map() };

  for (const name of names) {
    for (const row of split[name]) {
      let pairSet = pairsBySplit[name].get(row.userId);
      if (!pairSet) { pairSet = new Set(); pairsBySplit[name].set(row.userId, pairSet); }
      pairSet.add(row.movieId);
      let gradeMap = gradesBySplit[name].get(row.userId);
      if (!gradeMap) { gradeMap = new Map(); gradesBySplit[name].set(row.userId, gradeMap); }
      gradeMap.set(row.movieId, row.rating);
    }
  }

  for (const row of split.train) {
    candidateSet.add(row.movieId);
    trainUserSet.add(row.userId);
    if (row.rating >= POSITIVE) {
      let users = popularity.get(row.movieId);
      if (!users) { users = new Set(); popularity.set(row.movieId, users); }
      users.add(row.userId);
      positiveUsers.add(row.userId);
    }
  }
  for (const [movieId, users] of popularity) popularity.set(movieId, users.size);

  return {
    candidateIds: Array.from(candidateSet).sort((a, b) => a - b),
    candidateSet, trainUserIds: Array.from(trainUserSet).sort((a, b) => a - b),
    popularity, positiveUsers, pairsBySplit, gradesBySplit,
  };
}

function exclusionSet(idx, splitName, userId) {
  const out = new Set();
  const add = (name) => { const set = idx.pairsBySplit[name].get(userId); if (set) set.forEach((m) => out.add(m)); };
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

// ---------------------------------------------------------------------------
// Recommendation
// ---------------------------------------------------------------------------

/**
 * Top-K for one user, mirroring recommendTopK: filter exclusions, drop movies
 * without a finite score, sort score desc / movieId asc.
 */
function recommendTopK(map, idx, userId, splitName, k) {
  const excluded = exclusionSet(idx, splitName, userId);
  const rows = [];
  for (const movieId of idx.candidateIds) {
    if (excluded.has(movieId)) continue;
    if (!map.has(movieId)) continue;
    const score = map.get(movieId);
    if (!Number.isFinite(score)) continue;
    rows.push({ movieId, score, popularity: idx.popularity.get(movieId) || 0 });
  }
  rows.sort((a, b) => (b.score - a.score) || (a.movieId - b.movieId));
  return rows.slice(0, k);
}

// ---------------------------------------------------------------------------
// Score maps
// ---------------------------------------------------------------------------

function popularityMap(idx) {
  const map = new Map();
  for (const movieId of idx.candidateIds) map.set(movieId, idx.popularity.get(movieId) || 0);
  return map;
}

/**
 * Popularity by train rating COUNT (any rating, as in HW5) — not the readme 7.3
 * definition (distinct users with rating >= 4). Used only as a secondary
 * evidence variant in this report; the page's `popularity` method is unchanged.
 */
function ratingCountPopularityMap(idx, trainCountByMovie) {
  const map = new Map();
  for (const movieId of idx.candidateIds) map.set(movieId, trainCountByMovie.get(movieId) || 0);
  return map;
}

function pagerankMap(graph) {
  return movieScoreMap(propagate(graph, ALPHA, null).p, graph);
}

function pprMap(graph, userId) {
  const node = graph.userIndexOf.get(userId);
  if (node === undefined) return new Map();
  return movieScoreMap(propagate(graph, ALPHA, oneHot(graph.n, node)).p, graph);
}

/**
 * MF scores mu + p . q for one user over the HF5 checkpoint's movie rows.
 * Row index of itemFactors is the raw movieId (as in the page's mfScores).
 */
function mfMap(model, idx, userId) {
  const uid = Math.round(Number(userId));
  const map = new Map();
  if (!Number.isFinite(uid) || uid < 0 || uid >= model.userShape[0]) return map;
  const u = model.userFactors[uid];
  const F = u.length;
  for (const movieId of idx.candidateIds) {
    const row = model.itemFactors[movieId];
    if (!row) continue;
    let dot = 0;
    for (let f = 0; f < F; f++) dot += u[f] * row[f];
    map.set(movieId, model.globalMean + dot);
  }
  return map;
}

/** MF restricted to movies with at least `minRatings` train ratings. */
function mfMapWithFloor(model, idx, userId, trainCountByMovie, minRatings) {
  const full = mfMap(model, idx, userId);
  const out = new Map();
  for (const [movieId, score] of full) {
    if ((trainCountByMovie.get(movieId) || 0) >= minRatings) out.set(movieId, score);
  }
  return out;
}

// ---------------------------------------------------------------------------
// Metrics
// ---------------------------------------------------------------------------

/**
 * Evaluate one method over the whole cohort.
 *
 * @returns {{recall:number, evaluated:number, conversion:number,
 *   catalog:Set<number>, meanPopularity:number, tailSlots:number, slots:number,
 *   perUser:Map<number,number[]>}} perUser: userId -> top-K movieIds.
 */
function evaluateCohort(makeMap, idx, cohort, splitName, K, movieTrainCount) {
  const grades = idx.gradesBySplit[splitName];
  let recallSum = 0;
  let evaluated = 0;
  let conversion = 0;
  let popSum = 0;
  let tailSlots = 0;
  let slots = 0;
  const catalog = new Set();
  const perUser = new Map();

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

    const recs = recommendTopK(makeMap(userId), idx, userId, splitName, K);
    perUser.set(userId, recs.map((r) => r.movieId));

    let hit = 0;
    let converted = false;
    for (const rec of recs) {
      catalog.add(rec.movieId);
      popSum += rec.popularity;
      slots++;
      if ((movieTrainCount.get(rec.movieId) || 0) < 5) tailSlots++;
      if (relevantSet.has(rec.movieId)) { hit++; converted = true; }
    }
    recallSum += hit / relevantSet.size;
    if (converted) conversion++;
    evaluated++;
  }

  return {
    recall: evaluated > 0 ? recallSum / evaluated : 0,
    evaluated,
    conversion: conversion / (evaluated || 1),
    conversionUsers: conversion,
    catalog,
    meanPopularity: slots > 0 ? popSum / slots : 0,
    tailShare: slots > 0 ? tailSlots / slots : 0,
    tailSlots,
    slots,
    perUser,
  };
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

function main() {
  fs.mkdirSync(RESULTS, { recursive: true });
  const ratings = parseRatings(fs.readFileSync(path.join(WEEK6, "u.data"), "utf8"));
  const movies = parseMovies(fs.readFileSync(path.join(WEEK6, "u.item"), "latin1"));
  const model = JSON.parse(fs.readFileSync(path.join(WEEK6, "mf_checkpoint.json"), "utf8"));

  const split = splitByTimestamp(ratings);
  const idx = buildSplitIndices(split);
  const graph = buildGraph(split.train);
  const testCohort = sharedCohort(idx, "test");

  // Train rating count per movie (any rating, for the long-tail floor).
  const trainCountByMovie = new Map();
  for (const r of split.train) trainCountByMovie.set(r.movieId, (trainCountByMovie.get(r.movieId) || 0) + 1);

  // Train 4-5 per movie (popularity) for the mean-popularity metric.
  // idx.popularity already holds the distinct-user count of train ratings >= 4.

  const out = [];
  out.push("HW6 business comparison (pure JS reference)");
  out.push(`split train/val/test = ${split.train.length}/${split.val.length}/${split.test.length}`);
  out.push(`graph = ${graph.nUsers} users + ${graph.nMovies} movies = ${graph.n} nodes`);
  out.push(`test cohort = ${testCohort.length}`);
  out.push(`PPR alpha = ${ALPHA.toFixed(2)} (selected on validation); K = ${K}`);
  out.push(`MF model = week6/mf_checkpoint.json (mu=${model.globalMean.toFixed(10)}, ${model.userShape[1]} factors)`);
  out.push("");

  // ---- 1. Method table on the test cohort --------------------------------
  out.push(`=== Method comparison on the test cohort (${testCohort.length} users) ===`);
  out.push("method      Recall@10   conversion   catalog   meanPop   tail<5   slots");

  const pagerankScores = pagerankMap(graph);
  const pprCache = new Map();
  const pprMake = (userId) => {
    if (!pprCache.has(userId)) pprCache.set(userId, pprMap(graph, userId));
    return pprCache.get(userId);
  };
  const popMake = () => popularityMap(idx);
  const mfMake = (userId) => mfMap(model, idx, userId);

  const results = {};
  const defs = [
    ["popularity", popMake],
    ["pagerank", () => pagerankScores],
    ["ppr", pprMake],
    ["mf", mfMake],
  ];
  for (const [name, make] of defs) {
    const r = evaluateCohort(make, idx, testCohort, "test", K, trainCountByMovie);
    results[name] = r;
    out.push(
      `${name.padEnd(11)} ${r.recall.toFixed(6)}   ${(r.conversion * 100).toFixed(1).padStart(5)}%   ` +
      `${String(r.catalog.size).padStart(6)}   ${r.meanPopularity.toFixed(2).padStart(7)}   ` +
      `${(r.tailShare * 100).toFixed(1).padStart(5)}%   ${r.slots}`,
    );
  }
  out.push("");

  // ---- 1b. Secondary popularity: by train rating count (HW5 style) --------
  // The page's `popularity` method uses readme 7.3 (distinct users with >= 4).
  // This variant ranks by the number of train ratings (any value), as in HW5,
  // and reports the same metrics on the same cohort plus per-user hits for the
  // two displayed users.
  out.push("=== Popularity by train rating COUNT (HW5 style, secondary evidence) ===");
  {
    const countMap = ratingCountPopularityMap(idx, trainCountByMovie);
    const r = evaluateCohort(() => countMap, idx, testCohort, "test", K, trainCountByMovie);
    out.push(
      `popularity_by_count ${r.recall.toFixed(6)}   ${(r.conversion * 100).toFixed(1).padStart(5)}%   ` +
      `${String(r.catalog.size).padStart(6)}   meanTrainRatings=${r.meanPopularity.toFixed(2)}   ${r.slots} slots`,
    );
    const u1c = testCohort[0];
    const u2c = testCohort[1];
    for (const [n, uid] of [[1, u1c], [2, u2c]]) {
      const grades = idx.gradesBySplit.test.get(uid);
      const excluded = exclusionSet(idx, "test", uid);
      const relevantSet = new Set();
      if (grades) for (const [movieId, rating] of grades) {
        if (rating >= POSITIVE && idx.candidateSet.has(movieId) && !excluded.has(movieId)) relevantSet.add(movieId);
      }
      const recs = recommendTopK(countMap, idx, uid, "test", K);
      let hits = 0;
      for (const rec of recs) if (relevantSet.has(rec.movieId)) hits++;
      out.push(`  user ${n} (id ${uid}): hits in Top-10 = ${hits} / ${relevantSet.size} relevant`);
    }
  }
  out.push("");


  out.push("(averaged over the " + testCohort.length + " cohort users)");
  const pairOverlap = (recA, recB) => {
    let sum = 0;
    let users = 0;
    for (const userId of testCohort) {
      const a = recA.perUser.get(userId);
      const b = recB.perUser.get(userId);
      if (!a || !b) continue;
      const sb = new Set(b);
      let inter = 0;
      for (const m of a) if (sb.has(m)) inter++;
      sum += inter;
      users++;
    }
    return users > 0 ? sum / users : 0;
  };
  out.push(`ppr vs popularity = ${pairOverlap(results.ppr, results.popularity).toFixed(4)}`);
  out.push(`ppr vs mf         = ${pairOverlap(results.ppr, results.mf).toFixed(4)}`);
  out.push(`ppr vs pagerank   = ${pairOverlap(results.ppr, results.pagerank).toFixed(4)}`);
  out.push(`mf  vs popularity = ${pairOverlap(results.mf, results.popularity).toFixed(4)}`);
  out.push(`pagerank vs popularity = ${pairOverlap(results.pagerank, results.popularity).toFixed(4)}`);
  out.push("");

  // ---- 3. MF with a train-rating floor ------------------------------------
  out.push("=== MF with a train-rating floor on candidate movies ===");
  out.push("floor    Recall@10   conversion   catalog   meanPop   tail<5");
  let mf20 = null;
  for (const floor of [5, 20, 50]) {
    const make = (userId) => mfMapWithFloor(model, idx, userId, trainCountByMovie, floor);
    const r = evaluateCohort(make, idx, testCohort, "test", K, trainCountByMovie);
    if (floor === 20) mf20 = r;
    out.push(
      `>=${String(floor).padEnd(6)} ${r.recall.toFixed(6)}   ${(r.conversion * 100).toFixed(1).padStart(5)}%   ` +
      `${String(r.catalog.size).padStart(6)}   ${r.meanPopularity.toFixed(2).padStart(7)}   ${(r.tailShare * 100).toFixed(1).padStart(5)}%`,
    );
  }
  out.push("");

  // ---- 3b. Union conversion coverage: ppr Top-10 OR MF(>=20) Top-10 --------
  // How many cohort users have at least one test 4-5 movie in EITHER list.
  out.push("=== Union conversion coverage: ppr Top-10 OR MF(>=20 ratings) Top-10 ===");
  {
    const grades = idx.gradesBySplit.test;
    let unionConverted = 0;
    let pprConverted = 0;
    let mf20Converted = 0;
    let evaluated = 0;
    for (const userId of testCohort) {
      const gradeMap = grades.get(userId);
      const excluded = exclusionSet(idx, "test", userId);
      const relevantSet = new Set();
      if (gradeMap) for (const [movieId, rating] of gradeMap) {
        if (rating >= POSITIVE && idx.candidateSet.has(movieId) && !excluded.has(movieId)) relevantSet.add(movieId);
      }
      if (relevantSet.size === 0) continue;
      evaluated++;
      const pprList = results.ppr.perUser.get(userId) || [];
      const mfList = (mf20 && mf20.perUser.get(userId)) || [];
      let hitPpr = false;
      for (const m of pprList) if (relevantSet.has(m)) { hitPpr = true; break; }
      let hitMf = false;
      for (const m of mfList) if (relevantSet.has(m)) { hitMf = true; break; }
      if (hitPpr) pprConverted++;
      if (hitMf) mf20Converted++;
      if (hitPpr || hitMf) unionConverted++;
    }
    out.push(`cohort users evaluated = ${evaluated}`);
    out.push(`ppr Top-10 converted              = ${pprConverted} (${((pprConverted / (evaluated || 1)) * 100).toFixed(1)}%)`);
    out.push(`MF(>=20) Top-10 converted          = ${mf20Converted} (${((mf20Converted / (evaluated || 1)) * 100).toFixed(1)}%)`);
    out.push(`union (ppr OR MF>=20) converted   = ${unionConverted} (${((unionConverted / (evaluated || 1)) * 100).toFixed(1)}%)`);
  }
  out.push("");

  // ---- 4. User cards 1 and 2 ---------------------------------------------
  for (const n of [1, 2]) {
    const userId = testCohort[n - 1];
    const card = userCard(userId, idx, split, movies, graph, model, pprCache, trainCountByMovie);
    out.push(...card);
    out.push("");
  }

  // ---- 5. Within-method overlap for users 1 and 2 -------------------------
  out.push("=== Within-method Top-10 overlap between users 1 and 2 ===");
  const u1 = testCohort[0];
  const u2 = testCohort[1];
  for (const [name, record] of [["ppr", results.ppr], ["mf", results.mf], ["popularity", results.popularity], ["pagerank", results.pagerank]]) {
    const a = new Set(record.perUser.get(u1) || []);
    const b = new Set(record.perUser.get(u2) || []);
    let inter = 0;
    for (const m of a) if (b.has(m)) inter++;
    out.push(`${name.padEnd(11)} common movies (user ${u1} vs user ${u2}) = ${inter}`);
  }
  out.push("");

  // ---- 6. One-factor demo -------------------------------------------------
  out.push("=== One-factor demo ===");
  const list1_85 = demoList(graph, idx, movies, u1, 0.85, trainCountByMovie);
  const list2_85 = demoList(graph, idx, movies, u2, 0.85, trainCountByMovie);
  const list1_50 = demoList(graph, idx, movies, u1, 0.50, trainCountByMovie);
  const overlapIds = (a, b) => {
    const sb = new Set(b.ids);
    let inter = 0;
    for (const m of a.ids) if (sb.has(m)) inter++;
    return inter;
  };
  out.push("A. user 1 @ alpha 0.85");
  out.push(`   list = ${list1_85.list}`);
  out.push(`   mean list popularity = ${list1_85.meanPop.toFixed(2)}`);
  out.push(`   common with user 2 @ 0.85 = ${overlapIds(list1_85, list2_85)}`);
  out.push("B. user 2 @ alpha 0.85");
  out.push(`   list = ${list2_85.list}`);
  out.push(`   mean list popularity = ${list2_85.meanPop.toFixed(2)}`);
  out.push(`   common with user 1 @ 0.85 = ${overlapIds(list2_85, list1_85)}`);
  out.push("C. user 1 @ alpha 0.50");
  out.push(`   list = ${list1_50.list}`);
  out.push(`   mean list popularity = ${list1_50.meanPop.toFixed(2)}`);
  out.push(`   common with A (user 1 @ 0.85) = ${overlapIds(list1_50, list1_85)}`);
  out.push(`   mean popularity change vs A = ${(list1_50.meanPop - list1_85.meanPop).toFixed(2)}`);
  out.push("");

  const txt = out.join("\n") + "\n";
  fs.writeFileSync(path.join(RESULTS, "users.txt"), txt);
  console.log(txt);
  console.log("wrote results/users.txt");
}

/**
 * Full card for one user: train profile, genre mix, Top-10 ppr and mf.
 *
 * @returns {string[]} lines
 */
function userCard(userId, idx, split, movies, graph, model, pprCache, trainCountByMovie) {
  const lines = [];
  const trainRows = split.train.filter((r) => r.userId === userId);
  const testRows = split.test.filter((r) => r.userId === userId);
  const trainPos = trainRows.filter((r) => r.rating >= POSITIVE);
  const testPos = new Set(testRows.filter((r) => r.rating >= POSITIVE).map((r) => r.movieId));

  lines.push(`=== User card: user ${userId} ===`);
  lines.push(`train ratings = ${trainRows.length}; of them >= 4 = ${trainPos.length}`);
  lines.push(`test ratings = ${testRows.length}; of them >= 4 = ${testPos.size}`);

  // Genre profile over train ratings (share of ratings tagged with each genre).
  const genreCount = new Map();
  for (const r of trainRows) {
    const info = movies.get(r.movieId);
    if (!info) continue;
    for (const g of info.genres) genreCount.set(g, (genreCount.get(g) || 0) + 1);
  }
  const genreSorted = Array.from(genreCount.entries()).sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]));
  lines.push("train genre profile (share of train ratings):");
  for (const [g, c] of genreSorted.slice(0, 8)) {
    lines.push(`  ${g.padEnd(12)} ${((c / trainRows.length) * 100).toFixed(1)}%  (${c})`);
  }

  // Example train 5-star movies.
  const fives = trainRows.filter((r) => r.rating === 5).slice(0, 6);
  lines.push(`examples of train 5-star movies (${trainRows.filter((r) => r.rating === 5).length} total, first 6):`);
  for (const r of fives) {
    const info = movies.get(r.movieId);
    lines.push(`  ${String(r.movieId).padStart(4)}  ${info ? info.title : "Movie #" + r.movieId}  [${info ? info.genres.join(",") : "?"}]`);
  }

  for (const method of ["ppr", "mf"]) {
    let map;
    if (method === "ppr") {
      if (!pprCache.has(userId)) pprCache.set(userId, pprMap(graph, userId));
      map = pprCache.get(userId);
    } else {
      map = mfMap(model, idx, userId);
    }
    const recs = recommendTopK(map, idx, userId, "test", K);
    lines.push(`${method.toUpperCase()} Top-10 (alpha ${method === "ppr" ? ALPHA.toFixed(2) : "-"}):`);
    lines.push(`  ${"id".padStart(4)}  ${"title".padEnd(42)} ${"pop".padStart(4)}  conv  genres`);
    for (const rec of recs) {
      const info = movies.get(rec.movieId);
      const conv = testPos.has(rec.movieId) ? "yes" : "-";
      lines.push(
        `  ${String(rec.movieId).padStart(4)}  ${(info ? info.title : "Movie #" + rec.movieId).slice(0, 42).padEnd(42)} ` +
        `${String(rec.popularity).padStart(4)}  ${conv.padEnd(4)}  ${info ? info.genres.join(",") : "?"}`,
      );
    }
  }
  return lines;
}

/**
 * One demo list at a given alpha plus its ids and mean popularity.
 */
function demoList(graph, idx, movies, userId, alpha, trainCountByMovie) {
  const node = graph.userIndexOf.get(userId);
  const p = propagate(graph, alpha, oneHot(graph.n, node)).p;
  const map = movieScoreMap(p, graph);
  const recs = recommendTopK(map, idx, userId, "test", K);
  let meanPop = 0;
  for (const rec of recs) meanPop += rec.popularity;
  meanPop = recs.length ? meanPop / recs.length : 0;
  return {
    ids: recs.map((r) => r.movieId),
    list: recs.map((r) => r.movieId).join(","),
    meanPop,
  };
}

main();
