/**
 * week5-checks/users.js — §5.2 analysis: do the lists open something new or
 * repeat past tastes, and what do they give a streaming platform?
 *
 * Pure JavaScript, no packages, no TensorFlow.js. Uses results/factors_final.json
 * and the same chronological 80/10/10 split as week5/data.js (mirrored here, the
 * same way audit.js / reference.js do). week5/ is never modified.
 *
 * Part 0: re-check of user 1's first three test rows (from reference.js).
 * Part 1: users 1 and 2 — history size, mean rating, genre mix vs overall train
 *         mix, some 5-star films; their Top-10 (title, year, genres, score,
 *         train count); overlap between the two lists; whether the Top-10
 *         contains films they later rated 4–5 in test.
 * Part 2: across all 861 evaluable users, compare six Top-10 strategies:
 *         MF (page), Popular (most train-rated, excluding the user's own
 *         train/val), and MF restricted to train-count >= 5 / 20 / 50 / 100.
 *         For each: Recall@10, #users whose list has ≥1 later test-4–5 film,
 *         #distinct films, share of slots with <5 train ratings, and #slots
 *         filled by a same-title twin of a film the user already rated.
 *
 * Run:  node week5-checks/users.js
 * Writes: week5-checks/results/users.txt
 */

"use strict";

const fs = require("fs");
const path = require("path");

const WEEK5 = path.join(__dirname, "..", "week5");
const RESULTS = path.join(__dirname, "results");

const OUT = [];
const say = (s = "") => OUT.push(s);
const f = (x, d = 4) => (Number.isFinite(x) ? x.toFixed(d) : String(x));
const eff = (x, d = 4) => (x * 100).toFixed(d) + "%";

// ---------------------------------------------------------------------------
// Data layer — mirrors week5/data.js
// ---------------------------------------------------------------------------

const genreNames = [
  "Action", "Adventure", "Animation", "Children's", "Comedy",
  "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir",
  "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
  "Thriller", "War", "Western",
];

function parseItemData(text) {
  const movies = [];
  for (const line of text.split("\n")) {
    if (line.trim() === "") continue;
    const fields = line.split("|");
    if (fields.length < 24) continue;
    const id = parseInt(fields[0]);
    const title = fields[1];
    const genreValues = fields.slice(6, 24).map((v) => parseInt(v));
    const genres = genreNames.filter((_, i) => genreValues[i] === 1);
    movies.push({ id, title, genres });
  }
  return movies;
}

function parseRatingData(text) {
  const ratings = [];
  for (const line of text.split("\n")) {
    if (line.trim() === "") continue;
    const fields = line.split("\t");
    if (fields.length < 4) continue;
    ratings.push({
      userId: parseInt(fields[0]),
      movieId: parseInt(fields[1]),
      rating: parseFloat(fields[2]),
      timestamp: parseInt(fields[3]),
    });
  }
  return ratings;
}

function splitByTimestamp(ratings) {
  const MIN_RATINGS = 5;
  const byUser = new Map();
  for (const r of ratings) {
    if (!byUser.has(r.userId)) byUser.set(r.userId, []);
    byUser.get(r.userId).push(r);
  }
  const train = [];
  const val = [];
  const test = [];
  for (const [, userRatings] of byUser) {
    const n = userRatings.length;
    if (n < MIN_RATINGS) continue;
    const sorted = userRatings
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

const ratings = parseRatingData(fs.readFileSync(path.join(WEEK5, "u.data"), "utf8"));
const movies = parseItemData(fs.readFileSync(path.join(WEEK5, "u.item")).toString("latin1"));
const { train, val, test } = splitByTimestamp(ratings);

const titleById = new Map(movies.map((m) => [m.id, m.title]));
const genresById = new Map(movies.map((m) => [m.id, m.genres]));
const movieIds = movies.map((m) => m.id);
const candidateMovieSet = new Set(train.map((r) => r.movieId));

const yearOf = (id) => {
  const m = (titleById.get(id) || "").match(/\((\d{4})\)\s*$/);
  return m ? m[1] : "?";
};

// Duplicate-title groups (same title, different id).
const idsByTitle = new Map();
for (const m of movies) {
  if (!idsByTitle.has(m.title)) idsByTitle.set(m.title, []);
  idsByTitle.get(m.title).push(m.id);
}
const twinsOf = (id) => idsByTitle.get(titleById.get(id)) || [id];

// Train popularity.
const trainCount = new Map();
for (const r of train) trainCount.set(r.movieId, (trainCount.get(r.movieId) || 0) + 1);
const countOf = (id) => trainCount.get(id) || 0;

// Per-user facts.
const ratedTV = new Map(); // userId -> Set(movieId) over train + val
for (const r of train.concat(val)) {
  if (!ratedTV.has(r.userId)) ratedTV.set(r.userId, new Set());
  ratedTV.get(r.userId).add(r.movieId);
}
const historyByUser = new Map(); // userId -> [{movieId, rating, split}]
for (const r of train) {
  if (!historyByUser.has(r.userId)) historyByUser.set(r.userId, []);
  historyByUser.get(r.userId).push({ movieId: r.movieId, rating: r.rating });
}
for (const r of val) {
  if (!historyByUser.has(r.userId)) historyByUser.set(r.userId, []);
  historyByUser.get(r.userId).push({ movieId: r.movieId, rating: r.rating });
}

const testByUser = new Map();
for (const r of test) {
  if (!testByUser.has(r.userId)) testByUser.set(r.userId, []);
  testByUser.get(r.userId).push(r);
}

// Evaluable users: >=1 relevant test movie among the training candidate pool.
const relevantByUser = new Map();
for (const r of test) {
  if (r.rating >= 4.0 && candidateMovieSet.has(r.movieId)) {
    if (!relevantByUser.has(r.userId)) relevantByUser.set(r.userId, new Set());
    relevantByUser.get(r.userId).add(r.movieId);
  }
}
const likeByUser = new Map(); // test rating >= 4, any candidate status
for (const r of test) {
  if (r.rating >= 4.0) {
    if (!likeByUser.has(r.userId)) likeByUser.set(r.userId, new Set());
    likeByUser.get(r.userId).add(r.movieId);
  }
}
const evaluableUsers = [...relevantByUser.keys()];

// ---------------------------------------------------------------------------
// Factors + scoring
// ---------------------------------------------------------------------------

const fac = JSON.parse(fs.readFileSync(path.join(RESULTS, "factors_final.json"), "utf8"));
const F = fac.userShape[1];
const mu = fac.globalMean;
const U = fac.userFactors;
const V = fac.itemFactors;

const dot = (userId, movieId) => {
  const u = U[userId];
  const v = V[movieId];
  let d = 0;
  for (let k = 0; k < F; k++) d += u[k] * v[k];
  return d;
};
const scoreMF = (userId, movieId) => mu + dot(userId, movieId);

function recommendMF(userId, K, minCount = 0) {
  const excluded = ratedTV.get(userId) || new Set();
  const cand = [];
  for (const id of movieIds) {
    if (excluded.has(id)) continue;
    if (countOf(id) < minCount) continue;
    cand.push({ movieId: id, score: scoreMF(userId, id) });
  }
  cand.sort((a, b) => b.score - a.score || a.movieId - b.movieId);
  return cand.slice(0, K);
}

const popularRanked = movieIds
  .slice()
  .sort((a, b) => countOf(b) - countOf(a) || a - b);

function recommendPopular(userId, K) {
  const excluded = ratedTV.get(userId) || new Set();
  const out = [];
  for (const id of popularRanked) {
    if (excluded.has(id)) continue;
    out.push({ movieId: id, score: countOf(id) });
    if (out.length >= K) break;
  }
  return out;
}

// ---------------------------------------------------------------------------
// Part 0. Re-check of user 1's first three test rows
// ---------------------------------------------------------------------------

const user1Test = testByUser.get(1) || [];
const first3 = user1Test.slice(0, 3);

say("HW5 §5.2 — user analysis (pure JavaScript, from results/factors_final.json)");
say("=".repeat(78));
say("");

say("Part 0. User 1 — first three test rows (clamped prediction, 4 dp; compute RMSE yourself)");
say("-".repeat(78));
say("movieId\ttitle\tactual\tpredicted");
let sq = 0;
for (const r of first3) {
  const p = Math.min(5, Math.max(1, scoreMF(1, r.movieId)));
  sq += (p - r.rating) * (p - r.rating);
  say(`${r.movieId}\t${titleById.get(r.movieId)}\t${r.rating}\t${p.toFixed(4)}`);
}
const u1Rel = relevantByUser.get(1) || new Set();
const u1Top = recommendMF(1, 10).map((r) => r.movieId);
const u1Hits = [...u1Rel].filter((id) => u1Top.includes(id)).length;
say("");
say(`RMSE of the three rows        : ${f(Math.sqrt(sq / 3))}  (sum of squares ${f(sq, 5)})`);
say(`user 1 recall@10              : ${u1Hits} / ${u1Rel.size} = ${f(u1Rel.size ? u1Hits / u1Rel.size : 0)}`);

// ---------------------------------------------------------------------------
// Part 1. Users 1 and 2
// ---------------------------------------------------------------------------

function genreShare(ratingList) {
  const counts = new Map(genreNames.map((g) => [g, 0]));
  for (const r of ratingList) {
    for (const g of genresById.get(r.movieId) || []) counts.set(g, counts.get(g) + 1);
  }
  const total = ratingList.length || 1;
  return { counts, total };
}

const overallGenres = (() => {
  const counts = new Map(genreNames.map((g) => [g, 0]));
  for (const r of train) {
    for (const g of genresById.get(r.movieId) || []) counts.set(g, counts.get(g) + 1);
  }
  return { counts, total: train.length };
})();

function describeUser(userId) {
  const hist = historyByUser.get(userId) || [];
  const mean = hist.reduce((a, r) => a + r.rating, 0) / (hist.length || 1);
  const ges = genreShare(hist);
  const top = recommendMF(userId, 10);

  say("");
  say(`User ${userId}`);
  say("-".repeat(78));
  say(`history (train+val) ratings : ${hist.length}`);
  say(`mean rating                 : ${f(mean)}`);
  const fives = hist.filter((r) => r.rating === 5).map((r) => r.movieId);
  say(`5-star films (up to 6)      : ${fives.slice(0, 6).map((id) => `${titleById.get(id)} [${id}]`).join("; ") || "(none)"}`);

  say("");
  say("genre mix (share of this user's history vs share of train):");
  const rows = genreNames
    .map((g) => ({
      g,
      user: ges.counts.get(g) / ges.total,
      all: overallGenres.counts.get(g) / overallGenres.total,
    }))
    .sort((a, b) => b.user - a.user);
  for (const row of rows) {
    if (row.user === 0 && row.all === 0) continue;
    say(`  ${row.g.padEnd(12)} user ${eff(row.user, 2).padStart(7)}   train ${eff(row.all, 2).padStart(7)}`);
  }

  say("");
  say("Top-10 (recommendMF):");
  say("  #\tmovieId\ttitle\tyear\tgenres\tscore\ttrainCount");
  top.forEach((r, i) => {
    const id = r.movieId;
    const clean = (titleById.get(id) || "").replace(/\s*\(\d{4}\)\s*$/, "");
    say(`  ${i + 1}\t${id}\t${clean}\t${yearOf(id)}\t${(genresById.get(id) || []).join(", ") || "(none)"}\t${f(r.score)}\t${countOf(id)}`);
  });

  const like = likeByUser.get(userId) || new Set();
  const later = top.filter((r) => like.has(r.movieId)).map((r) => r.movieId);
  say("");
  say(`in Top-10 and later rated 4–5 in test: ${later.length ? later.map((id) => `${titleById.get(id)} [${id}]`).join("; ") : "(none)"}`);
  return top.map((r) => r.movieId);
}

say("");
say("Part 1. Users 1 and 2 — what the lists surface");
say("=".repeat(78));
const ids1 = describeUser(1);
const ids2 = describeUser(2);
const common = ids1.filter((id) => ids2.includes(id));
say("");
say(`Top-10 overlap between users 1 and 2: ${common.length} — ${common.map((id) => `${titleById.get(id)} [${id}]`).join("; ") || "(none)"}`);

// ---------------------------------------------------------------------------
// Part 2. All evaluable users — six strategies
// ---------------------------------------------------------------------------

const strategies = [
  { name: "MF (page)", list: (u) => recommendMF(u, 10, 0) },
  { name: "Popular", list: (u) => recommendPopular(u, 10) },
  { name: "MF >= 5", list: (u) => recommendMF(u, 10, 5) },
  { name: "MF >= 20", list: (u) => recommendMF(u, 10, 20) },
  { name: "MF >= 50", list: (u) => recommendMF(u, 10, 50) },
  { name: "MF >= 100", list: (u) => recommendMF(u, 10, 100) },
];

const results = [];
for (const strat of strategies) {
  let recallSum = 0;
  let usersWithNewLike = 0;
  let slots = 0;
  let lowPopSlots = 0;
  let twinSlots = 0;
  const distinct = new Set();

  for (const userId of evaluableUsers) {
    const list = strat.list(userId);
    const relevant = relevantByUser.get(userId);
    const like = likeByUser.get(userId) || new Set();
    const rated = ratedTV.get(userId) || new Set();

    let hit = 0;
    let sawLike = false;
    for (const { movieId } of list) {
      distinct.add(movieId);
      slots++;
      if (countOf(movieId) < 5) lowPopSlots++;
      const twins = twinsOf(movieId);
      if (twins.length > 1) {
        for (const t of twins) {
          if (t !== movieId && rated.has(t)) {
            twinSlots++;
            break;
          }
        }
      }
      if (relevant.has(movieId)) hit++;
      if (like.has(movieId)) sawLike = true;
    }
    recallSum += relevant.size ? hit / relevant.size : 0;
    if (sawLike) usersWithNewLike++;
  }

  results.push({
    name: strat.name,
    recall: recallSum / evaluableUsers.length,
    usersWithNewLike,
    distinct: distinct.size,
    slots,
    lowPopShare: slots ? lowPopSlots / slots : 0,
    twinSlots,
  });
}

say("");
say("Part 2. Six Top-10 strategies over all evaluable users");
say("=".repeat(78));
say(`evaluable users (>=1 relevant test movie in the train pool): ${evaluableUsers.length}`);
say("");
say("strategy      Recall@10   users w/ later 4–5   distinct films   <5-train slots   twin slots");
say("-".repeat(78));
for (const r of results) {
  say(
    `${r.name.padEnd(13)} ${f(r.recall, 4).padStart(8)}   ${String(r.usersWithNewLike).padStart(14)}   ${String(r.distinct).padStart(14)}   ${eff(r.lowPopShare, 2).padStart(13)}   ${String(r.twinSlots).padStart(9)}`,
  );
}
say("");
say(`(slots total per strategy: ${results[0].slots} = ${evaluableUsers.length} users x up to 10)`);

// ---------------------------------------------------------------------------
// Short conclusion
// ---------------------------------------------------------------------------

const mf = results.find((r) => r.name === "MF (page)");
const pop = results.find((r) => r.name === "Popular");
const mf100 = results.find((r) => r.name === "MF >= 100");

say("");
say("What follows");
say("-".repeat(78));
say(`- Popular wins Recall@10 (${f(pop.recall, 4)} vs MF ${f(mf.recall, 4)}) but shows only ${pop.distinct} distinct films total: almost every user gets the same safe hits, so it repeats global taste and opens nothing new.`);
say(`- MF shows many more distinct films (${mf.distinct} vs ${pop.distinct}) but has low recall and fills ${eff(mf.lowPopShare, 2)} of its slots with films that have <5 train ratings — it trusts barely-trained long-tail rows.`);
say(`- Restricting MF to popular films trades novelty for accuracy: MF >= 100 keeps recall ${f(mf100.recall, 4)} with 0.00% <5-train slots, and still ${mf100.distinct} distinct films.`);
say(`- "users w/ later 4–5": Popular ${pop.usersWithNewLike} vs MF ${mf.usersWithNewLike} — how often a list already contains a film the user later liked; Popular clears this mostly because popular films are widely liked.`);
say(`- twin slots (${mf.twinSlots} for MF, ${results.find((r) => r.name === "MF >= 50").twinSlots} for MF >= 50) are pure waste: a duplicate title of a film the user already rated, offered because the model has no title-level identity.`);
say(`- For a streaming platform: a high-novelty / low-precision MF list is an exploration and onboarding tool; a popular list is a safe fallback; a popularity floor sits in between.`);

const text = OUT.join("\n") + "\n";
fs.mkdirSync(RESULTS, { recursive: true });
fs.writeFileSync(path.join(RESULTS, "users.txt"), text);
process.stdout.write(text);
