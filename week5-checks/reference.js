/**
 * week5-checks/reference.js — independent, package-free check of the HW5 model.
 *
 * Node-only, no external packages (no TensorFlow.js). It re-implements the data
 * layer (mirroring week5/data.js), the metrics (clamped to [1, 5], like
 * week5/script.js `computeRMSEFor`), `recommendTopK` and `recallAtK` in plain
 * JavaScript, and replays the training with a hand-written Adam identical to
 * TF.js's `AdamOptimizer`.
 *
 * Parts:
 *   1. From results/factors_final.json: train/val/test RMSE, Top-10 for users
 *      1 and 2, Recall@10 (with exclusions) — compared against probe_fixed.txt.
 *   2. From results/factors_init.json: replay all 20 epochs with the same Adam
 *      (β1=0.9, β2=0.999, ε=1e-7, lr=0.01, reg=0.05, batch=64, mulberry32(19)
 *      Fisher–Yates) under two L2 variants — (a) penalty over the batch rows (as
 *      implemented) and (b) penalty over the whole factor matrix (as in the
 *      hint). Val RMSE per epoch is printed next to the page's history; the
 *      matching variant and the selected epoch are reported.
 *   3. Max absolute discrepancy between the page's saved factors and the best
 *      epoch's factors from the replay.
 *   4. The first three test rows of user 1 (movieId, title, actual, predicted to
 *      4 decimals), plus how many of that user's test movies are relevant
 *      (rating >= 4) and how many land in the Top-10. RMSE of those rows and
 *      user 1's recall are deliberately NOT computed here.
 *
 * Run:  node week5-checks/reference.js
 * Writes: week5-checks/results/reference.txt
 */

"use strict";

const fs = require("fs");
const path = require("path");

const WEEK5 = path.join(__dirname, "..", "week5");
const RESULTS = path.join(__dirname, "results");
const U_DATA = path.join(WEEK5, "u.data");
const U_ITEM = path.join(WEEK5, "u.item");

const HYPER = { epochs: 20, lr: 0.01, reg: 0.05, batchSize: 64, seed: 19 };
const ADAM = { beta1: 0.9, beta2: 0.999, epsilon: 1e-7 };
const OUT = [];
const say = (s = "") => OUT.push(s);
const f = (x, d = 6) => (Number.isFinite(x) ? x.toFixed(d) : String(x));

// ---------------------------------------------------------------------------
// Parsing — mirrors week5/data.js / week5-checks/audit.js
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
    const genreValues = fields.slice(6, 24).map((value) => parseInt(value));
    const genres = genreNames.filter((_, index) => genreValues[index] === 1);
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

// ---------------------------------------------------------------------------
// Data
// ---------------------------------------------------------------------------

const ratings = parseRatingData(fs.readFileSync(U_DATA, "utf8"));
const movies = parseItemData(fs.readFileSync(U_ITEM).toString("latin1"));
const { train, val, test } = splitByTimestamp(ratings);

const titleById = new Map(movies.map((m) => [m.id, m.title]));
const movieTitle = (id) => (titleById.has(id) ? titleById.get(id) : `Movie ${id}`);

// ---------------------------------------------------------------------------
// Factors
// ---------------------------------------------------------------------------

const finalFactors = JSON.parse(fs.readFileSync(path.join(RESULTS, "factors_final.json"), "utf8"));
const initFactors = JSON.parse(fs.readFileSync(path.join(RESULTS, "factors_init.json"), "utf8"));

const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

function flatten(nested) {
  const R = nested.length;
  const C = nested[0].length;
  const a = new Float64Array(R * C);
  for (let r = 0; r < R; r++) {
    const row = nested[r];
    for (let c = 0; c < C; c++) a[r * C + c] = row[c];
  }
  return { a, R, C };
}

// ---------------------------------------------------------------------------
// Metrics — clamped prediction, matching computeRMSEFor / predictRating
// ---------------------------------------------------------------------------

function dot(U, V, F, uRow, iRow) {
  const ub = uRow * F;
  const ib = iRow * F;
  let d = 0;
  for (let k = 0; k < F; k++) d += U[ub + k] * V[ib + k];
  return d;
}

function predict(U, V, F, mu, userId, movieId) {
  return clamp(mu + dot(U, V, F, userId, movieId), 1, 5);
}

function rmseFor(U, V, F, mu, split) {
  if (!split.length) return 0;
  let s = 0;
  for (const r of split) {
    const e = predict(U, V, F, mu, r.userId, r.movieId) - r.rating;
    s += e * e;
  }
  return Math.sqrt(s / split.length);
}

// ---------------------------------------------------------------------------
// recommendTopK / recallAtK — plain-JS mirrors of week5/script.js
// ---------------------------------------------------------------------------

function recommendTopK(U, V, F, mu, userId, K) {
  if (K <= 0) return [];
  const excluded = new Set();
  for (const r of train) if (r.userId === userId) excluded.add(r.movieId);
  for (const r of val) if (r.userId === userId) excluded.add(r.movieId);

  const cand = [];
  for (const m of movies) {
    if (excluded.has(m.id)) continue;
    cand.push({ movieId: m.id, title: m.title, score: mu + dot(U, V, F, userId, m.id) });
  }
  cand.sort((a, b) => b.score - a.score || a.movieId - b.movieId);
  return cand.slice(0, K);
}

function recallAtK(U, V, F, mu, K) {
  const candidates = new Set(train.map((r) => r.movieId));

  const relevantByUser = new Map();
  for (const r of test) {
    if (r.rating >= 4.0 && candidates.has(r.movieId)) {
      if (!relevantByUser.has(r.userId)) relevantByUser.set(r.userId, new Set());
      relevantByUser.get(r.userId).add(r.movieId);
    }
  }

  const unseenIds = [];
  const seen = new Set();
  for (const r of test) {
    if (r.rating >= 4.0 && !candidates.has(r.movieId) && !seen.has(r.movieId)) {
      seen.add(r.movieId);
      unseenIds.push(r.movieId);
    }
  }

  const usersWithoutRelevant = [];
  for (const userId of new Set(test.map((r) => r.userId))) {
    const rel = relevantByUser.get(userId);
    if (!rel || rel.size === 0) usersWithoutRelevant.push(userId);
  }

  const perUser = [];
  let sum = 0;
  for (const [userId, rel] of relevantByUser) {
    const recs = recommendTopK(U, V, F, mu, userId, K);
    const recommended = new Set(recs.map((r) => r.movieId));
    let hit = 0;
    for (const id of rel) if (recommended.has(id)) hit++;
    perUser.push({ userId, relevant: rel.size, hit });
    sum += hit / rel.size;
  }

  return {
    overallRecall: perUser.length ? sum / perUser.length : 0,
    perUser,
    excluded: { unseenIds, usersWithoutRelevant },
  };
}

// ---------------------------------------------------------------------------
// mulberry32 + hand-written Adam (identical to tfjs AdamOptimizer)
// ---------------------------------------------------------------------------

function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// newFirstMoment = β1·m + (1−β1)·g
// newSecondMoment = β2·v + (1−β2)·g²
// value += −lr · (m/(1−β1^t)) / (sqrt(v/(1−β2^t)) + ε)
function adamUpdate(param, m, v, grad, o1, o2) {
  const b1 = ADAM.beta1;
  const b2 = ADAM.beta2;
  const c1 = 1 - b1;
  const c2 = 1 - b2;
  const lr = HYPER.lr;
  const eps = ADAM.epsilon;
  for (let i = 0; i < param.length; i++) {
    const g = grad[i];
    const mi = b1 * m[i] + c1 * g;
    const vi = b2 * v[i] + c2 * g * g;
    m[i] = mi;
    v[i] = vi;
    param[i] -= lr * (mi / o1) / (Math.sqrt(vi / o2) + eps);
  }
}

function replay(init, variant) {
  const F = init.numFactors;
  const uRows = init.userShape[0];
  const iRows = init.itemShape[0];
  const mu = init.globalMean;

  const U = Float64Array.from(init.userFactors.flat());
  const V = Float64Array.from(init.itemFactors.flat());
  const mU = new Float64Array(uRows * F);
  const vU = new Float64Array(uRows * F);
  const mV = new Float64Array(iRows * F);
  const vV = new Float64Array(iRows * F);
  const gradU = new Float64Array(uRows * F);
  const gradV = new Float64Array(iRows * F);

  const n = train.length;
  const userIdx = Int32Array.from(train, (r) => r.userId);
  const itemIdx = Int32Array.from(train, (r) => r.movieId);
  const target = Float64Array.from(train, (r) => r.rating);

  const size = Math.min(Math.max(1, Math.round(HYPER.batchSize)), n);
  const order = new Int32Array(n);
  for (let i = 0; i < n; i++) order[i] = i;
  const rnd = mulberry32(HYPER.seed);

  let pow1 = 1;
  let pow2 = 1;

  const history = [];
  let bestVal = Infinity;
  let best = null;

  for (let epoch = 1; epoch <= HYPER.epochs; epoch++) {
    for (let k = n - 1; k > 0; k--) {
      const j = Math.floor(rnd() * (k + 1));
      const tmp = order[k];
      order[k] = order[j];
      order[j] = tmp;
    }

    for (let start = 0; start < n; start += size) {
      const end = Math.min(start + size, n);
      const B = end - start;
      const regc = (2 * HYPER.reg) / B;

      gradU.fill(0);
      gradV.fill(0);

      if (variant === "matrix") {
        for (let i = 0; i < gradU.length; i++) gradU[i] = 2 * HYPER.reg * U[i];
        for (let i = 0; i < gradV.length; i++) gradV[i] = 2 * HYPER.reg * V[i];
      }

      for (let b = start; b < end; b++) {
        const idx = order[b];
        const ru = userIdx[idx];
        const ri = itemIdx[idx];
        const err = mu + dot(U, V, F, ru, ri) - target[idx];
        const coef = ((2 * err) / B);
        const ub = ru * F;
        const ib = ri * F;
        if (variant === "matrix") {
          for (let k = 0; k < F; k++) {
            gradU[ub + k] += coef * V[ib + k];
            gradV[ib + k] += coef * U[ub + k];
          }
        } else {
          for (let k = 0; k < F; k++) {
            gradU[ub + k] += coef * V[ib + k] + regc * U[ub + k];
            gradV[ib + k] += coef * U[ub + k] + regc * V[ib + k];
          }
        }
      }

      pow1 *= ADAM.beta1;
      pow2 *= ADAM.beta2;
      const o1 = 1 - pow1;
      const o2 = 1 - pow2;
      adamUpdate(U, mU, vU, gradU, o1, o2);
      adamUpdate(V, mV, vV, gradV, o1, o2);
    }

    const trainRMSE = rmseFor(U, V, F, mu, train);
    const valRMSE = rmseFor(U, V, F, mu, val);
    history.push({ epoch, trainRMSE, valRMSE });
    if (valRMSE < bestVal) {
      bestVal = valRMSE;
      best = { epoch, valRMSE, U: U.slice(), V: V.slice() };
    }
  }

  return { history, best, F, mu };
}

// ---------------------------------------------------------------------------
// probe_fixed.txt parsing
// ---------------------------------------------------------------------------

function parseProbe() {
  const text = fs.readFileSync(path.join(RESULTS, "probe_fixed.txt"), "utf8");
  const num = (re) => {
    const m = text.match(re);
    return m ? Number(m[1]) : null;
  };
  const history = [];
  const histBlock = text.split("Training history (after Train #1)")[1] || "";
  for (const line of histBlock.split("\n")) {
    const m = line.match(/^(\d+)\t([\d.eE+-]+)\t([\d.eE+-]+)\s*$/);
    if (m) history.push({ epoch: Number(m[1]), trainRMSE: Number(m[2]), valRMSE: Number(m[3]) });
  }
  const top = (user) => {
    const after = text.split(`Top-10 for user ${user}`)[1] || "";
    const block = after.split("\n\n")[0];
    const ids = [];
    for (const line of block.split("\n")) {
      const m = line.match(/\[id (\d+)\]/);
      if (m) ids.push(Number(m[1]));
    }
    return ids;
  };
  return {
    testRMSE: num(/computeRMSEFor\(testSet\)\s*:\s*([\d.eE+-]+)/),
    baselineTestRMSE: num(/computeBaselineRMSE\(testSet\)\s*:\s*([\d.eE+-]+)/),
    overallRecall: num(/overallRecall\s*:\s*([\d.eE+-]+)/),
    evaluableUsers: num(/evaluable users\s*:\s*(\d+)/),
    unseenCount: num(/unseenIds \(count\)\s*:\s*(\d+)/),
    usersWithoutRelevantCount: num(/usersWithoutRelevant\s*:\s*(\d+)/),
    history,
    top1: top(1),
    top2: top(2),
  };
}

const maxAbsDiff = (a, b) => {
  let m = 0;
  for (let i = 0; i < a.length; i++) {
    const d = Math.abs(a[i] - b[i]);
    if (d > m) m = d;
  }
  return m;
};

const maxAbsDiffIdx = (a, b) => {
  let m = -1;
  let idx = -1;
  for (let i = 0; i < a.length; i++) {
    const d = Math.abs(a[i] - b[i]);
    if (d > m) {
      m = d;
      idx = i;
    }
  }
  return { m, idx };
};
const verdict = (ok) => (ok ? "MATCH" : "MISMATCH");

// ---------------------------------------------------------------------------
// Run
// ---------------------------------------------------------------------------

const F = finalFactors.userShape[1];
const fU = flatten(finalFactors.userFactors).a;
const fV = flatten(finalFactors.itemFactors).a;
const fMu = finalFactors.globalMean;

const probe = parseProbe();

say("HW5 reference check — pure JavaScript, no TensorFlow.js");
say("=".repeat(78));
say(`Train/val/test : ${train.length} / ${val.length} / ${test.length}`);
say(`Factors        : ${finalFactors.userShape} user, ${finalFactors.itemShape} item, F = ${F}`);
say(`Hyperparameters: epochs ${HYPER.epochs}, lr ${HYPER.lr}, reg ${HYPER.reg}, batch ${HYPER.batchSize}, seed ${HYPER.seed}`);
say(`Adam           : β1 ${ADAM.beta1}, β2 ${ADAM.beta2}, ε ${ADAM.epsilon}`);
say("");

// --- Part 1 ----------------------------------------------------------------
say("Part 1. Metrics recomputed from results/factors_final.json vs probe_fixed.txt");
say("-".repeat(78));

const refTrain = rmseFor(fU, fV, F, fMu, train);
const refVal = rmseFor(fU, fV, F, fMu, val);
const refTest = rmseFor(fU, fV, F, fMu, test);
const refTop1 = recommendTopK(fU, fV, F, fMu, 1, 10);
const refTop2 = recommendTopK(fU, fV, F, fMu, 2, 10);
const refRecall = recallAtK(fU, fV, F, fMu, 10);

// The page restores the best checkpoint (epoch 3); history row 3 is that model.
const pageBest = probe.history.find((h) => h.epoch === 3) || { trainRMSE: NaN, valRMSE: NaN };

say("");
say("RMSE (clamped prediction, matching computeRMSEFor):");
say(`  train : reference ${f(refTrain)}   |  probe(history ep.3) ${f(pageBest.trainRMSE)}   ${verdict(Math.abs(refTrain - pageBest.trainRMSE) < 5e-6)}`);
say(`  val   : reference ${f(refVal)}   |  probe(history ep.3) ${f(pageBest.valRMSE)}   ${verdict(Math.abs(refVal - pageBest.valRMSE) < 5e-6)}`);
say(`  test  : reference ${f(refTest)}   |  probe ${f(probe.testRMSE)}   ${verdict(Math.abs(refTest - probe.testRMSE) < 5e-6)}`);
say(`  baseline(test) probe: ${f(probe.baselineTestRMSE)}`);

say("");
say("Top-10 (recommendTopK):");
const idList = (xs) => xs.join(", ");
say(`  user 1 reference: ${idList(refTop1.map((r) => r.movieId))}`);
say(`  user 1 probe    : ${idList(probe.top1)}   ${verdict(idList(refTop1.map((r) => r.movieId)) === idList(probe.top1))}`);
say(`  user 2 reference: ${idList(refTop2.map((r) => r.movieId))}`);
say(`  user 2 probe    : ${idList(probe.top2)}   ${verdict(idList(refTop2.map((r) => r.movieId)) === idList(probe.top2))}`);

say("");
say("Recall@10 (recallAtK):");
say(`  overallRecall        : reference ${f(refRecall.overallRecall, 12)}  |  probe ${f(probe.overallRecall, 12)}   ${verdict(Math.abs(refRecall.overallRecall - probe.overallRecall) < 1e-9)}`);
say(`  evaluable users      : reference ${refRecall.perUser.length}  |  probe ${probe.evaluableUsers}   ${verdict(refRecall.perUser.length === probe.evaluableUsers)}`);
say(`  unseenIds (count)    : reference ${refRecall.excluded.unseenIds.length}  |  probe ${probe.unseenCount}   ${verdict(refRecall.excluded.unseenIds.length === probe.unseenCount)}`);
say(`  unseenIds            : ${JSON.stringify(refRecall.excluded.unseenIds)}`);
say(`  usersWithoutRelevant : reference ${refRecall.excluded.usersWithoutRelevant.length}  |  probe ${probe.usersWithoutRelevantCount}   ${verdict(refRecall.excluded.usersWithoutRelevant.length === probe.usersWithoutRelevantCount)}`);

// --- Part 2 ----------------------------------------------------------------
say("");
say("Part 2. Replay 20 epochs from results/factors_init.json (same Adam / seed)");
say("-".repeat(78));
say("Val RMSE per epoch: page history | batch-rows penalty (implemented) | whole-matrix penalty (hint)");
say("epoch\tpage_train\tpage_val\tbatchRows_val\tmatrix_val");

const t0 = Date.now();
const reBatch = replay(initFactors, "batch");
const tBatch = (Date.now() - t0) / 1000;
const t1 = Date.now();
const reMatrix = replay(initFactors, "matrix");
const tMatrix = (Date.now() - t1) / 1000;

let maxDiffBatch = 0;
let maxDiffMatrix = 0;
for (let i = 0; i < HYPER.epochs; i++) {
  const page = probe.history[i] || { trainRMSE: NaN, valRMSE: NaN };
  const b = reBatch.history[i];
  const m = reMatrix.history[i];
  maxDiffBatch = Math.max(maxDiffBatch, Math.abs(b.valRMSE - page.valRMSE));
  maxDiffMatrix = Math.max(maxDiffMatrix, Math.abs(m.valRMSE - page.valRMSE));
  say(
    `${b.epoch}\t${f(page.trainRMSE)}\t${f(page.valRMSE)}\t${f(b.valRMSE)}\t${f(m.valRMSE)}`,
  );
}
say("");
say(`Replay wall-clock: batch-rows ${tBatch.toFixed(1)} s, whole-matrix ${tMatrix.toFixed(1)} s`);
say(`Max |valRMSE − page| over 20 epochs: batch-rows ${f(maxDiffBatch)} , whole-matrix ${f(maxDiffMatrix)}`);
const matching = maxDiffBatch <= maxDiffMatrix ? "batch-rows" : "whole-matrix";
say(`Variant matching the page: ${matching}`);
say(`Selected epoch (min valRMSE, earliest tie): batch-rows ${reBatch.best.epoch} (val ${f(reBatch.best.valRMSE)}) , whole-matrix ${reMatrix.best.epoch} (val ${f(reMatrix.best.valRMSE)}) , page epoch 3`);

// --- Part 3 ----------------------------------------------------------------
say("");
say("Part 3. Page factors vs best-epoch replay factors (max abs discrepancy)");
say("-".repeat(78));
const bestRepU = matching === "whole-matrix" ? reMatrix.best.U : reBatch.best.U;
const bestRepV = matching === "whole-matrix" ? reMatrix.best.V : reBatch.best.V;
say(`matching variant : ${matching}, epoch ${matching === "whole-matrix" ? reMatrix.best.epoch : reBatch.best.epoch}`);
const uDiff = maxAbsDiffIdx(fU, bestRepU);
const iDiff = maxAbsDiffIdx(fV, bestRepV);
say(`userFactors max |Δ| : ${uDiff.m.toExponential(6)}  (user ${Math.floor(uDiff.idx / F)}, dim ${uDiff.idx % F}; ${train.filter((r) => r.userId === Math.floor(uDiff.idx / F)).length} train / ${val.filter((r) => r.userId === Math.floor(uDiff.idx / F)).length} val ratings)`);
const worstItem = Math.floor(iDiff.idx / F);
say(`itemFactors max |Δ| : ${iDiff.m.toExponential(6)}  (item ${worstItem}; ${train.filter((r) => r.movieId === worstItem).length} train / ${val.filter((r) => r.movieId === worstItem).length} val ratings)`);
say(`overall    max |Δ| : ${Math.max(uDiff.m, iDiff.m).toExponential(6)}`);
say("");
say("(for reference) batch-rows best-epoch vs page:");
say(`userFactors max |Δ| : ${maxAbsDiff(fU, reBatch.best.U).toExponential(6)}`);
say(`itemFactors max |Δ| : ${maxAbsDiff(fV, reBatch.best.V).toExponential(6)}`);

// --- Part 4 ----------------------------------------------------------------
say("");
say("Part 4. User 1 — first 3 test rows and relevant-item counts (compute RMSE yourself)");
say("-".repeat(78));
const user1Test = test.filter((r) => r.userId === 1);
const first3 = user1Test.slice(0, 3);
say("movieId\ttitle\tactual\tpredicted(clamped, 4dp)");
for (const r of first3) {
  const p = predict(fU, fV, F, fMu, r.userId, r.movieId);
  say(`${r.movieId}\t${movieTitle(r.movieId)}\t${r.rating}\t${p.toFixed(4)}`);
}

const candidates = new Set(train.map((r) => r.movieId));
const relTest = user1Test.filter((r) => r.rating >= 4.0);
const relCand = relTest.filter((r) => candidates.has(r.movieId));
const user1Top10 = refTop1.map((r) => r.movieId);
const topSet = new Set(user1Top10);
const relCandHit = relCand.filter((r) => topSet.has(r.movieId)).length;
say("");
say(`user 1 test ratings total              : ${user1Test.length}`);
say(`user 1 relevant (test rating >= 4.0)   : ${relTest.length}`);
say(`user 1 relevant AND in train candidates: ${relCand.length}`);
say(`user 1 relevant candidates in Top-10   : ${relCandHit}`);
say(`user 1 in usersWithoutRelevant         : ${refRecall.excluded.usersWithoutRelevant.includes(1)}`);

say("");
say("=".repeat(78));

const text = OUT.join("\n") + "\n";
fs.mkdirSync(RESULTS, { recursive: true });
fs.writeFileSync(path.join(RESULTS, "reference.txt"), text);
process.stdout.write(text);
