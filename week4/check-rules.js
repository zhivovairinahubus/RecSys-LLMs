#!/usr/bin/env node
/**
 * week4/check-rules.js — black-box checks for the counting and metric helpers.
 *
 * Loads the real `week4/transactions.js` and `week4/script.js` through `vm` in
 * a browser-shaped sandbox (`window` exists, `document` does not, so `script.js`
 * skips its DOM wiring) and calls the functions exactly where they live.
 * Nothing is copied into this file.
 *
 * Usage: node week4/check-rules.js
 * Exit code: 0 when every check passes, 1 otherwise.
 */

"use strict";

const fs = require("fs");
const path = require("path");
const vm = require("vm");

const DIR = __dirname;
const TRANSACTIONS_JS = path.join(DIR, "transactions.js");
const SCRIPT_JS = path.join(DIR, "script.js");

const sandbox = { window: {}, console };
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(TRANSACTIONS_JS, "utf8"), sandbox, {
  filename: TRANSACTIONS_JS,
});
vm.runInContext(fs.readFileSync(SCRIPT_JS, "utf8"), sandbox, {
  filename: SCRIPT_JS,
});

// Top-level `function` declarations become properties of the sandbox global.
const {
  tinyWorkedExample,
  countItem,
  countItemset,
  dedupeBasket,
  computeSupport,
  computeConfidence,
  computeLift,
  findFrequentItemsets,
  generateRules,
} = sandbox;

let passed = 0;
let failed = 0;

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

function close(actual, expected, eps = 1e-12) {
  return Math.abs(actual - expected) < eps;
}

function check(name, fn) {
  try {
    fn();
    passed += 1;
    console.log(`PASS  ${name}`);
  } catch (error) {
    failed += 1;
    console.log(`FAIL  ${name}`);
    console.log(`      ${error && error.message ? error.message : error}`);
  }
}

console.log(`Loaded ${path.basename(TRANSACTIONS_JS)} and ${path.basename(SCRIPT_JS)} in a vm sandbox.`);
console.log("");

// ---------------------------------------------------------------------------
// 1. Tiny worked example: item counts, metrics of the three fixture rules,
//    and both directions of the jam/eggs rule.
// ---------------------------------------------------------------------------

const example = tinyWorkedExample();
const baskets = example.baskets;

check("fixture: all five item counts are correct", () => {
  for (const [stock, expected] of Object.entries(example.itemCounts)) {
    const actual = countItem(baskets, stock);
    assert(actual === expected, `countItem(${stock}) = ${actual}, expected ${expected}`);
  }
  assert(dedupeBasket(baskets[0]).length === 4, "basket T1 must dedupe to 4 codes");
});

check("fixture: support / confidence / lift of the three rules", () => {
  for (const rule of example.rules) {
    const joint = countItemset(baskets, [...rule.antecedent, ...rule.consequent]);
    const support = computeSupport(joint, example.n);
    const confidence = computeConfidence(joint, countItem(baskets, rule.antecedent[0]));
    const lift = computeLift(confidence, countItem(baskets, rule.consequent[0]), example.n);
    const label = `${rule.antecedent[0]} -> ${rule.consequent[0]}`;
    assert(joint === rule.jointCount, `${label}: count(A u B) = ${joint}, expected ${rule.jointCount}`);
    assert(support.defined && close(support.value, rule.support), `${label}: support = ${support.value}, expected ${rule.support}`);
    assert(confidence.defined && close(confidence.value, rule.confidence), `${label}: confidence = ${confidence.value}, expected ${rule.confidence}`);
    assert(lift.defined && close(lift.value, rule.lift), `${label}: lift = ${lift.value}, expected ${rule.lift}`);
  }
});

check("fixture: confidence of jam -> eggs and eggs -> jam", () => {
  const joint = countPairOf(baskets, "jam", "eggs");
  const forward = computeConfidence(joint, countItem(baskets, "jam"));
  const reversed = computeConfidence(joint, countItem(baskets, "eggs"));
  assert(forward.defined && close(forward.value, 1 / 3), `jam -> eggs = ${forward.value}, expected ${1 / 3}`);
  assert(reversed.defined && close(reversed.value, 1 / 2), `eggs -> jam = ${reversed.value}, expected ${1 / 2}`);
  assert(!close(forward.value, reversed.value), "confidence must differ between the two directions");
});

function countPairOf(basketList, a, b) {
  return countItemset(basketList, [a, b]);
}

function powerSetUpTo(maxSize, itemsSorted) {
  // not needed; simple subset enumeration small
  return [];
}
function bruteFrequent(basketsStr, n, minSup) {
  const all = new Map();
  for (const b of basketsStr) {
    for (const s of b) all.set(s, true);
  }
  const items = Array.from(all.keys()).sort();
  const res = [];
  const total = 1 << items.length;
  for (let mask = 1; mask < total; mask += 1) {
    let size = 0;
    for (let i = 0; i < items.length; i += 1) if (mask & (1 << i)) size += 1;
    // check all sizes (tiny example has at most 5 items)
    const cand = [];
    for (let i = 0; i < items.length; i += 1) if (mask & (1 << i)) cand.push(items[i]);
    let c = 0;
    basket: for (const b of basketsStr) {
      for (let k = 0; k < cand.length; k += 1) {
        if (b.indexOf(cand[k]) === -1) continue basket;
      }
      c += 1;
    }
    if (c / n >= minSup) res.push({ items: cand.slice().sort(), count: c });
  }
  res.sort((a, b) => a.items.length - b.items.length || a.items.join('|').localeCompare(b.items.join('|')));
  return res;
}

// Enumerate, directly from the baskets, every rule that any correct miner should
// produce: each frequent itemset (support >= minSupport) split into every pair of
// non-empty sides A / B (both directions), kept when confidence(A -> B) >= minConfidence.
// Returns canonical "A|B->C|D" keys. Independent of `generateRules`.
function bruteRules(basketsStr, n, minSupport, minConfidence) {
  const items = Array.from(new Set(basketsStr.flat())).sort();
  const countOf = (subset) => basketsStr.filter((b) => subset.every((it) => b.includes(it))).length;
  const rules = [];
  const total = 1 << items.length;
  for (let mask = 1; mask < total; mask += 1) {
    const set = [];
    for (let i = 0; i < items.length; i += 1) if (mask & (1 << i)) set.push(items[i]);
    if (set.length < 2) continue;
    const joint = countOf(set);
    if (joint / n < minSupport) continue;
    const totalA = 1 << set.length;
    for (let am = 1; am < totalA - 1; am += 1) {
      const A = [];
      const B = [];
      for (let i = 0; i < set.length; i += 1) {
        if (am & (1 << i)) A.push(set[i]);
        else B.push(set[i]);
      }
      if (A.length === 0 || B.length === 0) continue;
      const aCount = countOf(A);
      if (aCount === 0) continue;
      if (joint / aCount < minConfidence) continue;
      rules.push(`${A.join('|')}->${B.join('|')}`);
    }
  }
  return rules.sort();
}




// ---------------------------------------------------------------------------
// 2. countItemset edge cases.
// ---------------------------------------------------------------------------

check("countItemset: empty request returns 0", () => {
  assert(countItemset(baskets, []) === 0, "empty stocks must give 0");
  assert(countItemset(baskets, []) === 0, "empty request must stay 0 on a second call");
});

check("countItemset: unknown stock returns 0 (never throws)", () => {
  assert(countItemset(baskets, ["NOT-A-REAL-CODE"]) === 0, "unknown single stock must give 0");
  assert(countItemset(baskets, ["bread", "NOT-A-REAL-CODE"]) === 0, "unknown stock in a pair must give 0");
});

check("countItemset: the same stock twice counts each basket once", () => {
  const twice = countItemset(baskets, ["milk", "milk"]);
  const once = countItem(baskets, "milk");
  assert(twice === once, `countItemset([milk, milk]) = ${twice}, expected ${once}`);
  const triple = countItemset(baskets, ["jam", "jam", "jam"]);
  assert(triple === countItem(baskets, "jam"), `countItemset([jam, jam, jam]) = ${triple}`);
});

check("countItemset: intersection equals a direct scan", () => {
  const pair = countItemset(baskets, ["bread", "jam"]);
  const naive = baskets.filter((b) => b.includes("bread") && b.includes("jam")).length;
  assert(pair === naive, `countItemset(bread, jam) = ${pair}, naive scan gives ${naive}`);
});

// ---------------------------------------------------------------------------
// 3. Zero denominators: defined = false and value = 0.
// ---------------------------------------------------------------------------

check("zero denominator: support with N = 0", () => {
  const support = computeSupport(0, 0);
  assert(support.defined === false, "support must be undefined when N = 0");
  assert(support.value === 0, `value = ${support.value}, expected 0`);
});

check("zero denominator: confidence with count(A) = 0", () => {
  const confidence = computeConfidence(0, 0);
  assert(confidence.defined === false, "confidence must be undefined when count(A) = 0");
  assert(confidence.value === 0, `value = ${confidence.value}, expected 0`);
});

check("zero denominator: lift with count(B) = 0", () => {
  const lift = computeLift({ value: 0.5, defined: true }, 0, 5);
  assert(lift.defined === false, "lift must be undefined when count(B) = 0");
  assert(lift.value === 0, `value = ${lift.value}, expected 0`);
});

check("zero denominator: lift with N = 0 and with an undefined confidence", () => {
  const noBaskets = computeLift({ value: 0.5, defined: true }, 0, 0);
  assert(noBaskets.defined === false && noBaskets.value === 0, "lift(N = 0) must be { 0, false }");
  const noConfidence = computeLift({ value: 0, defined: false }, 3, 5);
  assert(noConfidence.defined === false && noConfidence.value === 0, "lift of an undefined confidence must be { 0, false }");
  const missing = computeLift(undefined, 3, 5);
  assert(missing.defined === false && missing.value === 0, "lift without a confidence must be { 0, false }");
});

// ---------------------------------------------------------------------------
// 4. Real dataset: count(85123A).
// ---------------------------------------------------------------------------

check("real data: count(85123A) on the 17,080 baskets", () => {
  const hw4 = sandbox.window.HW4;
  const real = hw4.baskets.map((basket) => basket.map((i) => hw4.stocks[i]));
  const counted = countItem(real, "85123A");
  const naive = real.filter((b) => b.includes("85123A")).length;
  assert(counted === 1959, `count(85123A) = ${counted}, expected 1959 (top item of the dataset)`);
  assert(counted === naive, `index count ${counted} disagrees with the naive scan ${naive}`);
  const share = computeSupport(counted, real.length);
  assert(share.defined && close(share.value, 1959 / 17080), `support = ${share.value}, expected ${1959 / 17080}`);
  console.log(`      count(85123A) = ${counted}, N = ${real.length}, support = ${(share.value * 100).toFixed(2)}%`);
});

// ---------------------------------------------------------------------------
// Extra checks: independent brute force on tiny example and sanity on real data
// ---------------------------------------------------------------------------

check("brute force on tiny example (support >= 0.2) matches miner", () => {
  const ex = tinyWorkedExample();
  const baskets = ex.baskets.map((b) => b.slice().sort()); // strings
  const n = baskets.length;
  const minSup = 0.2;
  const fisMiner = findFrequentItemsets(ex.baskets, minSup);
  const fisBrute = bruteFrequent(baskets, n, minSup);
  const kMiner = new Map(fisMiner.map((f) => [f.items.slice().sort().join('|'), f.count]));
  const kBrute = new Map(fisBrute.map((f) => [f.items.join('|'), f.count]));
  assert(kMiner.size === kBrute.size, `sizes differ ${kMiner.size} vs ${kBrute.size}`);
  for (const [k, c] of kBrute) {
    assert(kMiner.has(k), `missing ${k} in miner`);
    assert(kMiner.get(k) === c, `count mismatch for ${k}: ${kMiner.get(k)} vs ${c}`);
  }
});

check("brute force on tiny example (support >= 0.4) matches miner", () => {
  const ex = tinyWorkedExample();
  const baskets = ex.baskets.map((b) => b.slice().sort());
  const n = baskets.length;
  const minSup = 0.4;
  const fisMiner = findFrequentItemsets(ex.baskets, minSup);
  const fisBrute = bruteFrequent(baskets, n, minSup);
  const kMiner = new Map(fisMiner.map((f) => [f.items.slice().sort().join('|'), f.count]));
  const kBrute = new Map(fisBrute.map((f) => [f.items.join('|'), f.count]));
  assert(kMiner.size === kBrute.size, `sizes differ ${kMiner.size} vs ${kBrute.size}`);
  for (const [k, c] of kBrute) {
    assert(kMiner.get(k) === c, `count mismatch for ${k}`);
  }
});

check("pairs on real data at support >= 1% match miner pairs", () => {
  const hw4 = sandbox.window.HW4;
  const baskets = hw4.baskets.map((b) => b.map((i) => hw4.stocks[i]));
  const n = baskets.length;
  const minSup = 0.01;
  const t0 = Date.now();
  const fisMiner = findFrequentItemsets(baskets, minSup);
  const t1 = Date.now();
  const minerPairs = fisMiner.filter((f) => f.items.length === 2);
  // brute force pairs
  const counts = new Map();
  for (let bi = 0; bi < baskets.length; bi += 1) {
    const b = dedupeBasket(baskets[bi]).sort();
    for (let i = 0; i < b.length; i += 1) {
      for (let j = i + 1; j < b.length; j += 1) {
        const key = b[i] + '|' + b[j];
        counts.set(key, (counts.get(key) || 0) + 1);
      }
    }
  }
  const brutePairs = [];
  for (const [k, c] of counts) {
    if (c / n >= minSup) {
      const [a, x] = k.split('|');
      brutePairs.push({ items: [a, x].sort(), count: c });
    }
  }
  brutePairs.sort((a, b) => a.items.join('|').localeCompare(b.items.join('|')));
  minerPairs.sort((a, b) => a.items.slice().sort().join('|').localeCompare(b.items.slice().sort().join('|')));
  assert(minerPairs.length === brutePairs.length, `pair counts differ: ${minerPairs.length} vs ${brutePairs.length}`);
  for (let i = 0; i < brutePairs.length; i += 1) {
    const mk = minerPairs[i].items.slice().sort().join('|');
    const bk = brutePairs[i].items.join('|');
    assert(mk === bk, `pair mismatch ${mk} vs ${bk}`);
    assert(minerPairs[i].count === brutePairs[i].count, `count mismatch for ${bk}`);
  }
  console.log(`      miner pairs (>=1%) = ${minerPairs.length}, brute pairs = ${brutePairs.length}, time miner @1% = ${(t1 - t0) / 1000}s`);
});

// ---------------------------------------------------------------------------
// 5. dedupeBasket: first-appearance order and dropping the description.
// ---------------------------------------------------------------------------

check("dedupeBasket: first-appearance order is preserved", () => {
  const order = dedupeBasket(["milk", "bread", "milk", "jam", "bread", "ham"]);
  assert(order.length === 4, `expected 4 unique codes, got ${order.length}`);
  assert(
    order.join(",") === "milk,bread,jam,ham",
    `first-appearance order wrong: ${order.join(",")}`,
  );
});

check("dedupeBasket: item description is dropped, stock codes are kept", () => {
  const items = [
    { stock: "milk", description: "MILK" },
    { stock: "bread", description: "BREAD" },
    { stock: "milk", description: "MILK (2)" },
  ];
  const codes = dedupeBasket(items);
  assert(
    codes.join(",") === "milk,bread",
    `expected stock codes ['milk','bread'], got ${JSON.stringify(codes)}`,
  );
});

// ---------------------------------------------------------------------------
// 6. generateRules on the tiny example: both directions, threshold, and
//    support / confidence / lift recomputed from the baskets.
// ---------------------------------------------------------------------------

check("generateRules: both directions, threshold, and metrics vs recomputation", () => {
  const ex = tinyWorkedExample();
  const exBaskets = ex.baskets;
  const n = ex.n;
  const minSupport = 0.4;
  const minConfidence = 0.5;
  const fis = findFrequentItemsets(exBaskets, minSupport);
  const rules = generateRules(fis, minConfidence);
  assert(rules.length > 0, "expected at least one rule at support >= 0.4, confidence >= 0.5");

  const seen = new Set(rules.map((r) => `${r.antecedent.join("+")}->${r.consequent.join("+")}`));
  const hasBothDirections = rules.some((r) =>
    seen.has(`${r.consequent.join("+")}->${r.antecedent.join("+")}`),
  );
  assert(hasBothDirections, "at least one pair must appear in both directions");

  for (const r of rules) {
    const aCount = countItemset(exBaskets, r.antecedent);
    const bCount = countItemset(exBaskets, r.consequent);
    const joint = countItemset(exBaskets, [...r.antecedent, ...r.consequent]);
    const support = joint / n;
    const confidence = joint / aCount;
    const lift = confidence / (bCount / n);
    const label = `${r.antecedent.join("+")} -> ${r.consequent.join("+")}`;
    assert(confidence + 1e-12 >= minConfidence, `${label}: confidence ${confidence} below ${minConfidence}`);
    assert(support + 1e-12 >= minSupport, `${label}: support ${support} below ${minSupport}`);
    assert(close(r.support, support), `${label}: support ${r.support} != ${support}`);
    assert(close(r.confidence, confidence), `${label}: confidence ${r.confidence} != ${confidence}`);
    assert(close(r.lift, lift), `${label}: lift ${r.lift} != ${lift}`);
  }
  console.log(`      generateRules @0.4/0.5 -> ${rules.length} rules, both directions present`);
});

// ---------------------------------------------------------------------------
// 6b. generateRules completeness: compare against an independent brute-force
//     enumeration over the baskets, at two support thresholds. Catches rules
//     lost because the miner dropped a frequent itemset, and rules invented.
// ---------------------------------------------------------------------------

check("generateRules is complete on the tiny example (two support thresholds)", () => {
  const ex = tinyWorkedExample();
  const exBaskets = ex.baskets;
  const n = ex.n;
  const minConfidence = 0.5;
  for (const minSupport of [0.2, 0.4]) {
    const fis = findFrequentItemsets(exBaskets, minSupport);
    const actual = generateRules(fis, minConfidence).map(
      (r) => `${r.antecedent.slice().sort().join("|")}->${r.consequent.slice().sort().join("|")}`,
    );
    const expected = bruteRules(exBaskets, n, minSupport, minConfidence);
    const actualSet = new Set(actual);
    const expectedSet = new Set(expected);
    const missing = [...expectedSet].filter((k) => !actualSet.has(k));
    const extra = [...actualSet].filter((k) => !expectedSet.has(k));
    assert(missing.length === 0, `support ${minSupport}: missing rules: ${missing.join(", ")}`);
    assert(extra.length === 0, `support ${minSupport}: unexpected rules: ${extra.join(", ")}`);
    console.log(`      generateRules @${minSupport}/0.5 matches brute force: ${expected.length} rules`);
  }
});

// ---------------------------------------------------------------------------
// 7. Lift is symmetric: lift(A -> B) = lift(B -> A).
// ---------------------------------------------------------------------------

check("lift is symmetric: lift(A -> B) = lift(B -> A)", () => {
  const ex = tinyWorkedExample();
  const exBaskets = ex.baskets;
  const n = ex.n;
  const fis = findFrequentItemsets(exBaskets, 0.4);
  const rules = generateRules(fis, 0.5);
  assert(rules.length > 0, "need rules to test lift symmetry");
  for (const r of rules) {
    const joint = countItemset(exBaskets, [...r.antecedent, ...r.consequent]);
    const forward = computeLift(
      computeConfidence(joint, countItemset(exBaskets, r.antecedent)),
      countItemset(exBaskets, r.consequent),
      n,
    );
    const reversed = computeLift(
      computeConfidence(joint, countItemset(exBaskets, r.consequent)),
      countItemset(exBaskets, r.antecedent),
      n,
    );
    const label = `${r.antecedent.join("+")} <-> ${r.consequent.join("+")}`;
    assert(forward.defined && reversed.defined, `${label}: both lifts must be defined`);
    assert(close(forward.value, reversed.value), `${label}: ${forward.value} != ${reversed.value}`);
  }
});

console.log("");
console.log(`checks: ${passed + failed} total, ${passed} passed, ${failed} failed`);
if (failed > 0) { console.log(`failed: ${failed}`); } else { console.log("all checks passed"); }
process.exitCode = failed > 0 ? 1 : 0;
