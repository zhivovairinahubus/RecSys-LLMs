/**
 * week4/script.js — Association-rule mining starter (HW4).
 *
 * This is a plain (classic) script, NOT an ES module. `week4/transactions.js`
 * (loaded first, in a regular <script> tag) assigns the dictionary-encoded UCI
 * Online Retail baskets to `window.HW4`. This file reads that global, decodes
 * the integer-index baskets back into `{ stock, description }` items, renders a
 * dataset summary, and wires two slider controls ("minimum support" and
 * "minimum confidence") plus a "Run rules" button. Because nothing is fetched,
 * the page works from a `file://` URL with no server.
 *
 * WHAT YOU MUST IMPLEMENT (`TODO(hw4)` — each stub throws until you write it):
 *   1. `dedupeBasket`         — unique stock codes in a basket, first-appearance order.
 *   2. `countItemset`         — baskets containing every requested stock (`0` if any is absent).
 *   3. `computeSupport`       — support = count(A union B) / N, guarded when N = 0.
 *   4. `computeConfidence`    — confidence = count(A union B) / count(A), guarded when count(A) = 0.
 *   5. `computeLift`          — lift = confidence / (count(B) / N), guarded.
 *   6. `findFrequentItemsets` — mine frequent itemsets (Apriori or equivalent).
 *   7. `generateRules`        — turn frequent itemsets into both-direction rules.
 *
 * PROVIDED for you (scaffolding): dataset loading/decoding from `window.HW4`, the
 * inverted-index builder (`buildIndex` / `asIndex` / `indexCache`), the thin
 * counting wrappers `countItem` / `countPair` (they call your `countItemset` and
 * therefore also throw until it is implemented), threshold validation and slider
 * readout, the DOM wiring, all formatting and rendering helpers, and the test
 * harness. Leave the provided code as-is and implement only the stubs above.
 *
 * Metrics (see week4/readme.md for definitions):
 *   support(A -> B)    = count(A union B) / N
 *   confidence(A -> B) = count(A union B) / count(A)
 *   lift(A -> B)       = confidence(A -> B) / (count(B) / N)
 *
 * @module week4/script
 */

// Defensive guard: `transactions.js` must run before this file. If `window.HW4`
// is missing (for example because `transactions.js` was not copied next to
// `index.html`), fail loudly and visibly instead of throwing an opaque error.
if (typeof window === "undefined" || typeof window.HW4 === "undefined") {
  if (typeof document !== "undefined") {
    document.body.innerHTML =
      '<p style="padding:2rem;font-family:sans-serif">' +
      "Failed to load `transactions.js`. Make sure it sits next to `index.html` " +
      "and is loaded before `script.js`.</p>";
  }
  throw new Error(
    "window.HW4 is not defined — load transactions.js before script.js.",
  );
}

/** The raw dataset global assigned by `week4/transactions.js`. */
const data = window.HW4;

/**
 * Baskets decoded from the dictionary-encoded arrays in `window.HW4`. Each
 * basket is an array of `{ stock, description }` items, matching the fixture
 * shape used by the tests.
 *
 * @type {Array<Array<Item>>}
 */
const TRANSACTIONS = data.baskets.map((basket) =>
  basket.map((stockIndex) => ({
    stock: data.stocks[stockIndex],
    description: data.descriptions[stockIndex],
  })),
);

/**
 * Dataset-wide inverted index, built once at startup. `null` until `init()` has
 * run, so the UI can tell "still loading" from "loaded".
 *
 * @type {BasketIndex|null}
 */
let DATASET_INDEX = null;

/** Number of baskets (the `N` used by support and lift). */
let N = data.N_BASKETS;

/**
 * @typedef {Object} Item
 * @property {string} stock       Stock code (the item identity).
 * @property {string} description Human-readable product description.
 */

/**
 * @typedef {Object} Rule
 * @property {string[]} antecedent    Item identities on the left-hand side (A).
 * @property {string[]} consequent    Item identities on the right-hand side (B).
 * @property {number} jointCount      count(A union B).
 * @property {number} antecedentCount count(A).
 * @property {number} consequentCount count(B).
 * @property {number} support         jointCount / N.
 * @property {number} confidence      jointCount / antecedentCount.
 * @property {number} lift            confidence / (consequentCount / N).
 */

/**
 * @typedef {Object} BasketIndex
 * @property {number} n                              Number of baskets.
 * @property {Map<string, Set<number>>} byStock      Stock code -> basket ids.
 */

// ---------------------------------------------------------------------------
// Provided helpers and student stubs
//
// Functions carrying a `TODO(hw4)` marker are stubs you must implement; every
// other function in this file is scaffolding and should be left as-is.
// ---------------------------------------------------------------------------

/**
 * Read the item identity out of a basket entry. Accepts either a plain string or
 * an object with a `stock` property so the helpers work with both the real
 * dataset and the small fixtures used by the tests.
 *
 * @param {Item|string} item
 * @returns {string}
 */
function stockOf(item) {
  return typeof item === "string" ? item : item.stock;
}

/**
 * Build an inverted index from stock code to the set of basket ids containing
 * that code. Counting a candidate itemset then becomes a set intersection.
 *
 * @param {Array<Array<Item|string>>} baskets
 * @returns {BasketIndex}
 */
function buildIndex(baskets) {
  /** @type {Map<string, Set<number>>} */
  const byStock = new Map();
  baskets.forEach((basket, basketId) => {
    for (const item of basket) {
      const stock = stockOf(item);
      let posting = byStock.get(stock);
      if (!posting) {
        posting = new Set();
        byStock.set(stock, posting);
      }
      posting.add(basketId);
    }
  });
  return { n: baskets.length, byStock };
}

/** Cache of lazily built indexes, keyed by the baskets array. */
const indexCache = new WeakMap();

/**
 * Accept either a ready-made index or a raw basket array and return an index.
 *
 * @param {BasketIndex|Array<Array<Item|string>>} basketsOrIndex
 * @returns {BasketIndex}
 */
function asIndex(basketsOrIndex) {
  if (basketsOrIndex && !Array.isArray(basketsOrIndex) && basketsOrIndex.byStock) {
    return basketsOrIndex;
  }
  let index = indexCache.get(basketsOrIndex);
  if (!index) {
    index = buildIndex(basketsOrIndex);
    indexCache.set(basketsOrIndex, index);
  }
  return index;
}

/**
 * Count the baskets that contain every stock code in `stocks`.
 *
 * TODO(hw4): build (or reuse) a stock -> basket ids inverted index, intersect the
 * posting lists of the requested stocks, and return the size of the
 * intersection.
 *
 * Contract:
 *  - Accept either a ready-made index or a raw basket array. The provided
 *    `asIndex` helper returns an index for either input.
 *  - Return `0` when `stocks` is empty, and also when any requested stock is
 *    absent from the dataset (never throw for an unknown stock).
 *  - A stock repeated within one basket must be counted at most once. Dedupe the
 *    request before intersecting.
 *
 * @param {BasketIndex|Array<Array<Item|string>>} basketsOrIndex
 * @param {Array<string|Item>} stocks
 * @returns {number} count(A) for a single-element `stocks`, count(A union B) for two.
 */
function countItemset(basketsOrIndex, stocks) {
  const index = asIndex(basketsOrIndex);
  const wanted = [];
  const requested = new Set();
  for (const stock of stocks) {
    const code = stockOf(stock);
    if (!requested.has(code)) {
      requested.add(code);
      wanted.push(code);
    }
  }
  if (wanted.length === 0) return 0;
  let intersection = null;
  for (const code of wanted) {
    const posting = index.byStock.get(code);
    if (!posting) return 0;
    if (!intersection) {
      intersection = new Set(posting);
    } else {
      const small = intersection.size <= posting.size ? intersection : posting;
      const large = intersection.size <= posting.size ? posting : intersection;
      const next = new Set();
      for (const basketId of small) {
        if (large.has(basketId)) next.add(basketId);
      }
      intersection = next;
    }
    if (intersection.size === 0) return 0;
  }
  return intersection.size;
}

/**
 * Count the baskets containing a single stock code.
 *
 * @param {BasketIndex|Array<Array<Item|string>>} basketsOrIndex
 * @param {string|Item} stock
 * @returns {number}
 */
function countItem(basketsOrIndex, stock) {
  return countItemset(basketsOrIndex, [stock]);
}

/**
 * Count the baskets containing both `stockA` and `stockB`.
 *
 * @param {BasketIndex|Array<Array<Item|string>>} basketsOrIndex
 * @param {string|Item} stockA
 * @param {string|Item} stockB
 * @returns {number}
 */
function countPair(basketsOrIndex, stockA, stockB) {
  return countItemset(basketsOrIndex, [stockA, stockB]);
}

/**
 * Remove repeated item identities from a raw basket, keeping first-appearance
 * order. A basket is a set of items, so duplicates must not be counted twice.
 *
 * TODO(hw4): walk the input once, map each entry to its stock code with the
 * provided `stockOf` helper, and return each distinct code the first time it
 * appears.
 *
 * Contract:
 *  - An empty input returns `[]`.
 *  - The input may mix plain strings and `{ stock, description }` objects.
 *  - Only the stock code is returned; the description is dropped.
 *
 * @param {Array<string|Item>} rawItems
 * @returns {Array<string>} unique stock codes, in first-appearance order.
 */
function dedupeBasket(rawItems) {
  const seen = new Set();
  const unique = [];
  for (const item of rawItems) {
    const stock = stockOf(item);
    if (!seen.has(stock)) {
      seen.add(stock);
      unique.push(stock);
    }
  }
  return unique;
}

/**
 * Compute support as `jointCount / n`.
 *
 * TODO(hw4): return the fraction and flag the `n === 0` case.
 *
 * Contract:
 *  - `defined: true` with `value = jointCount / n` when `n > 0`.
 *  - `defined: false` with `value = 0` when `n === 0` (never divide by zero).
 *
 * @param {number} jointCount count(A union B)
 * @param {number} n          number of baskets
 * @returns {{value: number, defined: boolean}} `defined` is false when `n === 0`.
 */
function computeSupport(jointCount, n) {
  if (n === 0) return { value: 0, defined: false };
  return { value: jointCount / n, defined: true };
}

/**
 * Compute confidence as `jointCount / antecedentCount`.
 *
 * TODO(hw4): return the fraction and flag the `antecedentCount === 0` case.
 *
 * Contract:
 *  - `defined: true` with `value = jointCount / antecedentCount` when count(A) > 0.
 *  - `defined: false` with `value = 0` when count(A) === 0: a rule whose
 *    left-hand side never occurs has no confidence.
 *
 * @param {number} jointCount      count(A union B)
 * @param {number} antecedentCount count(A)
 * @returns {{value: number, defined: boolean}}
 */
function computeConfidence(jointCount, antecedentCount) {
  if (antecedentCount === 0) return { value: 0, defined: false };
  return { value: jointCount / antecedentCount, defined: true };
}

/**
 * Compute lift as `confidence / (consequentCount / n)`.
 *
 * TODO(hw4): divide the incoming confidence by the consequent's baseline rate.
 *
 * Contract:
 *  - `defined: true` with `value = confidence.value / (consequentCount / n)`
 *    when every input is usable.
 *  - `defined: false` with `value = 0` when the incoming confidence is missing or
 *    undefined, when `n === 0`, or when the baseline `consequentCount / n` is 0
 *    (the consequent never occurs).
 *
 * @param {{value: number, defined: boolean}} confidence confidence(A -> B)
 * @param {number} consequentCount count(B)
 * @param {number} n               number of baskets
 * @returns {{value: number, defined: boolean}}
 */
function computeLift(confidence, consequentCount, n) {
  if (!confidence || !confidence.defined) return { value: 0, defined: false };
  if (n === 0) return { value: 0, defined: false };
  const baseline = consequentCount / n;
  if (baseline === 0) return { value: 0, defined: false };
  return { value: confidence.value / baseline, defined: true };
}

/**
 * Validate the two slider values, expressed as fractions in `(0, 1]`.
 *
 * @param {number} minSupport    minimum support fraction
 * @param {number} minConfidence minimum confidence fraction
 * @returns {{ok: boolean, errors: string[]}}
 */
function validateThresholds(minSupport, minConfidence) {
  const errors = [];
  for (const [label, value] of [
    ["Minimum support", minSupport],
    ["Minimum confidence", minConfidence],
  ]) {
    if (typeof value !== "number" || Number.isNaN(value)) {
      errors.push(`${label} must be a number.`);
    } else if (value <= 0 || value > 1) {
      errors.push(`${label} must be greater than 0 and at most 1.`);
    }
  }
  return { ok: errors.length === 0, errors };
}

/**
 * Mine all frequent itemsets whose support is at least `minSupport`.
 *
 * TODO(hw4): implement Apriori (level-wise candidate generation with a
 * downward-closure pruning step) or any equivalent frequent-itemset miner.
 *
 * Contract:
 *  - Return one entry per frequent itemset: `{ items, count, support }`, where
 *    `items` is the (deduplicated) set of stock codes, `count` is the number of
 *    baskets containing every item, and `support = count / N`.
 *  - Every returned itemset must satisfy `support >= minSupport`.
 *  - You may call `countItemset` while developing, but a per-candidate scan over
 *    17,080 baskets is slow — build your own occurrence/index structures.
 *  - `generateRules` consumes this exact shape, so keep the field names stable.
 *
 * @param {Array<Array<Item|string>>} transactions baskets
 * @param {number} minSupport minimum support fraction in `(0, 1]`
 * @returns {Array<{items: string[], count: number, support: number}>} frequent itemsets
 */
function findFrequentItemsets(transactions, minSupport) {
  const n = transactions.length;
  if (n === 0) return [];
  const passes = (count) => count / n >= minSupport;
  const baskets = transactions.map((raw) => dedupeBasket(raw));
  const index = buildIndex(baskets);
  const results = [];
  const countCandidate = (items) => {
    const postings = items.map((item) => index.byStock.get(item));
    if (postings.some((posting) => !posting)) return 0;
    let shortest = 0;
    for (let i = 1; i < postings.length; i += 1) {
      if (postings[i].size < postings[shortest].size) shortest = i;
    }
    let count = 0;
    basketLoop: for (const basketId of postings[shortest]) {
      for (let i = 0; i < postings.length; i += 1) {
        if (i !== shortest && !postings[i].has(basketId)) continue basketLoop;
      }
      count += 1;
    }
    return count;
  };
  let level = [];
  for (const [stock, posting] of index.byStock) {
    if (passes(posting.size)) level.push({ items: [stock], count: posting.size });
  }
  for (let k = 1; level.length > 0; k += 1) {
    for (const entry of level) {
      results.push({ items: entry.items.slice(), count: entry.count, support: entry.count / n });
    }
    const frequentKeys = new Set(level.map((entry) => entry.items.join(" ")));
    const counts = new Map();
    if (k === 1) {
      for (const basket of baskets) {
        const frequent = basket.filter((item) => frequentKeys.has(item)).sort();
        for (let i = 0; i < frequent.length; i += 1) {
          for (let j = i + 1; j < frequent.length; j += 1) {
            const key = frequent[i] + " " + frequent[j];
            counts.set(key, (counts.get(key) || 0) + 1);
          }
        }
      }
    } else {
      const buckets = new Map();
      for (const entry of level) {
        const prefix = entry.items.slice(0, k - 1).join(" ");
        let bucket = buckets.get(prefix);
        if (!bucket) { bucket = []; buckets.set(prefix, bucket); }
        bucket.push(entry.items);
      }
      for (const bucket of buckets.values()) {
        for (let i = 0; i < bucket.length; i += 1) {
          for (let j = i + 1; j < bucket.length; j += 1) {
            const a = bucket[i], b = bucket[j];
            if (a[k - 1] === b[k - 1]) continue;
            const union = a.concat(b[k - 1]).sort();
            const key = union.join(" ");
            if (counts.has(key)) continue;
            let keep = true;
            for (let m = 0; m < union.length; m += 1) {
              const subset = union.slice(0, m).concat(union.slice(m + 1));
              if (!frequentKeys.has(subset.join(" "))) { keep = false; break; }
            }
            if (keep) counts.set(key, countCandidate(union));
          }
        }
      }
    }
    level = [];
    for (const [key, count] of counts) {
      if (passes(count)) level.push({ items: key.split(" "), count });
    }
  }
  results.sort((a, b) => a.items.length - b.items.length || b.count - a.count);
  return results;
}

/**
 * Turn frequent itemsets into association rules and keep the ones whose
 * confidence is at least `minConfidence`.
 *
 * TODO(hw4): for each frequent itemset, split it into a non-empty antecedent `A`
 * and a non-empty, disjoint consequent `B` in BOTH directions, compute the
 * confidence for each direction, and keep the rules that pass the threshold.
 *
 * Contract:
 *  - Each returned rule follows the `Rule` shape documented at the top of this
 *    module. At minimum it carries `antecedent` and `consequent`; the renderer
 *    fills in the counts and metrics with `enrichRule`, but returning them
 *    yourself is fine and faster.
 *  - Generate both `A -> B` and `B -> A`: they are separate rules with (usually)
 *    different confidence. Skip a direction whose consequent is empty.
 *  - Keep only rules with `confidence >= minConfidence`. `count(A)` is non-zero
 *    for every generated rule, so the confidence is always defined.
 *
 * @param {Array<{items: string[], count: number, support: number}>} frequentItemsets
 * @param {number} minConfidence minimum confidence fraction in `(0, 1]`
 * @returns {Rule[]}
 */
function generateRules(frequentItemsets, minConfidence) {
  const rules = [];
  // Build a map of itemset key -> count and support for O(1) lookup of any subset.
  const countMap = new Map();
  let nMax = 0;
  for (const it of frequentItemsets) {
    if (it.count > nMax) nMax = it.count;
    countMap.set(it.items.slice().sort().join("|"), {
      count: it.count,
      support: it.support,
    });
  }
  // Infer N from support: N = count/support for itemsets with support > 0.
  let N = 0;
  for (const v of countMap.values()) {
    if (v.support > 0) {
      const cand = Math.round(v.count / v.support);
      if (cand > N) N = cand;
    }
  }
  if (N === 0) N = nMax;
  const getCount = (items) => {
    const key = items.slice().sort().join("|");
    const v = countMap.get(key);
    if (v) return v.count;
    return 0;
  };
  for (const itemset of frequentItemsets) {
    const items = itemset.items.slice().sort();
    const jointCount = itemset.count;
    const supportVal = itemset.support;
    if (items.length < 2) continue;
    const total = 1 << items.length;
    for (let mask = 1; mask < total; mask += 1) {
      const antecedent = [];
      const consequent = [];
      for (let i = 0; i < items.length; i += 1) {
        if (mask & (1 << i)) antecedent.push(items[i]);
        else consequent.push(items[i]);
      }
      if (antecedent.length === 0 || consequent.length === 0) continue;
      // count(A), count(B), count(A union B) are taken from frequentItemsets.
      const anteCount = getCount(antecedent);
      if (anteCount === 0) continue;
      const conseCount = getCount(consequent);
      const conf = computeConfidence(jointCount, anteCount);
      if (!conf.defined || conf.value < minConfidence) continue;
      const liftRes = computeLift(conf, conseCount, N);
      rules.push({
        antecedent: antecedent.slice(),
        consequent: consequent.slice(),
        jointCount,
        antecedentCount: anteCount,
        consequentCount: conseCount,
        support: supportVal,
        confidence: conf.value,
        lift: liftRes.defined ? liftRes.value : 0,
      });
    }
  }
  rules.sort((a, b) => {
    if (b.lift !== a.lift) return b.lift - a.lift;
    const ak = a.antecedent.join("|") + "->" + a.consequent.join("|");
    const bk = b.antecedent.join("|") + "->" + b.consequent.join("|");
    return ak.localeCompare(bk);
  });
  return rules;
}

// ---------------------------------------------------------------------------
// Tiny worked example (used by the automated tests and the readout panel)
// ---------------------------------------------------------------------------

/**
 * A five-basket fixture with hand-computed support / confidence / lift values.
 *
 * The fixture is deliberately small enough to check with a pencil:
 *
 *   T1: bread, milk, jam, ham
 *   T2: bread, milk, jam
 *   T3: bread, milk
 *   T4: bread, jam, eggs
 *   T5: bread, eggs
 *
 * Item counts: bread = 5, milk = 3, jam = 3, eggs = 2, ham = 1 (N = 5).
 *
 * Because `bread` appears in every basket, a rule has lift exactly 1 only when
 * `bread` is the sole item on its side (say `bread -> B`, where confidence equals
 * `support(B)`). If `bread` shares a side with another item, lift need not be 1 —
 * for example `{bread, jam} -> milk` has lift ≈ 1.11. That is the teaching point
 * of the fixture.
 *
 * @returns {{n: number, baskets: string[][], itemCounts: Object<string, number>, rules: Array<Object>}}
 */
function tinyWorkedExample() {
  const baskets = [
    ["bread", "milk", "jam", "ham"],
    ["bread", "milk", "jam"],
    ["bread", "milk"],
    ["bread", "jam", "eggs"],
    ["bread", "eggs"],
  ];
  const n = baskets.length;
  return {
    n,
    baskets,
    itemCounts: { bread: 5, milk: 3, jam: 3, eggs: 2, ham: 1 },
    rules: [
      // lift == 1: bread is in every basket, so confidence equals support(B).
      {
        antecedent: ["bread"],
        consequent: ["milk"],
        jointCount: 3,
        antecedentCount: 5,
        consequentCount: 3,
        support: 3 / n,
        confidence: 3 / 5,
        lift: 1,
      },
      // lift > 1: milk and jam co-occur more than independence predicts.
      {
        antecedent: ["milk"],
        consequent: ["jam"],
        jointCount: 2,
        antecedentCount: 3,
        consequentCount: 3,
        support: 2 / n,
        confidence: 2 / 3,
        lift: (2 / 3) / (3 / 5),
      },
      // lift < 1 (and not degenerate): jam and eggs co-occur less than expected.
      {
        antecedent: ["jam"],
        consequent: ["eggs"],
        jointCount: 1,
        antecedentCount: 3,
        consequentCount: 2,
        support: 1 / n,
        confidence: 1 / 3,
        lift: (1 / 3) / (2 / 5),
      },
    ],
  };
}

// ---------------------------------------------------------------------------
// Rule normalisation
// ---------------------------------------------------------------------------

/**
 * Fill in any missing counts / metrics on a rule using the dataset index, so the
 * table can render a rule even when the student returns only `antecedent` and
 * `consequent`.
 *
 * @param {Partial<Rule>} rule
 * @param {BasketIndex} index
 * @returns {Rule}
 */
function enrichRule(rule, index) {
  const antecedent = (rule.antecedent || []).map(stockOf);
  const consequent = (rule.consequent || []).map(stockOf);
  const n = index.n;
  const jointCount =
    typeof rule.jointCount === "number"
      ? rule.jointCount
      : countItemset(index, [...antecedent, ...consequent]);
  const antecedentCount =
    typeof rule.antecedentCount === "number"
      ? rule.antecedentCount
      : countItemset(index, antecedent);
  const consequentCount =
    typeof rule.consequentCount === "number"
      ? rule.consequentCount
      : countItemset(index, consequent);
  // Value and `defined` flag for each metric come from a single source: either
  // the recomputed metric or the rule's own number, never a mix of the two.
  const support =
    typeof rule.support === "number"
      ? { value: rule.support, defined: Number.isFinite(rule.support) }
      : computeSupport(jointCount, n);
  const confidence =
    typeof rule.confidence === "number"
      ? { value: rule.confidence, defined: Number.isFinite(rule.confidence) }
      : computeConfidence(jointCount, antecedentCount);
  const lift =
    typeof rule.lift === "number"
      ? { value: rule.lift, defined: Number.isFinite(rule.lift) }
      : computeLift(confidence, consequentCount, n);
  return {
    antecedent,
    consequent,
    jointCount,
    antecedentCount,
    consequentCount,
    support: support.value,
    confidence: confidence.value,
    lift: lift.value,
    supportDefined: support.defined,
    confidenceDefined: confidence.defined,
    liftDefined: lift.defined,
  };
}

/**
 * Swap the antecedent and consequent of a rule and recompute the metrics.
 *
 * Support is unchanged because `count(A union B)` and `N` are the same in both
 * directions. Confidence is not symmetric, so `B -> A` usually has a different
 * confidence from `A -> B`. Lift is unchanged because it is symmetric.
 *
 * @param {Rule} rule
 * @param {BasketIndex} index
 * @returns {Rule}
 */
function reverseRule(rule, index) {
  return enrichRule(
    {
      antecedent: rule.consequent,
      consequent: rule.antecedent,
    },
    index,
  );
}

// ---------------------------------------------------------------------------
// Formatting helpers
// ---------------------------------------------------------------------------

/**
 * Format a fraction as a percentage with two decimals.
 *
 * @param {number} fraction
 * @returns {string}
 */
function formatPercent(fraction) {
  if (!Number.isFinite(fraction)) return "n/a";
  const pct = fraction * 100;
  // Show enough precision to distinguish values just above/below 1%
  return `${pct.toFixed(3)}%`;
}

/**
 * Format a number with four decimal places for the results table.
 *
 * @param {number} value
 * @returns {string}
 */
function formatMetric(value) {
  if (!Number.isFinite(value)) return "n/a";
  return value.toFixed(4);
}

/**
 * Render a list of stock codes as readable text.
 *
 * @param {string[]} stocks
 * @param {BasketIndex} index
 * @returns {string}
 */
function formatItemset(stocks, index) {
  if (!stocks || stocks.length === 0) return "(empty)";
  return stocks.map((stock) => describe(stock, index)).join(", ");
}

/**
 * Lookup a stock code's canonical description, falling back to the code. Filled
 * by `primeDescriptions()` once `init()` runs.
 */
const descriptionByStock = new Map();

/**
 * Render `STOCK — human readable description` for one item.
 *
 * @param {string} stock
 * @param {BasketIndex} index
 * @returns {string}
 */
function describe(stock, index) {
  const description = descriptionByStock.get(stock);
  return description ? `${stock} — ${description}` : stock;
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

/**
 * Render the dataset summary at the top of the page: total baskets, distinct
 * items, and the top five items by basket count.
 *
 * @param {BasketIndex} index
 * @param {HTMLElement|null} container
 * @returns {void}
 */
function renderDatasetSummary(index, container) {
  const target = container || document.getElementById("dataset-summary-body");
  if (!target) return;

  const counts = [...index.byStock.entries()]
    .map(([stock, posting]) => ({ stock, count: posting.size }))
    .sort((a, b) => b.count - a.count || a.stock.localeCompare(b.stock));
  const topFive = counts.slice(0, 5);

  const rows = topFive
    .map(
      (entry, i) =>
        `<tr><td class="num">${i + 1}</td><td>${escapeHtml(
          describe(entry.stock, index),
        )}</td><td class="num">${entry.count}</td></tr>`,
    )
    .join("");

  target.innerHTML = `
    <p class="summary-line">
      <strong>${index.n.toLocaleString("en-US")}</strong> baskets ·
      <strong>${index.byStock.size.toLocaleString("en-US")}</strong> distinct items
    </p>
    <details class="top-items">
      <summary>Top 5 items by basket count</summary>
      <table class="data-table">
        <thead><tr><th scope="col">#</th><th scope="col">Item</th><th scope="col">Baskets</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>
    </details>
    <p class="provenance">${escapeHtml("UCI Online Retail (D. Chen, S. L. Sain, K. Guo, 2012): from 541,909 raw rows; removed C/A invoices, non-positive Quantity/UnitPrice, blank CustomerID (-132,219), non-product codes, deduplicated (InvoiceNo,StockCode) to 386,233 pairs, dropped baskets with < 2 distinct items → 384,911 (InvoiceNo,StockCode) pairs across 17,080 baskets. (Note: the embedded dataset_provenance in transactions.js has a name typo and omits blank-CustomerID removal; it also calls (InvoiceNo,StockCode) pairs 'rows'.)")}</p>
  `;
}

/**
 * Render the rule list into a table. Each row is clickable and shows the rule in
 * the detail panel.
 *
 * @param {Rule[]} rules
 * @param {BasketIndex} [index]
 * @param {HTMLElement|null} [container]
 * @returns {void}
 */
function renderResults(rules, index, container) {
  const target = container || document.getElementById("results");
  if (!target) return;
  const activeIndex = index || DATASET_INDEX;

  if (!rules || rules.length === 0) {
    target.innerHTML =
      '<p class="empty-state">No rules passed the current thresholds. ' +
      "Lower the minimum support or confidence and run again.</p>";
    return;
  }

  const enrichedRules = rules.map((rule) => enrichRule(rule, activeIndex));
  let liftLE1 = 0;
  for (const e of enrichedRules) {
    if (e.liftDefined && e.lift <= 1 + 1e-12) liftLE1 += 1;
  }
  const body = enrichedRules
    .map((enriched, rowIndex) => {
      const isLE1 = enriched.liftDefined && enriched.lift <= 1 + 1e-12;
      const cls = isLE1 ? ' class="rule-le1"' : '';
      return `
        <tr${cls} tabindex="0" data-rule-index="${rowIndex}">
          <td>${escapeHtml(formatItemset(enriched.antecedent, activeIndex))}</td>
          <td>${escapeHtml(formatItemset(enriched.consequent, activeIndex))}</td>
          <td class="num">${enriched.jointCount}</td>
          <td class="num">${enriched.antecedentCount}</td>
          <td class="num">${enriched.consequentCount}</td>
          <td class="num">${enriched.supportDefined ? formatPercent(enriched.support) : "n/a"}</td>
          <td class="num">${enriched.confidenceDefined ? formatPercent(enriched.confidence) : "n/a"}</td>
          <td class="num">${enriched.liftDefined ? formatMetric(enriched.lift) : "n/a"}</td>
        </tr>`;
    })
    .join("");

  target.innerHTML = `
    <p class="results-count">${rules.length} rule${rules.length === 1 ? "" : "s"}${liftLE1 > 0 ? `; ${liftLE1} with lift ≤ 1 (not used in the lift>1 analysis)` : ""}.</p>
    <div class="rules-wrapper">
    <table class="data-table rules-table">
      <thead>
        <tr>
          <th scope="col">Antecedent (A)</th>
          <th scope="col">Consequent (B)</th>
          <th scope="col">count(A∪B)</th>
          <th scope="col">count(A)</th>
          <th scope="col">count(B)</th>
          <th scope="col">support</th>
          <th scope="col">confidence</th>
          <th scope="col">lift</th>
        </tr>
      </thead>
      <tbody>${body}</tbody>
    </table>
    </div>`;

  target.querySelectorAll("tr[data-rule-index]").forEach((row) => {
    const activate = () => {
      const rule = rules[Number(row.dataset.ruleIndex)];
      renderRuleDetail(rule, activeIndex);
    };
    row.addEventListener("click", activate);
    row.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        activate();
      }
    });
  });
}

/**
 * Render the detail panel for one selected rule, including a "Reverse direction"
 * button that swaps A and B and re-renders the panel.
 *
 * Shows an inline note when `count(A)` or `count(B)` is zero, because confidence
 * and lift are then undefined.
 *
 * @param {Rule} rule
 * @param {BasketIndex} [index]
 * @param {HTMLElement|null} [container]
 * @returns {void}
 */
function renderRuleDetail(rule, index, container) {
  const target = container || document.getElementById("rule-detail");
  if (!target) return;
  const activeIndex = index || DATASET_INDEX;
  const enriched = enrichRule(rule, activeIndex);
  const reversed = reverseRule(enriched, activeIndex);

  const zeroDenominatorNotes = [];
  if (enriched.antecedentCount === 0) {
    zeroDenominatorNotes.push(
      "count(A) = 0, so confidence(A → B) is undefined (division by zero).",
    );
  }
  if (enriched.consequentCount === 0) {
    zeroDenominatorNotes.push(
      "count(B) = 0, so lift(A → B) is undefined (division by zero).",
    );
  }
  const noteHtml = zeroDenominatorNotes.length
    ? `<p class="warning">${zeroDenominatorNotes.map(escapeHtml).join(" ")}</p>`
    : "";

  target.innerHTML = `
    <dl class="rule-metrics">
      <dt>Antecedent (A)</dt><dd>${escapeHtml(formatItemset(enriched.antecedent, activeIndex))}</dd>
      <dt>Consequent (B)</dt><dd>${escapeHtml(formatItemset(enriched.consequent, activeIndex))}</dd>
      <dt>count(A∪B)</dt><dd class="num">${enriched.jointCount}</dd>
      <dt>count(A)</dt><dd class="num">${enriched.antecedentCount}</dd>
      <dt>count(B)</dt><dd class="num">${enriched.consequentCount}</dd>
      <dt>support</dt><dd class="num">${enriched.supportDefined ? formatPercent(enriched.support) : "n/a"}</dd>
      <dt>confidence</dt><dd class="num">${enriched.confidenceDefined ? formatPercent(enriched.confidence) : "n/a"}</dd>
      <dt>lift</dt><dd class="num">${enriched.liftDefined ? formatMetric(enriched.lift) : "n/a"}</dd>
    </dl>
    <p class="comparison">
      Reverse direction (B → A): confidence <strong>${reversed.confidenceDefined ? formatPercent(reversed.confidence) : "n/a"}</strong>, lift <strong>${reversed.liftDefined ? formatMetric(reversed.lift) : "n/a"}</strong>.
      ${(enriched.confidenceDefined && reversed.confidenceDefined) ? (Math.abs(enriched.confidence - reversed.confidence) > 1e-12 ? "Confidence differs between directions." : "Confidence is equal in both directions (count(A)=count(B)).") : ""}
      ${(!enriched.liftDefined || !reversed.liftDefined) ? "" : " Lift is symmetric."}
    </p>
    ${noteHtml}
    <button type="button" id="reverse-rule">Reverse direction (B → A)</button>
  `;

  const button = target.querySelector("#reverse-rule");
  button.addEventListener("click", () => {
    renderRuleDetail(reversed, activeIndex, target);
  });
}

/**
 * Render the optional worked-example readout panel.
 *
 * @param {ReturnType<typeof tinyWorkedExample>} example
 * @param {HTMLElement|null} [container]
 * @returns {void}
 */
function renderWorkedExample(example, container) {
  const target = container || document.getElementById("worked-example");
  if (!target) return;
  const rows = example.rules
    .map(
      (rule) => `
      <tr>
        <td>${rule.antecedent.join(", ")}</td>
        <td>${rule.consequent.join(", ")}</td>
        <td class="num">${rule.jointCount}</td>
        <td class="num">${rule.antecedentCount}</td>
        <td class="num">${rule.consequentCount}</td>
        <td class="num">${formatPercent(rule.support)}</td>
        <td class="num">${formatPercent(rule.confidence)}</td>
        <td class="num">${formatMetric(rule.lift)}</td>
      </tr>`,
    )
    .join("");
  target.innerHTML = `
    <p class="worked-example-intro">
      Five hand-built baskets, N = ${example.n}. The top row has lift exactly 1
      because <code>bread</code> appears in every basket; the second has lift &gt; 1;
      the third has lift &lt; 1.
    </p>
    <table class="data-table">
      <thead>
        <tr>
          <th scope="col">A</th><th scope="col">B</th>
          <th scope="col">count(A∪B)</th><th scope="col">count(A)</th><th scope="col">count(B)</th>
          <th scope="col">support</th><th scope="col">confidence</th><th scope="col">lift</th>
        </tr>
      </thead>
      <tbody>${rows}</tbody>
    </table>`;
}

/**
 * Escape text before inserting it into HTML.
 *
 * @param {string} text
 * @returns {string}
 */
function escapeHtml(text) {
  return String(text)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

// ---------------------------------------------------------------------------
// Test harness
// ---------------------------------------------------------------------------

/** Marker thrown by unimplemented `TODO(hw4)` stubs. */
const TODO_MARKER = "TODO(hw4)";

/**
 * Run the automated self-checks against the five-basket worked example and the
 * reference helpers. Unimplemented `TODO(hw4)` functions are reported as
 * `pending` rather than `fail`, so the harness is useful before and after the
 * assignment is implemented.
 *
 * @param {HTMLElement|null} [logElement] element that receives the text log
 * @returns {{passed: number, failed: number, pending: number, checks: Array<Object>}}
 */
function runTests(logElement) {
  const example = tinyWorkedExample();
  const checks = [];
  const basketObjects = example.baskets;

  const pass = (name, detail) => checks.push({ name, status: "PASS", detail });
  const fail = (name, detail) => checks.push({ name, status: "FAIL", detail });
  const pending = (name, detail) => checks.push({ name, status: "PENDING", detail });

  const close = (a, b, eps = 1e-9) => Math.abs(a - b) < eps;

  const check = (name, fn) => {
    try {
      const outcome = fn();
      if (outcome && outcome.pending) pending(name, outcome.pending);
      else pass(name, outcome === undefined ? "" : String(outcome));
    } catch (error) {
      if (String(error && error.message).includes(TODO_MARKER)) {
        pending(name, "TODO(hw4): not implemented yet.");
      } else {
        fail(name, String(error && error.message ? error.message : error));
      }
    }
  };

  // 1. Item counts on the fixture.
  check("fixture: item counts match the hand-computed values", () => {
    for (const [stock, expected] of Object.entries(example.itemCounts)) {
      const actual = countItem(basketObjects, stock);
      if (actual !== expected) {
        throw new Error(`countItem(${stock}) = ${actual}, expected ${expected}`);
      }
    }
    return "all five item counts correct";
  });

  // 2. lift == 1 case.
  check("fixture: bread -> milk has lift exactly 1", () => {
    const [breadMilk] = example.rules;
    const joint = countPair(basketObjects, "bread", "milk");
    const support = computeSupport(joint, example.n);
    const confidence = computeConfidence(joint, countItem(basketObjects, "bread"));
    const lift = computeLift(confidence, countItem(basketObjects, "milk"), example.n);
    if (!close(support.value, breadMilk.support)) throw new Error("support mismatch");
    if (!close(confidence.value, breadMilk.confidence)) throw new Error("confidence mismatch");
    if (!close(lift.value, breadMilk.lift)) throw new Error(`lift = ${lift.value}, expected 1`);
    return `support=${support.value.toFixed(4)} confidence=${confidence.value.toFixed(4)} lift=${lift.value.toFixed(4)}`;
  });

  // 3. lift > 1 case.
  check("fixture: milk -> jam has lift greater than 1", () => {
    const rule = example.rules[1];
    const joint = countPair(basketObjects, rule.antecedent[0], rule.consequent[0]);
    const confidence = computeConfidence(joint, countItem(basketObjects, rule.antecedent[0]));
    const lift = computeLift(confidence, countItem(basketObjects, rule.consequent[0]), example.n);
    if (!close(lift.value, rule.lift)) throw new Error(`lift = ${lift.value}, expected ${rule.lift}`);
    if (!(lift.value > 1)) throw new Error("expected lift > 1");
    return `lift=${lift.value.toFixed(4)} (> 1)`;
  });

  // 4. lift < 1 case.
  check("fixture: jam -> eggs has lift less than 1", () => {
    const rule = example.rules[2];
    const joint = countPair(basketObjects, rule.antecedent[0], rule.consequent[0]);
    const confidence = computeConfidence(joint, countItem(basketObjects, rule.antecedent[0]));
    const lift = computeLift(confidence, countItem(basketObjects, rule.consequent[0]), example.n);
    if (!close(lift.value, rule.lift)) throw new Error(`lift = ${lift.value}, expected ${rule.lift}`);
    if (!(lift.value < 1 && lift.value > 0)) throw new Error("expected 0 < lift < 1");
    return `lift=${lift.value.toFixed(4)} (0 < lift < 1)`;
  });

  // 5. Reversed confidence differs.
  check("fixture: reversed confidence differs (jam <-> eggs)", () => {
    const forwardJoint = countPair(basketObjects, "jam", "eggs");
    const forward = computeConfidence(forwardJoint, countItem(basketObjects, "jam"));
    const reversed = computeConfidence(forwardJoint, countItem(basketObjects, "eggs"));
    if (close(forward.value, reversed.value)) {
      throw new Error("confidence should differ between jam -> eggs and eggs -> jam");
    }
    if (!close(forward.value, 1 / 3) || !close(reversed.value, 1 / 2)) {
      throw new Error(`unexpected values: ${forward.value} vs ${reversed.value}`);
    }
    return `jam->eggs=${forward.value.toFixed(4)} vs eggs->jam=${reversed.value.toFixed(4)}`;
  });

  // 6. Duplicate items in a raw basket are counted once.
  check("duplicate items in a raw basket are counted once", () => {
    const deduped = dedupeBasket(["milk", "bread", "milk", "bread", "jam"]);
    if (deduped.length !== 3) throw new Error(`expected 3 unique items, got ${deduped.length}`);
    const duplicateBasket = [["milk", "bread", "milk", "bread"], ["milk"]];
    if (countItem(duplicateBasket, "milk") !== 2) {
      throw new Error("countItem must count each basket once, not each row");
    }
    if (countPair(duplicateBasket, "milk", "bread") !== 1) {
      throw new Error("countPair must count each basket once");
    }
    return "dedupeBasket removed repeats; counting is per basket";
  });

  // 7. Empty result set renders an empty-state note.
  check("empty rule set renders an empty-state note", () => {
    const scratch = document.createElement("div");
    renderResults([], DATASET_INDEX, scratch);
    if (!/no rules passed/i.test(scratch.textContent)) {
      throw new Error("expected an empty-state message");
    }
    return "empty state rendered";
  });

  // 8. Invalid thresholds are rejected.
  check("invalid thresholds are rejected", () => {
    if (validateThresholds(0.01, 0.3).ok !== true) throw new Error("valid thresholds rejected");
    if (validateThresholds(Number.NaN, 0.3).ok !== false) throw new Error("NaN support accepted");
    if (validateThresholds(0.01, 1.5).ok !== false) throw new Error("confidence > 1 accepted");
    if (validateThresholds(0, 0.3).ok !== false) throw new Error("zero support accepted");
    return "valid accepted, invalid rejected";
  });

  // 9. Zero-denominator guards.
  check("zero-denominator guards return undefined metrics", () => {
    if (computeConfidence(0, 0).defined !== false) throw new Error("count(A)=0 must be undefined");
    if (computeLift({ value: 0.5, defined: true }, 0, 5).defined !== false) {
      throw new Error("count(B)=0 must be undefined");
    }
    if (computeSupport(0, 0).defined !== false) throw new Error("N=0 must be undefined");
    return "zero denominators return defined=false";
  });

  // 10. Student function: frequent itemsets (pending until implemented).
  check("findFrequentItemsets reproduces the fixture's frequent itemsets", () => {
    const itemsets = findFrequentItemsets(basketObjects, 0.4);
    const pairs = itemsets.filter((set) => set.items.length === 2);
    if (pairs.length === 0) throw new Error("no frequent pairs found at support >= 0.4");
    return `${itemsets.length} itemsets`;
  });

  // 11. Student function: rules (pending until implemented).
  check("generateRules reproduces the fixture's rules", () => {
    const itemsets = findFrequentItemsets(basketObjects, 0.2);
    const rules = generateRules(itemsets, 0.5);
    if (rules.length === 0) throw new Error("no rules found at support >= 0.2, confidence >= 0.5");
    return `${rules.length} rules`;
  });

  const passed = checks.filter((c) => c.status === "PASS").length;
  const failed = checks.filter((c) => c.status === "FAIL").length;
  const pendingCount = checks.filter((c) => c.status === "PENDING").length;

  const lines = [
    `HW4 self-checks — pass ${passed}, fail ${failed}, pending ${pendingCount} (of ${checks.length})`,
    "",
    ...checks.map((c) => `${c.status.padEnd(7)} ${c.name}${c.detail ? ` — ${c.detail}` : ""}`),
  ];
  const text = lines.join("\n");

  const target = logElement || document.getElementById("testLog");
  if (target) target.textContent = text;

  return { passed, failed, pending: pendingCount, checks };
}

// ---------------------------------------------------------------------------
// Wiring
// ---------------------------------------------------------------------------

/**
 * Populate the stock -> description lookup used by the renderers from the
 * dictionary-encoded tables embedded in `window.HW4`.
 *
 * @returns {void}
 */
function primeDescriptions() {
  data.stocks.forEach((stock, i) => {
    if (!descriptionByStock.has(stock)) {
      descriptionByStock.set(stock, data.descriptions[i]);
    }
  });
}

/**
 * Read the two sliders and return the thresholds as fractions in `(0, 1]`.
 *
 * @returns {{minSupport: number, minConfidence: number}}
 */
function readThresholds() {
  const supportInput = document.getElementById("min-support");
  const confidenceInput = document.getElementById("min-confidence");
  return {
    minSupport: Number(supportInput.value) / 100,
    minConfidence: Number(confidenceInput.value) / 100,
  };
}

/** Update the `<output>` readouts next to the two sliders. */
function syncThresholdLabels() {
  const supportInput = document.getElementById("min-support");
  const confidenceInput = document.getElementById("min-confidence");
  const supportOutput = document.getElementById("min-support-value");
  const confidenceOutput = document.getElementById("min-confidence-value");
  if (supportOutput) supportOutput.textContent = `${Number(supportInput.value).toFixed(1)}%`;
  if (confidenceOutput) confidenceOutput.textContent = `${Number(confidenceInput.value).toFixed(0)}%`;
}

/**
 * Read the thresholds, run the student pipeline, and render the results.
 *
 * @returns {void}
 */
function runPipeline() {
  const status = document.getElementById("status");
  if (!DATASET_INDEX) {
    if (status) {
      status.textContent = "Dataset not ready yet — reload the page.";
    }
    return;
  }
  const { minSupport, minConfidence } = readThresholds();
  const validation = validateThresholds(minSupport, minConfidence);
  if (!validation.ok) {
    if (status) status.textContent = validation.errors.join(" ");
    return;
  }
  const runButton = document.getElementById("run-rules");
  try {
    if (status) status.textContent = "Mining frequent itemsets…";
    if (runButton) runButton.disabled = true;
    const doWork = () => {
      try {
        const itemsets = findFrequentItemsets(TRANSACTIONS, minSupport);
        const rules = generateRules(itemsets, minConfidence);
        renderResults(rules, DATASET_INDEX);
        if (status) status.textContent = `Done — ${rules.length} rule(s) at support \u2265 ${(minSupport * 100).toFixed(1)}% and confidence \u2265 ${(minConfidence * 100).toFixed(0)}%.`;
        if (runButton) runButton.disabled = false;
      } catch (error) {
        const message = String(error && error.message ? error.message : error);
        const resultsEl = document.getElementById("results");
        if (resultsEl) {
          resultsEl.innerHTML =
            '<p class="empty-state">Run failed &mdash; the rule miner did not complete. ' +

            `Error: ${escapeHtml(message)}</p>`;
        }
        if (status) status.textContent = message;
        if (runButton) runButton.disabled = false;
      }
    };
    if (typeof requestAnimationFrame !== "undefined") {
      requestAnimationFrame(() => setTimeout(doWork, 0));
    } else {
      setTimeout(doWork, 0);
    }
    return;
  } catch (error) {
    const message = String(error && error.message ? error.message : error);
    const resultsEl = document.getElementById("results");
    if (resultsEl) {
      resultsEl.innerHTML =
        '<p class="empty-state">Run failed &mdash; the rule miner did not complete. ' +

        `Error: ${escapeHtml(message)}</p>`;
    }
    if (status) status.textContent = message;
  }
}

/**
 * Wire the controls once the DOM is ready, then build the dataset index from the
 * baskets already decoded from `window.HW4` and render the summary.
 *
 * The data is already in memory (embedded by `transactions.js`), so there is no
 * fetch and no error path beyond the `window.HW4` guard at the top of this file.
 *
 * @returns {void}
 */
function init() {
  renderWorkedExample(tinyWorkedExample());
  syncThresholdLabels();

  const supportInput = document.getElementById("min-support");
  const confidenceInput = document.getElementById("min-confidence");
  if (supportInput) supportInput.addEventListener("input", syncThresholdLabels);
  if (confidenceInput) confidenceInput.addEventListener("input", syncThresholdLabels);

  const runButton = document.getElementById("run-rules");
  if (runButton) runButton.addEventListener("click", runPipeline);

  const testButton = document.getElementById("run-tests");
  if (testButton) testButton.addEventListener("click", () => runTests());

  const status = document.getElementById("status");
  primeDescriptions();
  N = TRANSACTIONS.length;
  DATASET_INDEX = buildIndex(TRANSACTIONS);
  try {
    renderDatasetSummary(DATASET_INDEX);
  } catch (error) {
    // Guard the summary so an unexpected failure in it cannot take the whole
    // page down at load time; the "Run tests" button stays usable regardless.
    const message = String(error && error.message ? error.message : error);
    const summary = document.getElementById("dataset-summary-body");
    if (summary) {
      summary.innerHTML =
        '<p class="empty-state">Could not render the dataset summary: ' +
        `${escapeHtml(message)}.</p>`;
    }
  }
  if (status) {
    status.textContent = `Dataset ready: ${N.toLocaleString("en-US")} baskets, ${data.N_ITEMS.toLocaleString("en-US")} distinct items. Adjust support/confidence and press “Run rules”.`;
  }
}

if (typeof document !== "undefined") {
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
}
