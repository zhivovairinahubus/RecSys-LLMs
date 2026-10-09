#!/usr/bin/env node
/**
 * week4/audit-data.js — read-only audit of the embedded HW4 dataset.
 *
 * Usage:  node week4/audit-data.js week4/transactions.js
 *
 * Loads `transactions.js` exactly as a browser would (via `vm`, no external
 * libraries, no network, no writes) and prints seven groups of checks:
 *
 *   1. declared vs. actual sizes, unique codes, code hygiene
 *   2. index range, duplicate items inside baskets, tiny baskets, pair total
 *   3. basket-size distribution (nearest-rank percentiles)
 *   4. top-10 items by basket count
 *   5. descriptions shared by several stock codes
 *   6. groups of byte-identical baskets
 *   7. basket x item matrix size and density
 *
 * Percentiles use the nearest-rank method: rank = ceil(p/100 * n), 1-based.
 * The script never modifies the dataset or any other file.
 */
"use strict";

const fs = require("fs");
const vm = require("vm");

// ---------------------------------------------------------------------------
// Loading
// ---------------------------------------------------------------------------

/**
 * Execute transactions.js in an isolated context and return window.HW4.
 *
 * @param {string} file path to transactions.js
 * @returns {Object} the HW4 dataset object
 */
function loadHW4(file) {
  const source = fs.readFileSync(file, "utf8");
  const sandbox = { window: {} };
  vm.createContext(sandbox);
  vm.runInContext(source, sandbox, { filename: file });
  const hw4 = sandbox.window.HW4;
  if (!hw4 || typeof hw4 !== "object") {
    throw new Error(`window.HW4 was not assigned by ${file}`);
  }
  return hw4;
}

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

/** @param {number} n @returns {string} locale-grouped integer */
function num(n) {
  return n.toLocaleString("en-US");
}

/** @param {number} n @returns {string} percentage with 4 decimals */
function pct(n) {
  return `${(n * 100).toFixed(4)}%`;
}

/** @param {number} n @returns {string} percentage with 2 decimals */
function pct2(n) {
  return `${(n * 100).toFixed(2)}%`;
}

/**
 * Nearest-rank percentile of an already sorted, non-empty array.
 *
 * @param {number[]} sorted ascending numbers
 * @param {number} p 0..100
 * @returns {number}
 */
function percentile(sorted, p) {
  if (sorted.length === 0) return NaN;
  const rank = Math.ceil((p / 100) * sorted.length);
  return sorted[Math.min(sorted.length - 1, Math.max(0, rank - 1))];
}

/** @param {string} text @param {number} width @returns {string} */
function truncate(text, width) {
  const flat = String(text).replace(/\s+/g, " ").trim();
  return flat.length <= width ? flat : `${flat.slice(0, width - 1)}…`;
}

/** Print a numbered section banner. */
function banner(title) {
  console.log("");
  console.log("=".repeat(74));
  console.log(title);
  console.log("=".repeat(74));
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

function main() {
  const file = process.argv[2];
  if (!file) {
    console.error("Usage: node week4/audit-data.js week4/transactions.js");
    process.exit(2);
  }

  const hw4 = loadHW4(file);
  const stocks = hw4.stocks || [];
  const descriptions = hw4.descriptions || [];
  const baskets = hw4.baskets || [];
  const nItems = stocks.length;

  console.log(`audit-data.js — ${file}`);

  // --- 1. declared sizes and code hygiene ---------------------------------
  banner("1. Declared sizes vs. actual arrays, code hygiene");

  const uniqueCodes = new Set(stocks);
  const withSpace = stocks.filter((c) => /\s/.test(c));
  const lowerCodes = stocks.filter((c) => /[a-z]/.test(c));

  console.log(`N_BASKETS (declared)        : ${num(hw4.N_BASKETS)}`);
  console.log(`baskets.length (actual)     : ${num(baskets.length)}`);
  console.log(`match                       : ${hw4.N_BASKETS === baskets.length ? "YES" : "NO"}`);
  console.log(`N_ITEMS (declared)          : ${num(hw4.N_ITEMS)}`);
  console.log(`stocks.length (actual)      : ${num(stocks.length)}`);
  console.log(`descriptions.length         : ${num(descriptions.length)}`);
  console.log(
    `match                       : ${
      hw4.N_ITEMS === stocks.length && stocks.length === descriptions.length
        ? "YES"
        : "NO"
    }`,
  );
  console.log(`unique stock codes          : ${num(uniqueCodes.size)}`);
  console.log(`duplicated codes            : ${num(stocks.length - uniqueCodes.size)}`);
  console.log(`codes containing whitespace : ${num(withSpace.length)}${withSpace.length ? ` -> ${withSpace.slice(0, 5).join(", ")}` : ""}`);
  console.log(`codes containing lowercase  : ${num(lowerCodes.length)}${lowerCodes.length ? ` -> ${lowerCodes.slice(0, 5).join(", ")}` : ""}`);

  // --- 1b. description hygiene and non-product codes -----------------------
  banner("1b. Description hygiene and non-product codes");

  const descEmpty = descriptions.filter((d) => !String(d).trim());
  const descLower = descriptions.filter((d) => /[a-z]/.test(d));
  const descRunSpace = descriptions.filter((d) => /\s{2,}/.test(d));
  const descEdgeSpace = descriptions.filter((d) => d !== d.trim());
  const descTrailingPunct = descriptions.filter((d) => /[.,;:!?\-]$/.test(d.trim()));

  console.log(`descriptions that are blank          : ${num(descEmpty.length)}`);
  console.log(`descriptions with lowercase letters  : ${num(descLower.length)}${descLower.length ? ` -> ${descLower.slice(0, 3).join(" | ")}` : ""}`);
  console.log(`descriptions with run of 2+ spaces   : ${num(descRunSpace.length)}`);
  console.log(`descriptions with edge whitespace    : ${num(descEdgeSpace.length)}`);
  console.log(`descriptions ending in punctuation   : ${num(descTrailingPunct.length)}${descTrailingPunct.length ? ` -> ${descTrailingPunct.slice(0, 3).join(" | ")}` : ""}`);
  console.log(`unique descriptions                  : ${num(new Set(descriptions).size)} (vs ${num(nItems)} codes)`);

  // Non-product StockCodes named by readme §3: postage, carriage, bank charges,
  // manual entries, samples, discounts, gift vouchers, packing charges, and the
  // code-level ids those map to in the raw UCI workbook.
  const NONPRODUCT_CODES = new Set([
    "POST", "CARRIAGE", "BANK CHARGES", "M", "D", "S", "DOT", "AM", "CRUK",
    "PADS", "C2", "DCDS2", "GIFT", "ADJUST", "ADJUSTMENT", "SAMPLE", "DISCOUNT",
  ]);
  // The same list expressed as canonical description labels (exact match).
  const NONPRODUCT_LABELS = new Set([
    "POSTAGE", "DOTCOM POSTAGE", "CARRIAGE", "BANK CHARGES", "MANUAL", "SAMPLE",
    "DISCOUNT", "GIFT VOUCHER", "GIFT VOUCHERS", "DONATION", "AMAZON FEE",
    "CRUK COMMISSION", "COMMISSION", "PACKING", "PACKING CHARGE",
    "PACKING CHARGES", "ADJUSTMENT",
  ]);

  const codeHits = [];
  const labelHits = [];
  stocks.forEach((code, i) => {
    if (NONPRODUCT_CODES.has(code)) codeHits.push(`${code} (${descriptions[i]})`);
    if (NONPRODUCT_LABELS.has(descriptions[i].trim().toUpperCase())) {
      labelHits.push(`${code} = ${descriptions[i]}`);
    }
  });

  console.log(`non-product codes present            : ${num(codeHits.length)}${codeHits.length ? ` -> ${codeHits.slice(0, 10).join(", ")}` : ""}`);
  console.log(`descriptions equal to a non-product label : ${num(labelHits.length)}${labelHits.length ? ` -> ${labelHits.slice(0, 10).join(", ")}` : ""}`);

  // Informational only: descriptions merely containing a keyword. Real products
  // such as "FRENCH CARRIAGE LANTERN" show up here and are NOT non-product.
  const KEYWORDS = /(POSTAGE|CARRIAGE|BANK CHARGE|MANUAL|SAMPLE|DISCOUNT|VOUCHER|DONATION|ADJUSTMENT|PACKING)/;
  const keywordHits = stocks
    .map((code, i) => ({ code, desc: descriptions[i] }))
    .filter((e) => KEYWORDS.test(e.desc) && !NONPRODUCT_LABELS.has(e.desc.trim().toUpperCase()));
  console.log(`descriptions containing a keyword (false positives kept): ${num(keywordHits.length)}`);
  keywordHits.slice(0, 10).forEach((e) => console.log(`      ${e.code} = ${e.desc}`));

  // --- 2. integrity --------------------------------------------------------
  banner("2. Index range, duplicate items, tiny baskets, pair total");

  let outOfRange = 0;
  const outOfRangeSamples = [];
  let basketsWithRepeat = 0;
  let basketsWithUnder2Distinct = 0;
  let totalPairs = 0;
  let totalDistinctPairs = 0;

  for (const basket of baskets) {
    totalPairs += basket.length;
    const distinct = new Set();
    for (const idx of basket) {
      if (!Number.isInteger(idx) || idx < 0 || idx >= nItems) {
        outOfRange += 1;
        if (outOfRangeSamples.length < 5) outOfRangeSamples.push(idx);
      } else {
        distinct.add(idx);
      }
    }
    if (distinct.size !== basket.length) basketsWithRepeat += 1;
    if (distinct.size < 2) basketsWithUnder2Distinct += 1;
    totalDistinctPairs += distinct.size;
  }

  console.log(`indices out of range [0..${nItems - 1}] : ${num(outOfRange)}${outOfRange ? ` -> sample ${outOfRangeSamples.join(", ")}` : ""}`);
  console.log(`baskets with a repeated item         : ${num(basketsWithRepeat)}`);
  console.log(`baskets with < 2 distinct items      : ${num(basketsWithUnder2Distinct)}`);
  console.log(`total (basket, item) pairs           : ${num(totalPairs)}`);
  console.log(`pairs after in-basket dedupe         : ${num(totalDistinctPairs)}`);

  // --- 3. basket size distribution ----------------------------------------
  banner("3. Basket size distribution (nearest-rank percentiles)");

  const sizes = baskets.map((b) => b.length).sort((a, b) => a - b);
  console.log(`min     : ${num(sizes[0])}`);
  console.log(`median  : ${num(percentile(sizes, 50))}  (p50)`);
  console.log(`p90     : ${num(percentile(sizes, 90))}`);
  console.log(`p99     : ${num(percentile(sizes, 99))}`);
  console.log(`max     : ${num(sizes[sizes.length - 1])}`);
  console.log(`mean    : ${(totalPairs / baskets.length).toFixed(2)}`);
  console.log(`baskets with >= 50 items : ${num(sizes.filter((s) => s >= 50).length)} (${pct2(sizes.filter((s) => s >= 50).length / baskets.length)})`);
  console.log(`baskets with >= 100 items: ${num(sizes.filter((s) => s >= 100).length)} (${pct2(sizes.filter((s) => s >= 100).length / baskets.length)})`);

  // --- 4. top-10 items -----------------------------------------------------
  banner("4. Top-10 items by number of baskets");

  const counts = new Array(nItems).fill(0);
  for (const basket of baskets) {
    for (const idx of new Set(basket)) {
      if (idx >= 0 && idx < nItems) counts[idx] += 1;
    }
  }
  const top = counts
    .map((count, idx) => ({ idx, count }))
    .sort((a, b) => b.count - a.count || a.idx - b.idx)
    .slice(0, 10);

  console.log("rank  code        baskets    share    description");
  top.forEach((entry, i) => {
    console.log(
      `${String(i + 1).padStart(4)}  ${stocks[entry.idx].padEnd(10)} ${num(entry.count).padStart(9)}  ${pct2(entry.count / baskets.length).padStart(7)}  ${truncate(descriptions[entry.idx], 48)}`,
    );
  });

  // --- 5. shared descriptions ---------------------------------------------
  banner("5. Descriptions used by several stock codes");

  /** @type {Map<string, Set<string>>} */
  const codesByDescription = new Map();
  stocks.forEach((code, i) => {
    const desc = descriptions[i];
    if (!codesByDescription.has(desc)) codesByDescription.set(desc, new Set());
    codesByDescription.get(desc).add(code);
  });
  const shared = [...codesByDescription.entries()]
    .filter(([, codes]) => codes.size > 1)
    .sort((a, b) => b[1].size - a[1].size || a[0].localeCompare(b[0]));
  const sharedCodes = shared.reduce((acc, [, codes]) => acc + codes.size, 0);

  console.log(`descriptions mapped to > 1 code: ${shared.length}`);
  console.log(`codes involved                 : ${sharedCodes}`);
  for (const [desc, codes] of shared) {
    console.log(`  ${truncate(desc, 46).padEnd(47)} ${[...codes].sort().join(", ")}`);
  }

  // --- 6. identical baskets ------------------------------------------------
  banner("6. Groups of byte-identical baskets");

  /** @type {Map<string, number[]>} */
  const groups = new Map();
  baskets.forEach((basket, id) => {
    const key = [...basket].sort((a, b) => a - b).join(",");
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(id);
  });
  const duplicated = [...groups.entries()].filter(([, ids]) => ids.length > 1);
  const extraCopies = duplicated.reduce((acc, [, ids]) => acc + (ids.length - 1), 0);
  const largest = [...duplicated]
    .sort((a, b) => b[1].length - a[1].length || a[0].localeCompare(b[0]))
    .slice(0, 5);

  console.log(`distinct basket patterns : ${num(groups.size)}`);
  console.log(`patterns appearing > 1   : ${num(duplicated.length)}`);
  console.log(`extra (redundant) copies : ${num(extraCopies)} (${pct2(extraCopies / baskets.length)} of all baskets)`);
  console.log("five largest groups:");
  largest.forEach(([key, ids], i) => {
    const items = key === "" ? 0 : key.split(",").length;
    console.log(
      `  ${i + 1}. ${num(ids.length)} copies, ${items} items, first basket ids: ${ids.slice(0, 6).join(", ")}${ids.length > 6 ? ", …" : ""}`,
    );
    console.log(`      codes: ${key.split(",").slice(0, 12).map((c) => stocks[Number(c)]).join(", ")}${items > 12 ? ", …" : ""}`);
  });

  // --- 7. matrix size and density -----------------------------------------
  banner("7. Basket x item matrix");

  const cells = baskets.length * nItems;
  console.log(`rows (baskets)            : ${num(baskets.length)}`);
  console.log(`cols (distinct items)     : ${num(nItems)}`);
  console.log(`cells (dense matrix)      : ${num(cells)}`);
  console.log(`non-zero cells (pairs)    : ${num(totalPairs)}`);
  console.log(`density (non-zero share)  : ${pct(totalPairs / cells)}`);
  console.log(`sparsity                  : ${pct(1 - totalPairs / cells)}`);
}

main();
