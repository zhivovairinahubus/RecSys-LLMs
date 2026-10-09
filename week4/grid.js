#!/usr/bin/env node
/**
 * week4/grid.js — threshold grid over the real dataset.
 *
 * Loads `week4/transactions.js` and `week4/script.js` through `vm` in a
 * browser-shaped sandbox (`window` exists, `document` does not) and calls the
 * student functions exactly where they live. Nothing is copied into this file.
 *
 * For every (support, confidence) pair it prints the number of frequent
 * itemsets by size, the number of rules, how many rules have lift <= 1, the
 * smallest lift, and the timing.
 *
 * Usage: node week4/grid.js
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

const { findFrequentItemsets, generateRules } = sandbox;
const hw4 = sandbox.window.HW4;
const baskets = hw4.baskets.map((basket) => basket.map((i) => hw4.stocks[i]));
const N = baskets.length;

const SUPPORTS = [0.005, 0.01, 0.02, 0.03];
const CONFIDENCES = [0.1, 0.3, 0.6];

function sizeHistogram(itemsets) {
  const hist = new Map();
  for (const itemset of itemsets) {
    const size = itemset.items.length;
    hist.set(size, (hist.get(size) || 0) + 1);
  }
  return hist;
}

function histogramToString(hist) {
  const sizes = [...hist.keys()].sort((a, b) => a - b);
  if (sizes.length === 0) return "-";
  return sizes.map((size) => `${size}:${hist.get(size)}`).join(" ");
}

function ms(startNs, endNs) {
  return Number(endNs - startNs) / 1e6;
}

function pad(value, width, right) {
  const text = String(value);
  return right ? text.padStart(width) : text.padEnd(width);
}

const rows = [];
console.log(`Loaded ${path.basename(TRANSACTIONS_JS)} + ${path.basename(SCRIPT_JS)}; N = ${N.toLocaleString("en-US")} baskets.`);
console.log("");

for (const support of SUPPORTS) {
  const mineStart = process.hrtime.bigint();
  const itemsets = findFrequentItemsets(baskets, support);
  const mineEnd = process.hrtime.bigint();
  const mineMs = ms(mineStart, mineEnd);
  const histogram = histogramToString(sizeHistogram(itemsets));

  for (const confidence of CONFIDENCES) {
    const rulesStart = process.hrtime.bigint();
    const rules = generateRules(itemsets, confidence);
    const rulesEnd = process.hrtime.bigint();

    let liftLE1 = 0;
    let minLift = Infinity;
    for (const rule of rules) {
      if (!Number.isFinite(rule.lift)) continue;
      if (rule.lift <= 1) liftLE1 += 1;
      if (rule.lift < minLift) minLift = rule.lift;
    }

    rows.push({
      support,
      confidence,
      histogram,
      rules: rules.length,
      liftLE1,
      minLift: Number.isFinite(minLift) ? minLift.toFixed(4) : "n/a",
      mineMs: mineMs.toFixed(0),
      rulesMs: ms(rulesStart, rulesEnd).toFixed(0),
    });
  }
}

const columns = [
  { key: "support", title: "support", width: 8 },
  { key: "confidence", title: "conf", width: 5 },
  { key: "histogram", title: "itemsets (size:count)", width: 24 },
  { key: "rules", title: "rules", width: 7, right: true },
  { key: "liftLE1", title: "lift<=1", width: 8, right: true },
  { key: "minLift", title: "min lift", width: 9, right: true },
  { key: "mineMs", title: "mine ms", width: 8, right: true },
  { key: "rulesMs", title: "rules ms", width: 9, right: true },
];

const header = columns.map((c) => pad(c.title, c.width, false)).join("  ");
console.log(header);
console.log("-".repeat(header.length));
for (const row of rows) {
  const supportPct = `${(row.support * 100).toFixed(1)}%`;
  const confidencePct = `${(row.confidence * 100).toFixed(0)}%`;
  const line = columns
    .map((c) => {
      const value =
        c.key === "support" ? supportPct : c.key === "confidence" ? confidencePct : row[c.key];
      return pad(value, c.width, c.right);
    })
    .join("  ");
  console.log(line);
}
