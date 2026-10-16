/**
 * week5-checks/probe.js — headless probe of the fixed HW5 page.
 *
 * Serves week5/ over local HTTP, opens index.html in the locally installed
 * Google Chrome (playwright-core, channel "chrome" — no browser download),
 * waits for the data layer, clicks "Run tests", then starts "Train" WITHOUT
 * waiting on the click (scheduled via setTimeout inside page.evaluate) and
 * waits up to 20 min for the status to leave "Training…". Train is run twice.
 *
 * Captures: tests, training history, best checkpoint, test RMSE vs baseline,
 * tensor counts, Top-10 for users 1 and 2 (with each movie's train-rating
 * count), Recall@10 with exclusions, and every `tf.*` call in script.js with
 * an existence check. Also dumps initial and final factor matrices.
 *
 * Every step is guarded: a failure is printed and the probe continues.
 *
 * TensorFlow.js CDN note
 * ----------------------
 * The page loads tfjs from cdn.jsdelivr.net. Where the launched Chrome has no
 * outbound network, the request fails; the probe caches the real tf.min.js via
 * Node's https client and serves that exact bundle through a Playwright route.
 * week5/ is never modified.
 *
 * Run:  node week5-checks/probe.js
 * Writes: week5-checks/results/probe_fixed.txt
 *         week5-checks/results/factors_init.json
 *         week5-checks/results/factors_final.json
 */

"use strict";

const http = require("http");
const https = require("https");
const fs = require("fs");
const path = require("path");
const { chromium } = require("playwright-core");

const WEEK5 = path.join(__dirname, "..", "week5");
const RESULTS = path.join(__dirname, "results");
const CACHE_DIR = path.join(__dirname, "node_modules", ".cache", "tfjs");
const TF_URL = "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js";

const TRAIN_TIMEOUT_MS = 20 * 60 * 1000;

const MIME = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".data": "text/plain; charset=iso-8859-1",
  ".item": "text/plain; charset=iso-8859-1",
};

function fetchUrl(url, redirects = 0) {
  return new Promise((resolve, reject) => {
    if (redirects > 5) return reject(new Error("too many redirects"));
    https
      .get(url, { headers: { "user-agent": "week5-checks-probe" } }, (res) => {
        if (res.statusCode >= 300 && res.statusCode < 400 && res.headers.location) {
          res.resume();
          return resolve(fetchUrl(new URL(res.headers.location, url).toString(), redirects + 1));
        }
        if (res.statusCode !== 200) {
          res.resume();
          return reject(new Error(`HTTP ${res.statusCode} for ${url}`));
        }
        const chunks = [];
        res.on("data", (d) => chunks.push(d));
        res.on("end", () => resolve(Buffer.concat(chunks)));
      })
      .on("error", reject);
  });
}

async function tfBundle() {
  const file = path.join(CACHE_DIR, "tf.min.js");
  if (fs.existsSync(file)) return fs.readFileSync(file);
  process.stderr.write(`[probe] caching ${TF_URL}\n`);
  const buf = await fetchUrl(TF_URL);
  fs.mkdirSync(CACHE_DIR, { recursive: true });
  fs.writeFileSync(file, buf);
  return buf;
}

function startServer(rootDir) {
  const server = http.createServer((req, res) => {
    const urlPath = decodeURIComponent(new URL(req.url, "http://localhost").pathname);
    const rel = urlPath === "/" ? "index.html" : urlPath.replace(/^\/+/, "");
    const filePath = path.join(rootDir, rel);
    if (!filePath.startsWith(rootDir)) {
      res.writeHead(403);
      res.end("Forbidden");
      return;
    }
    fs.readFile(filePath, (err, data) => {
      if (err) {
        res.writeHead(404);
        res.end("Not found");
        return;
      }
      res.writeHead(200, { "Content-Type": MIME[path.extname(filePath)] || "application/octet-stream" });
      res.end(data);
    });
  });
  return new Promise((resolve) => {
    server.listen(0, "127.0.0.1", () => resolve({ server, port: server.address().port }));
  });
}

// Collect every distinct `tf.*` path referenced in week5/script.js.
function collectTfCalls(file) {
  const src = fs.readFileSync(file, "utf8");
  const matches = src.match(/\btf(?:\.[A-Za-z_$][\w$]*)+/g) || [];
  return [...new Set(matches)].filter((t) => !/\.js$/.test(t)).sort();
}

async function main() {
  const bundle = await tfBundle();

  const { server, port } = await startServer(WEEK5);
  const url = `http://127.0.0.1:${port}/`;

  const browser = await chromium.launch({
    channel: "chrome",
    headless: true,
    args: ["--no-sandbox", "--disable-ipv6"],
  });
  const version = await browser.version();
  const page = await browser.newPage();

  const captured = { pageErrors: [], consoleErrors: [] };
  page.on("pageerror", (err) => captured.pageErrors.push(String(err && err.message ? err.message : err)));
  page.on("console", (msg) => {
    if (msg.type() === "error") captured.consoleErrors.push(msg.text());
  });

  await page.route("**/tf.min.js", (route) =>
    route.fulfill({ status: 200, contentType: "text/javascript; charset=utf-8", body: bundle }),
  );

  await page.goto(url, { waitUntil: "load", timeout: 60000 });
  await page.waitForFunction(
    () => {
      const el = document.getElementById("status");
      return el && /Dataset ready|Data not loaded|Error/i.test(el.textContent || "");
    },
    null,
    { timeout: 60000 },
  );

  const steps = [];

  // --- Run tests -----------------------------------------------------------
  let summaryLine = "(run tests failed)";
  let testLog = "";
  try {
    await page.click("#run-tests");
    await page.waitForFunction(
      () => {
        const el = document.getElementById("testLog");
        return el && /self-checks/i.test(el.textContent || "");
      },
      null,
      { timeout: 60000 },
    );
    const r = await page.evaluate(() => {
      const el = document.getElementById("testLog");
      const t = el ? (el.textContent || "").trim() : "";
      return { log: t, summary: (t.split("\n").find((l) => /self-checks/i.test(l)) || "(no summary line)").trim() };
    });
    testLog = r.log;
    summaryLine = r.summary;
  } catch (err) {
    steps.push(`Run tests — ERROR: ${err && err.message ? err.message : err}`);
  }

  const tfStatus = await page.textContent("#tf-status").catch(() => "(status unavailable)");

  // --- Capture the seeded initial factors ----------------------------------
  let factorsInit = null;
  try {
    factorsInit = await page.evaluate(() => {
      initializeFactors(trainSet, NUM_FACTORS);
      return {
        numFactors: NUM_FACTORS,
        globalMean: model.globalMean,
        userShape: model.userFactors.shape,
        itemShape: model.itemFactors.shape,
        userFactors: model.userFactors.arraySync(),
        itemFactors: model.itemFactors.arraySync(),
      };
    });
  } catch (err) {
    steps.push(`initializeFactors capture — ERROR: ${err && err.message ? err.message : err}`);
  }

  const memBefore = await page.evaluate(() => tf.memory().numTensors).catch(() => "(unavailable)");

  // Train without waiting on the click: schedule it and wait on the status.
  const trainOnce = async (label) => {
    const t0 = Date.now();
    try {
      await page.evaluate(() => {
        setTimeout(() => document.getElementById("train").click(), 0);
      });
      await page.waitForFunction(
        () => {
          const el = document.getElementById("status");
          const t = (el ? el.textContent : "").trim();
          return t.length > 0 && !/^Training/.test(t) && !/^Dataset ready/.test(t) && !/^Loading/.test(t);
        },
        null,
        { timeout: TRAIN_TIMEOUT_MS },
      );
    } catch (err) {
      steps.push(`${label} — ERROR: ${err && err.message ? err.message : err}`);
    }
    return (Date.now() - t0) / 1000;
  };

  // --- Train #1 ------------------------------------------------------------
  const train1Seconds = await trainOnce("Train #1");
  const statusAfterTrain1 = await page.textContent("#status").catch(() => "(status unavailable)");
  const memAfter1 = await page.evaluate(() => tf.memory().numTensors).catch(() => "(unavailable)");
  let history = [];
  let bestCheckpoint = null;
  try {
    history = await page.evaluate(() =>
      trainingHistory.map((r) => ({ epoch: r.epoch, trainRMSE: r.trainRMSE, valRMSE: r.valRMSE })),
    );
    bestCheckpoint = await page.evaluate(() =>
      bestCheckpoint ? { epoch: bestCheckpoint.epoch, trainRMSE: bestCheckpoint.trainRMSE, valRMSE: bestCheckpoint.valRMSE } : null,
    );
  } catch (err) {
    steps.push(`history/checkpoint after Train #1 — ERROR: ${err && err.message ? err.message : err}`);
  }

  // --- Train #2 ------------------------------------------------------------
  const train2Seconds = await trainOnce("Train #2");
  const statusAfterTrain2 = await page.textContent("#status").catch(() => "(status unavailable)");
  const memAfter2 = await page.evaluate(() => tf.memory().numTensors).catch(() => "(unavailable)");
  const historyLenAfter2 = await page.evaluate(() =>
    Array.isArray(trainingHistory) ? trainingHistory.length : "(not an array)",
  ).catch(() => "(unavailable)");

  // --- Metrics -------------------------------------------------------------
  let metrics = null;
  try {
    metrics = await page.evaluate(() => ({
      testRMSE: computeRMSEFor(testSet),
      baselineTestRMSE: computeBaselineRMSE(testSet),
    }));
  } catch (err) {
    steps.push(`metrics — ERROR: ${err && err.message ? err.message : err}`);
  }

  // --- Top-K for users 1 and 2 --------------------------------------------
  const topFor = async (userId) => {
    try {
      return await page.evaluate((uid) => {
        const counts = new Map();
        for (const r of trainSet) counts.set(r.movieId, (counts.get(r.movieId) || 0) + 1);
        return recommendTopK(uid, 10).map((r) => ({
          movieId: r.movieId,
          title: r.title,
          score: r.score,
          trainRatings: counts.get(r.movieId) || 0,
        }));
      }, userId);
    } catch (err) {
      steps.push(`recommendTopK(${userId}, 10) — ERROR: ${err && err.message ? err.message : err}`);
      return null;
    }
  };
  const top1 = await topFor(1);
  const top2 = await topFor(2);

  // --- Recall@10 -----------------------------------------------------------
  let recall = null;
  try {
    recall = await page.evaluate(() => {
      const r = recallAtK(10);
      return {
        overallRecall: r.overallRecall,
        evaluableUsers: r.perUser.length,
        unseenIds: r.excluded.unseenIds,
        usersWithoutRelevant: r.excluded.usersWithoutRelevant,
      };
    });
  } catch (err) {
    steps.push(`recallAtK(10) — ERROR: ${err && err.message ? err.message : err}`);
  }

  // --- Final factors -------------------------------------------------------
  let factorsFinal = null;
  try {
    factorsFinal = await page.evaluate(() => ({
      globalMean: model.globalMean,
      userShape: model.userFactors.shape,
      itemShape: model.itemFactors.shape,
      userFactors: model.userFactors.arraySync(),
      itemFactors: model.itemFactors.arraySync(),
    }));
  } catch (err) {
    steps.push(`final factors capture — ERROR: ${err && err.message ? err.message : err}`);
  }

  // --- tf.* usage and existence -------------------------------------------
  const tfCalls = collectTfCalls(path.join(WEEK5, "script.js"));
  let tfChecks = [];
  try {
    tfChecks = await page.evaluate((list) => {
      return list.map((p) => {
        const parts = p.split(".").slice(1);
        let cur = tf;
        let ok = true;
        for (const part of parts) {
          if (cur && (typeof cur === "object" || typeof cur === "function") && part in cur) cur = cur[part];
          else {
            ok = false;
            break;
          }
        }
        return { path: p, type: ok ? typeof cur : "MISSING" };
      });
    }, tfCalls);
  } catch (err) {
    steps.push(`tf API existence check — ERROR: ${err && err.message ? err.message : err}`);
  }

  const memBeforeDisp = memBefore === null ? "(unavailable)" : memBefore;

  // --- Report --------------------------------------------------------------
  const L = [];
  L.push("HW5 probe — fixed trainMF (Koren L2, seeded init, shuffled batches, best checkpoint)");
  L.push("=".repeat(78));
  L.push(`URL            : ${url}`);
  L.push(`Served folder  : ${WEEK5}`);
  L.push(`Browser        : ${version}`);
  L.push(`TF.js bundle   : ${TF_URL}`);
  L.push("                 (served from a Node-cached copy via a Playwright route because");
  L.push("                  the sandboxed Chrome has no outbound network)");
  L.push("");
  L.push("Run tests");
  L.push("-".repeat(78));
  L.push(summaryLine);
  L.push("");
  L.push(testLog || "(no test log)");
  L.push("");
  L.push("Status");
  L.push("-".repeat(78));
  L.push(`TF.js status         : ${tfStatus}`);
  L.push(`Status after Train#1 : ${statusAfterTrain1}`);
  L.push(`Status after Train#2 : ${statusAfterTrain2}`);
  L.push("");
  L.push("Training timing");
  L.push("-".repeat(78));
  L.push(`Train #1: epochs ${history.length}, wall-clock ${train1Seconds} s`);
  L.push(`Train #2: wall-clock ${train2Seconds} s`);
  L.push("");
  L.push("Training history (after Train #1)");
  L.push("-".repeat(78));
  if (Array.isArray(history) && history.length) {
    L.push("epoch\ttrainRMSE\tvalRMSE");
    for (const r of history) L.push(`${r.epoch}\t${r.trainRMSE}\t${r.valRMSE}`);
  } else {
    L.push("(empty)");
  }
  L.push("");
  L.push("Best checkpoint (selectBestCheckpoint)");
  L.push("-".repeat(78));
  L.push(JSON.stringify(bestCheckpoint));
  L.push("");
  L.push("Test RMSE vs baseline");
  L.push("-".repeat(78));
  if (metrics) {
    L.push(`computeRMSEFor(testSet)        : ${metrics.testRMSE}`);
    L.push(`computeBaselineRMSE(testSet)   : ${metrics.baselineTestRMSE}`);
    L.push(`improvement over baseline      : ${metrics.baselineTestRMSE - metrics.testRMSE}`);
  } else {
    L.push("(unavailable)");
  }
  L.push("");
  L.push("Tensor memory (tf.memory().numTensors)");
  L.push("-".repeat(78));
  L.push(`before first Train              : ${memBeforeDisp}`);
  L.push(`after  first Train              : ${memAfter1}`);
  L.push(`after second Train              : ${memAfter2}`);
  L.push(`trainingHistory length after #2 : ${historyLenAfter2}`);
  L.push("");
  for (const [userId, top] of [[1, top1], [2, top2]]) {
    L.push(`Top-10 for user ${userId} (score + train-rating count of the movie)`);
    L.push("-".repeat(78));
    if (Array.isArray(top)) {
      if (top.length === 0) L.push("(empty)");
      else top.forEach((r, i) => L.push(`${i + 1}. ${r.title}  [id ${r.movieId}]  score ${r.score}  trainRatings ${r.trainRatings}`));
    } else {
      L.push(String(top));
    }
    L.push("");
  }
  L.push("Recall@10");
  L.push("-".repeat(78));
  if (recall) {
    L.push(`overallRecall        : ${recall.overallRecall}`);
    L.push(`evaluable users      : ${recall.evaluableUsers}`);
    L.push(`unseenIds (count)    : ${recall.unseenIds.length} — ${JSON.stringify(recall.unseenIds)}`);
    L.push(`usersWithoutRelevant : ${recall.usersWithoutRelevant.length} — ${JSON.stringify(recall.usersWithoutRelevant)}`);
  } else {
    L.push("(unavailable)");
  }
  L.push("");
  L.push("tf.* calls in week5/script.js (existence check in the loaded tf)");
  L.push("-".repeat(78));
  if (tfChecks.length) {
    const missing = tfChecks.filter((c) => c.type === "MISSING");
    for (const c of tfChecks) L.push(`${c.type === "MISSING" ? "MISSING " : "ok      "} ${c.path}  -> ${c.type}`);
    L.push("");
    L.push(`total ${tfChecks.length}, missing ${missing.length}`);
  } else {
    L.push("(no tf.* calls found or check unavailable)");
  }

  if (steps.length) {
    L.push("");
    L.push("Step errors");
    L.push("-".repeat(78));
    for (const s of steps) L.push(s);
  }
  if (captured.pageErrors.length || captured.consoleErrors.length) {
    L.push("");
    L.push("Browser errors");
    L.push("-".repeat(78));
    for (const e of captured.pageErrors) L.push(`pageerror: ${e}`);
    for (const e of captured.consoleErrors) L.push(`console.error: ${e}`);
  }
  L.push("");

  fs.mkdirSync(RESULTS, { recursive: true });
  fs.writeFileSync(path.join(RESULTS, "probe_fixed.txt"), L.join("\n"));
  if (factorsInit) fs.writeFileSync(path.join(RESULTS, "factors_init.json"), JSON.stringify(factorsInit));
  if (factorsFinal) fs.writeFileSync(path.join(RESULTS, "factors_final.json"), JSON.stringify(factorsFinal));

  process.stdout.write(L.join("\n"));

  await browser.close();
  await new Promise((resolve) => server.close(resolve));
}

main().catch((err) => {
  console.error("probe failed:", err);
  process.exit(1);
});
