/**
 * week6-checks/probe.js — headless probe of the week6 page (final implementation).
 *
 * Serves week6/ over local HTTP, opens index.html in Google Chrome
 * (playwright-core, channel "chrome"), routes tf.min.js through a cache
 * to avoid external network fetches if desired, waits for dataset load,
 * clicks "Run tests", and writes results/probe_final.txt.
 */

"use strict";

const http = require("http");
const https = require("https");
const fs = require("fs");
const path = require("path");
const { chromium } = require("playwright-core");

const WEEK6 = path.join(__dirname, "..", "week6");
const RESULTS = path.join(__dirname, "results");
const CACHE_DIR = path.join(__dirname, "node_modules", ".cache", "tfjs");
const TF_URL = "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js";

const MIME = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".svg": "image/svg+xml",
  ".png": "image/png",
  ".data": "text/plain; charset=iso-8859-1",
  ".item": "text/plain; charset=iso-8859-1",
};

function ensureDir(p) {
  if (!fs.existsSync(p)) fs.mkdirSync(p, { recursive: true });
}

function fetchUrl(url, redirects = 0) {
  return new Promise((resolve, reject) => {
    if (redirects > 5) return reject(new Error("too many redirects"));
    https
      .get(url, { headers: { "user-agent": "week6-checks-probe" } }, (res) => {
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
  try {
    const buf = await fetchUrl(TF_URL);
    ensureDir(CACHE_DIR);
    fs.writeFileSync(file, buf);
    return buf;
  } catch (e) {
    return null;
  }
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
      const ext = path.extname(filePath);
      res.writeHead(200, { "Content-Type": MIME[ext] || "application/octet-stream" });
      res.end(data);
    });
  });
  return new Promise((resolve) => {
    server.listen(0, "127.0.0.1", () => resolve({ server, port: server.address().port }));
  });
}

async function main() {
  ensureDir(RESULTS);
  const bundle = await tfBundle();
  const { server, port } = await startServer(WEEK6);
  const url = `http://127.0.0.1:${port}/index.html`;
  const outPath = path.join(RESULTS, "probe_final.txt");

  const browser = await chromium.launch({
    channel: "chrome",
    headless: true,
    protocolTimeout: 1800000,
    args: ["--no-sandbox", "--disable-ipv6"],
  });
  const version = await browser.version();
  const page = await browser.newPage();
  page.setDefaultTimeout(1200000);
  page.setDefaultNavigationTimeout(1200000);

  const logs = [];
  page.on("console", (msg) => logs.push(`[console] ${msg.text()}`));
  page.on("pageerror", (err) => logs.push(`[pageerror] ${err.message}`));

  if (bundle) {
    await page.route("**/tf.min.js", (route) =>
      route.fulfill({ status: 200, contentType: "text/javascript; charset=utf-8", body: bundle }),
    );
  }

  await page.goto(url, { waitUntil: "load" });

  const loadStart = Date.now();

  try {
    await page.waitForSelector("#dataset-summary-body", { timeout: 60000 });
    await page.waitForFunction(() => {
      const el = document.getElementById("dataset-summary-body");
      return el && el.textContent && !el.textContent.includes("Loading dataset");
    }, { timeout: 120000 });
  } catch (e) {
    logs.push(`[wait] dataset summary timeout: ${e.message}`);
  }

  // Dataset ready = the split exists and the candidate pool is populated.
  let datasetReady = null;
  try {
    await page.waitForFunction(() => {
      return typeof HW6_STATE !== "undefined" && HW6_STATE.splitReady && HW6_STATE.loaded;
    }, { timeout: 120000 });
    datasetReady = Date.now();
  } catch (e) {
    logs.push(`[wait] dataset/split ready timeout: ${e.message}`);
  }
  const pageLoadToReadyMs = datasetReady === null ? null : datasetReady - loadStart;

  await page.waitForTimeout(1000);

  const tfBackend = await page.evaluate(() => (typeof tf !== "undefined" ? tf.getBackend() : "undefined"));
  const datasetSummary = await page.evaluate(() => document.getElementById("dataset-summary-body")?.innerText || "");

  // u.item encoding check: data.js uses `Response.text()`, which the Fetch spec
  // always decodes as UTF-8 regardless of any charset header. Accented latin1
  // bytes therefore become U+FFFD. Count and list the corrupted titles.
  const encodingCheck = await page.evaluate(() => {
    try {
      const broken = movies
        .filter((m) => typeof m.title === "string" && m.title.includes("\uFFFD"))
        .map((m) => ({ id: m.id, title: m.title }));
      const total = movies.length;
      // Titles that look correct would have no replacement char; also record how
      // many carry any non-ASCII character at all (expected: only the broken ones).
      const nonAscii = movies.filter((m) => typeof m.title === "string" && /[^\x00-\x7F]/.test(m.title)).length;
      return { total: total, corrupted: broken.length, nonAscii: nonAscii, list: broken };
    } catch (e) {
      return { err: e.message };
    }
  });

  // Do the corrupted titles appear in the submitted Top-10 for users 1 and 2?
  const corruptedInSubmission = (() => {
    try {
      const ids = new Set((encodingCheck.list || []).map((b) => b.id));
      const csv = fs.readFileSync(path.join(WEEK6, "submission", "recommendations.csv"), "utf8");
      const rows = csv.trim().split("\n").slice(1);
      const hits = [];
      for (const row of rows) {
        const cells = row.split(",");
        const movieId = Number(cells[4]);
        if (ids.has(movieId)) hits.push({ movieId: movieId, title: cells[5] });
      }
      return { hits: hits, count: hits.length, movieCount: rows.length };
    } catch (e) {
      return { err: e.message };
    }
  })();

  // Build graph first
  let buildGraphMs = null;
  try {
    buildGraphMs = await page.evaluate(async () => {
      const t0 = performance.now();
      const btn = document.getElementById('build-graph');
      if (btn) btn.click();
      // graphReady flips once buildGraph() has populated GRAPH / the view.
      const deadline = t0 + 60000;
      while (performance.now() < deadline) {
        const g = (typeof getGraph === 'function') ? getGraph() : (typeof GRAPH !== 'undefined' ? GRAPH : null);
        if (g && g.nodeCount) break;
        await new Promise((r) => setTimeout(r, 50));
      }
      return performance.now() - t0;
    });
    await page.waitForTimeout(3000);
  } catch (e) {
    logs.push(`[click] build-graph failed: ${e.message}`);
  }

  // Memory before
  const memBefore = await page.evaluate(() => {
    if (typeof tf !== 'undefined' && tf.memory) return tf.memory().numTensors;
    return null;
  });

  // Run pageRank twice
  const pr1 = await page.evaluate(() => { try { const r = pageRank(0.85); return {it:r.iterations, res:r.residual, conv:r.converged}; } catch(e){ return {err:e.message}; } });
  const pr2 = await page.evaluate(() => { try { const r = pageRank(0.85); return {it:r.iterations, res:r.residual, conv:r.converged}; } catch(e){ return {err:e.message}; } });
  const pr95 = await page.evaluate(() => { try { const r = pageRank(0.95); return {it:r.iterations, res:r.residual, conv:r.converged}; } catch(e){ return {err:e.message}; } });
  const ppr0 = await page.evaluate(() => { try { const g = (typeof getGraph==='function')?getGraph():GRAPH; const n = g?g.nodeCount:0; const r = personalizedPageRank(0,0.85); return {it:r.iterations, res:r.residual, conv:r.converged, n:n}; } catch(e){ return {err:e.message}; } });
  await page.waitForTimeout(2000);

  const memAfter = await page.evaluate(() => {
    if (typeof tf !== 'undefined' && tf.memory) return tf.memory().numTensors;
    return null;
  });

  let runTestsMs = null;
  try {
    runTestsMs = await page.evaluate(async () => {
      const t0 = performance.now();
      if (typeof runSelfChecks === 'function') {
        runSelfChecks();
      } else {
        const btn = document.getElementById('run-tests');
        if (btn) btn.click();
      }
      return performance.now() - t0;
    });
    await page.waitForFunction(() => {
      const el = document.getElementById('self-checks');
      if (!el) return false;
      const t = el.innerText || '';
      return t.trim() && !t.includes('Press “Run tests”');
    }, { timeout: 60000 });
  } catch (e) {
    logs.push(`[click] run-tests failed: ${e.message}`);
  }

  await page.waitForTimeout(1000);
  const selfChecks = await page.evaluate(() => document.getElementById("self-checks")?.innerText || "");

  const memAfterRunTests = await page.evaluate(() => {
    if (typeof tf !== 'undefined' && tf.memory) return tf.memory().numTensors;
    return null;
  });

  // Re-run tests a second time: the check harness must not leak tensors.
  let memAfterRunTests2 = null;
  try {
    await page.evaluate(() => { if (typeof runSelfChecks === 'function') runSelfChecks(); });
    await page.waitForFunction(() => {
      const el = document.getElementById('self-checks');
      const t = el ? el.innerText || '' : '';
      return t.trim() && !t.includes('Press “Run tests”');
    }, { timeout: 60000 });
    await page.waitForTimeout(500);
    memAfterRunTests2 = await page.evaluate(() => {
      if (typeof tf !== 'undefined' && tf.memory) return tf.memory().numTensors;
      return null;
    });
  } catch (e) {
    logs.push(`[rerun] run-tests failed: ${e.message}`);
  }

  // Per-alpha convergence of global pageRank and PPR(user 1), readme 3.4/6.1.
  const convergenceByAlpha = await page.evaluate(() => {
    const out = { pageRank: [], pprUser1: [] };
    for (const alpha of [0.50, 0.70, 0.85, 0.95]) {
      try {
        const pr = pageRank(alpha);
        out.pageRank.push({ alpha: alpha, iterations: pr.iterations, residual: pr.residual, converged: pr.converged });
      } catch (e) {
        out.pageRank.push({ alpha: alpha, error: e.message });
      }
      try {
        const ppr = personalizedPageRank(1, alpha);
        out.pprUser1.push({ alpha: alpha, iterations: ppr.iterations, residual: ppr.residual, converged: ppr.converged });
      } catch (e) {
        out.pprUser1.push({ alpha: alpha, error: e.message });
      }
    }
    return out;
  });

  // tf.* API audit: every tf.* member referenced in the source must exist in the
  // loaded runtime. Source is read on disk; existence is checked in the page.
  let tfAudit = null;
  try {
    const files = ["graph.js", "script.js"].map((f) => ({
      file: f,
      src: fs.readFileSync(path.join(WEEK6, f), "utf8"),
    }));
    const perFile = {};
    const allNames = new Set();
    for (const { file, src } of files) {
      const names = new Set();
      const re = /\btf\.([A-Za-z_$][\w$]*)/g;
      let m;
      while ((m = re.exec(src)) !== null) names.add(m[1]);
      perFile[file] = Array.from(names).sort();
      for (const n of names) allNames.add(n);
    }
    const existence = await page.evaluate((names) => {
      const res = {};
      for (const n of names) res[n] = typeof tf !== "undefined" && typeof tf[n] !== "undefined";
      return res;
    }, Array.from(allNames));
    tfAudit = { perFile: perFile, existence: existence };
  } catch (e) {
    tfAudit = { err: e.message };
  }


  // PRIMARY path (item 3): after Build graph, the ONLY evaluation action is a
  // click on "Evaluate Recall@10 — all four methods" (#recall-btn). No manual
  // slider move and no alpha-sweep button. The handler must select alpha on
  // validation, apply it to the slider, then evaluate test at that alpha.
  let evaluateClick = null;
  try {
    evaluateClick = await page.evaluate(async () => {
      const t0 = performance.now();
      const btn = document.getElementById("recall-btn");
      if (!btn) return { err: "#recall-btn not found" };
      btn.click();
      // Wait for the two async panels the handler fills: the comparison table
      // and the recall detail. Watch HW6_STATE for the recorded rows.
      const deadline = t0 + 600000;
      while (performance.now() < deadline) {
        const rows = (typeof HW6_STATE !== "undefined" && HW6_STATE.comparisonRows) || null;
        if (rows && rows.length >= 4) break;
        await new Promise((r) => setTimeout(r, 100));
      }
      const ms = performance.now() - t0;
      const slider = document.getElementById("alpha");
      const comparisonRows = (typeof HW6_STATE !== "undefined" && HW6_STATE.comparisonRows) || [];
      const sweepRows = (typeof HW6_STATE !== "undefined" && HW6_STATE.alphaSweepRows) || [];
      const status = document.getElementById("status") ? document.getElementById("status").innerText : "";
      return {
        ms: ms,
        sliderValue: slider ? Number(slider.value) : null,
        sliderReadout: document.getElementById("alpha-value") ? document.getElementById("alpha-value").textContent : null,
        status: status,
        sweep: sweepRows.map((row) => ({ alpha: row.alpha, recall: row.recall, error: row.error })),
        rows: comparisonRows.map((row) => ({
          method: row.method,
          recall: row.recall,
          cohortSize: row.cohortSize,
          excludedUnseen: row.excludedUnseen,
          excludedNoEligible: row.excludedNoEligible,
          meanPopularity: row.meanPopularity,
          iterations: row.iterations,
          residual: row.residual,
          converged: row.converged,
          error: row.error,
          note: row.note,
        })),
      };
    });
  } catch (e) {
    evaluateClick = { err: e.message };
  }

  const memAfterEvaluate = await page.evaluate(() => {
    if (typeof tf !== 'undefined' && tf.memory) return tf.memory().numTensors;
    return null;
  });

  // Derived views so the rest of the report (cross-check, exports) keeps working.
  const sweep = evaluateClick && !evaluateClick.err
    ? {
        ms: evaluateClick.ms,
        best: (function () {
          let best = null; let bestRecall = -Infinity;
          for (const row of evaluateClick.sweep || []) {
            if (Number.isFinite(row.recall) && row.recall > bestRecall) { bestRecall = row.recall; best = row.alpha; }
          }
          return best;
        })(),
        note: "",
        rows: evaluateClick.sweep || [],
      }
    : { err: evaluateClick ? evaluateClick.err : "evaluate click failed" };
  const comparison = evaluateClick && !evaluateClick.err
    ? { ms: evaluateClick.ms, rows: evaluateClick.rows || [] }
    : { err: evaluateClick ? evaluateClick.err : "evaluate click failed" };
  const sliderSet = evaluateClick && !evaluateClick.err
    ? { value: evaluateClick.sliderValue, readout: evaluateClick.sliderReadout }
    : { err: evaluateClick ? evaluateClick.err : "evaluate click failed" };

  // MF_INFO after Evaluate (the handler loads MF on first need).
  const mfLoad = await page.evaluate(() => {
    try {
      return {
        checkpoint: MF_INFO.checkpoint,
        factors: MF_INFO.factors,
        source: MF_INFO.source,
        validationRMSE: MF_INFO.validationRMSE,
      };
    } catch (e) {
      return { err: e.message };
    }
  });


  // Repeat the Evaluate click: it must not leak tensors and must keep the same
  // selected alpha.
  const repeatSweep = await page.evaluate(async () => {
    try {
      const t0 = performance.now();
      const btn = document.getElementById("recall-btn");
      if (btn) btn.click();
      const deadline = t0 + 600000;
      while (performance.now() < deadline) {
        const rows = (typeof HW6_STATE !== "undefined" && HW6_STATE.comparisonRows) || null;
        if (rows && rows.length >= 4) break;
        await new Promise((r) => setTimeout(r, 100));
      }
      const slider = document.getElementById("alpha");
      return { ms: performance.now() - t0, best: slider ? Number(slider.value) : null };
    } catch (e) {
      return { err: e.message };
    }
  });
  await page.waitForTimeout(500);
  const memAfterRepeat = await page.evaluate(() => {
    if (typeof tf !== 'undefined' && tf.memory) return tf.memory().numTensors;
    return null;
  });

  // PPR top-K for the two smallest test-cohort users.
  const pprRecs = await page.evaluate(() => {
    const out = {};
    try {
      const cohort = hw6SharedCohort("test");
      if (cohort.length < 2) return { err: "test cohort smaller than 2" };
      [1, 2].forEach((n) => {
        const userId = cohort[n - 1];
        const recs = recommendTopK("ppr", userId, 10, "test");
        out["user" + n] = {
          userId: userId,
          recs: recs.map((r) => ({ movieId: r.movieId, title: r.title, score: r.score, popularity: r.popularity })),
        };
      });
    } catch (e) {
      out.err = e.message;
    }
    return out;
  });

  // (c) Raw PPR through the page's adapter at alpha 0.85: which node receives the
  // restart mass, and does that vector match the independent reference PPR for
  // user 1?
  const rawPpr = await page.evaluate(() => {
    try {
      const graph = getGraph();
      const result = hw6RunPPR(graph, 1, 0.85);
      const scores = result.scores;
      // node with the largest component right after one restart is the restart node;
      // here we instead report which node index the adapter seeded. The adapter
      // calls personalizedPageRank(rawUserId=1, alpha): node index 1.
      const nUsers = hw6UserIds(graph).length;
      const userNode1 = hw6UserNodeIndex(graph, 1);
      // Report the highest-mass node over the whole vector and its kind.
      let argmax = 0;
      for (let i = 1; i < scores.length; i++) if (scores[i] > scores[argmax]) argmax = i;
      const movieLocal = hw6MovieLocalIndex(graph);
      const argmaxMovieId = (() => {
        if (argmax < nUsers) return null;
        for (const [mid, local] of movieLocal) if (nUsers + local === argmax) return mid;
        return null;
      })();
      return {
        nUsers: nUsers,
        userNodeOfUser1: userNode1,
        // With the corrected convention the adapter passes the RAW user id, and
        // graph.js maps it through GRAPH.nodeOfUser; the effective restart node is
        // user 1's node (0), not the literal index 1.
        adapterRestartNode: userNode1,
        argmaxNode: argmax,
        argmaxIsUserNode: argmax < nUsers,
        argmaxMovieId: argmaxMovieId,
        argmaxScore: scores[argmax],
        sum: Array.from(scores).reduce((a, b) => a + b, 0),
        iterations: result.iterations,
        residual: result.residual,
        converged: result.converged,
        movieScoresUser1Node1: Array.from(movieScores(scores, graph)).slice(0, 5),
      };
    } catch (e) {
      return { err: e.message };
    }
  });

  // The reference PPR vector for user 1 (node 1) computed independently is in
  // results/reference.json (pprUser1Node085). Read it for the comparison table.
  let refJson = null;
  try {
    refJson = JSON.parse(fs.readFileSync(path.join(RESULTS, "reference.json"), "utf8"));
  } catch (e) {
    refJson = null;
  }

  // MF_INFO: checkpoint, validation RMSE, per-epoch history.
  const mfInfo = await page.evaluate(() => {
    try {
      return JSON.parse(JSON.stringify(MF_INFO));
    } catch (e) {
      return { err: e.message };
    }
  });

  // Submission export: drive the page through TWO-USER DEMO and Top-10 for users
  // 1 and 2 across all four methods, then read the three serializers. Order per
  // readme 9: build graph -> run tests -> alpha sweep -> slider -> comparison ->
  // two-user demo -> Top-10 ppr/mf/popularity/pagerank. The previous sections
  // already covered build/tests/sweep/slider/comparison.
  const submission = await page.evaluate(async () => {
    try {
      const cohort = hw6SharedCohort("test");
      const u1 = cohort[0];
      const u2 = cohort[1];

      // Two-user one-factor demonstration (records its own panel; the demo does
      // not write recommendation rows, so the Top-10 pass below does).
      const demo = hw6RunTwoUserDemo();
      const demoUsers = demo && demo.users ? demo.users.slice() : [];

      // Top-10 for both users across all four methods. Reuse the page's own
      // recorder so the recommendation CSV matches what the tables would hold.
      const methods = ["ppr", "mf", "popularity", "pagerank"];
      const alpha = hw6ReadAlpha();
      for (const userId of [u1, u2]) {
        for (const method of methods) {
          if (method === "mf") await hw6EnsureMfForNeed();
          const recs = recommendTopK(method, userId, 10, "test");
          hw6RecordRecommendationRows(userId, method, method === "mf" || method === "popularity" ? null : alpha, recs);
        }
      }

      return {
        demoUsers: demoUsers,
        user1: u1,
        user2: u2,
        methods: methods,
        verification: serializeVerification(),
        results: serializeResults(),
        recommendations: serializeRecommendations(),
      };
    } catch (e) {
      return { err: e.message };
    }
  });

  const SUBMISSION = path.join(WEEK6, "submission");
  ensureDir(SUBMISSION);
  if (!submission.err) {
    fs.writeFileSync(path.join(SUBMISSION, "verification.json"), submission.verification);
    fs.writeFileSync(path.join(SUBMISSION, "results.csv"), submission.results);
    fs.writeFileSync(path.join(SUBMISSION, "recommendations.csv"), submission.recommendations);
  }

  // Cross-check the page against the independent reference.
  const crossCheck = [];
  if (refJson) {
    const refSweep = refJson.alphaSweep || [];
    crossCheck.push("(a) Recall / selected alpha");
    crossCheck.push(`    page alpha sweep: ${(sweep.rows || []).map((r) => r.alpha + "=" + (r.recall === null ? "-" : r.recall)).join(", ")}; best=${sweep.best}`);
    crossCheck.push(`    ref  alpha sweep: ${refSweep.map((r) => r.alpha + "=" + r.recall).join(", ")}; best=${refJson.bestAlpha}`);
    const refTest = (refJson.recall && refJson.recall.test) || {};
    for (const method of ["popularity", "pagerank", "ppr"]) {
      const pageRow = (comparison.rows || []).find((r) => r.method === method) || {};
      const refRow = refTest[method] || {};
      const pageVal = pageRow.recall;
      const refVal = refRow.recall;
      const same = pageVal !== undefined && refVal !== undefined && Math.abs(pageVal - refVal) < 1e-9;
      crossCheck.push(`    ${method}: page=${pageVal} ref=${refVal} -> ${same ? "MATCH" : "DIFF"}`);
    }
    crossCheck.push("(b) Top-10 PPR users 1 and 2 (page recommendTopK vs ref)");
    for (const n of [1, 2]) {
      const pageList = (pprRecs["user" + n] && pprRecs["user" + n].recs) || [];
      const refList = (refJson.top10 && refJson.top10["user" + n] && refJson.top10["user" + n].methods.ppr) || [];
      let idsSame = pageList.length === refList.length;
      let maxScoreDiff = 0;
      for (let i = 0; i < Math.max(pageList.length, refList.length); i++) {
        if (!pageList[i] || !refList[i] || pageList[i].movieId !== refList[i].movieId) idsSame = false;
        if (pageList[i] && refList[i]) maxScoreDiff = Math.max(maxScoreDiff, Math.abs(pageList[i].score - refList[i].score));
      }
      crossCheck.push(`    user ${(refJson.top10["user" + n] || {}).userId}: ids ${idsSame ? "MATCH" : "DIFF"}; max |score diff| = ${maxScoreDiff.toExponential(3)}`);
    }
    crossCheck.push("(c) Raw adapter PPR hw6RunPPR(getGraph(), 1, 0.85)");
    crossCheck.push(`    adapter restart node = ${rawPpr.adapterRestartNode} (user 1's real node = ${rawPpr.userNodeOfUser1})`);
    crossCheck.push(`    argmax node = ${rawPpr.argmaxNode}, isUserNode=${rawPpr.argmaxIsUserNode}, argmaxMovieId=${rawPpr.argmaxMovieId}, score=${rawPpr.argmaxScore}`);
    const refPpr = refJson.pprUser1Node085 || {};
    crossCheck.push(`    ref PPR(user 1 node ${refPpr.node}) iterations=${refPpr.iterations} residual=${refPpr.residual} sum=${refPpr.sum}`);
    crossCheck.push(`    page raw sum=${rawPpr.sum} iterations=${rawPpr.iterations} residual=${rawPpr.residual}`);
    crossCheck.push(`    seeds same node? ${rawPpr.adapterRestartNode === refPpr.node ? "YES" : "NO (" + rawPpr.adapterRestartNode + " != " + refPpr.node + ")"}`);
  } else {
    crossCheck.push("results/reference.json not found; run `node week6-checks/reference.js` first.");
  }

  const report = [
    `Chrome: ${version}`,
    `TF backend: ${tfBackend}`,
    "",
    "=== Dataset summary ===",
    datasetSummary,
    "",
    "=== Memory ===",
    `tf.memory().numTensors before two pageRank(0.85): ${memBefore}`,
    `tf.memory().numTensors after two pageRank(0.85): ${memAfter}`,
    "",
    "=== Runs ===",
    `pageRank(0.85) #1: ${JSON.stringify(pr1)}`,
    `pageRank(0.85) #2: ${JSON.stringify(pr2)}`,
    `pageRank(0.95): ${JSON.stringify(pr95)}`,
    `personalizedPageRank(0,0.85): ${JSON.stringify(ppr0)}`,
    "",
    "=== Self-checks (#self-checks) ===",
    selfChecks,
    "",
    "=== Alpha sweep (validation Recall@10, PPR) ===",
    `time_ms: ${sweep.ms === undefined ? "-" : sweep.ms.toFixed(1)}`,
    `best_alpha: ${sweep.best}`,
    `note: ${sweep.note || ""}`,
    ...(sweep.rows || []).map((row) => `alpha ${row.alpha}: recall=${row.recall}${row.error ? " error=" + row.error : ""}`),
    sweep.err ? `error: ${sweep.err}` : "",
    "",
    "=== Slider set to selected alpha ===",
    JSON.stringify(sliderSet),
    "",
    "=== Comparison (test split, 4 methods) ===",
    `time_ms: ${comparison.ms === undefined ? "-" : comparison.ms.toFixed(1)}`,
    ...(comparison.rows || []).map((row) => JSON.stringify(row)),
    comparison.err ? `error: ${comparison.err}` : "",
    "",
    "=== recommendTopK(\"ppr\", u, 10, \"test\") ===",
    JSON.stringify(pprRecs, null, 2),
    "",
    "=== Raw adapter PPR (hw6RunPPR(getGraph(), 1, 0.85)) ===",
    JSON.stringify(rawPpr, null, 2),
    "",
    "=== Cross-check: page vs independent reference ===",
    crossCheck.join("\n"),
    "",
    "=== MF_INFO ===",
    JSON.stringify(mfInfo, null, 2),
    "",
    "=== Submission export (week6/submission/) ===",
    submission.err ? `error: ${submission.err}` :
      `demo users = ${JSON.stringify(submission.demoUsers)}; Top-10 users = ${submission.user1}, ${submission.user2}; methods = ${JSON.stringify(submission.methods)}\n` +
      `verification.json = ${submission.verification.length} bytes\n` +
      `results.csv = ${submission.results.length} bytes\n` +
      `recommendations.csv = ${submission.recommendations.length} bytes`,
    "",
    "=== u.item encoding (data.js Response.text() = UTF-8) ===",
    encodingCheck.err ? `error: ${encodingCheck.err}` :
      `movies: ${encodingCheck.total}; titles with U+FFFD: ${encodingCheck.corrupted}; titles with any non-ASCII char: ${encodingCheck.nonAscii}\n` +
      (encodingCheck.list || []).map((b) => `  id ${b.id}: ${b.title}`).join("\n"),
    corruptedInSubmission.err ? `submission check error: ${corruptedInSubmission.err}` :
      `corrupted titles present in recommendations.csv: ${corruptedInSubmission.count} (of ${corruptedInSubmission.movieCount} rows)` +
      (corruptedInSubmission.hits.length ? "\n" + corruptedInSubmission.hits.map((h) => `  movieId ${h.movieId}: ${h.title}`).join("\n") : ""),
    "",
    "=== PRIMARY FLOW: only \"Evaluate Recall@10\" clicked after Build graph ===",
    evaluateClick && evaluateClick.err ? `error: ${evaluateClick.err}` :
      `selected alpha on slider: ${evaluateClick.sliderValue} (readout ${evaluateClick.sliderReadout})\n` +
      `validation sweep (inside Evaluate): ${(evaluateClick.sweep || []).map((r) => r.alpha + "=" + r.recall).join(", ")}\n` +
      `status: ${evaluateClick.status}\n` +
      `test comparison rows:\n` + (evaluateClick.rows || []).map((r) => "  " + JSON.stringify(r)).join("\n"),
    "",
    "=== Timings (ms) ===",
    `page load -> dataset/split ready: ${pageLoadToReadyMs === null ? "-" : pageLoadToReadyMs}`,
    `Build graph: ${buildGraphMs === null ? "-" : buildGraphMs.toFixed(1)}`,
    `Run tests: ${runTestsMs === null ? "-" : runTestsMs.toFixed(1)}`,
    `Evaluate click (sweep -> slider -> test -> recall): ${evaluateClick && evaluateClick.ms !== undefined ? evaluateClick.ms.toFixed(1) : "-"}`,
    `repeat Evaluate click: ${repeatSweep.ms === undefined ? "-" : repeatSweep.ms.toFixed(1)}`,
    "",
    "=== Tensor counts (tf.memory().numTensors) ===",
    `after two pageRank(0.85): ${memAfter}`,
    `after Run tests: ${memAfterRunTests}`,
    `after second Run tests: ${memAfterRunTests2}`,
    `after Evaluate click (MF loaded): ${memAfterEvaluate}`,
    `after repeat Evaluate click: ${memAfterRepeat}`,
    "",
    "=== MF model after Evaluate ===",
    JSON.stringify(mfLoad),
    "",
    "=== Convergence by alpha (pageRank and PPR user 1) ===",
    ...(convergenceByAlpha.pageRank || []).map((r) => `pageRank alpha ${r.alpha}: ${r.error ? "error=" + r.error : "iterations=" + r.iterations + " residual=" + r.residual + " converged=" + r.converged}`),
    ...(convergenceByAlpha.pprUser1 || []).map((r) => `ppr(user 1) alpha ${r.alpha}: ${r.error ? "error=" + r.error : "iterations=" + r.iterations + " residual=" + r.residual + " converged=" + r.converged}`),
    "",
    "=== tf.* API audit (source references vs loaded runtime) ===",
    tfAudit && tfAudit.err ? `error: ${tfAudit.err}` :
      Object.entries(tfAudit ? tfAudit.perFile : {}).map(([file, names]) => {
        const missing = names.filter((n) => !tfAudit.existence[n]);
        return `${file}: ${names.length} distinct tf.* members; missing in runtime: ${missing.length ? missing.join(", ") : "none"}\n    ${names.join(", ")}`;
      }).join("\n"),
    "",
    "=== Console/logs (last 80) ===",
    logs.slice(-80).join("\n"),
  ].join("\n");

  fs.writeFileSync(outPath, report);
  console.log(report);

  await browser.close();
  server.close();
}

main().catch((e) => {
  console.error("probe error", e);
  process.exit(1);
});
