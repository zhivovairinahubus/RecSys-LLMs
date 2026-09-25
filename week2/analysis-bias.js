// analysis-bias.js — analytics over the recommender's candidates. Runs data.js and
// cbf-core.js in Node against the real u.item / u.data (fetch() stubbed like the
// check-* scripts). For every query movie it builds a Top-5 by three similarities
// and aggregates the results; it also tests the "multi-genre movies are
// blockbusters" premise and the exact tie rule at the 5th boundary.
// Usage: node week2/analysis-bias.js
// No external dependencies.

const fs = require('fs');
const path = require('path');

const dataJs = fs.readFileSync(path.join(__dirname, 'data.js'), 'utf8');

const stubs = `
const util = require('util');
const dir = ${JSON.stringify(__dirname)};
global.TextDecoder = util.TextDecoder;
global.document = { getElementById: () => null };
global.fetch = async (name) => {
    const fsr = require('fs');
    const pth = require('path');
    const buf = fsr.readFileSync(pth.join(dir, name));
    return {
        ok: true,
        arrayBuffer: async () => buf,
        text: async () => buf.toString('latin1'),
    };
};
`;

const checks = `
;(async () => {
    await loadData();

    const core = require(${JSON.stringify(path.join(__dirname, 'cbf-core.js'))});

    const ratingCount = {};
    for (const r of ratings) {
        ratingCount[r.itemId] = (ratingCount[r.itemId] || 0) + 1;
    }
    const ratingsOf = m => ratingCount[m.id] || 0;
    const genreCount = m => m.genres.length;

    // Candidate pool = distinct movies with a non-zero genre vector.
    const queries = distinctMovies.filter(m => m.genreVector.some(v => v !== 0));

    function median(sorted) {
        const n = sorted.length;
        if (n === 0) return 0;
        const mid = Math.floor(n / 2);
        return n % 2 === 1 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
    }

    // Average ranks (ties share the averaged rank); Spearman = Pearson on ranks.
    function rankAverage(values) {
        const order = values.map((v, i) => [v, i]).sort((a, b) => a[0] - b[0]);
        const ranks = new Array(values.length);
        let i = 0;
        while (i < order.length) {
            let j = i;
            while (j + 1 < order.length && order[j + 1][0] === order[i][0]) j++;
            const avg = (i + j) / 2 + 1;
            for (let k = i; k <= j; k++) ranks[order[k][1]] = avg;
            i = j + 1;
        }
        return ranks;
    }
    function spearman(xs, ys) {
        const n = xs.length;
        if (n < 2) return 0;
        const rx = rankAverage(xs);
        const ry = rankAverage(ys);
        const mean = (n + 1) / 2;
        let cov = 0, vx = 0, vy = 0;
        for (let i = 0; i < n; i++) {
            cov += (rx[i] - mean) * (ry[i] - mean);
            vx += (rx[i] - mean) * (rx[i] - mean);
            vy += (ry[i] - mean) * (ry[i] - mean);
        }
        return cov / Math.sqrt(vx * vy);
    }

    const pad = (s, n) => (s + ' '.repeat(n)).slice(0, n);

    const byTitle = t => queries.find(m => m.title === t);
    const toy = byTitle('Toy Story (1995)');

    const catalog4plusShare = (queries.filter(m => genreCount(m) >= 4).length / queries.length) * 100;

    console.log('catalog: ' + queries.length + ' movies (distinct, non-zero genre vector);');
    console.log('Good Morning (1971) excluded as a zero-vector query; ' +
                'query loop: ' + queries.length + ' queries x 3 similarities x Top-5');

    // --- 1) bias of the three similarities, aggregated over all queries ---
    const scoreFns = { dot: core.dot, jaccard: core.jaccard, cosine: core.cosine };

    console.log('');
    console.log('=== 1) aggregated Top-5 bias per similarity ===');
    console.log('  per similarity: avg #genres | % slots with 4+ genres | catalog % 4+ | ' +
                'mean ratings | median ratings | distinct movies in any Top-5');

    const results = {};
    for (const [key, fn] of Object.entries(scoreFns)) {
        let sumGenres = 0, slots4plus = 0, sumRatings = 0, slotCount = 0;
        const ratingValues = [];
        const seenInTop = new Set();
        for (const q of queries) {
            const res = core.rankCandidates(q.genreVector, movies, new Set([q.titleKey]), 5, fn);
            for (const m of res.top) {
                slotCount++;
                sumGenres += genreCount(m);
                if (genreCount(m) >= 4) slots4plus++;
                const rc = ratingsOf(m);
                sumRatings += rc;
                ratingValues.push(rc);
                seenInTop.add(m.id);
            }
        }
        ratingValues.sort((a, b) => a - b);
        results[key] = {
            avgGenres: sumGenres / slotCount,
            share4plus: (slots4plus / slotCount) * 100,
            meanRatings: sumRatings / slotCount,
            medianRatings: median(ratingValues),
            distinctInTop: seenInTop.size
        };
        console.log(
            '  ' + pad(key, 8) +
            ' avgGenres=' + pad(results[key].avgGenres.toFixed(2), 6) +
            ' | %4+= ' + pad(results[key].share4plus.toFixed(1) + '%', 7) +
            ' | cat%4+=' + pad(catalog4plusShare.toFixed(1) + '%', 9) +
            ' | meanRat=' + pad(results[key].meanRatings.toFixed(1), 7) +
            ' | medRat=' + pad(results[key].medianRatings.toFixed(1), 8) +
            ' | distinctInTop=' + results[key].distinctInTop
        );
    }

    // --- 2) premise: more genres -> more ratings ("blockbusters") ---
    console.log('');
    console.log('=== 2) ratings vs genre count (' + queries.length + ' catalog movies) ===');
    const buckets = { '1': [], '2': [], '3': [], '4+': [] };
    const gx = [], rx = [];
    for (const m of queries) {
        const g = genreCount(m);
        const rc = ratingsOf(m);
        gx.push(g);
        rx.push(rc);
        (g < 4 ? buckets[String(g)] : buckets['4+']).push(rc);
    }
    console.log('  bucket   n      meanRatings  medianRatings');
    for (const [b, arr] of Object.entries(buckets)) {
        arr.sort((a, b2) => a - b2);
        const mean = arr.reduce((s, v) => s + v, 0) / arr.length;
        console.log('  ' + pad(b, 7) + pad(String(arr.length), 6) +
                    '  ' + pad(mean.toFixed(1), 12) + '  ' + median(arr).toFixed(1));
    }
    console.log('  Spearman(#genres, #ratings) = ' + spearman(gx, rx).toFixed(4));
    console.log('  mean ratings across all: ' +
                (rx.reduce((s, v) => s + v, 0) / rx.length).toFixed(1) +
                ',  median: ' + median(rx.slice().sort((a, b) => a - b)).toFixed(1));

    // --- 3) Toy Story (1995): dot Top-5 against cosine Top-5 ---
    console.log('');
    console.log('=== 3) Toy Story (1995) [id ' + toy.id + '] Top-5: dot vs cosine ===');
    const dotRes = core.rankCandidates(toy.genreVector, movies, new Set([toy.titleKey]), 5, core.dot);
    const cosRes = core.rankCandidates(toy.genreVector, movies, new Set([toy.titleKey]), 5, core.cosine);
    const cellText = (m, i) => {
        const label = (i + 1) + '. ' + m.displayTitle;
        return pad(label, 40) + ' g=' + pad(String(genreCount(m)), 1) + ' rt=' + pad(String(ratingsOf(m)), 3);
    };
    console.log('  ' + pad('dot Top-5', 49) + ' | ' + pad('cosine Top-5', 49));
    for (let i = 0; i < 5; i++) {
        console.log('  ' + cellText(dotRes.top[i], i) + ' | ' + cellText(cosRes.top[i], i));
    }
    console.log('  (g = number of genres, rt = number of ratings in u.data)');

    // --- 4) refined tie metric at the 5th boundary (cosine, app rule) ---
    console.log('');
    console.log('=== 4) tie rule at the Top-5 boundary (cosine) ===');
    let oldRule = 0;      // previous: candidates with score == 5th score > 5
    let newRule = 0;      // inside-tie boundary: candidates with score >= 5th score > 5
    const onlyNew = [];
    for (const q of queries) {
        const res = core.rankCandidates(q.genreVector, movies, new Set([q.titleKey]), 5, core.cosine);
        if (res.top.length === 0) continue;
        const kth = res.top[res.top.length - 1].score;
        let above = 0, equal = 0;
        for (const m of res.top) {
            if (m.score === kth) equal++;
        }
        // count equal across ALL eligible candidates (not just top-5)
        equal = res.tiesAtK;
        // candidates strictly above the 5th score: recompute over full ranking
        // (rankCandidates only returns top-5, so count by score over the pool again)
        const pool = movies.filter(m =>
            m.duplicateOf === undefined &&
            !(new Set([q.titleKey])).has(m.titleKey) &&
            m.genreVector.some(v => v !== 0)
        );
        let aboveCount = 0;
        for (const m of pool) {
            const s = core.cosine(q.genreVector, m.genreVector);
            if (s === null) continue;
            if (s > kth) aboveCount++;
        }
        if (equal > 5) oldRule++;
        if (aboveCount + equal > 5) {
            newRule++;
            if (equal <= 5) onlyNew.push(q.displayTitle);
        }
    }
    console.log('  old rule (equal-to-5th-score candidates > 5): ' + oldRule + ' of ' + queries.length);
    console.log('  new rule (candidates with score >= 5th > 5, i.e. boundary inside a tie group): ' +
                newRule + ' of ' + queries.length);
    console.log('  movies counted by the new rule but NOT the old rule: ' + onlyNew.length);
    const shown = onlyNew.slice(0, 5);
    for (const t of shown) console.log('      e.g. ' + t);
})();
`;

eval(stubs + dataJs + checks);