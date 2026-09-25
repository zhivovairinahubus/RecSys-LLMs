// analysis-discovery.js — how well each approach opens the catalog's "long tail".
// Runs data.js and cbf-core.js in Node against the real u.item / u.data (fetch()
// stubbed like the check-* scripts). For every user with at least three distinct
// 4-5 rated movies (non-zero genre vector, id-copies replaced by the first copy),
// builds two Top-5 exactly like the app does — item-to-item (last film only) and
// profile-based (all three) — then aggregates discovery metrics, also under a
// "filter" mode that removes everything the user has rated from the candidates.
// Usage: node week2/analysis-discovery.js
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

    const movieById = new Map(movies.map(m => [m.id, m]));
    const idToTitleKey = new Map(movies.map(m => [m.id, m.titleKey]));
    const ratingCount = {};
    for (const r of ratings) ratingCount[r.itemId] = (ratingCount[r.itemId] || 0) + 1;

    // id-copy -> id of its first instance (normalized id)
    const normId = itemId => {
        const m = movieById.get(itemId);
        return m && m.duplicateOf !== undefined ? m.duplicateOf : itemId;
    };

    const CATALOG_MEDIAN_RATINGS = 27; // median rating count across the catalog

    // Group ratings per user, keep original order index for stable timestamp ties
    const userRatings = new Map();
    ratings.forEach((r, idx) => {
        if (!userRatings.has(r.userId)) userRatings.set(r.userId, []);
        userRatings.get(r.userId).push({ time: r.timestamp, idx, itemId: r.itemId, rating: r.rating });
    });

    // Three last DISTINCT 4-5 rated movies (normalized ids, non-zero genre vector)
    const analyzed = [];
    let skipped = 0;
    for (const [userId, list] of userRatings) {
        list.sort((a, b) => (a.time - b.time) || (a.idx - b.idx));
        const historyIds = new Set(list.map(r => normId(r.itemId)));
        const three = [];
        const seen = new Set();
        for (let i = list.length - 1; i >= 0 && three.length < 3; i--) {
            const r = list[i];
            if (r.rating < 4) continue;
            const normalized = normId(r.itemId);
            if (seen.has(normalized)) continue;
            const movie = movieById.get(normalized);
            if (!movie) continue; // rating refers to an invalid row (e.g. id 267)
            if (!movie.genreVector.some(v => v !== 0)) continue;
            seen.add(normalized);
            three.unshift(movie); // chronological order, three[2] is the latest
        }
        if (three.length < 3) {
            skipped++;
            continue;
        }
        analyzed.push({ userId, three, historyIds, list });
    }

    const usersTotal = userRatings.size;
    const historyKeys = user => new Set(user.list.map(r => {
        const normalized = normId(r.itemId);
        return idToTitleKey.has(normalized) ? idToTitleKey.get(normalized) : null;
    }).filter(k => k !== null));

    function median(sorted) {
        const n = sorted.length;
        if (n === 0) return 0;
        const mid = Math.floor(n / 2);
        return n % 2 === 1 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
    }

    // Aggregator for one (approach x filter) configuration.
    // Weights: leak / long-tail / diversity are averaged per user (equal weight per
    // user); rating counts and slot shares are pooled over all slots.
    function aggregate(seedFn, excludeFn) {
        const acc = {
            slots: 0,
            leakPerUser: [],
            ratingValues: [],
            tailPerUser: [],
            divPerUser: [],
            divUsers: 0,
            union: new Set(),
            freq: new Map()
        };
        for (const user of analyzed) {
            const { top } = core.rankCandidates(
                seedFn(user), movies, excludeFn(user), 5, core.cosine
            );
            let leaks = 0, tail = 0;
            for (const m of top) {
                acc.slots++;
                if (user.historyIds.has(m.id)) leaks++;
                acc.ratingValues.push(ratingCount[m.id] || 0);
                if ((ratingCount[m.id] || 0) <= CATALOG_MEDIAN_RATINGS) tail++;
                acc.union.add(m.id);
                acc.freq.set(m.id, (acc.freq.get(m.id) || 0) + 1);
            }
            acc.leakPerUser.push(top.length ? leaks / top.length : 0);
            acc.tailPerUser.push(top.length ? tail / top.length : 0);
            if (top.length >= 2) {
                let sumPairs = 0;
                for (let i = 0; i < top.length; i++) {
                    for (let j = i + 1; j < top.length; j++) {
                        sumPairs += 1 - core.cosine(top[i].genreVector, top[j].genreVector);
                    }
                }
                const pairs = top.length * (top.length - 1) / 2;
                acc.divPerUser.push(sumPairs / pairs);
                acc.divUsers++;
            }
        }
        acc.ratingValues.sort((a, b) => a - b);
        acc.leakPerUser.sort((a, b) => a - b);
        acc.tailPerUser.sort((a, b) => a - b);
        acc.divPerUser.sort((a, b) => a - b);
        const top10 = Array.from(acc.freq.values()).sort((a, b) => b - a).slice(0, 10);
        const mean = arr => arr.reduce((s, v) => s + v, 0) / arr.length;
        return {
            slots: acc.slots,
            leak: mean(acc.leakPerUser) * 100,
            meanRatings: mean(acc.ratingValues),
            medianRatings: median(acc.ratingValues),
            tailPct: mean(acc.tailPerUser) * 100,
            diversity: acc.divUsers ? mean(acc.divPerUser) : 0,
            diversityUsers: acc.divUsers,
            coverage: acc.union.size,
            top10Share: acc.slots ? (top10.reduce((s, v) => s + v, 0) / acc.slots) * 100 : 0
        };
    }

    const itemSeed = u => u.three[u.three.length - 1].genreVector; // latest film
    const profileSeed = u => core.buildProfile(u.three.map(m => m.genreVector));
    const itemExclude = u => new Set([u.three[u.three.length - 1].titleKey]);
    const profileExclude = u => new Set(u.three.map(m => m.titleKey));
    const filterExclude = u => historyKeys(u);

    const cfg = [
        { label: 'item-to-item (last film)', seed: itemSeed, exclude: itemExclude },
        { label: 'profile (three films)', seed: profileSeed, exclude: profileExclude }
    ];

    console.log('=== discovery analysis ===');
    console.log('users in u.data       : ' + usersTotal);
    console.log('skipped (fewer than three distinct 4-5 rated, non-zero vector films): ' + skipped);
    console.log('analyzed              : ' + analyzed.length);
    console.log('long-tail threshold (catalog median ratings) = ' + CATALOG_MEDIAN_RATINGS);
    console.log('metrics: leak/long-tail/diversity are averaged per user; mean/median ratings pooled;');
    console.log('         coverage & top-10 share are pooled over all slots.');
    console.log('');

    for (const c of cfg) {
        const base = aggregate(c.seed, c.exclude);
        const filt = aggregate(c.seed, u => {
            const s = c.exclude(u);
            for (const k of filterExclude(u)) s.add(k);
            return s;
        });
        console.log('=== ' + c.label + ' ===');
        console.log(padStr('metric', 34) + padStr('baseline', 14) + padStr('with filter', 14));
        console.log(padStr('1) leak (watched earlier)', 34) +
                    padStr(base.leak.toFixed(2) + '%', 14) +
                    padStr(filt.leak.toFixed(2) + '%', 14));
        console.log(padStr('2) mean ratings of recs', 34) +
                    padStr(base.meanRatings.toFixed(1), 14) +
                    padStr(filt.meanRatings.toFixed(1), 14));
        console.log(padStr('2) median ratings of recs', 34) +
                    padStr(base.medianRatings.toFixed(1), 14) +
                    padStr(filt.medianRatings.toFixed(1), 14));
        console.log(padStr('3) long-tail share (<=27)', 34) +
                    padStr(base.tailPct.toFixed(2) + '%', 14) +
                    padStr(filt.tailPct.toFixed(2) + '%', 14));
        console.log(padStr('4) intra-list diversity (1-cos)', 34) +
                    padStr(base.diversity.toFixed(4), 14) +
                    padStr(filt.diversity.toFixed(4), 14));
        console.log(padStr('5) distinct movies covered', 34) +
                    padStr(String(base.coverage), 14) + padStr(String(filt.coverage), 14));
        console.log(padStr('5) top-10 films slot share', 34) +
                    padStr(base.top10Share.toFixed(2) + '%', 14) +
                    padStr(filt.top10Share.toFixed(2) + '%', 14));
        console.log(padStr('   total slots', 34) +
                    padStr(String(base.slots), 14) + padStr(String(filt.slots), 14));
        console.log('');
    }

    function padStr(s, n) { return (s + ' '.repeat(n)).slice(0, n); }
})();
`;

eval(stubs + dataJs + checks);