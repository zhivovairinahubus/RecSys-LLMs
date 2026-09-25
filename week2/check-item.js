// check-item.js — runs data.js + cbf-core.js in Node against the real u.item / u.data
// files, mirroring the browser loading order (fetch() stubbed to read the local files).
// Usage: node week2/check-item.js
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

    const byTitle = t => distinctMovies.find(m => m.title === t);
    const pad = (s, n) => (s + ' '.repeat(n)).slice(0, n);
    const genreCol = g => '[' + g.join(', ') + ']';

    const queries = [
        'Toy Story (1995)',
        'Unforgiven (1992)',
        '101 Dalmatians (1996)',
        'Chasing Amy (1997)',
        'Good Morning (1971)'
    ];

    for (const t of queries) {
        const q = byTitle(t);
        if (!q) { console.log('NOT FOUND: ' + t); continue; }
        console.log('');
        console.log('=== ' + q.displayTitle + '  (id=' + q.id + ')  ratings=' + (ratingCount[q.id] || 0) + ' ===');
        console.log('  query vector: ' + q.genreVector.join(''));

        if (q.genreVector.every(v => v === 0)) {
            console.log('  zero genre vector -> app shows the "no genre information" message; no recommendations.');
            continue;
        }

        const res = core.rankCandidates(q.genreVector, movies, new Set([q.titleKey]), 5, core.cosine);

        console.log('  ' + pad('displayTitle', 52) + ' ' + pad('genres', 34) + ' cosine  jaccard   dot  ratings');
        res.top.forEach((m, i) => {
            const j = core.jaccard(q.genreVector, m.genreVector);
            const d = core.dot(q.genreVector, m.genreVector);
            console.log(
                '  ' + pad((i + 1) + '. ' + m.displayTitle + '  [id ' + m.id + ']', 52) + ' ' +
                pad(genreCol(m.genres), 34) + ' ' +
                (m.score === null ? '   n/a' : m.score.toFixed(3)) + '   ' +
                (j === null ? '  n/a' : j.toFixed(3)) + '   ' +
                String(d).padStart(3) + '   ' +
                String(ratingCount[m.id] || 0)
            );
        });

        const selfById = res.top.some(m => m.id === q.id);
        const selfByKey = res.top.some(m => m.titleKey === q.titleKey);
        const kth = res.top.length > 0 ? res.top[res.top.length - 1].score : null;
        const inTop = kth === null ? 0 : res.top.filter(m => m.score === kth).length;
        const extra = Math.max(0, res.tiesAtK - inTop);
        console.log('  checks: no self by id: ' + (selfById ? 'NOT (failed!)' : 'OK') +
                    ',  no self by titleKey: ' + (selfByKey ? 'NOT (failed!)' : 'OK'));
        console.log('  ties at 5th score (extra beyond top-5): ' + extra +
                    '   [total candidates tied at the kth score: ' + res.tiesAtK + ']');
    }

    // Pair: Toy Story (1995) and Aladdin (1992)
    const toy = byTitle('Toy Story (1995)');
    const ala = byTitle('Aladdin (1992)');
    console.log('');
    console.log('=== pair: Toy Story (1995) & Aladdin (1992) ===');
    console.log('  Toy Story vector: ' + toy.genreVector.join('') + '  genres=' + genreCol(toy.genres) +
                '  norm=' + core.norm(toy.genreVector).toFixed(6));
    console.log('  Aladdin   vector: ' + ala.genreVector.join('') + '  genres=' + genreCol(ala.genres) +
                '  norm=' + core.norm(ala.genreVector).toFixed(6));
    console.log('  dot     = ' + core.dot(toy.genreVector, ala.genreVector));
    console.log('  cosine  = ' + core.cosine(toy.genreVector, ala.genreVector).toFixed(6));
    console.log('  jaccard = ' + core.jaccard(toy.genreVector, ala.genreVector).toFixed(6));

    // How many dropdown movies have MORE than 5 candidates tied at the 5th-place
    // score (rule-of-ties instead of rule-of-similarity deciding the Top-5).
    console.log('');
    console.log('=== tie-rule scan: distinctMovies with tiesAtK > 5 ===');
    let tieRule = 0;
    const worst = [];
    for (const m of distinctMovies) {
        if (m.genreVector.every(v => v === 0)) continue; // app blocks zero-vector queries
        const r = core.rankCandidates(m.genreVector, movies, new Set([m.titleKey]), 5, core.cosine);
        const kth = r.top.length > 0 ? r.top[r.top.length - 1].score : null;
        const inTop = kth === null ? 0 : r.top.filter(x => x.score === kth).length;
        const extra = Math.max(0, r.tiesAtK - inTop);
        if (r.tiesAtK > 5) {
            tieRule++;
            worst.push({ title: m.title, desc: m.displayTitle,
                         kth: (kth === null ? 'n/a' : kth.toFixed(3)),
                         extra: extra, total: r.tiesAtK });
        }
    }
    console.log('  dropdown movies with tiesAtK > 5: ' + tieRule + ' of ' +
                (distinctMovies.length - 1) + ' (excluding the one zero-vector movie)');
    worst.sort((a, b) => (b.total - a.total) || a.title.localeCompare(b.title));
    console.log('  largest tiesAtK:');
    for (const w of worst.slice(0, 8)) {
        console.log('    tiesAtK=' + String(w.total).padStart(3) + '  kth score=' + w.kth +
                    '  extra beyond top-5=' + String(w.extra).padStart(3) + '  ' + w.desc);
    }
})();
`;

eval(stubs + dataJs + checks);