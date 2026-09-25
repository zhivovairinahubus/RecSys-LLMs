// check-profile.js — runs data.js + cbf-core.js in Node against the real u.item /
// u.data files, mirroring the browser loading order (fetch() stubbed to read the
// local files). Verifies profile-building: two hand-picked profiles (A and B),
// their Top-5 by profile, and the item-to-item Top-5 for the same films.
// Usage: node week2/check-profile.js
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

    const pad = (s, n) => (s + ' '.repeat(n)).slice(0, n);
    const genreCol = g => '[' + g.join(', ') + ']';
    const header = '  ' + pad('displayTitle', 52) + ' ' + pad('genres', 34) + ' cosine  ratings';

    const findMovie = t => distinctMovies.find(m => m.title === t) || movies.find(m => m.title === t);
    const findMovies = ts => ts.map(findMovie);

    function printTop5(label, queryVector, excludedTitleKeys, watched) {
        const res = core.rankCandidates(queryVector, movies, excludedTitleKeys, 5, core.cosine);
        console.log('  ' + label + ':');
        console.log(header);
        res.top.forEach((m, i) => {
            console.log(
                '  ' + pad((i + 1) + '. ' + m.displayTitle + '  [id ' + m.id + ']', 52) + ' ' +
                pad(genreCol(m.genres), 34) + ' ' +
                (m.score === null ? '   n/a' : m.score.toFixed(3)) + '   ' +
                String(ratingCount[m.id] || 0)
            );
        });

        // No watched film may appear in its own Top-5 (dedup by id and by titleKey)
        const hitById = res.top.filter(x => watched.some(w => w.id === x.id)).length;
        const hitByKey = res.top.filter(x => watched.some(w => w.titleKey === x.titleKey)).length;
        console.log('  checks: watched in Top-5 by id: ' + hitById + ' (expect 0),  by titleKey: ' +
                    hitByKey + ' (expect 0)');

        const kth = res.top.length > 0 ? res.top[res.top.length - 1].score : null;
        const inTop = kth === null ? 0 : res.top.filter(m => m.score === kth).length;
        const extra = Math.max(0, res.tiesAtK - inTop);
        console.log('  ties beyond 5th place: ' + extra +
                    '   [total candidates tied at the kth score: ' + res.tiesAtK + ']');
    }

    const profiles = [
        {
            name: 'A',
            films: ['Star Wars (1977)', 'Empire Strikes Back, The (1980)', 'Return of the Jedi (1983)']
        },
        {
            name: 'B',
            films: ['Scream (1996)', 'Sleepless in Seattle (1993)', 'Fargo (1996)']
        }
    ];

    const profileResults = {};
    for (const p of profiles) {
        const watched = findMovies(p.films);
        console.log('');
        console.log('=== Profile ' + p.name + ': ' + watched.map(w => w.displayTitle).join(' | ') + ' ===');

        const profile = core.buildProfile(watched.map(w => w.genreVector));
        const weights = [];
        realGenreNames.forEach((name, i) => {
            if (profile[i] !== 0) weights.push(name + ': ' + profile[i].toFixed(2));
        });
        console.log('  profile vector: ' + (weights.length > 0 ? weights.join(', ') : '(empty)'));
        console.log('  full profile vector: [' + profile.map(v => v.toFixed(4)).join(', ') + ']');

        console.log('  watched films, cosine with profile:');
        for (const w of watched) {
            const c = core.cosine(profile, w.genreVector);
            console.log('    ' + pad(w.displayTitle + '  [id ' + w.id + ']', 52) + (c === null ? 'n/a' : c.toFixed(3)));
        }

        const excluded = new Set(watched.map(w => w.titleKey));
        printTop5('Top-5 by profile', profile, excluded, watched);
        profileResults[p.name] = { watched, profile };
    }

    // Item-to-item Top-5 for each film of the two profiles, for comparison
    for (const p of profiles) {
        console.log('');
        console.log('=== item-to-item Top-5 for profile ' + p.name + ' films ===');
        for (const w of profileResults[p.name].watched) {
            console.log('');
            console.log('  --- ' + w.displayTitle + '  [id ' + w.id + '] ---');
            printTop5('Top-5 by item', w.genreVector, new Set([w.titleKey]), [w]);
        }
    }
})();
`;

eval(stubs + dataJs + checks);