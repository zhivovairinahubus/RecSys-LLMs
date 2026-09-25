// check-data.js — runs week2/data.js against the real u.item / u.data files in Node.
// Usage: node week2/check-data.js
// No external dependencies. fetch() is stubbed so it reads the local files; the
// stub returns the raw bytes for arrayBuffer() and a latin-1 string for text(),
// mirroring what a browser would hand to the real loadData().

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

    const copies = movies.filter(m => m.duplicateOf !== undefined);
    const copyIds = copies.map(m => m.id);
    const copyIdSet = new Set(copyIds);
    const ratingsOnCopies = ratings.filter(r => copyIdSet.has(r.itemId)).length;
    const emptyGenre = distinctMovies.filter(m => m.genreVector.every(v => v === 0));

    console.log('movies in movies[]             :', movies.length);
    console.log('movies for dropdown (distinct) :', distinctMovies.length);
    console.log('id-copies (duplicateOf set)    :', copyIds.length, JSON.stringify(copyIds));
    console.log('ratings on id-copies (u.data)  :', ratingsOnCopies);
    console.log('invalid records                :', invalidRecords.length, JSON.stringify(invalidRecords));
    console.log('movies with empty genre vector :');
    if (emptyGenre.length === 0) console.log('   (none)');
    for (const m of emptyGenre) {
        console.log('  ', m.id, JSON.stringify(m.title), 'unknownGenre=' + m.unknownGenre);
    }

    console.log('');
    console.log('copies mapping (copyId -> firstId, title):');
    for (const m of copies) console.log('  ', m.id + ' -> ' + m.duplicateOf + '  ' + JSON.stringify(m.title));
    console.log('  (also in distinctMovies: ' + (distinctMovies.every(m => m.duplicateOf === undefined)) + ')');

    console.log('');
    console.log('genre spot checks:');
    for (const id of [1, 203]) {
        const m = movies.find(x => x.id === id);
        console.log('id ' + id + ' ' + m.title + ':');
        console.log('  genres:', JSON.stringify(m.genres));
        console.log('  vector:', m.genreVector.join(''));
    }

    console.log('');
    console.log('displayTitle spot checks:');
    const byId = id => movies.find(m => m.id === id);
    const byTitle = t => movies.find(m => m.title === t);
    const cases = [
        [byId(1), 'Toy Story (1995)', 'id 1 displayTitle'],
        [byId(211), 'M*A*S*H (1970)', 'id 211 displayTitle (must stay)'],
        [byId(1609), 'B.A.P.S (1997)', 'id 1609 displayTitle (manual override)'],
        [byId(543), 'Les Misérables (1995)', 'id 543 displayTitle'],
        [byTitle('Paris, Texas (1984)'), 'Paris, Texas (1984)', 'Paris, Texas stays'],
        [byTitle("Enfer, L' (1994)"), "L'Enfer (1994)", "Enfer, L' -> L'Enfer"],
        [byTitle('Good, The Bad and The Ugly, The (1966)'), 'The Good, The Bad and The Ugly (1966)', 'Good/Bad/Ugly The -> front'],
    ];
    for (const [m, expected, label] of cases) {
        if (!m) { console.log('NOT ' + label + ': movie not found'); continue; }
        console.log(m.id + ' | ' + m.title + ' -> ' + m.displayTitle);
        console.log((m.displayTitle === expected ? 'OK  ' : 'NOT ') + label);
    }

    console.log('');
    const uniqKeys = new Set(movies.map(m => m.titleKey)).size;
    const withCopies = distinctMovies.some(m => m.duplicateOf !== undefined) ? 'yes (!)' : 'no';
    console.log('unique titleKeys:', uniqKeys);
    console.log('distinctMovies contains any duplicateOf:', withCopies);
})();
`;

eval(stubs + dataJs + checks);