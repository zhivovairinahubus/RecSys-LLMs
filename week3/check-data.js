#!/usr/bin/env node
'use strict';

const fs = require('fs');
const vm = require('vm');
const path = require('path');

const dataDir = __dirname;
const dataJsPath = path.join(dataDir, 'data.js');
const scriptJsPath = path.join(dataDir, 'script.js');

if (!fs.existsSync(dataJsPath) || !fs.existsSync(scriptJsPath)) {
    console.error('Не найден data.js или script.js в ' + dataDir);
    process.exit(1);
}

const areas = new Map();

function makeFetchStub() {
    return async function (url) {
        const file = path.join(dataDir, url);
        if (!fs.existsSync(file)) {
            return { ok: false, status: 404, text: async () => '', arrayBuffer: async () => new ArrayBuffer(0) };
        }
        const buf = fs.readFileSync(file);
        return {
            ok: true,
            status: 200,
            text: async () => buf.toString('utf8'),
            arrayBuffer: async () => buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength),
        };
    };
}

const documentStub = {
    getElementById(id) {
        if (!areas.has(id)) areas.set(id, { id: id, innerHTML: '' });
        return areas.get(id);
    },
};

const ctx = vm.createContext({
    console: console,
    TextDecoder: TextDecoder,
    fetch: makeFetchStub(),
    document: documentStub,
    window: {},
    setTimeout: setTimeout,
});

vm.runInContext(fs.readFileSync(dataJsPath, 'utf8'), ctx, { filename: 'week3/data.js' });
// script.js is loaded for cosineAndCommonUsers and displayTitleOf only; its
// window.onload never runs here, and the DOM stub absorbs any element lookup.
vm.runInContext(fs.readFileSync(scriptJsPath, 'utf8'), ctx, { filename: 'week3/script.js' });

async function main() {
    const raw = await vm.runInContext(`(async () => {
        await loadData();

        const byId = {};
        for (const m of movies) byId[m.id] = {
            title: m.title, displayTitle: m.displayTitle,
            genres: m.genres, isInvalid: m.isInvalid, duplicateOf: m.duplicateOf
        };

        let nonZero = 0;
        const perUser = new Array(numUsers + 1).fill(0);
        const perMovie = new Array(numMovies + 1).fill(0);
        for (let u = 1; u <= numUsers; u++) {
            const row = ratingMatrix[u];
            for (let m = 1; m <= numMovies; m++) {
                if (row[m] !== 0) { nonZero++; perUser[u]++; perMovie[m]++; }
            }
        }

        let row0Sum = 0, col0Sum = 0, row0NonZero = 0, col0NonZero = 0;
        for (let m = 0; m <= numMovies; m++) {
            if (ratingMatrix[0][m] !== 0) row0NonZero++;
            row0Sum += ratingMatrix[0][m];
        }
        for (let u = 0; u <= numUsers; u++) {
            if (ratingMatrix[u][0] !== 0) col0NonZero++;
            col0Sum += ratingMatrix[u][0];
        }

        let minPerUser = Infinity, maxPerUser = -Infinity;
        for (let u = 1; u <= numUsers; u++) {
            if (perUser[u] < minPerUser) minPerUser = perUser[u];
            if (perUser[u] > maxPerUser) maxPerUser = perUser[u];
        }

        // Per-FILM statistics count only real films: a duplicate copy shares a
        // column-less twin and an invalid row is not a film at all, so both
        // would drag the minimum down to zero for a reason that is not about
        // the data. isRealFilm() is the single definition the page uses too.
        let minPerMovie = Infinity, maxPerMovie = -Infinity;
        let realFilmCount = 0, copyCount = 0, invalidCount = 0;
        for (let m = 1; m <= numMovies; m++) {
            if (invalidMovieId[m]) { invalidCount++; continue; }
            if (canonicalMovieId[m] !== m) { copyCount++; continue; }
            realFilmCount++;
            if (perMovie[m] < minPerMovie) minPerMovie = perMovie[m];
            if (perMovie[m] > maxPerMovie) maxPerMovie = perMovie[m];
        }

        return JSON.stringify({
            movieCount: movies.length,
            ratingCount: ratings.length,
            numUsers: numUsers,
            numMovies: numMovies,
            rows: ratingMatrix.length,
            cols: ratingMatrix[0].length,
            nonZero: nonZero,
            row0Sum: row0Sum, row0NonZero: row0NonZero,
            col0Sum: col0Sum, col0NonZero: col0NonZero,
            minPerUser: minPerUser, maxPerUser: maxPerUser,
            minPerMovie: minPerMovie, maxPerMovie: maxPerMovie,
            realFilmCount: realFilmCount, copyCount: copyCount, invalidCount: invalidCount,
            skippedInvalidRatings: lastSkippedInvalidRatings,
            c196_242: ratingMatrix[196][242],
            c1_1: ratingMatrix[1][1],
            c7_599: ratingMatrix[7][599],
            c19_267: ratingMatrix[19][267],
            invalid267NonZero: (function(){ let n=0; for (let u=1;u<=numUsers;u++) if (ratingMatrix[u][267]!==0) n++; return n; })(),
            byId: byId
        });
    })()`, ctx);

    const r = JSON.parse(raw);
    const j = JSON.stringify;

    console.log('=== 1. фильмы ===');
    console.log('  число фильмов (movies.length): ' + r.movieCount);
    console.log('  число оценок  (ratings.length): ' + r.ratingCount);

    console.log('\n=== 2. названия с не-ASCII байтами (проверка кодировки) ===');
    console.log('  id=543  : ' + j(r.byId['543'].title));
    console.log('  id=1633 : ' + j(r.byId['1633'].title));
    const fffd = s => [...s].filter(c => c.charCodeAt(0) === 0xfffd).length;
    console.log('  символов U+FFFD в id=543:  ' + fffd(r.byId['543'].title));
    console.log('  символов U+FFFD в id=1633: ' + fffd(r.byId['1633'].title));

    console.log('\n=== 3. жанры ===');
    for (const id of [1, 2, 203, 267]) {
        const m = r.byId[String(id)];
        console.log('  id=' + String(id).padEnd(5) + j(m.title).padEnd(24) + 'genres=' + j(m.genres));
    }

    console.log('\n=== 4. размеры ===');
    console.log('  numUsers:  ' + r.numUsers);
    console.log('  numMovies: ' + r.numMovies);
    console.log('  ratingMatrix.length (строк, пользователи):  ' + r.rows);
    console.log('  ratingMatrix[0].length (столбцов, фильмы):   ' + r.cols);
    console.log('  ожидаемое (numUsers+1) x (numMovies+1):      ' +
        (r.numUsers + 1) + ' x ' + (r.numMovies + 1));

    console.log('\n=== 5. заполненность ===');
    console.log('  ненулевых ячеек:      ' + r.nonZero);
    console.log('  всего ячеек:          ' + r.rows * r.cols);
    console.log('  доля:                ' + (r.nonZero / (r.rows * r.cols) * 100).toFixed(4) + '%');

    console.log('\n=== 6. нулевая строка и нулевой столбец ===');
    console.log('  сумма строки 0 (ratingMatrix[0]):        ' + r.row0Sum +
        '   ненулевых элементов: ' + r.row0NonZero);
    console.log('  сумма столбца 0 (ratingMatrix[u][0]):    ' + r.col0Sum +
        '   ненулевых элементов: ' + r.col0NonZero);

    console.log('\n=== 7. точечные значения ===');
    console.log('  ratingMatrix[196][242] = ' + r.c196_242);
    console.log('  ratingMatrix[1][1]     = ' + r.c1_1);
    console.log('  ratingMatrix[7][599]   = ' + r.c7_599);

    console.log('\n=== 8. оценок на пользователя / на фильм ===');
    console.log('  пользователь:  min=' + r.minPerUser + '  max=' + r.maxPerUser);
    console.log('  фильм:         min=' + r.minPerMovie + '  max=' + r.maxPerMovie +
        '   (только реальные фильмы: ' + r.realFilmCount + ')');
    console.log('  исключено из статистики по фильмам: ' + r.copyCount + ' копий, ' +
        r.invalidCount + ' невалидных строк');

    console.log('\n=== 8a. невалидная строка u.item ===');
    console.log('  невалидных строк:        ' + r.invalidCount + '  -> id 267, title "unknown"');
    console.log('  оценок в u.data на них:  ' + r.skippedInvalidRatings + '  (пропущено при построении матрицы)');
    console.log('  ненулевых ячеек ratingMatrix[u][267]: ' + r.invalid267NonZero + '  (ожидаем 0)');
    console.log('  ratingMatrix[19][267]:   ' + r.c19_267 + '  (ожидаем 0)');

    console.log('\n=== 9. что записалось в DOM (fetch не падал, значит пусто) ===');
    for (const [id, el] of areas) {
        console.log('  #' + id + ': ' + j(el.innerHTML));
    }

    // -----------------------------------------------------------------------
    // Ручной расчёт сходства для двух пар пользователей. Всё берётся из
    // ratingMatrix, то есть из тех же данных, что использует script.js, и
    // сходство считается ровно той же функцией cosineAndCommonUsers.
    // -----------------------------------------------------------------------
    const pairsRaw = JSON.parse(await vm.runInContext(`(() => {
        function dump(userId) {
            let count = 0, sum = 0, sumSquares = 0;
            const rated = {};
            for (let m = 1; m <= numMovies; m++) {
                const rating = ratingMatrix[userId][m];
                if (rating === 0) continue;
                rated[m] = rating;
                count++; sum += rating; sumSquares += rating * rating;
            }
            return { userId: userId, count: count, sum: sum, sumSquares: sumSquares, rated: rated };
        }

        function common(a, b) {
            const rows = [];
            let dot = 0, normA = 0, normB = 0, n = 0;
            for (let m = 1; m <= numMovies; m++) {
                const ra = ratingMatrix[a][m], rb = ratingMatrix[b][m];
                if (ra === 0 || rb === 0) continue;
                rows.push({
                    id: m,
                    displayTitle: displayTitleOf(m),
                    ratingA: ra, ratingB: rb,
                    product: ra * rb
                });
                dot += ra * rb; normA += ra * ra; normB += rb * rb; n++;
            }
            const rawCosine = (normA === 0 || normB === 0) ? 0 : dot / Math.sqrt(normA * normB);
            const pair = cosineAndCommonUsers(a, b);
            return {
                a: a, b: b, rows: rows,
                common: n, dot: dot, normA: normA, normB: normB,
                rawCosine: rawCosine,
                similarity: pair.similarity, commonFromFn: pair.common
            };
        }

        return JSON.stringify({
            users: [dump(19), dump(37), dump(13)],
            pairs: [common(19, 37), common(19, 13)]
        });
    })()`, ctx));

    console.log('\n=== 10. ручной расчёт: сходство пар пользователей ===');
    console.log('  сходство считает cosineAndCommonUsers из script.js: сырой косинус');
    console.log('  по совместным оценкам, умноженный на min(common, ' +
        'CO_RATED_CAP) / CO_RATED_CAP.');

    console.log('\n  --- сводка по пользователям ---');
    for (const u of pairsRaw.users) {
        console.log('    user ' + String(u.userId).padEnd(4) +
            'оценок: ' + String(u.count).padStart(4) +
            '   сумма: ' + String(u.sum).padStart(6) +
            '   сумма квадратов: ' + String(u.sumSquares).padStart(6));
    }

    for (const p of pairsRaw.pairs) {
        console.log('\n  --- пара ' + p.a + ' и ' + p.b + ' ---');
        console.log('    общих фильмов: ' + p.common + '  (cosineAndCommonUsers.common = ' +
            p.commonFromFn + ')');
        console.log('    id    название                              оценка ' + p.a + '   оценка ' + p.b + '   произведение');
        for (const row of p.rows) {
            console.log('    ' + String(row.id).padEnd(6) +
                (row.displayTitle.length > 36 ? row.displayTitle.slice(0, 35) + '…' : row.displayTitle).padEnd(38) +
                String(row.ratingA).padStart(6) + String(row.ratingB).padStart(13) +
                String(row.product).padStart(15));
        }
        console.log('    сумма произведений (dot):      ' + p.dot);
        console.log('    сумма квадратов пользователя ' + p.a + ':  ' + p.normA);
        console.log('    сумма квадратов пользователя ' + p.b + ':  ' + p.normB);
        console.log('    сырой косинус dot/sqrt(normA*normB): ' + p.rawCosine.toFixed(6));
        console.log('    min(common, 50) / 50:                  ' +
            (Math.min(p.common, 50) / 50).toFixed(6));
        console.log('    сходство пары (script.js):              ' + p.similarity.toFixed(6));
    }
}

main().catch(e => { console.error('FATAL:', e); process.exit(1); });
