#!/usr/bin/env node
'use strict';

const fs = require('fs');
const vm = require('vm');
const path = require('path');

const dataDir = __dirname;

const GROUPS = [
    { label: '1',         test: n => n === 1 },
    { label: '2',         test: n => n === 2 },
    { label: '3-5',       test: n => n >= 3 && n <= 5 },
    { label: '6-10',      test: n => n >= 6 && n <= 10 },
    { label: '11-50',     test: n => n >= 11 && n <= 50 },
    { label: 'более 50',  test: n => n > 50 },
];

function median(values) {
    if (values.length === 0) return null;
    const s = Float64Array.from(values).sort();
    const mid = s.length >> 1;
    return s.length % 2 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
}

// ---------------------------------------------------------------------------
// Данные берём из настоящего data.js — ровно так же, как их получает
// приложение. Поэтому здесь уже учтено слияние дубликатов фильмов: оценки,
// разнесённые по двум id одного фильма, лежат в столбце первого экземпляра.
// ---------------------------------------------------------------------------
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

const areas = new Map();
const ctx = vm.createContext({
    console: console,
    TextDecoder: TextDecoder,
    fetch: makeFetchStub(),
    window: {},
    document: {
        getElementById(id) {
            if (!areas.has(id)) areas.set(id, { id: id, innerHTML: '' });
            return areas.get(id);
        },
    },
});

vm.runInContext(fs.readFileSync(path.join(dataDir, 'data.js'), 'utf8'), ctx, { filename: 'week3/data.js' });

async function main() {
    await vm.runInContext('loadData()', ctx);

    const numUsers = vm.runInContext('numUsers', ctx);
    const numMovies = vm.runInContext('numMovies', ctx);
    const ratingMatrix = vm.runInContext('ratingMatrix', ctx);
    const canonicalMovieId = vm.runInContext('canonicalMovieId', ctx);
    const invalidMovieId = vm.runInContext('invalidMovieId', ctx);

    const userIndex = new Map();
    const itemIndex = new Map();

    // id-копии считаем по canonicalMovieId, а не по вхождениям в матрицу: их
    // столбцы пустые, поэтому при обходе они просто не встретятся.
    let copyCount = 0;
    let invalidCount = 0;
    for (let movieId = 1; movieId <= numMovies; movieId++) {
        if (canonicalMovieId[movieId] !== movieId) { copyCount++; continue; }
        if (invalidMovieId[movieId]) invalidCount++;
    }

    for (let userId = 1; userId <= numUsers; userId++) {
        const row = ratingMatrix[userId];
        const rated = new Map();
        for (let movieId = 1; movieId <= numMovies; movieId++) {
            const rating = row[movieId];
            if (rating === 0) continue;
            rated.set(movieId, rating);

            // Копия фильма не считается отдельным фильмом, а строка "unknown"
            // не считается фильмом вовсе, поэтому в выборку пар фильмов обе
            // не попадают. Проверка та же, что isRealFilm() в data.js.
            if (canonicalMovieId[movieId] !== movieId || invalidMovieId[movieId]) continue;
            let viewers = itemIndex.get(movieId);
            if (!viewers) { viewers = new Map(); itemIndex.set(movieId, viewers); }
            viewers.set(userId, rating);
        }
        userIndex.set(userId, rated);
    }

    console.log('=== ИСТОЧНИК ДАННЫХ ===');
    console.log('  data.js: numUsers=' + numUsers + '  numMovies=' + numMovies);
    console.log('  фильмов всего в u.item:            ' + numMovies);
    console.log('  из них id-копий (duplicateOf):      ' + copyCount +
        '   исключены из пар фильмов');
    console.log('  из них невалидных строк ("unknown"):' + invalidCount +
        '   исключены из пар фильмов');
    console.log('  фильмов в подсчёте пар:            ' + itemIndex.size);
    console.log('  пользователей в подсчёте пар:      ' + userIndex.size);

    analyze('ПОЛЬЗОВАТЕЛИ', userIndex);
    analyze('ФИЛЬМЫ (без id-копий и невалидных)', itemIndex);
}

main().catch(e => { console.error('FATAL:', e); process.exit(1); });

function analyze(title, index) {
    const ids = [...index.keys()].sort((a, b) => a - b);
    const n = ids.length;
    const started = Date.now();

    const common = [];
    let commonMin = Infinity, commonMax = -Infinity;
    let zero = 0, le2 = 0, le5 = 0, ge1 = 0;
    let cosSum = 0, cosMin = Infinity, cosHigh = 0;
    const g = GROUPS.map(() => ({ count: 0, sum: 0, min: Infinity, high: 0 }));

    for (let i = 0; i < n; i++) {
        const a = index.get(ids[i]);
        for (let j = i + 1; j < n; j++) {
            const b = index.get(ids[j]);
            const small = a.size <= b.size ? a : b;
            const large = a.size <= b.size ? b : a;

            let cnt = 0, sab = 0, sa2 = 0, sb2 = 0;
            for (const [k, r] of small) {
                const r2 = large.get(k);
                if (r2 === undefined) continue;
                cnt++;
                sab += r * r2;
                sa2 += r * r;
                sb2 += r2 * r2;
            }

            common.push(cnt);
            if (cnt < commonMin) commonMin = cnt;
            if (cnt > commonMax) commonMax = cnt;
            if (cnt === 0) zero++;
            if (cnt <= 2) le2++;
            if (cnt <= 5) le5++;
            if (cnt === 0) continue;

            const cos = sab / Math.sqrt(sa2 * sb2);
            ge1++;
            cosSum += cos;
            if (cos < cosMin) cosMin = cos;
            if (cos >= 0.99) cosHigh++;

            for (let gi = 0; gi < GROUPS.length; gi++) {
                if (GROUPS[gi].test(cnt)) {
                    g[gi].count++;
                    g[gi].sum += cos;
                    if (cos < g[gi].min) g[gi].min = cos;
                    if (cos >= 0.99) g[gi].high++;
                }
            }
        }
    }

    const totalPairs = n * (n - 1) / 2;
    const pct = x => (totalPairs ? (x / totalPairs * 100).toFixed(2) + '%' : 'н/д');

    console.log('\n' + '='.repeat(64));
    console.log(title + ' (' + n + ' шт.)');
    console.log('='.repeat(64));

    console.log('1) всего пар: ' + totalPairs);

    console.log('2) общих оценок в паре:   min=' + commonMin +
        '   median=' + median(common) + '   max=' + commonMax);

    console.log('3) доля пар с 0 общих оценок:  ' + pct(zero) + '  (' + zero + ')');
    console.log('   доля пар с <=2 общими:       ' + pct(le2) + '  (' + le2 + ')');
    console.log('   доля пар с <=5 общими:       ' + pct(le5) + '  (' + le5 + ')');

    console.log('4) косинус по сырым оценкам, co-rated (только пары с >=1 общей: ' +
        ge1 + ', это ' + pct(ge1) + '):');
    console.log('     средний:     ' + (ge1 ? (cosSum / ge1).toFixed(6) : 'н/д'));
    console.log('     минимальный: ' + (ge1 ? cosMin.toFixed(6) : 'н/д'));
    console.log('     доля >= 0.99: ' + (ge1 ? (cosHigh / ge1 * 100).toFixed(2) + '%' : 'н/д') +
        '  (' + cosHigh + ')');

    console.log('5) по группам числа общих оценок:');
    console.log('   ' + 'группа'.padEnd(12) + 'пар'.padStart(12) + 'средний'.padStart(14) +
        'минимум'.padStart(14) + 'доля>=0.99'.padStart(14));
    for (let gi = 0; gi < GROUPS.length; gi++) {
        const s = g[gi];
        if (s.count === 0) {
            console.log('   ' + GROUPS[gi].label.padEnd(12) + String(0).padStart(12) +
                'нет пар'.padStart(22));
            continue;
        }
        console.log('   ' + GROUPS[gi].label.padEnd(12) + String(s.count).padStart(12) +
            (s.sum / s.count).toFixed(6).padStart(14) +
            s.min.toFixed(6).padStart(14) +
            ((s.high / s.count) * 100).toFixed(2).padStart(13) + '%');
    }
    console.log('   время: ' + ((Date.now() - started) / 1000).toFixed(1) + ' с');
}
