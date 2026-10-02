#!/usr/bin/env node
'use strict';

// week3/analysis-cf.js — аналитика поверх НЕИЗМЕНЁННОГО week3/script.js.
//
// Приложение здесь только исполняется: все Top-5 получены вызовом
// getUserBasedRecommendations / getItemBasedRecommendations, а сходство для
// массовых подсчётов берётся однопроходной функцией из script.js
// (cosineAndCommonUsers / cosineAndCommonItems). Никакая логика рекомендаций
// здесь не переписывается.
//
// Отличие от check-cf.js: здесь НЕТ сверки с эталоном. Это аналитика, а не
// проверка, и её вывод предсказуемо расходится с эталонными числами там, где
// сознательно считается другая стратегия пропусков или сырое сходство.

const fs = require('fs');
const vm = require('vm');
const path = require('path');

const dataDir = __dirname;

// ---------------------------------------------------------------------------
// Загрузка настоящих data.js и script.js, ровно как в check-cf.js
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
vm.runInContext(fs.readFileSync(path.join(dataDir, 'script.js'), 'utf8'), ctx, { filename: 'week3/script.js' });

const ms = (startNs) => Number(process.hrtime.bigint() - startNs) / 1e6;

// Тот же генератор и то же зерно, что в check-cf.js
function makeRandom(seed) {
    let a = seed >>> 0;
    return function () {
        a = (a + 0x6D2B79F5) >>> 0;
        let t = a;
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}

const f2 = x => x.toFixed(2);
const f3 = x => x.toFixed(3);
const f4 = x => x.toFixed(4);
const f6 = x => x.toFixed(6);
const int = n => n.toLocaleString('ru-RU').replace(/ /g, ' ');
const padR = (s, n) => String(s).padStart(n);
const short = (s, n) => s.length > n ? s.slice(0, n - 1) + '…' : s;

function median(values) {
    if (!values.length) return NaN;
    const v = values.slice().sort((a, b) => a - b);
    const mid = v.length >> 1;
    return v.length % 2 ? v[mid] : (v[mid - 1] + v[mid]) / 2;
}
const mean = a => a.length ? a.reduce((x, y) => x + y, 0) / a.length : NaN;

// Спирмен через средние ранги при равенствах.
function spearman(xs, ys) {
    const n = xs.length;
    if (n < 2) return NaN;
    function ranks(values) {
        const order = values.map((v, i) => [v, i]).sort((a, b) => a[0] - b[0]);
        const r = new Array(n);
        let i = 0;
        while (i < n) {
            let j = i;
            while (j + 1 < n && order[j + 1][0] === order[i][0]) j++;
            const averageRank = (i + j) / 2 + 1;
            for (let k = i; k <= j; k++) r[order[k][1]] = averageRank;
            i = j + 1;
        }
        return r;
    }
    const rx = ranks(xs), ry = ranks(ys);
    const mx = mean(rx), my = mean(ry);
    let num = 0, dx = 0, dy = 0;
    for (let i = 0; i < n; i++) {
        const a = rx[i] - mx, b = ry[i] - my;
        num += a * b; dx += a * a; dy += b * b;
    }
    return (dx === 0 || dy === 0) ? NaN : num / Math.sqrt(dx * dy);
}

function section(title) {
    console.log('\n' + '='.repeat(78));
    console.log(title);
    console.log('='.repeat(78));
}

async function main() {
    await vm.runInContext('loadData()', ctx);

    const meta = JSON.parse(vm.runInContext(`(() => {
        const perMovie = new Array(numMovies + 1).fill(0);
        for (let m = 1; m <= numMovies; m++) for (let u = 1; u <= numUsers; u++) if (ratingMatrix[u][m] !== 0) perMovie[m]++;
        const titleById = {};
        for (const mv of movies) {
            titleById[mv.id] = mv.title;
        }
        // Статистика по фильмам считает только реальные фильмы: копия дубликата
        // и строка "unknown" не являются фильмами, поэтому обе исключены из
        // perMovieReal, а исходный perMovie остаётся для проверок самой матрицы.
        const perMovieReal = new Array(numMovies + 1).fill(0);
        for (let m = 1; m <= numMovies; m++) if (isRealFilm(m)) perMovieReal[m] = perMovie[m];
        return JSON.stringify({ perMovie, perMovieReal, titleById, numUsers, numMovies });
    })()`, ctx));

    const { perMovie, perMovieReal, titleById } = meta;
    let numUsers = vm.runInContext('numUsers', ctx);
    let numMovies = vm.runInContext('numMovies', ctx);

    const M = () => vm.runInContext('ratingMatrix', ctx);
    const cosUsers = vm.runInContext('cosineAndCommonUsers', ctx);
    const cosItems = vm.runInContext('cosineAndCommonItems', ctx);
    const CAP = 50;

    // u.item держит 18 фильмов дважды, под двумя id. data.js сливает их оценки в
    // столбец первого экземпляра, а сами копии исключает из кандидатов, поэтому
    // в Top-5 они не попадают и названия больше не нужно разрешать в id: функции
    // рекомендаций возвращают movieId напрямую.
    const canonicalMovieId = vm.runInContext('canonicalMovieId', ctx);
    const invalidMovieId = vm.runInContext('invalidMovieId', ctx);
    const canonicalIds = [];
    const duplicateIds = [];
    const invalidIds = [];
    for (let m = 1; m <= numMovies; m++) {
        if (canonicalMovieId[m] !== m) { duplicateIds.push(m); continue; }
        if (invalidMovieId[m]) { invalidIds.push(m); continue; }
        canonicalIds.push(m);
    }
    // Тот же фильтр, что у приложения: копия дубликата или строка "unknown" не
    // является фильмом и не участвует в статистике по фильмам.
    const isReal = m => canonicalMovieId[m] === m && !invalidMovieId[m];

    const getUB = id => vm.runInContext(`getUserBasedRecommendations(${id})`, ctx);
    const getIB = id => vm.runInContext(`getItemBasedRecommendations(${id})`, ctx);

    // Рекомендации возвращают { movieId, title, score }, поэтому id известен
    // напрямую и разрешать название в id не нужно.

    const header = [
        'Факты о данных',
        `  пользователей:            ${int(numUsers)}`,
        `  фильмов:                  ${int(numMovies)}`,
        `  оценок:                   ${int(fs.readFileSync(path.join(dataDir, 'u.data'), 'latin1').split('\n').filter(l => l.trim()).length)}`,
        `  фильмов с ровно 1 оценкой: ${int(perMovieReal.filter(v => v === 1).length)}`,
        `  фильмов с 1..4 оценками:  ${int(perMovieReal.filter(v => v >= 1 && v <= 4).length)}`,
        '  (счёт по фильмам — только реальные: копии дубликатов и невалидная строка 267 исключены)',
    ];
    console.log(header.join('\n'));

    // =====================================================================
    section('1. РАЗМЕР И СТОИМОСТЬ');
    // =====================================================================
    const userPairs = (numUsers * (numUsers - 1)) / 2;
    const moviePairs = (numMovies * (numMovies - 1)) / 2;
    const canonicalCount = canonicalIds.length;
    const moviePairsCanonical = (canonicalCount * (canonicalCount - 1)) / 2;
    console.log(`  пар пользователей:  ${int(userPairs)}`);
    console.log(`  пар фильмов:        ${int(moviePairs)}`);
    console.log(`    из них по фильмам без duplicateOf: ${int(moviePairsCanonical)}` +
        `  (${canonicalCount} фильмов вместо ${numMovies}: ${duplicateIds.length} id-копий ` +
        `и ${invalidIds.length} невалидных строк исключены)`);

    let t = process.hrtime.bigint();
    let userSimPos = 0, userMax = 0, userNonZeroCommon = 0;
    for (let a = 1; a <= numUsers; a++) {
        for (let b = a + 1; b <= numUsers; b++) {
            const r = cosUsers(a, b);
            if (r.similarity > 0) {
                userSimPos++;
                if (r.similarity > userMax) userMax = r.similarity;
            }
            if (r.common > 0) userNonZeroCommon++;
        }
    }
    const userAllMs = ms(t);
    t = process.hrtime.bigint();
    let movieSimPos = 0, movieMax = 0, movieNonZeroCommon = 0;
    // Только фильмы без duplicateOf: столбцы копий пустые, поэтому любая пара
    // с копией дала бы нулевое сходство и лишь раздувала число пар.
    for (let i = 0; i < canonicalCount; i++) {
        const a = canonicalIds[i];
        for (let j = i + 1; j < canonicalCount; j++) {
            const r = cosItems(a, canonicalIds[j]);
            if (r.similarity > 0) {
                movieSimPos++;
                if (r.similarity > movieMax) movieMax = r.similarity;
            }
            if (r.common > 0) movieNonZeroCommon++;
        }
    }
    const movieAllMs = ms(t);

    console.log(`\n  время на ВСЕ пары (однопроходная функция из script.js):`);
    console.log(`    пары пользователей: ${int(userPairs)}   за ${int(userAllMs)} мс` +
        `   (${(userAllMs * 1000 / userPairs).toFixed(3)} мкс на пару)`);
    console.log(`    пары фильмов:       ${int(moviePairsCanonical)}   за ${int(movieAllMs)} мс` +
        `   (${(movieAllMs * 1000 / moviePairsCanonical).toFixed(3)} мкс на пару)   только по фильмам без duplicateOf`);
    console.log(`\n  пар со сходством > 0:  пользователи ${int(userSimPos)} (${f2(100 * userSimPos / userPairs)}%),` +
        ` фильмы ${int(movieSimPos)} (${f2(100 * movieSimPos / moviePairsCanonical)}%)`);
    console.log(`  пар с >=1 общей оценкой: пользователи ${int(userNonZeroCommon)} (${f2(100 * userNonZeroCommon / userPairs)}%),` +
        ` фильмы ${int(movieNonZeroCommon)} (${f2(100 * movieNonZeroCommon / moviePairsCanonical)}%)`);
    console.log(`  максимальное сходство:   пользователи ${f6(userMax)}, фильмы ${f6(movieMax)}`);
    console.log(`\n  отношение числа пар: фильмы к пользователям = ${int(moviePairsCanonical / userPairs)}x`);

    // =====================================================================
    section('2. ВЫБОРКА: каждый пятый пользователь (id 1, 6, 11, ...)');
    // =====================================================================
    const sample = [];
    for (let u = 1; u <= numUsers; u++) if (u % 5 === 1) sample.push(u);
    console.log(`  размер выборки: ${sample.length} пользователей` +
        ` (с id ${sample[0]} до ${sample[sample.length - 1]})`);

    const stats = {
        userBased: { times: [], short: 0, preds: [], counts: [], ids: [] },
        itemBased: { times: [], short: 0, preds: [], counts: [], ids: [] },
    };
    const overlaps = [];
    const duplicateHits = { userBased: 0, itemBased: 0 };

    for (const u of sample) {
        const a = process.hrtime.bigint();
        const ub = getUB(u);
        const ubMs = ms(a);
        const b = process.hrtime.bigint();
        const ib = getIB(u);
        const ibMs = ms(b);

        const idSets = [];
        for (const [name, list, ms_] of [['userBased', ub, ubMs], ['itemBased', ib, ibMs]]) {
            const st = stats[name];
            st.times.push(ms_);
            if (list.length < 5) st.short++;
            for (const item of list) {
                const id = item.movieId;
                if (duplicateIds.includes(id)) duplicateHits[name]++;
                st.preds.push(item.score);
                st.counts.push(perMovieReal[id]);
                st.ids.push(id);
            }
            idSets.push(new Set(list.map(item => item.movieId)));
        }
        let inter = 0;
        for (const x of idSets[0]) if (idSets[1].has(x)) inter++;
        overlaps.push(inter);
    }

    for (const [name, ru] of [['userBased', 'user-based'], ['itemBased', 'item-based']]) {
        const st = stats[name];
        const slots = st.preds.length;
        const freq = new Map();
        for (const id of st.ids) freq.set(id, (freq.get(id) || 0) + 1);
        const top10 = [...freq.entries()].sort((x, y) => y[1] - x[1]).slice(0, 10);
        const top10Slots = top10.reduce((s, [, c]) => s + c, 0);
        const times = st.times;

        console.log(`\n  --- ${ru} ---`);
        console.log(`    время одного Top-5: среднее ${f2(mean(times))} мс, медиана ${f2(median(times))} мс, максимум ${f2(Math.max(...times))} мс`);
        console.log(`    всего мест: ${slots}`);
        console.log(`    пользователей с списком короче 5: ${st.short} из ${sample.length} (${f2(100 * st.short / sample.length)}%)`);
        console.log(`    средний прогноз: ${f4(mean(st.preds))}   диапазон: ${f3(Math.min(...st.preds))} .. ${f3(Math.max(...st.preds))}`);
        console.log(`    число оценок у рекомендованных фильмов: среднее ${f2(mean(st.counts))}, медиана ${f2(median(st.counts))}` +
            `   (всего оценок у фильмов из списков: ${int(st.counts.reduce((s, v) => s + v, 0))})`);
        console.log(`    разных фильмов во всех Top-5: ${freq.size}`);
        console.log(`    доля мест, занятых 10 самыми частыми фильмами: ${f2(100 * top10Slots / slots)}%` +
            `   (эти фильмы: ${top10.length}, места ${top10Slots} из ${slots})`);
    }
    console.log(`\n    среднее число общих фильмов в двух списках одного пользователя: ${f3(mean(overlaps))}` +
        `   (медиана ${f3(median(overlaps))}, максимум ${Math.max(...overlaps)})`);
    console.log(`    id-копий, попавших в Top-5 выборки: user-based ${duplicateHits.userBased}, ` +
        `item-based ${duplicateHits.itemBased}   (ожидаем 0: копии исключены из кандидатов)`);

    // для пункта 4: какие фильмы вообще попадали в списки выборки
    const sampleMovieIds = new Set([...stats.userBased.ids, ...stats.itemBased.ids]);
    const oneRating = [], lowRating = [];
    for (let m = 1; m <= numMovies; m++) {
        if (!isReal(m)) continue;
        if (perMovieReal[m] === 1) oneRating.push(m);
        if (perMovieReal[m] >= 1 && perMovieReal[m] <= 4) lowRating.push(m);
    }
    const hit = list => list.filter(m => sampleMovieIds.has(m));
    const hitOne = hit(oneRating), hitLow = hit(lowRating);
    console.log(`\n    из ${oneRating.length} фильмов с 1 оценкой в списки попали: ${hitOne.length}` +
        `  (${hitOne.map(m => titleById[m]).join(', ') || '—'})`);
    console.log(`    из ${lowRating.length} фильмов с 1..4 оценками в списки попали: ${hitLow.length}` +
        `  (${hitLow.map(m => titleById[m] + ' [' + perMovieReal[m] + ']').join(', ') || '—'})`);
    // подробная разбивка по числу оценок
    for (const [lo, hi] of [[1, 1], [2, 4], [5, 19], [20, 999]]) {
        const bucket = [];
        for (let m = 1; m <= numMovies; m++) {
            if (!isReal(m)) continue;
            if (perMovieReal[m] >= lo && perMovieReal[m] <= hi) bucket.push(m);
        }
        const got = hit(bucket).length;
        const label = lo === hi ? `${lo} оценкой` : `${lo}..${hi === 999 ? '∞' : hi} оценками`;
        console.log(`    ${padR(bucket.length, 4)} фильмов с ${label.padEnd(18)}-> попали ${padR(got, 4)} (${f2(100 * got / bucket.length)}%)`);
    }

    // =====================================================================
    section('3. ТРИ СТРАТЕГИИ ПРОПУСКОВ НА 10000 ПАРАХ ПОЛЬЗОВАТЕЛЕЙ');
    // =====================================================================
    console.log('  генератор mulberry32, зерно 20241017 (тот же, что в check-cf.js)');
    const PAIR_COUNT = 10000;
    const rnd = makeRandom(20241017);
    const pairs = [];
    for (let i = 0; i < PAIR_COUNT; i++) {
        let a = 1 + Math.floor(rnd() * numUsers);
        let b = 1 + Math.floor(rnd() * numUsers);
        if (a === b) b = b >= numUsers ? 1 : b + 1;
        pairs.push([a, b]);
    }

    // стратегия (б) требует среднего пользователя по всем фильмам
    const userMean = new Array(numUsers + 1).fill(0);
    {
        const Mx = M();
        for (let u = 1; u <= numUsers; u++) {
            let s = 0, c = 0;
            for (let m = 1; m <= numMovies; m++) if (Mx[u][m] !== 0) { s += Mx[u][m]; c++; }
            userMean[u] = c ? s / c : 0;
        }
    }

    function cosineCoRatedOnly(a, b) {
        // (a) только общие оценки, без всякого взвешивания
        const Mx = M(), ra = Mx[a], rb = Mx[b];
        let dot = 0, na = 0, nb = 0;
        for (let m = 1; m <= numMovies; m++) {
            const x = ra[m], y = rb[m];
            if (x === 0 || y === 0) continue;
            dot += x * y; na += x * x; nb += y * y;
        }
        return na === 0 || nb === 0 ? 0 : dot / Math.sqrt(na * nb);
    }

    function cosineMeanImputed(a, b) {
        // (б) пропуски заполнены средней оценкой пользователя по всем фильмам
        const Mx = M(), ra = Mx[a], rb = Mx[b], ma = userMean[a], mb = userMean[b];
        let dot = 0, na = 0, nb = 0;
        for (let m = 1; m <= numMovies; m++) {
            const x = ra[m] !== 0 ? ra[m] : ma;
            const y = rb[m] !== 0 ? rb[m] : mb;
            dot += x * y; na += x * x; nb += y * y;
        }
        return na === 0 || nb === 0 ? 0 : dot / Math.sqrt(na * nb);
    }

    const strategies = [
        ['(а) только общие оценки', cosineCoRatedOnly],
        ['(б) заполнение пропусков средней', cosineMeanImputed],
        ['(в) взвешивание min(n,50)/50', null],   // берём однопроходную функцию приложения
    ];

    console.log(`\n  ${'стратегия'.padEnd(30)}${'среднее'.padStart(9)}${'минимум'.padStart(9)}` +
        `${'максимум'.padStart(9)}${'доля >=0.99'.padStart(12)}${'Спирмен'.padStart(10)}${'время'.padStart(11)}`);
    const commonAll = pairs.map(([a, b]) => {
        const Mx = M(), ra = Mx[a], rb = Mx[b];
        let c = 0;
        for (let m = 1; m <= numMovies; m++) if (ra[m] !== 0 && rb[m] !== 0) c++;
        return c;
    });

    for (const [name, fn] of strategies) {
        const sims = new Array(PAIR_COUNT);
        const t0 = process.hrtime.bigint();
        for (let i = 0; i < PAIR_COUNT; i++) {
            const [a, b] = pairs[i];
            sims[i] = fn ? fn(a, b) : cosUsers(a, b).similarity;
        }
        const elapsed = ms(t0);
        const hi = sims.filter(v => v >= 0.99).length;
        console.log(`  ${name.padEnd(30)}${f4(mean(sims)).padStart(9)}${f4(Math.min(...sims)).padStart(9)}` +
            `${f4(Math.max(...sims)).padStart(9)}${(f2(100 * hi / PAIR_COUNT) + '%').padStart(12)}` +
            `${spearman(sims, commonAll).toFixed(4).padStart(10)}${(f1(elapsed) + ' мс').padStart(11)}`);
    }
    function f1(x) { return x.toFixed(1); }

    console.log(`\n  Спирмен посчитан по всем ${PAIR_COUNT} парам. Для справки — по подмножеству пар`);
    console.log('  с ненулевым пересечением (иначе сходство почти везде 0 и рангов мало):');
    const idxNonZero = commonAll.map((c, i) => [c, i]).filter(([c]) => c > 0).map(([, i]) => i);
    console.log(`  пар с ненулевым пересечением: ${idxNonZero.length}`);
    console.log(`\n  ${'стратегия'.padEnd(30)}${'Спирмен (все пары)'.padStart(20)}${'Спирмен (только пересечение)'.padStart(30)}`);
    for (const [name, fn] of strategies) {
        const sims = new Array(PAIR_COUNT);
        for (let i = 0; i < PAIR_COUNT; i++) {
            const [a, b] = pairs[i];
            sims[i] = fn ? fn(a, b) : cosUsers(a, b).similarity;
        }
        const sub = idxNonZero.map(i => sims[i]);
        const subCommon = idxNonZero.map(i => commonAll[i]);
        console.log(`  ${name.padEnd(30)}${spearman(sims, commonAll).toFixed(4).padStart(20)}` +
            `${spearman(sub, subCommon).toFixed(4).padStart(30)}`);
    }

    // =====================================================================
    section('4. ХОЛОДНЫЙ СТАРТ: пользователь 19 и его первые k оценок');
    // =====================================================================
    const COLD_USER = 19;
    const allRatings = vm.runInContext('JSON.stringify(ratings)', ctx);
    const parsedRatings = JSON.parse(allRatings);
    const coldAll = parsedRatings.filter(r => r.userId === COLD_USER)
        .sort((a, b) => a.timestamp - b.timestamp);
    console.log(`  пользователь ${COLD_USER}: всего оценок ${coldAll.length}, берём первые k по времени`);
    console.log(`  первые ${Math.min(20, coldAll.length)} оценок по времени:`);
    for (const r of coldAll.slice(0, 20)) {
        console.log(`    ${titleById[r.itemId].padEnd(42)} оценка ${r.rating}   id ${r.itemId}`);
    }

    const KS = [0, 1, 3, 5, 10, 20];
    for (const k of KS) {
        const keep = new Set(coldAll.slice(0, k).map(r => r.itemId));
        const rebuilt = parsedRatings.filter(r => r.userId !== COLD_USER || keep.has(r.itemId));
        vm.runInContext(`ratings = ${JSON.stringify(rebuilt)}`, ctx);
        vm.runInContext('buildRatingMatrix()', ctx);
        numUsers = vm.runInContext('numUsers', ctx);
        numMovies = vm.runInContext('numMovies', ctx);

        let pos = 0, maxSim = 0, maxRaw = 0, bestCommon = 0, bestUser = -1;
        for (let u = 1; u <= numUsers; u++) {
            if (u === COLD_USER) continue;
            const r = cosUsers(COLD_USER, u);
            if (r.similarity > 0) {
                pos++;
                if (r.similarity > maxSim) maxSim = r.similarity;
            }
            if (r.common > 0 && r.common > bestCommon) { bestCommon = r.common; bestUser = u; }
        }
        // сырое сходство по общим оценкам для интерпретации
        {
            const Mx = M(), ra = Mx[COLD_USER];
            for (let u = 1; u <= numUsers; u++) {
                if (u === COLD_USER) continue;
                const v = cosineCoRatedOnly(COLD_USER, u);
                if (v > maxRaw) maxRaw = v;
            }
        }

        const ub = getUB(COLD_USER);
        const ib = getIB(COLD_USER);

        console.log(`\n  --- k = ${k} (оценок у ${COLD_USER}: ${k}) ---`);
        console.log(`    пользователей с положительным сходством (взвешенным, как в приложении): ${pos}`);
        console.log(`    максимальное сходство: взвешенное ${f6(maxSim)}, сырое по общим ${f6(maxRaw)}` +
            `   лучший сосед по числу общих: user ${bestUser} (${bestCommon} общих)`);
        console.log(`    длина Top-5: user-based ${ub.length}, item-based ${ib.length}`);
        for (const [ru, list] of [['user-based', ub], ['item-based', ib]]) {
            if (!list.length) { console.log(`    ${ru}: список пуст`); continue; }
            console.log(`    ${ru}:`);
            list.forEach((item, i) => {
                console.log(`      ${i + 1}. ${short(item.title, 40).padEnd(42)}${f4(item.score)}` +
                    `   id ${item.movieId}   оценок у фильма: ${perMovieReal[item.movieId]}`);
            });
        }
    }
    // вернуть полную матрицу
    vm.runInContext(`ratings = ${JSON.stringify(parsedRatings)}`, ctx);
    vm.runInContext('buildRatingMatrix()', ctx);
    numUsers = vm.runInContext('numUsers', ctx);
    numMovies = vm.runInContext('numMovies', ctx);
    console.log('\n  матрица восстановлена в полном виде');

    // =====================================================================
    section('5. СРАВНЕНИЕ С ПРОШЛОЙ РАБОТОЙ');
    // =====================================================================
    const TARGETS = [1, 203];
    const genreNamesAll = vm.runInContext('genreNames', ctx);
    const realGenres = genreNamesAll.filter(g => g !== 'unknown');   // 18 жанров
    const moviesMeta = vm.runInContext('movies', ctx);

    console.log(`  жанров без "unknown": ${realGenres.length}`);
    console.log(`  (жанры из u.item, поля 6..23; поле 5 — unknown, в косинус не входит)`);
    const genreVec = {};
    for (const mv of moviesMeta) {
        const v = new Array(realGenres.length).fill(0);
        for (const name of mv.genres) {
            const idx = realGenres.indexOf(name);
            if (idx >= 0) v[idx] = 1;
        }
        genreVec[mv.id] = v;
    }
    const noGenre = Object.keys(genreVec).filter(id => genreVec[id].every(v => v === 0)).map(Number);
    console.log(`  фильмов без ни одного из этих жанров: ${noGenre.length}` +
        ` (${noGenre.map(i => i + ': "' + titleById[i] + '"').join(', ')})`);

    function genreCosine(a, b) {
        const va = genreVec[a], vb = genreVec[b];
        let dot = 0, na = 0, nb = 0;
        for (let i = 0; i < va.length; i++) {
            dot += va[i] * vb[i]; na += va[i] * va[i]; nb += vb[i] * vb[i];
        }
        return na === 0 || nb === 0 ? 0 : dot / Math.sqrt(na * nb);
    }
    function genreList(mv) {
        const v = genreVec[mv];
        return realGenres.filter((_, i) => v[i] === 1).join(', ') || '(только unknown)';
    }

    for (const target of TARGETS) {
        console.log(`\n  ${'#'.repeat(74)}`);
        console.log(`  # ${titleById[target]} (id ${target})`);
        console.log(`  #   жанры: ${genreList(target)}`);
        console.log(`  #   оценок у фильма: ${perMovieReal[target]}`);
        console.log(`  ${'#'.repeat(74)}`);

        // --- по жанрам ---
        // id-копии не рассматриваем: у них те же жанры и пустой столбец оценок,
        // поэтому в Top-5 они бы дублировали исходный фильм.
        const byGenre = [];
        for (const m of canonicalIds) {
            if (m === target) continue;
            byGenre.push({ id: m, title: titleById[m], sim: genreCosine(target, m) });
        }
        byGenre.sort((x, y) => (y.sim - x.sim) || (x.title < y.title ? -1 : x.title > y.title ? 1 : 0));

        console.log(`\n  --- Top-5 по ЖАНРАМ (косинус по 18 жанрам, при равенстве — по названию) ---`);
        console.log('    ' + '#'.padStart(2) + '  ' + 'название'.padEnd(40) + 'сходство'.padStart(9) +
            'оценок'.padStart(8) + 'общих зрителей'.padStart(16));
        byGenre.slice(0, 5).forEach((r, i) => {
            const common = cosItems(target, r.id).common;
            console.log('    ' + padR(i + 1, 2) + '  ' + short(r.title, 40).padEnd(40) + f4(r.sim).padStart(9) +
                padR(perMovieReal[r.id], 8) + padR(common, 16));
            console.log('       жанры: ' + genreList(r.id));
        });

        // --- по оценкам (взвешенный косинус столбцов) ---
        const byRatings = [];
        for (const m of canonicalIds) {
            if (m === target) continue;
            byRatings.push({ id: m, title: titleById[m], ...cosItems(target, m) });
        }
        byRatings.sort((x, y) => (y.similarity - x.similarity) || (x.id - y.id));

        console.log(`\n  --- Top-5 по ОЦЕНКАМ (взвешенный косинус столбцов, как в приложении) ---`);
        console.log('    ' + '#'.padStart(2) + '  ' + 'название'.padEnd(40) + 'сходство'.padStart(9) +
            'оценок'.padStart(8) + 'общих зрителей'.padStart(16));
        byRatings.slice(0, 5).forEach((r, i) => {
            console.log('    ' + padR(i + 1, 2) + '  ' + short(r.title, 40).padEnd(40) + f4(r.similarity).padStart(9) +
                padR(perMovieReal[r.id], 8) + padR(r.common, 16));
            console.log('       жанры: ' + genreList(r.id));
        });

        // сколько из жанровых попали в рейтинговые и наоборот
        const gIds = new Set(byGenre.slice(0, 5).map(r => r.id));
        const rIds = new Set(byRatings.slice(0, 5).map(r => r.id));
        let shared = 0;
        for (const id of gIds) if (rIds.has(id)) shared++;
        console.log(`\n  совпадений между двумя Top-5: ${shared} из 5`);
        if (shared === 0) {
            console.log(`  жанровый Top-5: ${[...gIds].map(i => titleById[i]).join(' | ')}`);
            console.log(`  рейтинговый Top-5: ${[...rIds].map(i => titleById[i]).join(' | ')}`);
        }
    }

    console.log('\n' + '='.repeat(78));
    console.log('ГОТОВО');
    console.log('='.repeat(78));
}

main().catch(e => { console.error('FATAL:', e); process.exit(1); });