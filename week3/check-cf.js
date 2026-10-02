#!/usr/bin/env node
'use strict';

const fs = require('fs');
const vm = require('vm');
const path = require('path');

const dataDir = __dirname;

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

// Эталонный вывод, снят ДО оптимизации (cosineSimilarity + countCommonRatings
// по всей длине вектора, колонки строились на каждый вызов). Здесь только id и
// прогноз с 6 знаками — этого достаточно, чтобы поймать любое изменение.
const BASELINE = {
    1: {
        userBased: [[408, '4.856821'], [1039, '4.856652'], [603, '4.800629'], [919, '4.800277'], [525, '4.799036']],
        itemBased: [[319, '4.751347'], [1065, '4.751334'], [710, '4.749769'], [919, '4.749466'], [1110, '4.709082']],
    },
    7: {
        userBased: [[209, '4.667172'], [48, '4.571417'], [124, '4.500198'], [87, '4.400609'], [467, '4.399956']],
        itemBased: [[1107, '4.901968'], [715, '4.799974'], [469, '4.799381'], [1112, '4.799020'], [1149, '4.780974']],
    },
    19: {
        userBased: [[127, '4.639504'], [50, '4.552752'], [56, '4.508928'], [168, '4.483094'], [357, '4.437006']],
        itemBased: [[414, '3.698578'], [524, '3.676530'], [493, '3.674648'], [503, '3.674095'], [502, '3.673561']],
    },
};

// Детерминированный генератор с фиксированным зерном: mulberry32.
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

const ms = (startNs) => Number(process.hrtime.bigint() - startNs) / 1e6;

async function main() {
    const loadStart = process.hrtime.bigint();
    await vm.runInContext('loadData()', ctx);
    const loadMs = ms(loadStart);

    const buildStats = vm.runInContext('JSON.stringify(lastBuildStats)', ctx);

    const st = vm.runInContext(`(() => {
        const M = ratingMatrix, NU = numUsers, NM = numMovies;
        const perUser = new Array(NU + 1).fill(0);
        const perMovie = new Array(NM + 1).fill(0);
        for (let u = 1; u <= NU; u++) for (let m = 1; m <= NM; m++) if (M[u][m] !== 0) { perUser[u]++; perMovie[m]++; }
        const titleById = {};
        const displayTitleById = {};
        const idsByTitle = {};
        for (const mv of movies) {
            titleById[mv.id] = mv.title;
            displayTitleById[mv.id] = mv.displayTitle;
            (idsByTitle[mv.title] = idsByTitle[mv.title] || []).push(mv.id);
        }
        let minRatedId = -1, count20 = 0;
        for (let u = 1; u <= NU; u++) if (perUser[u] === 20) { count20++; if (minRatedId < 0) minRatedId = u; }
        return JSON.stringify({ perUser, perMovie, titleById, displayTitleById, idsByTitle, minRatedId, count20, numUsers: NU, numMovies: NM });
    })()`, ctx);

    const { perUser, perMovie, titleById, displayTitleById, idsByTitle, minRatedId, count20, numUsers, numMovies } = JSON.parse(st);

    const M = vm.runInContext('ratingMatrix', ctx);
    const cos = vm.runInContext('cosineSimilarity', ctx);
    const ratingColumns = vm.runInContext('ratingColumns', ctx);
    const cosUsers = vm.runInContext('cosineAndCommonUsers', ctx);
    const cosItems = vm.runInContext('cosineAndCommonItems', ctx);
    const canonicalMovieId = vm.runInContext('canonicalMovieId', ctx);
    const duplicateIds = new Set();
    for (let m = 1; m <= numMovies; m++) if (canonicalMovieId[m] !== m) duplicateIds.add(m);
    const isCopy = m => canonicalMovieId[m] !== m;

    function commonCount(rowA, rowB) {
        let n = 0;
        const length = Math.min(rowA.length, rowB.length);
        for (let i = 0; i < length; i++) {
            if (!rowA[i] || !rowB[i]) continue;
            n++;
        }
        return n;
    }

    const CAP = 50, THR = 5, K = 20;
    const ev = c => Math.min(c, CAP) / CAP;

    // -----------------------------------------------------------------
    // Эталонные реализации ровно в том виде, в каком они были ДО оптимизации:
    // cosineSimilarity и countCommonRatings идут по всей длине вектора, а
    // колонки фильмов перестраиваются на каждый вызов. Используются только для
    // проверки (b) и для честного сравнения времени.
    // -----------------------------------------------------------------
    function ubReference(activeUserId) {
        const activeRow = M[activeUserId];
        const neighbours = [];
        for (let userId = 1; userId <= numUsers; userId++) {
            if (userId === activeUserId) continue;
            const similarity = cos(activeRow, M[userId]);
            if (similarity > 0) neighbours.push({ userId, similarity });
        }
        neighbours.sort((a, b) => b.similarity - a.similarity);
        const top = neighbours.slice(0, K);
        for (const n of top) n.evidence = ev(commonCount(activeRow, M[n.userId]));
        const candidates = [];
        for (let movieId = 1; movieId <= numMovies; movieId++) {
            if (isCopy(movieId)) continue;
            if (activeRow[movieId] !== 0) continue;
            let weighted = 0, simSum = 0, voters = 0, evidence = 0;
            for (const n of top) {
                const rating = M[n.userId][movieId];
                if (rating === 0) continue;
                weighted += n.similarity * rating; simSum += n.similarity; voters++;
                evidence += n.evidence;
            }
            if (voters === 0) continue;
            if (evidence < THR) continue;
            candidates.push({ movieId, score: weighted / simSum });
        }
        candidates.sort((a, b) => b.score - a.score);
        return candidates.slice(0, 5);
    }

    function ibReference(activeUserId) {
        const activeRow = M[activeUserId];
        const ratedMovieIds = [];
        for (let movieId = 1; movieId <= numMovies; movieId++) {
            if (activeRow[movieId] !== 0) ratedMovieIds.push(movieId);
        }
        const columns = [];   // как раньше: пересобираются на каждый вызов
        for (let movieId = 0; movieId <= numMovies; movieId++) {
            const column = new Array(numUsers + 1).fill(0);
            for (let userId = 1; userId <= numUsers; userId++) column[userId] = M[userId][movieId];
            columns.push(column);
        }
        const candidates = [];
        for (let movieId = 1; movieId <= numMovies; movieId++) {
            if (isCopy(movieId)) continue;
            if (activeRow[movieId] !== 0) continue;
            const contributors = [];
            for (const ratedId of ratedMovieIds) {
                const common = commonCount(columns[ratedId], columns[movieId]);
                if (common === 0) continue;
                contributors.push({ ratedId, similarity: cos(columns[ratedId], columns[movieId]), evidence: ev(common) });
            }
            contributors.sort((a, b) => b.similarity - a.similarity);
            const top = contributors.slice(0, K);
            let weighted = 0, simSum = 0, evidence = 0;
            for (const c of top) {
                weighted += c.similarity * activeRow[c.ratedId]; simSum += c.similarity; evidence += c.evidence;
            }
            if (simSum === 0) continue;
            if (evidence < THR) continue;
            candidates.push({ movieId, score: weighted / simSum });
        }
        candidates.sort((a, b) => b.score - a.score);
        return candidates.slice(0, 5);
    }

    function allNeighbours(userId) {
        const list = [];
        for (let u = 1; u <= numUsers; u++) {
            if (u === userId) continue;
            const s = cos(M[userId], M[u]);
            if (s > 0) list.push({ userId: u, sim: s, common: commonCount(M[userId], M[u]) });
        }
        list.sort((a, b) => b.sim - a.sim);
        return list;
    }

    function userBasedCandidates(userId, nbrs) {
        const out = [];
        const rejected = [];
        let before = 0;
        for (let m = 1; m <= numMovies; m++) {
            if (isCopy(m)) continue;
            if (M[userId][m] !== 0) continue;
            let num = 0, den = 0, voters = 0, evidence = 0;
            for (const nb of nbrs) {
                const r = M[nb.userId][m];
                if (r === 0) continue;
                num += nb.sim * r; den += nb.sim; voters++;
                evidence += ev(nb.common);
            }
            if (voters === 0) continue;
            before++;
            if (evidence < THR) {
                // Keep the strongest ones by sum of weights: a candidate rejected
                // on evidence is still interesting, because a high sum of
                // neighbour weights with low evidence is exactly the case where
                // the weights themselves are small.
                rejected.push({ movieId: m, simSum: den, voters, evidence });
                continue;
            }
            out.push({ movieId: m, score: num / den, voters, evidence });
        }
        out.sort((a, b) => b.score - a.score);
        out.доПорога = before;
        rejected.sort((a, b) => b.simSum - a.simSum);
        out.отсеяно = rejected;
        return out;
    }

    const cols = ratingColumns;   // строятся один раз в data.js
    function itemBasedCandidates(userId) {
        const rated = [];
        for (let m = 1; m <= numMovies; m++) if (M[userId][m] !== 0) rated.push(m);
        const out = [];
        let before = 0;
        for (let m = 1; m <= numMovies; m++) {
            if (isCopy(m)) continue;
            if (M[userId][m] !== 0) continue;
            const contrib = [];
            for (const r of rated) {
                const c = commonCount(cols[r], cols[m]);
                if (c === 0) continue;
                contrib.push({ ratedId: r, sim: cos(cols[r], cols[m]), evidence: ev(c) });
            }
            contrib.sort((a, b) => b.sim - a.sim);
            const top = contrib.slice(0, K);
            let num = 0, den = 0, evidence = 0;
            for (const t of top) { num += t.sim * M[userId][t.ratedId]; den += t.sim; evidence += t.evidence; }
            if (den === 0) continue;
            before++;
            if (evidence < THR) continue;
            out.push({ movieId: m, score: num / den, voters: top.length, evidence });
        }
        out.sort((a, b) => b.score - a.score);
        out.доПорога = before;
        return out;
    }

    const f4 = x => x.toFixed(4);
    const f6 = x => x.toFixed(6);
    const f2 = x => x.toFixed(2);
    const short = (s, n) => s.length > n ? s.slice(0, n - 1) + '…' : s;

    // -----------------------------------------------------------------
    // displayTitle: артикль из конца названия переносится в начало. Поле title
    // при этом не меняется, поэтому оба напечатаны рядом. Сортировки по
    // displayTitle нигде нет: порядок при равных баллах остаётся по id.
    // -----------------------------------------------------------------
    console.log('='.repeat(72));
    console.log('ПРОВЕРКА: displayTitle');
    console.log('='.repeat(72));
    console.log('  id    title в u.item                              displayTitle                        изменилось');
    const titleIds = [127, 408, 919, 543, 1609];
    for (const id of titleIds) {
        const raw_ = titleById[id], disp = displayTitleById[id];
        console.log('  ' + String(id).padEnd(6) + short(raw_, 44).padEnd(46) +
            short(disp, 44).padEnd(46) + (raw_ === disp ? 'нет' : 'да'));
    }
    for (const wanted of ['Paris, Texas (1984)', 'M*A*S*H (1970)']) {
        const ids = idsByTitle[wanted] || [];
        for (const id of ids) {
            const raw_ = titleById[id], disp = displayTitleById[id];
            console.log('  ' + String(id).padEnd(6) + short(raw_, 44).padEnd(46) +
                short(disp, 44).padEnd(46) + (raw_ === disp ? 'нет (ожидаемо)' : 'ДА (ожидаемо нет)'));
        }
    }
    let changedCount = 0;
    for (let id = 1; id <= numMovies; id++) {
        if (titleById[id] !== displayTitleById[id]) changedCount++;
    }
    console.log('  всего фильмов с изменённым displayTitle: ' + changedCount + ' из ' + numMovies);
    console.log('');

    // -----------------------------------------------------------------
    // (a) однопроходная функция против эталонной, на 10000 пар каждого вида
    // -----------------------------------------------------------------
    console.log('='.repeat(72));
    console.log('ПРОВЕРКА (a): новая однопроходная функция против эталонной');
    console.log('='.repeat(72));

    const PAIR_COUNT = 10000;
    const rnd = makeRandom(20241017);
    const randInt = hi => 1 + Math.floor(rnd() * hi);

    function comparePairs(kind, count) {
        let maxCos = 0, maxCommon = 0, mismatchedCos = 0, mismatchedCommon = 0;
        let nonZeroCommon = 0;
        const tRef = process.hrtime.bigint();
        for (let i = 0; i < count; i++) {
            const a = randInt(kind === 'user' ? numUsers : numMovies);
            const b = randInt(kind === 'user' ? numUsers : numMovies);
            const refCos = kind === 'user'
                ? cos(M[a], M[b])
                : cos(ratingColumns[a], ratingColumns[b]);
            const refCommon = commonCount(
                kind === 'user' ? M[a] : ratingColumns[a],
                kind === 'user' ? M[b] : ratingColumns[b]);
            const fast = kind === 'user'
                ? cosUsers(a, b)
                : cosItems(a, b);
            const dCos = Math.abs(fast.similarity - refCos);
            const dCommon = Math.abs(fast.common - refCommon);
            if (dCos > maxCos) maxCos = dCos;
            if (dCommon > maxCommon) maxCommon = dCommon;
            if (dCos !== 0) mismatchedCos++;
            if (dCommon !== 0) mismatchedCommon++;
            if (refCommon > 0) nonZeroCommon++;
        }
        const refMs = ms(tRef);
        const tFast = process.hrtime.bigint();
        for (let i = 0; i < count; i++) {
            const a = randInt(kind === 'user' ? numUsers : numMovies);
            const b = randInt(kind === 'user' ? numUsers : numMovies);
            if (kind === 'user') cosUsers(a, b); else cosItems(a, b);
        }
        const fastMs = ms(tFast);
        return { maxCos, maxCommon, mismatchedCos, mismatchedCommon, nonZeroCommon, refMs, fastMs };
    }

    console.log('\n  пар фильмов:  ' + PAIR_COUNT + '   (генератор mulberry32, зерно 20241017)');
    console.log('  ' + 'параметр'.padEnd(34) + 'эталон'.padStart(12) + 'однопроходная'.padStart(16) + 'разница'.padStart(12));
    const itemRes = comparePairs('movie', PAIR_COUNT);
    const userRes = comparePairs('user', PAIR_COUNT);
    for (const [label, r] of [['фильмы: макс. расхождение косинуса', itemRes.maxCos],
    ['фильмы: макс. расхождение общих', itemRes.maxCommon],
    ['пользователи: макс. расхождение косинуса', userRes.maxCos],
    ['пользователи: макс. расхождение общих', userRes.maxCommon]]) {
        console.log('  ' + label.padEnd(34) + ''.padStart(12) + r.toExponential(3).padStart(16));
    }
    console.log('  пар с ненулевым пересечением: фильмы ' + itemRes.nonZeroCommon + ', пользователи ' + userRes.nonZeroCommon);
    console.log('  пар с ненулевым расхождением:   фильмы ' + itemRes.mismatchedCos + '/' + itemRes.mismatchedCommon +
        ', пользователи ' + userRes.mismatchedCos + '/' + userRes.mismatchedCommon);
    console.log('  время ' + PAIR_COUNT + ' пар: эталон ' + itemRes.refMs.toFixed(1) + ' мс -> однопроходная ' + itemRes.fastMs.toFixed(1) + ' мс');

    // -----------------------------------------------------------------
    // (c) время построения новых структур при загрузке
    // -----------------------------------------------------------------
    console.log('\n' + '='.repeat(72));
    console.log('ПРОВЕРКА (c): построение структур при загрузке данных');
    console.log('='.repeat(72));
    const bs = JSON.parse(buildStats);
    console.log('  матрица ratingMatrix:            ' + bs.matrixMs.toFixed(1) + ' мс');
    console.log('  колонки + разрежённые списки:   ' + bs.derivedMs.toFixed(1) + ' мс');
    console.log('  loadData() целиком:             ' + loadMs.toFixed(1) + ' мс');

    const ambiguousTitles = Object.values(idsByTitle).filter(v => v.length > 1);
    console.log('\n  --- слияние дубликатов u.item ---');
    console.log('    id-копий (duplicateOf задан): ' + duplicateIds.size +
        '   исключены из кандидатов в обоих методах');
    console.log('    столбцов-копий, ставших пустыми: ' + duplicateIds.size +
        '   (ненулевых ячеек в матрице: ' +
        Array.from({ length: numUsers }, (_, i) => {
            const row = M[i + 1];
            let n = 0;
            for (let m = 1; m <= numMovies; m++) if (row[m] !== 0) n++;
            return n;
        }).reduce((a, b) => a + b, 0) + ')');
    console.log('    названий у 2 фильмов: ' + ambiguousTitles.length +
        '  -> это ровно те же группы копий, теперь с оценками в одном столбце');

    // 34 and 242 join 1, 7 and 19. Both have 20 ratings like 19, but neither
    // gets a full user-based list: 34 only clears the threshold for one movie
    // and 242 for none, which is what the rejected-candidate block below shows.
    const REPORT_USERS = [1, 7, minRatedId, 34, 242];
    for (const userId of REPORT_USERS) {
        const label = userId === minRatedId
            ? `user ${userId}  (наименьший id среди ${count20} пользователей ровно с 20 оценками)`
            : `user ${userId}`;

        const myRatedIds = new Set();
        for (let m = 1; m <= numMovies; m++) if (M[userId][m] !== 0) myRatedIds.add(m);

        console.log('\n' + '#'.repeat(72));
        console.log('# ' + label);
        console.log('#'.repeat(72));
        console.log('  число оценок пользователя: ' + perUser[userId] +
            '   (фильмов всего ' + numMovies + ', не оценено ' + (numMovies - perUser[userId]) + ')');

        const all = allNeighbours(userId);
        const nbrs = all.slice(0, 20);
        console.log('\n  --- 20 соседей (из ' + all.length + ' с sim > 0) ---');
        console.log('    ' + 'id'.padStart(4) + '   сходство     общих');
        for (const nb of nbrs) {
            console.log('    ' + String(nb.userId).padStart(4) + '   ' + f6(nb.sim) + '  ' + String(nb.common).padStart(5));
        }
        console.log('    из всех соседей с сходством ровно 1.000000: ' +
            all.filter(n => n.sim >= 0.999999).length);

        const t0 = process.hrtime.bigint();
        const ubReturned = vm.runInContext(`getUserBasedRecommendations(${userId})`, ctx);
        const ubMs = ms(t0);
        const ubMine = userBasedCandidates(userId, nbrs);

        // эталонный вариант user-based целиком, для сравнения времени
        const tRefUb = process.hrtime.bigint();
        ubReference(userId);
        const ubRefMs = ms(tRefUb);

        console.log('\n  --- Top-5 user-based ---');
        console.log('    время расчёта (script.js): ' + ubMs.toFixed(1) + ' мс');
        console.log('    кандидатов до порога: ' + ubMine.доПорога +
            '   после порога (evidence >= ' + THR + '): ' + ubMine.length +
            '   отсеяно: ' + (ubMine.доПорога - ubMine.length));
        console.log('    ' + 'название'.padEnd(38) + 'прогноз'.padStart(9) +
            'соседей'.padStart(9) + 'суммаВесов'.padStart(12) + 'оценок'.padStart(9) + '  id');
        for (const item of ubReturned) {
            const id = item.movieId;
            const cand = ubMine.find(c => c.movieId === id);
            console.log('    ' + short(item.displayTitle || item.title, 38).padEnd(38) + f4(item.score).padStart(9) +
                String(cand ? cand.voters : '?').padStart(9) +
                f2(cand ? cand.evidence : NaN).padStart(12) +
                String(perMovie[id] || 0).padStart(9) +
                '  ' + id);
        }

        // What the threshold actually removed, and whether the neighbour pool could
        // ever have cleared it: the ceiling is the sum of the evidence weights
        // of all 20 neighbours, so if that sum is below THR no candidate can
        // pass no matter how many of them rated the movie.
        const evidenceCeiling = nbrs.reduce((s, nb) => s + ev(nb.common), 0);
        console.log('\n  --- user-based: что отсеял порог ---');
        console.log('    сумма весов всех 20 соседей (потолок evidence): ' + f2(evidenceCeiling) +
            '   порог: ' + THR + '   ' +
            (evidenceCeiling < THR ? 'ПОТОЛКА НЕ ХВАТАЕТ: ни один кандидат не может пройти'
                                   : 'потолка хватает, но не каждый набор соседей дотягивает'));
        console.log('    три кандидата с наибольшей суммой весов, отсечённых по evidence:');
        console.log('    ' + 'название'.padEnd(38) + 'суммаВесов'.padStart(12) + 'соседей'.padStart(9) + 'evidence'.padStart(11));
        const rejected = ubMine.отсеяно || [];
        if (rejected.length === 0) {
            console.log('    (отсеянных кандидатов нет)');
        } else {
            for (const rej of rejected.slice(0, 3)) {
                console.log('    ' + short(titleById[rej.movieId], 38).padEnd(38) +
                    f2(rej.simSum).padStart(12) +
                    String(rej.voters).padStart(9) +
                    f2(rej.evidence).padStart(11) +
                    '  ' + rej.movieId);
            }
            console.log('    всего отсеяно по evidence: ' + rejected.length);
        }

        const t1 = process.hrtime.bigint();
        const ibReturned = vm.runInContext(`getItemBasedRecommendations(${userId})`, ctx);
        const ibMs = ms(t1);
        const ibMine = itemBasedCandidates(userId);

        // эталонный вариант item-based целиком, для сравнения времени
        const tRefIb = process.hrtime.bigint();
        ibReference(userId);
        const ibRefMs = ms(tRefIb);

        console.log('\n  --- Top-5 item-based ---');
        console.log('    время расчёта (script.js): ' + ibMs.toFixed(1) + ' мс');
        console.log('    кандидатов до порога: ' + ibMine.доПорога +
            '   после порога (evidence >= ' + THR + '): ' + ibMine.length +
            '   отсеяно: ' + (ibMine.доПорога - ibMine.length));
        console.log('    ' + 'название'.padEnd(38) + 'прогноз'.padStart(9) +
            'соседей'.padStart(9) + 'суммаВесов'.padStart(12) + 'оценок'.padStart(9) +
            'общих'.padStart(8) + '  id');
        for (const item of ibReturned) {
            const id = item.movieId;
            const cand = ibMine.find(c => c.movieId === id);
            // общие зрители с самым похожим оценённым фильмом пользователя
            let bestRated = -1, bestSim = -1, bestCommon = 0;
            for (let r = 1; r <= numMovies; r++) {
                if (M[userId][r] === 0) continue;
                const s = cos(cols[r], cols[id]);
                if (s > bestSim) { bestSim = s; bestRated = r; bestCommon = commonCount(cols[r], cols[id]); }
            }
            console.log('    ' + short(item.displayTitle || item.title, 38).padEnd(38) + f4(item.score).padStart(9) +
                String(cand ? cand.voters : '?').padStart(9) +
                f2(cand ? cand.evidence : NaN).padStart(12) +
                String(perMovie[id] || 0).padStart(9) +
                String(bestCommon).padStart(8) +
                '  ' + id +
                (bestRated > 0 ? '  <- ближайший оценённый id ' + bestRated : ''));
        }

        console.log('\n  --- пересечение Top-5 с фильмами, которые пользователь УЖЕ оценил ---');
        for (const [name, arr] of [['user-based', ubReturned], ['item-based', ibReturned]]) {
            const hits = [];
            for (const item of arr) {
                if (myRatedIds.has(item.movieId)) hits.push(titleById[item.movieId] + ' (id ' + item.movieId + ')');
            }
            console.log('    ' + name.padEnd(11) + ' совпадений: ' + hits.length +
                (hits.length ? '  -> ' + hits.join(', ') : '  (ожидаем 0: алгоритм пропускает оценённое)'));
        }

        const dupInList = [...ubReturned, ...ibReturned].filter(item => duplicateIds.has(item.movieId));
        console.log('\n  --- проверка на дубликаты ---');
        console.log('    id-копий в u.item: ' + duplicateIds.size +
            '   попало в Top-5: ' + dupInList.length +
            (dupInList.length ? '  -> СБОЙ: ' + dupInList.map(i => i.movieId).join(', ') : '  (ожидаем 0)'));

        console.log('\n  --- кандидаты с тем же прогнозом, что 5-е место ---');
        for (const [name, mine] of [['user-based', ubMine], ['item-based', ibMine]]) {
            const fifth = mine[4];
            if (!fifth) { console.log('    ' + name + ': кандидатов меньше пяти'); continue; }
            const tied = mine.filter(c => Math.abs(c.score - fifth.score) < 1e-12);
            console.log('    ' + name.padEnd(11) + ' кандидатов: ' + mine.length +
                '   5-е место = ' + f6(fifth.score) + ' (id ' + fifth.movieId + ')' +
                '   с тем же прогнозом: ' + tied.length);
        }

        // ---------------------------------------------------------------
        // (b) сверка с эталоном из пункта 1 + (c) время до/после
        // ---------------------------------------------------------------
        const base = BASELINE[userId];
        console.log('\n  --- сверка с эталоном (снят ДО оптимизации и ДО слияния дубликатов) ---');
        if (!base) {
            // Эталон снимался только для 1, 7 и 19. Для 34 и 242 сравнивать
            // не с чем — эталонных значений по ним просто не существует.
            console.log('    эталонных значений для этого пользователя нет (эталон снят для 1, 7 и 19),');
            console.log('    сверка не выполняется; выше показан текущий результат.');
        }
        for (const [meth, ru, list] of base
            ? [['userBased', 'user-based', ubReturned], ['itemBased', 'item-based', ibReturned]]
            : []) {
            const expect = base[meth];
            let changed = 0;
            console.log('    ' + ru + ':');
            console.log('      ' + '#'.padStart(2) + '  ' + 'прежний id'.padStart(10) + 'прежний'.padStart(10) +
                '  ' + 'новый id'.padStart(8) + 'новый'.padStart(10) + '   итог');
            for (let i = 0; i < expect.length; i++) {
                const [expId, expScore] = expect[i];
                const got = list[i];
                const gotId = got ? got.movieId : null;
                const gotScore = got ? got.score.toFixed(6) : '(нет)';
                let verdict;
                if (!got) verdict = 'ИСЧЕЗ из списка';
                else if (gotId !== expId) verdict = 'ИЗМЕНЁН id';
                else if (gotScore !== expScore) verdict = 'изменён прогноз';
                else verdict = 'без изменений';
                if (verdict !== 'без изменений') changed++;
                console.log('      ' + String(i + 1).padStart(2) + '  ' + String(expId).padStart(10) +
                    expScore.padStart(10) + '  ' + String(gotId === null ? '-' : gotId).padStart(8) +
                    gotScore.padStart(10) + '   ' + verdict);
            }
            console.log('      изменений: ' + changed + ' из ' + expect.length);
        }

        console.log('\n  --- время расчёта списка: до / после ---');
        console.log('    ' + 'список'.padEnd(12) + 'эталонный вариант'.padStart(20) +
            'новый вариант'.padStart(16) + 'разница'.padStart(14));
        const row = (name, before, after) => {
            const speed = before > 0 ? (before / after) : 0;
            console.log('    ' + name.padEnd(12) + (before.toFixed(1) + ' мс').padStart(20) +
                (after.toFixed(1) + ' мс').padStart(16) + ('x' + speed.toFixed(1)).padStart(14));
        };
        row('user-based', ubRefMs, ubMs);
        row('item-based', ibRefMs, ibMs);
    }
}

main().catch(e => { console.error('FATAL:', e); process.exit(1); });
