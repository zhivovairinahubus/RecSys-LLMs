#!/usr/bin/env node
'use strict';

const fs = require('fs');

const itemPath = process.argv[2];
const dataPath = process.argv[3];

if (!itemPath || !dataPath) {
    console.error('Использование: node audit-data.js <путь-к-u.item> <путь-к-u.data>');
    process.exit(1);
}

const LF = 0x0a;
const PIPE = 0x7c;
const TAB = 0x09;
const ASCII_LIMIT = 0x80;

function section(n, title) {
    console.log('\n' + '='.repeat(64));
    console.log(n + '. ' + title);
    console.log('='.repeat(64));
}

function median(values) {
    if (values.length === 0) return null;
    const s = [...values].sort((a, b) => a - b);
    const mid = Math.floor(s.length / 2);
    return s.length % 2 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
}

function stats(values) {
    if (values.length === 0) return { min: null, max: null, med: null };
    let min = Infinity, max = -Infinity;
    for (const v of values) {
        if (v < min) min = v;
        if (v > max) max = v;
    }
    return { min, max, med: median(values) };
}

const itemBuf = fs.readFileSync(itemPath);
const dataBuf = fs.readFileSync(dataPath);

const itemLines = itemBuf.toString('binary').split('\n');
const dataLines = dataBuf.toString('binary').split('\n');

const itemNonEmpty = itemLines.filter(l => l.trim() !== '');
const dataNonEmpty = dataLines.filter(l => l.trim() !== '');

section(1, 'u.item: строки и распределение числа полей');
console.log('  всего частей после split(\\n):      ' + itemLines.length);
console.log('  непустых строк:                    ' + itemNonEmpty.length);
console.log('  завершается символом LF:           ' + (itemBuf[itemBuf.length - 1] === LF));

const itemFieldCounts = {};
for (const l of itemNonEmpty) {
    const n = l.split('|').length;
    itemFieldCounts[n] = (itemFieldCounts[n] || 0) + 1;
}
console.log('  распределение числа полей (после split по "|", непустые строки):');
for (const k of Object.keys(itemFieldCounts).map(Number).sort((a, b) => a - b)) {
    console.log('    ' + String(k).padStart(3) + ' полей -> ' + itemFieldCounts[k] + ' строк');
}
console.log('  строк с числом полей != 24:        ' +
    (itemNonEmpty.length - (itemFieldCounts[24] || 0)));

section(2, 'u.data: объём данных');
const perUser = new Map();
const perMovie = new Map();
const pairCounts = new Map();
const ratingHist = new Map();
let outOfRange = 0;
let badFieldLines = 0;
let maxUserId = 0;
let maxMovieId = 0;
let ratingCount = 0;

for (const line of dataNonEmpty) {
    const f = line.split('\t');
    if (f.length !== 4) badFieldLines++;

    const userId = parseInt(f[0], 10);
    const itemId = parseInt(f[1], 10);
    const rating = parseFloat(f[2]);

    if (Number.isNaN(userId) || Number.isNaN(itemId) || Number.isNaN(rating)) continue;
    ratingCount++;

    if (userId > maxUserId) maxUserId = userId;
    if (itemId > maxMovieId) maxMovieId = itemId;

    ratingHist.set(rating, (ratingHist.get(rating) || 0) + 1);
    if (rating < 1 || rating > 5) outOfRange++;

    perUser.set(userId, (perUser.get(userId) || 0) + 1);
    perMovie.set(itemId, (perMovie.get(itemId) || 0) + 1);

    const key = userId + ':' + itemId;
    pairCounts.set(key, (pairCounts.get(key) || 0) + 1);
}

console.log('  всего частей после split(\\n):      ' + dataLines.length);
console.log('  непустых строк (оценок):           ' + dataNonEmpty.length);
console.log('  разных пользователей:              ' + perUser.size);
console.log('  разных фильмов:                    ' + perMovie.size);
console.log('  максимальный id пользователя:      ' + maxUserId);
console.log('  максимальный id фильма:             ' + maxMovieId);

section(3, 'Распределение оценок');
for (const r of [1, 2, 3, 4, 5]) {
    const c = ratingHist.get(r) || 0;
    console.log('    оценка ' + r + ': ' + String(c).padStart(7) + '  (' +
        (ratingCount ? (c / ratingCount * 100).toFixed(2) : '0.00') + '%)');
}
console.log('  значений вне диапазона 1..5:        ' + outOfRange);
const other = [...ratingHist.keys()].filter(r => r < 1 || r > 5);
console.log('  ненайденные значения оценки:       ' + (other.length ? other.join(', ') : 'нет'));

section(4, 'Дубли и целостность строк');
let dupPairs = 0, extraRows = 0;
for (const c of pairCounts.values()) {
    if (c > 1) { dupPairs++; extraRows += c - 1; }
}
console.log('  повторяющихся пар (пользователь,фильм): ' + dupPairs);
console.log('  лишних строк сверх одной на пару:        ' + extraRows);
console.log('  строк u.data с числом полей != 4:        ' + badFieldLines);

section(5, 'Плотность матрицы пользователь x фильм');
const distinctCells = perUser.size * perMovie.size;
const shapeCells = (maxUserId + 1) * (maxMovieId + 1);
console.log('  заполнено ячеек (уникальных пар): ' + pairCounts.size);
console.log('  размер по разным пользователям x фильмам: ' + perUser.size + ' x ' + perMovie.size +
    ' = ' + distinctCells);
console.log('      плотность: ' + (pairCounts.size / distinctCells * 100).toFixed(4) + '%');
console.log('  размер по форме из data.js (maxId+1 x maxId+1): ' + (maxUserId + 1) + ' x ' +
    (maxMovieId + 1) + ' = ' + shapeCells);
console.log('      плотность: ' + (pairCounts.size / shapeCells * 100).toFixed(4) + '%');
console.log('  свободных ячеек (по форме data.js): ' + (shapeCells - pairCounts.size));

section(6, 'Оценок на пользователя и на фильм');
const perUserVals = [...perUser.values()];
const perMovieVals = [...perMovie.values()];
const u = stats(perUserVals);
const m = stats(perMovieVals);
console.log('  на пользователя:  min=' + u.min + '  медиана=' + u.med + '  max=' + u.max +
    '  (пользователей: ' + perUserVals.length + ')');
console.log('  на фильм:         min=' + m.min + '  медиана=' + m.med + '  max=' + m.max +
    '  (фильмов: ' + perMovieVals.length + ')');
let single = 0;
for (const v of perMovieVals) if (v === 1) single++;
console.log('  фильмов ровно с одной оценкой:  ' + single);
console.log('  фильмов с 1..4 оценками:        ' + perMovieVals.filter(v => v <= 4).length);
let singleUser = 0;
for (const v of perUserVals) if (v === 1) singleUser++;
console.log('  пользователей ровно с одной оценкой: ' + singleUser);

section(7, 'u.item: строки с байтами вне ASCII');
const bad = [];
{
    let start = 0;
    for (let i = 0; i <= itemBuf.length; i++) {
        if (i === itemBuf.length || itemBuf[i] === LF) {
            const lineBuf = itemBuf.slice(start, i);
            start = i + 1;
            let hasHigh = false;
            for (const b of lineBuf) if (b >= ASCII_LIMIT) { hasHigh = true; break; }
            if (!hasHigh) continue;
            const parts = [];
            let fs = 0, cur = [];
            for (let j = 0; j < lineBuf.length; j++) {
                if (lineBuf[j] === PIPE) { parts.push(Buffer.from(cur)); cur = []; } else cur.push(lineBuf[j]);
            }
            parts.push(Buffer.from(cur));
            bad.push({ id: parts[0].toString('latin1'), titleBuf: parts[1] || Buffer.alloc(0) });
        }
    }
}
    console.log('  строк с байтом >= 0x80: ' + bad.length);
for (const r of bad) {
    const high = [...r.titleBuf].filter(b => b >= ASCII_LIMIT).length;
    const utf8 = r.titleBuf.toString('utf8');
    const replaced = [...utf8].filter(c => c.charCodeAt(0) === 0xfffd).length;
    console.log('    id=' + r.id + '   байтов >= 0x80: ' + high +
        '   заменено на U+FFFD: ' + replaced);
    console.log('        latin1: ' + r.titleBuf.toString('latin1'));
    console.log('        utf8:   ' + utf8.replace(/�/g, '<U+FFFD>'));
}
console.log('');

section(8, 'u.item: группы строк, совпадающих во всех полях кроме id');

// Ключ группы — все поля строки, кроме первого (id). Порядок строк в u.item
// сохраняем, поэтому «первый» экземпляр группы это первая встретившаяся строка.
const itemGroups = new Map();
for (const line of itemNonEmpty) {
    const f = line.split('|');
    const key = f.slice(1).join('|');
    if (!itemGroups.has(key)) itemGroups.set(key, []);
    itemGroups.get(key).push({ id: parseInt(f[0], 10), title: f[1] });
}
const dupGroups = [...itemGroups.entries()]
    .filter(([, list]) => list.length > 1)
    .sort((a, b) => a[1][0].id - b[1][0].id);

// Оценка пользователя на конкретный id — нужна, чтобы сравнить оценки двух
// копий у тех, кто оценил обе.
const ratingByUserMovie = new Map();
for (const line of dataNonEmpty) {
    const f = line.split('\t');
    const userId = parseInt(f[0], 10);
    const itemId = parseInt(f[1], 10);
    const rating = parseFloat(f[2]);
    if (Number.isNaN(userId) || Number.isNaN(itemId) || Number.isNaN(rating)) continue;
    ratingByUserMovie.set(userId + ':' + itemId, rating);
}
const ratingsOf = (userId, itemId) => ratingByUserMovie.get(userId + ':' + itemId);

console.log('  всего групп-дубликатов (строки совпадают во всех полях кроме id): ' + dupGroups.length);
console.log('  строк в u.item: ' + itemNonEmpty.length + ',   уникальных групп: ' + itemGroups.size);
console.log('  экземпляров всего в группах-дубликатах: ' +
    dupGroups.reduce((s, [, l]) => s + l.length, 0));

console.log('');
console.log('  ' + 'id'.padEnd(14) + 'название'.padEnd(38) + 'оценок на id'.padStart(12) +
    'на копиях'.padStart(11) + 'оценили обе'.padStart(13) + 'одинаковые'.padStart(12) + 'разные'.padStart(8));

let totalOnCopies = 0;
let totalBoth = 0, totalSame = 0, totalDiff = 0;
for (const [, list] of dupGroups) {
    const counts = list.map(m => perMovie.get(m.id) || 0);
    const onCopies = counts.slice(1).reduce((a, b) => a + b, 0);

    let both = 0, same = 0, diff = 0;
    for (let i = 0; i < list.length; i++) {
        for (let j = i + 1; j < list.length; j++) {
            for (let userId = 1; userId <= maxUserId; userId++) {
                const a = ratingsOf(userId, list[i].id);
                const b = ratingsOf(userId, list[j].id);
                if (a === undefined || b === undefined) continue;
                both++;
                if (a === b) same++; else diff++;
            }
        }
    }

    totalOnCopies += onCopies;
    totalBoth += both; totalSame += same; totalDiff += diff;

    console.log('  ' + list.map(m => m.id).join(', ').padEnd(14) +
        (list[0].title.length > 36 ? list[0].title.slice(0, 35) + '…' : list[0].title).padEnd(38) +
        counts.map(c => String(c).padStart(8)).join('') +
        String(onCopies).padStart(11) +
        String(both).padStart(13) + String(same).padStart(12) + String(diff).padStart(8));
}

console.log('');
console.log('  ИТОГО оценок на id-копиях (кроме первого экземпляра): ' + totalOnCopies);
console.log('  ИТОГО оценок на всех id группы:                    ' +
    dupGroups.reduce((s, [, l]) => s + l.reduce((x, m) => x + (perMovie.get(m.id) || 0), 0), 0));
console.log('  ИТОГО пользователей оценили обе копии:              ' + totalBoth);
console.log('      из них с одинаковой оценкой:                     ' + totalSame +
    '  (' + (totalBoth ? (totalSame / totalBoth * 100).toFixed(2) : '0.00') + '%)');
console.log('      из них с разной оценкой:                        ' + totalDiff +
    '  (' + (totalBoth ? (totalDiff / totalBoth * 100).toFixed(2) : '0.00') + '%)');
console.log('  при слиянии в столбец первого экземпляра потеряется ячеек: ' + totalBoth +
    '  (ненулевых ячеек станет ' + (pairCounts.size - totalBoth) + ' вместо ' + pairCounts.size + ')');
console.log('');
