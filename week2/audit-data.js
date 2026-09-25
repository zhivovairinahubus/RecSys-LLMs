// audit-data.js — read-only audit of MovieLens 100K u.item and u.data files.
// Usage: node audit-data.js <path-to-u.item> <path-to-u.data>
// No external dependencies. Files are read as latin1 (their real encoding).

const fs = require('fs');
const path = require('path');

const itemPath = process.argv[2];
const dataPath = process.argv[3];
if (!itemPath || !dataPath) {
    console.error('Usage: node audit-data.js <path-to-u.item> <path-to-u.data>');
    process.exit(1);
}

// Decode a latin1 string as if its bytes were UTF-8 (what a browser does
// when the page is declared UTF-8 but the file is latin1).
function toUtf8(latin1Str) {
    return Buffer.from(latin1Str, 'latin1').toString('utf8');
}

// Percent-decode an IMDb URL on raw latin1 bytes and strip "title-exact?" prefix.
function decodeUrlTitle(url) {
    const decoded = url.replace(/%([0-9A-Fa-f]{2})/g, (_, hex) => String.fromCharCode(parseInt(hex, 16)));
    const q = decoded.indexOf('?');
    return q >= 0 ? decoded.slice(q + 1) : decoded;
}

function median(sorted) {
    const n = sorted.length;
    if (n === 0) return 0;
    const mid = Math.floor(n / 2);
    return n % 2 === 1 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

const itemText = fs.readFileSync(itemPath, 'latin1');
const dataText = fs.readFileSync(dataPath, 'latin1');

const itemLines = itemText.split('\n').filter(l => l.trim() !== '');
const rows = itemLines.map((line, i) => {
    const f = line.split('|');
    return { row: i + 1, id: f[0], title: f[1], relDate: f[2], url: f[4] || '', flags: f.slice(5, 24) };
});

console.log('==== 1) u.item: line count and field-count distribution ====');
console.log('total data rows:', itemLines.length);
const fieldDist = {};
for (const ln of itemLines) {
    const n = ln.split('|').length;
    fieldDist[n] = (fieldDist[n] || 0) + 1;
}
for (const n of Object.keys(fieldDist).map(Number).sort((a, b) => a - b)) {
    console.log(`  ${n} fields: ${fieldDist[n]} rows`);
}

console.log('\n==== 2) only-unknown genre + rows without release date ====');
const onlyUnknown = rows.filter(r => r.flags[0] === '1' && r.flags.slice(1).every(v => v === '0'));
console.log('movies with ONLY the first (unknown) flag set:', onlyUnknown.length);
for (const r of onlyUnknown) console.log(`  row ${r.row}: id=${r.id} title=${JSON.stringify(r.title)}`);
const noDate = rows.filter(r => r.relDate === undefined || r.relDate.trim() === '');
console.log('rows without release date:', noDate.length);
for (const r of noDate) console.log(`  row ${r.row}: id=${r.id} title=${JSON.stringify(r.title)}`);

console.log('\n==== 3) titles occurring more than once ====');
const byTitle = {};
for (const r of rows) (byTitle[r.title] = byTitle[r.title] || []).push(r.id);
const dups = Object.entries(byTitle).filter(([, ids]) => ids.length > 1);
console.log('duplicate titles:', dups.length);
for (const [t, ids] of dups) console.log(`  ${JSON.stringify(t)} -> ids: ${ids.join(', ')}`);

console.log('\n==== 4) rows with non-ASCII bytes (latin1 vs utf8 reading) ====');
const nonAscii = rows.filter(r => [...r.title].some(ch => ch.charCodeAt(0) > 127));
console.log('rows with non-ASCII title bytes:', nonAscii.length);
for (const r of nonAscii) {
    console.log(`  row ${r.row} (id=${r.id}):`);
    console.log(`    latin1: ${r.title}`);
    console.log(`    utf8  : ${toUtf8(r.title)}`);
}

console.log('\n==== 5) titles ending with an article (explicit list) before the year ====');
const articleList = ['The', 'A', 'An', 'La', 'Le', 'Les', "L'", 'Il', 'Das', 'Der', 'Die', 'El', 'Los'];
const escapeRe = s => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
const articleCount = {};
const articleTitles = {};
for (const r of rows) {
    const t = r.title.trim();
    for (const a of articleList) {
        // article must sit right after the LAST comma, directly before "(year)"
        const re = new RegExp(`,\\s*${escapeRe(a)}\\s*\\(\\d{4}\\)$`, 'i');
        if (re.test(t)) {
            articleCount[a] = (articleCount[a] || 0) + 1;
            (articleTitles[a] = articleTitles[a] || []).push(t);
            break; // one article per title
        }
    }
}
console.log('distinct articles matched:', Object.keys(articleCount).length);
for (const a of articleList) {
    const n = articleCount[a] || 0;
    const examples = (articleTitles[a] || []).slice(0, 5);
    console.log(`  ${a}: ${n}${examples.length ? `   e.g. ${examples.map(e => JSON.stringify(e)).join(', ')}` : ''}`);
}

// Delta vs the previous flawed run: old rule counted the segment from the FIRST comma,
// so it only labeled "The" when the whole segment was exactly "The".
const oldArticleRe = /,\s+([^()]+?)\s*\(\d{4}\)\s*$/;
function oldLabel(t) {
    const m = oldArticleRe.exec(t.trim());
    return m ? m[1].trim() : null;
}
const addedThe = (articleTitles['The'] || []).filter(t => oldLabel(t) !== 'The');
console.log(`  -> "The": titles NOT counted as "The" by the old first-comma logic: ${addedThe.length}`);
for (const t of addedThe) console.log(`      added: ${JSON.stringify(t)}`);

console.log('\n==== 6) genre count per movie (ignoring unknown) ====');
const genreDist = {};
for (const r of rows) {
    const cnt = r.flags.slice(1).filter(v => v === '1').length;
    genreDist[cnt] = (genreDist[cnt] || 0) + 1;
}
for (const n of Object.keys(genreDist).map(Number).sort((a, b) => a - b)) {
    console.log(`  ${n} genre(s): ${genreDist[n]} movies`);
}

console.log('\n==== 7) u.data: ratings per movie (min / median / max) ====');
const dataLines = dataText.split('\n').filter(l => l.trim() !== '');
const perMovie = {};
for (const ln of dataLines) {
    const f = ln.split('\t');
    perMovie[f[1]] = (perMovie[f[1]] || 0) + 1;
}
const counts = Object.values(perMovie).sort((a, b) => a - b);
console.log(`total ratings: ${dataLines.length}`);
console.log(`distinct movies rated: ${counts.length}`);
if (counts.length) {
    console.log(`min:    ${counts[0]}`);
    console.log(`median: ${median(counts)}`);
    console.log(`max:    ${counts[counts.length - 1]}`);
}

console.log('\n==== 8) titles containing "*" and their title decoded from IMDb URL ====');
const starred = rows.filter(r => r.title.includes('*'));
console.log('titles with "*":', starred.length);
for (const r of starred) {
    const decoded = decodeUrlTitle(r.url).replace(/\s*$/, '');
    console.log(`  row ${r.row} (id=${r.id}) title=${JSON.stringify(r.title)}`);
    console.log(`    imdb url: ${r.url}`);
    console.log(`    decoded : ${decoded}`);
}