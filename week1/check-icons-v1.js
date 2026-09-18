const fs = require('fs');

const [htmlPath, cssPath] = process.argv.slice(2);

if (!htmlPath || !cssPath) {
    console.error('Usage: node check-icons-v1.js <html-path> <css-path>');
    process.exit(2);
}

const html = fs.readFileSync(htmlPath, 'utf8');
const css = fs.readFileSync(cssPath, 'utf8');

const serviceTokens = new Set([
    'fa-solid', 'fa-regular', 'fa-brands', 'fa-classic', 'fa-sharp', 'fa-duotone',
    'fa-spin', 'fa-pulse', 'fa-fw', 'fa-border', 'fa-2x', 'fa-3x', 'fa-lg', 'fa-sm',
    'fa-xs', 'fa-beat', 'fa-fade', 'fa-flip', 'fa-shake', 'fa-bounce', 'fa-inverse'
]);

const tokens = [];
const seen = new Set();

for (const match of html.matchAll(/fa-[a-z0-9]+(?:-[a-z0-9]+)*/g)) {
    const token = match[0].toLowerCase();
    if (!serviceTokens.has(token) && !seen.has(token)) {
        seen.add(token);
        tokens.push(token);
    }
}

const missing = [];
let resolved = 0;

for (const token of tokens) {
    if (css.includes(`.${token}:before`)) {
        resolved++;
        console.log(`${token}: resolved`);
    } else {
        missing.push(token);
        console.log(`${token}: missing`);
    }
}

const total = tokens.length;
const percent = total === 0 ? '0.00' : ((resolved / total) * 100).toFixed(2);
console.log(`resolved ${resolved}/${total} (${percent}%)`);

if (missing.length > 0) {
    console.log(`MISSING: ${missing.join(', ')}`);
    process.exit(1);
}