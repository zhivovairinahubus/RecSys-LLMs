/**
 * week5-checks/audit.js — independent audit of the HW5 dataset layer.
 *
 * Node-only, no external packages. Reads week5/u.data and week5/u.item and
 * re-implements the parsing and the chronological 80/10/10 split EXACTLY as
 * week5/data.js does, then reports dataset facts the page does not show.
 *
 * Run:  node week5-checks/audit.js
 * Writes: week5-checks/results/audit.txt (and prints the same text).
 */

"use strict";

const fs = require("fs");
const path = require("path");

const WEEK5 = path.join(__dirname, "..", "week5");
const RESULTS = path.join(__dirname, "results");
const U_DATA = path.join(WEEK5, "u.data");
const U_ITEM = path.join(WEEK5, "u.item");

// ---------------------------------------------------------------------------
// Parsing — mirrors week5/data.js verbatim
// ---------------------------------------------------------------------------

const genreNames = [
  "Action", "Adventure", "Animation", "Children's", "Comedy",
  "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir",
  "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
  "Thriller", "War", "Western",
];

function parseItemData(text) {
  const lines = text.split("\n");
  const movies = [];
  for (const line of lines) {
    if (line.trim() === "") continue;
    const fields = line.split("|");
    if (fields.length < 24) continue;
    const id = parseInt(fields[0]);
    const title = fields[1];
    const genreValues = fields.slice(6, 24).map((value) => parseInt(value));
    const genres = genreNames.filter((_, index) => genreValues[index] === 1);
    movies.push({ id, title, genres });
  }
  return movies;
}

function parseRatingData(text) {
  const lines = text.split("\n");
  const ratings = [];
  for (const line of lines) {
    if (line.trim() === "") continue;
    const fields = line.split("\t");
    if (fields.length < 4) continue;
    const userId = parseInt(fields[0]);
    const movieId = parseInt(fields[1]);
    const rating = parseFloat(fields[2]);
    const timestamp = parseInt(fields[3]);
    ratings.push({ userId, movieId, rating, timestamp });
  }
  return ratings;
}

// Mirrors splitByTimestamp() in week5/data.js verbatim (MIN_RATINGS = 5,
// nTrain = floor(0.8*n), nVal = floor(0.1*n), nTest = remainder, ties by movieId).
function splitByTimestamp(ratings) {
  const MIN_RATINGS = 5;

  const byUser = new Map();
  for (const r of ratings) {
    if (!byUser.has(r.userId)) byUser.set(r.userId, []);
    byUser.get(r.userId).push(r);
  }

  const train = [];
  const val = [];
  const test = [];
  const exclusions = [];

  for (const [userId, userRatings] of byUser) {
    const n = userRatings.length;
    if (n < MIN_RATINGS) {
      exclusions.push({ userId, count: n, reason: `fewer than ${MIN_RATINGS} ratings` });
      continue;
    }

    const sorted = userRatings
      .slice()
      .sort((a, b) => a.timestamp - b.timestamp || a.movieId - b.movieId);

    const nTrain = Math.floor(0.8 * n);
    const nVal = Math.floor(0.1 * n);

    for (let i = 0; i < n; i++) {
      if (i < nTrain) train.push(sorted[i]);
      else if (i < nTrain + nVal) val.push(sorted[i]);
      else test.push(sorted[i]);
    }
  }

  return { train, val, test, exclusions };
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const fmt = (x, d = 4) => (Number.isFinite(x) ? x.toFixed(d) : String(x));
const pct = (x, d = 4) => `${(x * 100).toFixed(d)}%`;
const num = (x) => x.toLocaleString("en-US");

function mean(xs) {
  return xs.reduce((a, b) => a + b, 0) / xs.length;
}

function rmseConstant(values, mu) {
  const sum = values.reduce((acc, v) => acc + (mu - v) * (mu - v), 0);
  return Math.sqrt(sum / values.length);
}

// ---------------------------------------------------------------------------
// Load
// ---------------------------------------------------------------------------

// u.data is ASCII; read as utf8 (identical to latin1 here).
const ratingsText = fs.readFileSync(U_DATA, "utf8");
const ratings = parseRatingData(ratingsText);

// u.item is ISO-8859-1 (Latin-1). Read the raw bytes and decode as latin1 for
// the CORRECT titles, and keep the utf8 decode to demonstrate the breakage.
const itemBuf = fs.readFileSync(U_ITEM);
const moviesTextLatin1 = itemBuf.toString("latin1");
const moviesTextUtf8 = itemBuf.toString("utf8");
const movies = parseItemData(moviesTextLatin1);

const numUsers = ratings.reduce((max, r) => Math.max(max, r.userId), 0);
const distinctUsers = new Set(ratings.map((r) => r.userId)).size;
const maxMovieId = movies.reduce((max, m) => Math.max(max, m.id), 0);
const distinctMoviesRated = new Set(ratings.map((r) => r.movieId)).size;

// ---------------------------------------------------------------------------
// Split
// ---------------------------------------------------------------------------

const { train, val, test, exclusions } = splitByTimestamp(ratings);

const trainRatings = train.map((r) => r.rating);
const valRatings = val.map((r) => r.rating);
const testRatings = test.map((r) => r.rating);

const mu = trainRatings.length ? mean(trainRatings) : 0;
const baselineTrain = trainRatings.length ? rmseConstant(trainRatings, mu) : 0;
const baselineVal = valRatings.length ? rmseConstant(valRatings, mu) : 0;
const baselineTest = testRatings.length ? rmseConstant(testRatings, mu) : 0;

// ---------------------------------------------------------------------------
// Movie coverage of the splits
// ---------------------------------------------------------------------------

const trainMovieIds = new Set(train.map((r) => r.movieId));
const moviesAbsentFromTrain = movies.filter((m) => !trainMovieIds.has(m.id));
const valOnMissing = val.filter((r) => !trainMovieIds.has(r.movieId));
const testOnMissing = test.filter((r) => !trainMovieIds.has(r.movieId));

// ---------------------------------------------------------------------------
// u.item encoding damage (Latin-1 read as UTF-8)
// ---------------------------------------------------------------------------

// Parse titles from the utf8-decoded text to show the mojibake title.
function titlesFrom(text) {
  const out = new Map();
  for (const line of text.split("\n")) {
    if (line.trim() === "") continue;
    const fields = line.split("|");
    if (fields.length < 24) continue;
    out.set(parseInt(fields[0]), fields[1]);
  }
  return out;
}
const titlesModern = titlesFrom(moviesTextLatin1); // correct
const titlesBroken = titlesFrom(moviesTextUtf8); // mojibake

const damaged = [];
for (const [id, title] of titlesModern) {
  if (/[^\x00-\x7F]/.test(title)) {
    damaged.push({ id, latin1: title, utf8: titlesBroken.get(id) });
  }
}
damaged.sort((a, b) => a.id - b.id);

// ---------------------------------------------------------------------------
// Duplicate titles under different ids
// ---------------------------------------------------------------------------

const byTitle = new Map();
for (const m of movies) {
  if (!byTitle.has(m.title)) byTitle.set(m.title, []);
  byTitle.get(m.title).push(m.id);
}
const duplicateTitles = [...byTitle.entries()]
  .filter(([, ids]) => ids.length > 1)
  .sort((a, b) => a[0].localeCompare(b[0]));

// ---------------------------------------------------------------------------
// Genres
// ---------------------------------------------------------------------------

const noGenreMovies = movies.filter((m) => m.genres.length === 0);

// Count genre flags in a raw u.item row (first well-formed line).
const firstItemLine = moviesTextLatin1.split("\n").find((l) => l.trim() !== "");
const firstFields = firstItemLine.split("|");
const totalFields = firstFields.length;
const genreFlagColumns = firstFields.slice(5, 24); // indices 5..23 = 19 columns incl. "unknown"
const realGenreFlags = firstFields.slice(6, 24); // indices 6..23 = 18 real flags (data.js)

// ---------------------------------------------------------------------------
// Report
// ---------------------------------------------------------------------------

const L = [];
const line = (s = "") => L.push(s);
const hr = () => line("=".repeat(78));
const sub = (s) => {
  line("");
  line("-".repeat(78));
  line(s);
  line("-".repeat(78));
};

hr();
line("HW5 dataset audit — week5/u.data + week5/u.item (logic mirrors week5/data.js)");
hr();

sub("1. Shapes");
line(`Users (distinct userIds in u.data) : ${num(distinctUsers)}`);
line(`Users (max userId, numUsers)       : ${num(numUsers)}`);
line(`Movies (parsed from u.item)        : ${num(movies.length)}`);
line(`Movies (distinct movieIds rated)   : ${num(distinctMoviesRated)}`);
line(`Movies (max movieId)               : ${num(maxMovieId)}`);
line(`Ratings                            : ${num(ratings.length)}`);

sub("2. Sparsity of the rating matrix");
const observed = ratings.length;
const cells943 = 943 * 1682;
const cells944 = 944 * 1683;
line(`Observed ratings                   : ${num(observed)}`);
line(`Matrix 943 x 1682 = ${num(cells943)} cells`);
line(`  density  = observed / cells      = ${pct(observed / cells943, 6)}`);
line(`  sparsity = 1 - density            = ${pct(1 - observed / cells943, 4)}`);
line(`Matrix 944 x 1683 = ${num(cells944)} cells   (the size shown on the page)`);
line(`  density  = observed / cells      = ${pct(observed / cells944, 6)}`);
line(`  sparsity = 1 - density            = ${pct(1 - observed / cells944, 4)}`);
line("");
line(`Formula: 1 - ${num(observed)} / (943 x 1682) = ${pct(1 - observed / cells943, 4)}`);

sub("3. Chronological 80/10/10 split");
line(`Train ratings : ${num(train.length)}  (${pct(train.length / ratings.length, 2)})`);
line(`Val ratings   : ${num(val.length)}  (${pct(val.length / ratings.length, 2)})`);
line(`Test ratings  : ${num(test.length)}  (${pct(test.length / ratings.length, 2)})`);
line(`Sum           : ${num(train.length + val.length + test.length)}`);
line("");
line(`Skipped users (fewer than 5 ratings): ${num(exclusions.length)}`);
for (const e of exclusions) {
  line(`  userId ${e.userId}: ${e.count} rating(s) — ${e.reason}`);
}

sub("4. Training mean and mean-rating baseline");
line(`mu (mean rating over TRAIN)        : ${fmt(mu, 6)}`);
line(`Baseline RMSE (predict mu) on TRAIN: ${fmt(baselineTrain, 6)}`);
line(`Baseline RMSE (predict mu) on VAL  : ${fmt(baselineVal, 6)}`);
line(`Baseline RMSE (predict mu) on TEST : ${fmt(baselineTest, 6)}`);

sub("5. Movies absent from the training split");
line(`Movies with no rating in TRAIN     : ${num(moviesAbsentFromTrain.length)}`);
line(`VAL ratings on those movies        : ${num(valOnMissing.length)}`);
line(`TEST ratings on those movies       : ${num(testOnMissing.length)}`);
if (moviesAbsentFromTrain.length) {
  line("");
  const shown = moviesAbsentFromTrain.slice(0, 20);
  for (const m of shown) line(`  #${m.id} ${m.title}`);
  if (moviesAbsentFromTrain.length > shown.length) {
    line(`  ... and ${num(moviesAbsentFromTrain.length - shown.length)} more`);
  }
}

sub("6. Titles that break when u.item is read as UTF-8 (file is Latin-1)");
line(`Titles with non-ASCII bytes        : ${num(damaged.length)}`);
if (damaged.length) {
  line("");
  for (const d of damaged) {
    line(`  #${d.id}`);
    line(`    Latin-1 (correct): ${d.latin1}`);
    line(`    UTF-8   (broken) : ${d.utf8}`);
  }
}

sub("7. Duplicate titles under different ids");
line(`Duplicate title groups             : ${num(duplicateTitles.length)}`);
if (duplicateTitles.length) {
  line("");
  for (const [title, ids] of duplicateTitles) {
    line(`  "${title}"  -> ids ${ids.join(", ")}`);
  }
}

sub("8. Genre flags and movies without genres");
line(`Fields in a u.item row (split by |): ${num(totalFields)}`);
line(`Genre-flag columns (indices 5..23) : ${num(genreFlagColumns.length)}  (incl. legacy "unknown")`);
line(`Real genre flags (indices 6..23)   : ${num(realGenreFlags.length)}  (used by data.js)`);
line("");
line(`Movies with zero genres             : ${num(noGenreMovies.length)}`);
if (noGenreMovies.length) {
  const shown = noGenreMovies.slice(0, 20);
  for (const m of shown) line(`  #${m.id} ${m.title}`);
  if (noGenreMovies.length > shown.length) {
    line(`  ... and ${num(noGenreMovies.length - shown.length)} more`);
  }
}

hr();

// ---------------------------------------------------------------------------
// Output
// ---------------------------------------------------------------------------

const text = L.join("\n") + "\n";
fs.mkdirSync(RESULTS, { recursive: true });
fs.writeFileSync(path.join(RESULTS, "audit.txt"), text);
process.stdout.write(text);
