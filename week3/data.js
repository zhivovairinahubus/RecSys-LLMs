// Global variables for storing movie and rating data
let movies = [];
let ratings = [];

// Collaborative filtering structures (populated by buildRatingMatrix)
let numUsers = 0;          // highest user id found in u.data
let numMovies = 0;         // number of parsed movies
let ratingMatrix = null;   // (numUsers + 1) x (numMovies + 1); 0 = "not rated"

// Derived lookup structures, built once by buildRatingMatrix so that the
// recommendation functions never rebuild them per click.
//
// ratingColumns[movieId] is the film vector (a column) as a plain array of
// length numUsers + 1, i.e. the same thing script.js used to recompute on
// every call to getItemBasedRecommendations.
//
// userRatings[userId] and movieRatings[movieId] are flat sparse lists
// [id1, rating1, id2, rating2, ...] holding only the non-zero entries, sorted
// by ascending id. Iterating one of these visits only the ratings that
// actually exist, which is what makes the one-pass cosine cheap.
let ratingColumns = null;
let userRatings = null;
let movieRatings = null;

// Number of values in the sparse structures, filled in by buildRatingMatrix.
let lastBuildStats = null;

// u.item contains 18 films written twice: two rows identical in every field
// except the id. The second and every further instance of such a group carries
// duplicateOf = the id of the first instance, and null for a normal film.
//
// canonicalMovieId[movieId] is the id whose COLUMN holds this film's ratings:
// the id itself for a normal film, the first instance's id for a copy. Copies
// keep their own column, but that column stays zero. It is a plain array
// because both recommendation functions read it once per candidate.
let canonicalMovieId = null;
let duplicateMovieIds = [];

// u.item also holds one row that describes no film at all: id 267 carries the
// title "unknown" and nothing else useful - no release date, no video link, no
// IMDb URL and no genre set (the only genre bit set is the "unknown" one).
// Such a row is marked invalid, and an invalid film is never treated as a film:
//
//   - its ratings are not written into ratingMatrix at all,
//   - it never becomes a candidate in either recommendation method,
//   - it never appears in the "Because you liked ..." line,
//   - it is left out of every per-film statistic in the check scripts.
//
// invalidMovieIds lists the ids, and invalidMovieId[movieId] is the O(1) lookup.
let invalidMovieIds = [];
let invalidMovieId = null;

// O(1) access to the parsed record of a film, filled in by buildDuplicateLookup.
let movieById = null;

// Number of ratings that buildRatingMatrix refused to write because their film
// is invalid. Reported through console.info so the number is visible.
let lastSkippedInvalidRatings = 0;

// Leading articles that are written after the last comma of a title, right
// before the year, and belong at the front of the title when it is displayed.
const TRAILING_ARTICLES = [
    'The', 'A', 'An', 'La', 'Le', 'Les', "L'", 'Il',
    'Das', 'Der', 'Die', 'El', 'Los'
];

// Titles whose display form is fixed by hand because the u.item title itself
// cannot be fixed by moving an article: id 1609 is spelled "B*A*P*S (1997)" in
// the title column, while the IMDb URL of that same row spells the real title
// "B.A.P.S. (1997)". The task asks for "B.A.P.S (1997)" - without the final
// dot. Keyed by movie id, because the title alone would also match nothing else.
const DISPLAY_TITLE_OVERRIDES = {
    1609: 'B.A.P.S (1997)',
};

// Genre names as defined in the u.item file
const genreNames = [
    "unknown", "Action", "Adventure", "Animation", "Children's",
    "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
    "Film-Noir", "Horror", "Musical", "Mystery", "Romance",
    "Sci-Fi", "Thriller", "War", "Western"
];

// Primary function to load data from files
async function loadData() {
    try {
        // Load and parse movie data
        const moviesResponse = await fetch('u.item');
        if (!moviesResponse.ok) {
            throw new Error(`Failed to load movie data: ${moviesResponse.status}`);
        }
        // u.item is single-byte latin1, not UTF-8. Decoding it as UTF-8
        // (what response.text() does) turns every accented letter into
        // U+FFFD, so read the bytes and decode them explicitly.
        const moviesBuffer = await moviesResponse.arrayBuffer();
        const moviesText = new TextDecoder('iso-8859-1').decode(moviesBuffer);
        parseItemData(moviesText);

        // Load and parse rating data
        const ratingsResponse = await fetch('u.data');
        if (!ratingsResponse.ok) {
            throw new Error(`Failed to load rating data: ${ratingsResponse.status}`);
        }
        const ratingsText = await ratingsResponse.text();
        parseRatingData(ratingsText);

        // Derive matrix dimensions, then build the rating matrix
        numUsers = ratings.reduce((max, r) => Math.max(max, r.userId), 0);
        numMovies = movies.length;
        buildRatingMatrix();
    } catch (error) {
        console.error('Error loading data:', error);
        const errorHtml = `<p class="error">Error: ${error.message}. Please make sure u.item and u.data are in the correct location.</p>`;
        for (const targetId of ['user-based-result', 'item-based-result']) {
            const target = document.getElementById(targetId);
            if (target) {
                target.innerHTML = errorHtml;
            }
        }
        throw error; // Re-throw so script.js can handle the error
    }
}

// Parse movie data from u.item format
function parseItemData(text) {
    const lines = text.split('\n');

    // Everything except the id decides whether two rows describe one film.
    // First occurrence in file order is the original, the rest are copies.
    const firstIdBySignature = new Map();

    for (const line of lines) {
        if (line.trim() === '') continue;

        const fields = line.split('|');
        if (fields.length < 5) continue; // Skip invalid lines

        const id = parseInt(fields[0]);
        const title = fields[1];
        const releaseDate = fields[2] || '';
        const videoLink = fields[3] || '';
        const imdbUrl = fields[4] || '';

        // Extract genres (last 19 fields)
        const genreValues = fields.slice(5, 24).map(value => parseInt(value));
        const genres = genreNames.filter((_, index) => genreValues[index] === 1);

        // A row is invalid when it says nothing about a film: the placeholder
        // title "unknown", no release date, no video link and no IMDb URL. Every
        // one of those has to fail at once, so a real film that merely happens
        // to lack one of the columns is still kept.
        //
        // Note: id 267 is NOT genre-empty — its first genre flag is 1 — so a
        // genre condition cannot be part of this test without hiding the row.
        const isInvalid =
            title.trim().toLowerCase() === 'unknown' &&
            releaseDate === '' &&
            videoLink === '' &&
            imdbUrl === '';

        const signature = fields.slice(1).join('|');
        let duplicateOf = null;
        if (firstIdBySignature.has(signature)) {
            duplicateOf = firstIdBySignature.get(signature);
        } else {
            firstIdBySignature.set(signature, id);
        }

        movies.push({
            id,
            title,
            displayTitle: buildDisplayTitle(id, title),
            genres,
            duplicateOf,
            isInvalid,
        });
    }
}

// ---------------------------------------------------------------------------
// Display title.
//
// MovieLens writes an English article after the last comma, right before the
// year: "Godfather, The (1972)". For display it belongs at the front, so
// "The Godfather (1972)". The stored `title` is never modified.
//
// Only a trailing word that is one of TRAILING_ARTICLES, and that sits
// immediately before the year, is moved. That keeps titles such as
// "Paris, Texas (1984)" (the tail is a place, not an article) and
// "Mrs. Brown (Her Majesty, Mrs. Brown) (1997)" (the last comma is inside the
// parentheses and is followed by more words) untouched, and it leaves titles
// with no comma at all alone - "M*A*S*H (1970)" needs no change.
// ---------------------------------------------------------------------------
function buildDisplayTitle(movieId, title) {
    if (Object.prototype.hasOwnProperty.call(DISPLAY_TITLE_OVERRIDES, movieId)) {
        return DISPLAY_TITLE_OVERRIDES[movieId];
    }

    // "Head, Article (Year)" at the very end of the string. The tail may not
    // contain a comma (it is what follows the LAST one) nor a parenthesis (it
    // is a single word sitting immediately before the year).
    const match = /^(.*),\s*([^,()]*?)\s*\((\d{4})\)$/.exec(title);
    if (match === null) return title;

    const head = match[1].trim();
    const tail = match[2].trim();
    const year = match[3];

    if (!TRAILING_ARTICLES.includes(tail)) return title;
    if (head === '') return title;

    return `${tail} ${head} (${year})`;
}

// Parse rating data from u.data format
function parseRatingData(text) {
    const lines = text.split('\n');

    for (const line of lines) {
        if (line.trim() === '') continue;

        const fields = line.split('\t');
        if (fields.length < 4) continue; // Skip invalid lines

        const userId = parseInt(fields[0]);
        const itemId = parseInt(fields[1]);
        const rating = parseFloat(fields[2]);
        const timestamp = parseInt(fields[3]);

        ratings.push({ userId, itemId, rating, timestamp });
    }
}

// ---------------------------------------------------------------------------
// Build the user-item rating matrix.
//
// Shape: (numUsers + 1) x (numMovies + 1). ORIENTATION: ROWS ARE USERS,
// COLUMNS ARE FILMS. That is, ratingMatrix[userId][movieId] holds the rating
// user `userId` gave to movie `movieId`.
//
// So a user vector is a ROW (ratingMatrix[userId], length numMovies + 1)
// and a film vector is a COLUMN (ratingMatrix[0..numUsers][movieId], length
// numUsers + 1).
//
// A missing entry is 0. MovieLens ratings are 1-5, so 0 is unambiguous.
// Row 0 and column 0 are padding only: they are never written to and stay 0.
//
// If you adopt a different missing-value convention (for example mean
// imputation, which week3/readme.md section 6 allows), keep
// cosineSimilarity in script.js consistent with it.
// ---------------------------------------------------------------------------
function buildRatingMatrix() {
    const buildStart = now();
    buildDuplicateLookup();

    ratingMatrix = [];
    for (let userId = 0; userId <= numUsers; userId++) {
        ratingMatrix.push(new Array(numMovies + 1).fill(0));
    }

    // A duplicate copy and the film it copies are ONE film, so a rating sitting
    // on a copy is written into the column of the first instance and the copy's
    // own column stays zero. When a user rated both copies the cell would be
    // written twice, so the later timestamp wins and the other row is dropped.
    const timestampOfCell = new Map();
    let ratingsOnCopies = 0;
    let mergedPairs = 0;
    let mergedRatings = 0;
    let skipped = 0;
    let skippedInvalid = 0;
    for (const { userId, itemId, rating, timestamp } of ratings) {
        if (userId < 1 || userId > numUsers || itemId < 1 || itemId > numMovies) {
            skipped++;
            continue;
        }

        // A row that describes no film has no ratings to place: they are counted
        // and dropped rather than written into a column nobody may recommend.
        if (invalidMovieId[itemId]) {
            skippedInvalid++;
            continue;
        }

        const targetId = canonicalMovieId[itemId];
        if (targetId !== itemId) ratingsOnCopies++;

        const cellKey = userId * (numMovies + 1) + targetId;
        const previousTimestamp = timestampOfCell.get(cellKey);
        if (previousTimestamp !== undefined) {
            mergedPairs++;
            mergedRatings++;
            if (timestamp <= previousTimestamp) continue;   // the later one wins
        }

        ratingMatrix[userId][targetId] = rating;
        timestampOfCell.set(cellKey, timestamp);
    }

    lastSkippedInvalidRatings = skippedInvalid;

    if (duplicateMovieIds.length > 0) {
        console.info(
            `buildRatingMatrix: ${duplicateMovieIds.length} film(s) in u.item are duplicate copies; ` +
            `${ratingsOnCopies} rating(s) moved into the first instance's column, ` +
            `${mergedPairs} pair(s) merged, ${mergedRatings} rating(s) merged away ` +
            `(later timestamp wins)`
        );
    }

    if (invalidMovieIds.length > 0) {
        console.info(
            `buildRatingMatrix: ${invalidMovieIds.length} invalid film row(s) in u.item ` +
            `(${invalidMovieIds.join(', ')}); ${skippedInvalid} rating(s) on them were not written ` +
            `into the matrix and they are excluded from both recommendation methods`
        );
    }

    if (skipped > 0) {
        console.warn(
            `buildRatingMatrix: skipped ${skipped} rating(s) with ids outside ` +
            `1..${numUsers} (users) / 1..${numMovies} (films)`
        );
    }

    const derivedStart = now();
    buildDerivedStructures();
    lastBuildStats = {
        matrixMs: derivedStart - buildStart,
        derivedMs: now() - derivedStart,
    };
}

// Fill canonicalMovieId from the duplicateOf markers set by parseItemData, and
// collect the ids of the rows that describe no film.
function buildDuplicateLookup() {
    canonicalMovieId = new Array(numMovies + 1).fill(0);
    duplicateMovieIds = [];
    invalidMovieIds = [];
    invalidMovieId = new Array(numMovies + 1).fill(false);
    movieById = new Array(numMovies + 1).fill(null);
    for (const movie of movies) {
        movieById[movie.id] = movie;
        canonicalMovieId[movie.id] = movie.duplicateOf === null ? movie.id : movie.duplicateOf;
        if (movie.duplicateOf !== null) duplicateMovieIds.push(movie.id);
        if (movie.isInvalid) {
            invalidMovieIds.push(movie.id);
            invalidMovieId[movie.id] = true;
        }
    }
}

// True when the film may be recommended, counted as a film, or listed as one of
// the user's favourites. Both a duplicate copy and an invalid row are excluded.
// Kept in data.js so that the page and the check scripts all agree on which
// films count.
function isRealFilm(movieId) {
    return !invalidMovieId[movieId] && canonicalMovieId[movieId] === movieId;
}

// Build the column view and the two sparse views. Called once, at load time.
function buildDerivedStructures() {
    // Film vectors, i.e. the columns, materialised as plain arrays.
    ratingColumns = new Array(numMovies + 1);
    for (let movieId = 0; movieId <= numMovies; movieId++) {
        ratingColumns[movieId] = new Array(numUsers + 1).fill(0);
    }
    for (let userId = 1; userId <= numUsers; userId++) {
        const row = ratingMatrix[userId];
        for (let movieId = 1; movieId <= numMovies; movieId++) {
            if (row[movieId] !== 0) ratingColumns[movieId][userId] = row[movieId];
        }
    }

    // Sparse [id, rating, id, rating, ...] lists in ascending id order, so a
    // single pass visits exactly the existing ratings and never a zero.
    const userCounts = new Array(numUsers + 1).fill(0);
    const movieCounts = new Array(numMovies + 1).fill(0);
    for (let userId = 1; userId <= numUsers; userId++) {
        const row = ratingMatrix[userId];
        for (let movieId = 1; movieId <= numMovies; movieId++) {
            if (row[movieId] !== 0) { userCounts[userId]++; movieCounts[movieId]++; }
        }
    }

    userRatings = new Array(numUsers + 1);
    for (let userId = 0; userId <= numUsers; userId++) userRatings[userId] = new Array(userCounts[userId] * 2);
    movieRatings = new Array(numMovies + 1);
    for (let movieId = 0; movieId <= numMovies; movieId++) movieRatings[movieId] = new Array(movieCounts[movieId] * 2);

    const userFill = new Array(numUsers + 1).fill(0);
    const movieFill = new Array(numMovies + 1).fill(0);
    for (let userId = 1; userId <= numUsers; userId++) {
        const row = ratingMatrix[userId];
        for (let movieId = 1; movieId <= numMovies; movieId++) {
            const rating = row[movieId];
            if (rating === 0) continue;
            userRatings[userId][userFill[userId]++] = movieId;
            userRatings[userId][userFill[userId]++] = rating;
            movieRatings[movieId][movieFill[movieId]++] = userId;
            movieRatings[movieId][movieFill[movieId]++] = rating;
        }
    }
}

// now() exists so the load-time timings work both in the browser and in the
// Node based checks, where performance is available but not guaranteed.
function now() {
    if (typeof performance !== 'undefined' && performance.now) return performance.now();
    return Date.now();
}
