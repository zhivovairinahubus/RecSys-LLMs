// Global variables for storing movie and rating data
let movies = [];
let ratings = [];

// Rows that failed validation and therefore never enter `movies` (see parser).
let invalidRecords = [];

// Movies without a `duplicateOf` marker: the ones shown in the dropdown and used as
// recommendation candidates (see parseItemData).
let distinctMovies = [];

// Genre name per column of the u.item file, in the order used by the MovieLens
// 100K documentation (ml-100k-README.txt): the first of the last 19 fields is the
// "unknown" flag, followed by the 18 real genres from "Action" to "Western".
const genreNames = [
    "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir",
    "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
    "Thriller", "War", "Western"
];

// The 18 real genres, without "unknown".
const realGenreNames = genreNames.slice(1);

// Articles that are moved to the front of a title for display purposes.
const displayArticles = ["The", "A", "An", "La", "Le", "Les", "L'", "Il", "Das", "Der", "Die", "El", "Los"];

function escapeRegExp(str) {
    return str.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

// Title key: the original title without redundant whitespace.
function makeTitleKey(title) {
    return title.trim().replace(/\s+/g, ' ');
}

// Move a trailing article to the front of the title for display, e.g.
// "Santa Clause, The (1994)" -> "The Santa Clause (1994)", "Enfer, L' (1994)" -> "L'Enfer (1994)".
// Only an article from `displayArticles` sitting right after the last comma and right
// before the year is moved; plain commas such as in "Paris, Texas (1984)" are left alone.
function makeDisplayTitle(title) {
    const t = title.trim();
    for (const article of displayArticles) {
        const match = new RegExp(',\\s*' + escapeRegExp(article) + '\\s*\\((\\d{4})\\)$', 'i').exec(t);
        if (match) {
            const head = t.slice(0, match.index).replace(/,\s*$/, '').trim();
            const separator = article.endsWith("'") ? '' : ' ';
            return article + separator + head + ' (' + match[1] + ')';
        }
    }
    return t;
}

// Primary function to load data from files
async function loadData() {
    try {
        // Load and parse movie data. The u.item file is latin-1 encoded; response.text()
        // always decodes as UTF-8 (Fetch Standard, "text()" = UTF-8 decode), which would
        // turn e.g. "é" into U+FFFD. Read the raw bytes and decode them explicitly.
        const moviesResponse = await fetch('u.item');
        if (!moviesResponse.ok) {
            throw new Error(`Failed to load movie data: ${moviesResponse.status}`);
        }
        const moviesBytes = await moviesResponse.arrayBuffer();
        const moviesText = new TextDecoder('iso-8859-1').decode(moviesBytes);
        parseItemData(moviesText);

        // Load and parse rating data
        const ratingsResponse = await fetch('u.data');
        if (!ratingsResponse.ok) {
            throw new Error(`Failed to load rating data: ${ratingsResponse.status}`);
        }
        const ratingsText = await ratingsResponse.text();
        parseRatingData(ratingsText);
    } catch (error) {
        console.error('Error loading data:', error);
        const resultElement = document.getElementById('result');
        if (resultElement) {
            resultElement.textContent = `Error: ${error.message}. Please make sure u.item and u.data files are in the correct location.`;
            resultElement.className = 'error';
        }
        throw error; // Re-throw to allow script.js to handle the error
    }
}

// Parse movie data from u.item format
function parseItemData(text) {
    const lines = text.split('\n');
    const seenTitleKeys = new Set();
    const firstMovieId = new Map();

    for (const line of lines) {
        if (line.trim() === '') continue;

        const fields = line.split('|');
        if (fields.length < 5) continue; // Skip invalid lines

        const id = parseInt(fields[0]);
        const title = fields[1];

        // Rows without a real title (the literal "unknown" placeholder row, id 267 in this
        // dataset: no release date, no IMDb URL, no genres) are not movies. They must not
        // appear in the dropdown nor in the recommendation candidates.
        if (!title || title.toLowerCase() === 'unknown') {
            invalidRecords.push({ id, title, releaseDate: fields[2] || '' });
            continue;
        }

        // MovieLens ships a few movies twice (same title, different id). All valid ids stay
        // in `movies` because u.data can carry ratings for the copies (e.g. ids 268, 680,
        // 303), and a user profile built from ratings would lose those movies. Each copy is
        // marked with duplicateOf = id of the first occurrence; only movies without
        // duplicateOf enter the dropdown and the recommendation candidates.
        const key = makeTitleKey(title);
        let duplicateOf = null;
        if (seenTitleKeys.has(key)) {
            duplicateOf = firstMovieId.get(key);
        } else {
            seenTitleKeys.add(key);
            firstMovieId.set(key, id);
        }

        // The last 19 fields are the genre flags; the first one is the "unknown" flag.
        // "unknown" means absence of information, not a feature movies are similar on,
        // so it is kept separately and excluded from the similarity vector.
        const genreValues = fields.slice(5, 24);
        const unknownGenre = genreValues[0] === '1';
        const genreVector = genreValues.slice(1).map(value => (value === '1' ? 1 : 0));
        const genres = realGenreNames.filter((_, index) => genreVector[index] === 1);

        // id 1609 "B*A*P*S" is titled "B.A.P.S (1997)" on Rotten Tomatoes, Letterboxd and
        // Wikipedia, and the IMDb URL field of the same row encodes exactly that spelling
        // (B%2EA%2EP%2ES%2E). "M*A*S*H" is left untouched: its asterisks are official.
        let displayTitle = makeDisplayTitle(title);
        if (id === 1609) {
            displayTitle = 'B.A.P.S (1997)';
        }

        movies.push({
            id, title, displayTitle, titleKey: key,
            genres, genreVector, unknownGenre,
            ...(duplicateOf !== null ? { duplicateOf } : {})
        });
    }

    distinctMovies = movies.filter(movie => !movie.duplicateOf);
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
