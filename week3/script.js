// ---------------------------------------------------------------------------
// HW3 — Collaborative Filtering core
//
// Missing-value strategy (see week3/readme.md section 6). Choose EXACTLY ONE
// and keep it consistent in cosineSimilarity below:
//
//   [x] weight similarity by the number of co-rated items
// ---------------------------------------------------------------------------

// Number of co-rated items at which a similarity is considered fully
// trustworthy. A pair with n co-rated items is scaled by min(n, 50) / 50,
// so pairs built on a single shared rating are pushed far down the ranking.
const CO_RATED_CAP = 50;

// Number of neighbours used by user-based CF, and also the number of rated
// movies that contribute to a single item-based prediction.
const NEIGHBOUR_COUNT = 20;

// Minimum total evidence a prediction must be built on. The evidence of one
// contributing pair is min(common, CO_RATED_CAP) / CO_RATED_CAP, so a movie
// predicted from a handful of thin pairs is discarded instead of being
// recommended on almost no data.
const EVIDENCE_THRESHOLD = 5;

// Initialize the application when the window loads
window.onload = async function() {
    const userBased = document.getElementById('user-based-result');
    const itemBased = document.getElementById('item-based-result');

    try {
        userBased.innerHTML = '<p>Loading movie data...</p>';
        itemBased.innerHTML = '<p>Loading movie data...</p>';

        await loadData();

        populateUserDropdown();
        wireUpUserControls();
        // Offer every user id as a suggestion before anything has been typed.
        refreshUserSuggestions('');

        userBased.innerHTML = '<p>Data loaded. Select a user.</p>';
        itemBased.innerHTML = '<p>Data loaded. Select a user.</p>';
        updateUserStatsLine();
    } catch (error) {
        console.error('Initialization error:', error);
        // The error message is already shown by data.js
    }
};

// Populate the user dropdown with one option per user id found in u.data
function populateUserDropdown() {
    const selectElement = document.getElementById('user-select');

    // Clear existing options except the first placeholder
    while (selectElement.options.length > 1) {
        selectElement.remove(1);
    }

    for (let userId = 1; userId <= numUsers; userId++) {
        const option = document.createElement('option');
        option.value = userId;
        option.textContent = `User ${userId}`;
        selectElement.appendChild(option);
    }
}

// ---------------------------------------------------------------------------
// User controls: a <select> of all users and a free-text search field that
// narrows to numbers as you type. The two are bound to each other in both
// directions, so typing a number selects that user and picking a user fills the
// field. Enter in the search field runs the calculation.
// ---------------------------------------------------------------------------
function wireUpUserControls() {
    const selectElement = document.getElementById('user-select');
    const searchElement = document.getElementById('user-search');
    const listElement = document.getElementById('user-search-options');

    // select -> search field, and refresh the "N ratings, average" line
    selectElement.addEventListener('change', function() {
        searchElement.value = selectElement.value;
        updateUserStatsLine();
        // A different user means the previous lists describe somebody else.
        clearResults('User changed. Press "Get Recommendations" to recalculate.');
    });

    // search field -> datalist suggestions, narrowed by the digits typed so far.
    // This handler only ever writes to the select, never back into the field:
    // mirroring the select's value back would refill the field the moment it was
    // emptied, so the user could not clear it and type another number.
    searchElement.addEventListener('input', function() {
        const digits = searchElement.value.replace(/\D/g, '');
        refreshUserSuggestions(digits);

        const typed = parseInt(digits, 10);
        if (digits !== '' && isValidUserId(typed)) selectElement.value = digits;

        updateUserStatsLine();
        clearResults('User changed. Press "Get Recommendations" to recalculate.');
    });

    // Enter in the search field starts the calculation
    searchElement.addEventListener('keydown', function(event) {
        if (event.key !== 'Enter') return;
        event.preventDefault();
        getRecommendations();
    });
}

// Rebuild the datalist from the digits typed so far. An empty or partial input
// offers everything that starts with those digits, so "1" lists 1, 10..19, 1xx,
// and an exact id leaves just that id.
//
// The options are collected into a fragment and swapped in with a single
// replaceChildren() call. Removing 943 options one by one instead would mean
// 943 separate DOM mutations per keystroke, which is slow enough to be felt
// while typing and stalls the renderer outright in headless Chrome.
function refreshUserSuggestions(digits) {
    const listElement = document.getElementById('user-search-options');

    const fragment = document.createDocumentFragment();
    for (let userId = 1; userId <= numUsers; userId++) {
        const asString = String(userId);
        if (digits !== '' && !asString.startsWith(digits)) continue;
        const option = document.createElement('option');
        option.value = userId;
        fragment.appendChild(option);
    }

    listElement.replaceChildren(fragment);
}

function isValidUserId(userId) {
    return Number.isInteger(userId) && userId >= 1 && userId <= numUsers;
}

// The id the two controls currently agree on, or NaN.
function selectedUserId() {
    const selectElement = document.getElementById('user-select');
    const userId = parseInt(selectElement.value, 10);
    return isValidUserId(userId) ? userId : NaN;
}

// "User 19 rated 271 films, average 3.71."
function updateUserStatsLine() {
    const statsLine = document.getElementById('user-stats');
    if (!statsLine) return;

    const userId = selectedUserId();
    if (!Number.isInteger(userId)) {
        statsLine.textContent = '';
        return;
    }

    const { count, average } = ratingSummary(userId);
    statsLine.textContent =
        `User ${userId} rated ${count} film${count === 1 ? '' : 's'}, average ${average.toFixed(2)}.`;
}

// Replace both columns with the same notice.
function clearResults(message) {
    setExplanationVisible(false);
    renderMessage('user-based-result', message);
    renderMessage('item-based-result', message);
}

// ---------------------------------------------------------------------------
// Cosine similarity between two rating vectors, on the co-rated entries only.
// See week3/readme.md section 5.3 and the missing-value strategy marked above.
//
// Missing-value convention: a rating of 0 means "not rated". Only positions
// where BOTH vectors are non-zero are compared; every other position is
// ignored and contributes to neither the dot product nor either norm. If the
// two vectors share no rated positions, return 0 (a zero denominator).
//
// Confidence weighting (the strategy selected above): the raw cosine is
// multiplied by min(common, CO_RATED_CAP) / CO_RATED_CAP, where "common" is
// the number of co-rated entries the two vectors actually share. A pair that
// shares a single rating therefore scores at most 1 / CO_RATED_CAP, while a
// pair sharing CO_RATED_CAP or more keeps its unmodified cosine value.
// ---------------------------------------------------------------------------
function cosineSimilarity(a, b) {
    let dot = 0;
    let normA = 0;
    let normB = 0;
    let common = 0;

    const length = Math.min(a.length, b.length);
    for (let i = 0; i < length; i++) {
        if (a[i] === 0 || b[i] === 0) continue;
        dot += a[i] * b[i];
        normA += a[i] * a[i];
        normB += b[i] * b[i];
        common++;
    }

    return combineCosineAndCommon(dot, normA, normB, common);
}

// ---------------------------------------------------------------------------
// REFERENCE implementation. Kept deliberately unchanged next to the fast
// path below so week3/check-cf.js can prove the two agree; the recommendation
// code itself uses the one-pass versions instead.
// ---------------------------------------------------------------------------
function countCommonRatings(a, b) {
    let common = 0;
    const length = Math.min(a.length, b.length);
    for (let i = 0; i < length; i++) {
        if (a[i] === 0 || b[i] === 0) continue;
        common++;
    }
    return common;
}

// ---------------------------------------------------------------------------
// One-pass cosine for two users. Walks the active user's SHORT sparse list of
// actual ratings instead of the full width of the matrix row, and resolves
// the other user's rating for the same film with a single array lookup.
//
// Returns both numbers the caller needs, so the co-rated count never has to
// be recomputed in a second scan:
//
//   similarity - identical to cosineSimilarity(ratingMatrix[a], ratingMatrix[b])
//   common     - identical to countCommonRatings(ratingMatrix[a], ratingMatrix[b])
// ---------------------------------------------------------------------------
function cosineAndCommonUsers(userA, userB) {
    const rowB = ratingMatrix[userB];
    const sparseA = userRatings[userA];
    let dot = 0;
    let normA = 0;
    let normB = 0;
    let common = 0;

    for (let i = 0; i < sparseA.length; i += 2) {
        const ratingB = rowB[sparseA[i]];
        if (ratingB === 0) continue;
        const ratingA = sparseA[i + 1];
        dot += ratingA * ratingB;
        normA += ratingA * ratingA;
        normB += ratingB * ratingB;
        common++;
    }

    return { similarity: combineCosineAndCommon(dot, normA, normB, common), common: common };
}

// ---------------------------------------------------------------------------
// One-pass cosine for two movies, walking the first film's sparse viewer list.
// Returns the same pair as cosineAndCommonUsers, with the same meaning.
// ---------------------------------------------------------------------------
function cosineAndCommonItems(movieA, movieB) {
    const sparseA = movieRatings[movieA];
    let dot = 0;
    let normA = 0;
    let normB = 0;
    let common = 0;

    for (let i = 0; i < sparseA.length; i += 2) {
        const ratingB = ratingMatrix[sparseA[i]][movieB];
        if (ratingB === 0) continue;
        const ratingA = sparseA[i + 1];
        dot += ratingA * ratingB;
        normA += ratingA * ratingA;
        normB += ratingB * ratingB;
        common++;
    }

    return { similarity: combineCosineAndCommon(dot, normA, normB, common), common: common };
}

// Shared tail of every cosine above: zero denominator or no overlap gives 0,
// otherwise the raw cosine scaled by the confidence factor.
function combineCosineAndCommon(dot, normA, normB, common) {
    if (common === 0) return 0;
    const denominator = Math.sqrt(normA * normB);
    if (denominator === 0) return 0;
    return (dot / denominator) * Math.min(common, CO_RATED_CAP) / CO_RATED_CAP;
}

// Evidence contributed by one pair that shares "common" co-rated entries.
function evidenceWeight(common) {
    return Math.min(common, CO_RATED_CAP) / CO_RATED_CAP;
}

// ---------------------------------------------------------------------------
// User-Based CF. See week3/readme.md section 5.4.
//
//   1. compare the active user's rating vector (a matrix ROW) against every
//      other user
//   2. keep the N = NEIGHBOUR_COUNT most similar users with positive similarity
//   3. for each movie the active user has NOT rated, predict a score as the
//      similarity-weighted average of those users' ratings
//   4. drop movies whose total co-rated evidence is below EVIDENCE_THRESHOLD
//   5. sort by score descending and take the top K
//
// Duplicate films: u.item holds some films twice under two ids. data.js merges
// their ratings into the column of the first instance and marks the extra ids
// with duplicateOf, so the copies are skipped as candidates here and are never
// recommended twice. One row (id 267, "unknown") describes no film at all: it
// is marked invalid, its ratings are not in the matrix, and isRealFilm() keeps
// it out of the candidates as well.
//
// Returns { movieId, title, displayTitle, score } per recommended film. `title`
// is the raw u.item title, `displayTitle` is the form shown to a reader.
//
// Ties: no tie-breaking rule is imposed. Neighbours are generated in ascending
// userId order and sorted with a stable descending sort, so users with equal
// similarity keep ascending userId order. The same holds for the candidate
// movies below: they are generated in ascending movieId order, so equal
// predicted scores come out in ascending movieId order.
// ---------------------------------------------------------------------------
function getUserBasedRecommendations(activeUserId, topK = 5) {
    const activeRow = ratingMatrix[activeUserId];

    const neighbours = [];
    for (let userId = 1; userId <= numUsers; userId++) {
        if (userId === activeUserId) continue;
        const { similarity } = cosineAndCommonUsers(activeUserId, userId);
        if (similarity > 0) neighbours.push({ userId: userId, similarity: similarity });
    }
    neighbours.sort((a, b) => b.similarity - a.similarity);
    const topNeighbours = neighbours.slice(0, NEIGHBOUR_COUNT);

    // The similarity already carries the co-rated count's effect, but the
    // threshold needs the raw count as evidence, so ask for it once more.
    for (const neighbour of topNeighbours) {
        const { common } = cosineAndCommonUsers(activeUserId, neighbour.userId);
        neighbour.common = common;
        neighbour.evidence = evidenceWeight(common);
    }

    const candidates = [];
    let beforeThreshold = 0;

    for (let movieId = 1; movieId <= numMovies; movieId++) {
        // A duplicate copy never stands for itself: its ratings live in the
        // column of the first instance, so it must not become a candidate. An
        // invalid row is not a film at all, so it is skipped as well.
        if (!isRealFilm(movieId)) continue;
        if (activeRow[movieId] !== 0) continue;

        let weighted = 0;
        let similaritySum = 0;
        let voters = 0;
        let evidence = 0;
        for (const neighbour of topNeighbours) {
            const rating = ratingMatrix[neighbour.userId][movieId];
            if (rating === 0) continue;
            weighted += neighbour.similarity * rating;
            similaritySum += neighbour.similarity;
            voters++;
            evidence += neighbour.evidence;
        }
        if (voters === 0) continue;
        beforeThreshold++;

        // Reject predictions resting on too little co-rated evidence.
        if (evidence < EVIDENCE_THRESHOLD) continue;

        candidates.push({
            movieId: movieId,
            score: weighted / similaritySum,
            voters: voters,
            evidence: evidence,
        });
    }
    candidates.sort((a, b) => b.score - a.score);

    return candidates.slice(0, topK).map(candidate => ({
        movieId: candidate.movieId,
        title: titleOf(candidate.movieId),
        displayTitle: displayTitleOf(candidate.movieId),
        score: candidate.score,
    }));
}

// ---------------------------------------------------------------------------
// Item-Based CF. See week3/readme.md section 5.5.
//
//   1. for each movie the active user has rated, compute the item-item
//      similarity against every other movie's rating column
//   2. for each candidate movie the active user has NOT rated, keep only the
//      NEIGHBOUR_COUNT rated movies with the highest weighted similarity, and
//      aggregate those similarities weighted by the user's rating
//   3. drop movies whose total co-rated evidence is below EVIDENCE_THRESHOLD
//   4. sort by the aggregated score descending and take the top K
//
// Duplicate films: the same skip as in user-based, so neither a copy nor an
// invalid row appears as a candidate or competes with the film it copies.
//
// Returns { movieId, title, displayTitle, score } per recommended film.
//
// A film vector is a matrix COLUMN, so the columns are materialised once and
// then handed to cosineSimilarity as plain arrays.
//
// Ties: no tie-breaking rule is imposed. Candidates are generated in ascending
// movieId order and sorted with a stable descending sort, so equal predicted
// scores come out in ascending movieId order.
// ---------------------------------------------------------------------------
function getItemBasedRecommendations(activeUserId, topK = 5) {
    const activeRow = ratingMatrix[activeUserId];

    const ratedMovieIds = [];
    for (let movieId = 1; movieId <= numMovies; movieId++) {
        if (activeRow[movieId] !== 0) ratedMovieIds.push(movieId);
    }

    const candidates = [];
    let beforeThreshold = 0;

    for (let movieId = 1; movieId <= numMovies; movieId++) {
        // Same rule as user-based: neither a duplicate copy nor an invalid row
        // is a separate film.
        if (!isRealFilm(movieId)) continue;
        if (activeRow[movieId] !== 0) continue;

        // Only the NEIGHBOUR_COUNT rated movies closest to this candidate take
        // part in the prediction, ranked by the weighted similarity.
        const contributors = [];
        for (const ratedId of ratedMovieIds) {
            const { similarity, common } = cosineAndCommonItems(ratedId, movieId);
            if (common === 0) continue;
            contributors.push({
                ratedId: ratedId,
                similarity: similarity,
                evidence: evidenceWeight(common),
            });
        }
        contributors.sort((a, b) => b.similarity - a.similarity);
        const topContributors = contributors.slice(0, NEIGHBOUR_COUNT);

        let weighted = 0;
        let similaritySum = 0;
        let evidence = 0;
        for (const contributor of topContributors) {
            weighted += contributor.similarity * activeRow[contributor.ratedId];
            similaritySum += contributor.similarity;
            evidence += contributor.evidence;
        }
        if (similaritySum === 0) continue;
        beforeThreshold++;

        // Same threshold as user-based: too little shared evidence, no candidate.
        if (evidence < EVIDENCE_THRESHOLD) continue;

        candidates.push({
            movieId: movieId,
            score: weighted / similaritySum,
            voters: topContributors.length,
            evidence: evidence,
        });
    }
    candidates.sort((a, b) => b.score - a.score);

    return candidates.slice(0, topK).map(candidate => ({
        movieId: candidate.movieId,
        title: titleOf(candidate.movieId),
        displayTitle: displayTitleOf(candidate.movieId),
        score: candidate.score,
    }));
}

// The film vectors (matrix columns) and the sparse rating lists are built once
// by data.js at load time and live in ratingColumns / movieRatings. Nothing in
// this file rebuilds them per call.

function titleOf(movieId) {
    const movie = movieById[movieId];
    return movie ? movie.title : `unknown movie ${movieId}`;
}

// Title as it should be shown to a reader: the article moved to the front.
// Kept separate from titleOf so that the raw u.item title stays available.
function displayTitleOf(movieId) {
    const movie = movieById[movieId];
    return movie ? movie.displayTitle : `unknown movie ${movieId}`;
}

// The user's three highest-rated films, highest rating first, ties broken by
// ascending movieId. Copies and invalid rows are left out, so the
// "Because you liked ..." line only ever names real films.
//
// The list is read straight off the matrix row, so the ratings it reports are
// exactly the ones the recommendations were computed from: a rating that sat on
// a duplicate copy is already counted under the film's first id.
function topRatedMovies(userId, count = 3) {
    const row = ratingMatrix[userId];
    const rated = [];
    for (let movieId = 1; movieId <= numMovies; movieId++) {
        const rating = row[movieId];
        if (rating === 0) continue;
        if (!isRealFilm(movieId)) continue;
        rated.push({ movieId, rating });
    }
    // Descending by rating; on equal ratings the smaller id comes first.
    rated.sort((a, b) => (b.rating - a.rating) || (a.movieId - b.movieId));
    return rated.slice(0, count);
}

// Number of ratings the user has and their average, for the line under the
// search field.
function ratingSummary(userId) {
    const row = ratingMatrix[userId];
    let count = 0;
    let sum = 0;
    for (let movieId = 1; movieId <= numMovies; movieId++) {
        const rating = row[movieId];
        if (rating === 0) continue;
        count++;
        sum += rating;
    }
    return { count, sum, average: count === 0 ? 0 : sum / count };
}

// Provided — read the selected user and render both recommendation lists
async function getRecommendations() {
    const button = document.getElementById('recommend-btn');
    const userId = selectedUserId();

    if (!Number.isInteger(userId)) {
        clearResults('Please select a user first.');
        return;
    }

    // Both methods are pure functions of the loaded data, so there is nothing
    // genuinely asynchronous left to await. The button is still disabled for the
    // duration so a second click cannot start a run while this one is going, and
    // so a slow machine still shows the "Calculating…" state.
    setCalculating(true);
    try {
        // Yield once so the browser actually paints the disabled button before
        // the item-based pass blocks the main thread.
        await new Promise(resolve => setTimeout(resolve, 0));

        const userBased = getUserBasedRecommendations(userId);
        const itemBased = getItemBasedRecommendations(userId);
        const liked = topRatedMovies(userId, 3);

        // Only reached when at least one column has a list to show, so the
        // explanation of what a predicted rating means is visible exactly when
        // predictions are on screen.
        setExplanationVisible(true);

        renderList('user-based-result', userBased,
            'Because viewers with similar taste rated these highly, we recommend:',
            null,
            'Not enough evidence for a reliable recommendation: the most ' +
            'similar users share too few rated movies with this user.');
        renderList('item-based-result', itemBased,
            'Because you liked …, we recommend:',
            liked.map(movie => displayTitleOf(movie.movieId)),
            'Not enough evidence for a reliable recommendation: too few of the movies ' +
            'you rated are similar enough to the rest of the catalogue.');
    } finally {
        setCalculating(false);
    }
}

// Disable the button and label it "Calculating…" while a run is in progress.
function setCalculating(isCalculating) {
    const button = document.getElementById('recommend-btn');
    if (!button) return;
    button.disabled = isCalculating;
    button.textContent = isCalculating ? 'Calculating…' : 'Get Recommendations';
}

// Show a plain notice in a column instead of a list. Used for the states where
// there is nothing to recommend yet, or an explanation why there is nothing.
function renderMessage(elementId, message) {
    const el = document.getElementById(elementId);
    el.innerHTML = `<p>${message}</p>`;
}

// The "predicted rating is…" note only makes sense while predictions are shown,
// so it is hidden whenever the columns hold a message instead of a list.
function setExplanationVisible(isVisible) {
    const el = document.getElementById('rating-explanation');
    if (el) el.hidden = !isVisible;
}

// Render a list of { title, displayTitle, score } into the element.
//
// `heading` is the line printed above the entries. For the item-based column it
// contains a "…" where the three films the user liked best belong; `likedTitles`
// fills that gap. `items` decides whether a list or an explanation is shown.
//
// `emptyMessage` is column-specific on purpose: an empty user-based list and an
// empty item-based list have different causes, so they must not share wording.
// It is NOT the same text as "please select a user" — that is a UI state, this
// is a computed result.
function renderList(elementId, items, heading, likedTitles, emptyMessage) {
    const el = document.getElementById(elementId);

    if (!items || items.length === 0) {
        renderMessage(elementId, emptyMessage);
        return;
    }

    let html = '';
    if (heading) {
        const leadIn = likedTitles && likedTitles.length > 0
            ? heading.replace('…', likedTitles.join(', '))
            : heading;
        html += `<p class="list-heading">${leadIn}</p>`;
    }

    // displayTitle is the readable form: an article moved to the front.
    // Three decimals: enough to tell neighbouring predictions apart, which two
    // decimals did not (four films of one user all rounded to 4.75).
    const entries = items
        .map(item =>
            `<li>${item.displayTitle || item.title} &mdash; predicted rating ` +
            `${Number(item.score).toFixed(3)} out of 5</li>`)
        .join('');
    html += `<ul>${entries}</ul>`;

    // A short list is still a list, so the entries stay — but say so, because a
    // single recommendation out of 1663 films is a weak result and the reader
    // should not have to infer that from the length alone.
    if (items.length < 5) {
        html += `<p class="short-list-note">Only ${items.length} ` +
            `${items.length === 1 ? 'movie has' : 'movies have'} enough evidence.</p>`;
    }
    el.innerHTML = html;
}
