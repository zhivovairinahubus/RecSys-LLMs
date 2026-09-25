// cbf-core.js — content-based filtering helpers. Pure functions, no DOM access.
// Works both in browsers (declared as globals) and in Node (module.exports).
// Vectors here are the 18-value 0/1 genre vectors produced by data.js.

// Dot product of two equally sized numeric vectors.
function dot(a, b) {
    let sum = 0;
    for (let i = 0; i < a.length; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

// Euclidean norm of a numeric vector.
function norm(a) {
    return Math.sqrt(a.reduce((sum, x) => sum + x * x, 0));
}

// Cosine similarity: dot / (norm(a) * norm(b)).
// Returns null when either vector has zero norm (undefined similarity).
function cosine(a, b) {
    const na = norm(a);
    const nb = norm(b);
    if (na === 0 || nb === 0) return null;
    return dot(a, b) / (na * nb);
}

// Jaccard similarity for binary (0/1) vectors: intersection / union.
// Returns null when both vectors are all-zero (undefined similarity), kept for
// comparison against the cosine score.
function jaccard(a, b) {
    let intersection = 0;
    let union = 0;
    for (let i = 0; i < a.length; i++) {
        const ai = a[i];
        const bi = b[i];
        if (ai === 1 || bi === 1) {
            union++;
            if (ai === 1 && bi === 1) intersection++;
        }
    }
    if (union === 0) return null;
    return intersection / union;
}

// Rank candidates against a query vector.
//   queryVector        - genre vector of the selected movie
//   candidates         - full movies array
//   excludedTitleKeys  - Set of titleKeys to drop (e.g. the selected movie)
//   k                  - how many top items to return
//   scoreFn            - (queryVector, candidateVector) => score
// Excluded: movies with duplicateOf, movies whose titleKey is in excludedTitleKeys,
// and movies with an all-zero genre vector. Sorted by score (desc), ties broken by
// the original title ascending (same key the dropdown is sorted by; sorting by
// displayTitle would let moved articles like "A"/"The" bias tie order). Returns
// { top, tiesAtK } where tiesAtK is the number of candidates whose score equals
// the score of the k-th item.
function rankCandidates(queryVector, candidates, excludedTitleKeys, k, scoreFn) {
    const eligible = candidates.filter(movie =>
        movie.duplicateOf === undefined &&
        !excludedTitleKeys.has(movie.titleKey) &&
        movie.genreVector.some(v => v !== 0)
    );

    const scored = eligible.map(movie => {
        const raw = scoreFn(queryVector, movie.genreVector);
        return Object.assign({}, movie, {
            score: (raw === null || raw === undefined) ? null : raw
        });
    });

    scored.sort((a, b) => {
        const sa = a.score === null ? Number.NEGATIVE_INFINITY : a.score;
        const sb = b.score === null ? Number.NEGATIVE_INFINITY : b.score;
        if (sa !== sb) return sb - sa;
        return a.title.localeCompare(b.title);
    });

    const top = scored.slice(0, k);

    let tiesAtK = 0;
    if (top.length > 0) {
        const kth = top[top.length - 1].score;
        tiesAtK = scored.filter(movie => movie.score === kth).length;
    }

    return { top, tiesAtK };
}

// Build a user profile vector as the arithmetic mean of the given genre vectors.
// All-zero vectors (movies without genre information) are skipped. Returns an
// all-zero vector when nothing is left to average.
function buildProfile(vectors) {
    const dim = vectors.length > 0 ? vectors[0].length : 0;
    const profile = new Array(dim).fill(0);
    let count = 0;
    for (const vector of vectors) {
        if (!vector || vector.length !== dim || !vector.some(value => value !== 0)) continue;
        for (let i = 0; i < dim; i++) {
            profile[i] += vector[i];
        }
        count++;
    }
    if (count === 0) return profile;
    for (let i = 0; i < dim; i++) {
        profile[i] /= count;
    }
    return profile;
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = { dot, norm, cosine, jaccard, rankCandidates, buildProfile };
}