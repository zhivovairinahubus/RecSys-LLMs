// Initialize the application when the window loads
window.onload = async function() {
    try {
        // Display loading message
        const resultElement = document.getElementById('result');
        resultElement.textContent = "Loading movie data...";
        resultElement.className = 'loading';

        // Load data
        await loadData();

        // Populate dropdowns and update status
        populateMoviesDropdown();
        populateProfileDropdowns();

        // Clear stale results as soon as the user changes a selection
        clearResultOnChange('movie-select', 'result', "Select a movie and click Get Recommendations.");
        clearResultOnChange('watched-select-1', 'profile-result', "Select two or three watched movies and click Recommend from profile.");
        clearResultOnChange('watched-select-2', 'profile-result', "Select two or three watched movies and click Recommend from profile.");
        clearResultOnChange('watched-select-3', 'profile-result', "Select two or three watched movies and click Recommend from profile.");

        resultElement.textContent = "Data loaded. Please select a movie.";
        resultElement.className = 'success';
    } catch (error) {
        console.error('Initialization error:', error);
        // Error message already set in data.js
    }
};

function clearResultOnChange(selectId, resultId, message) {
    document.getElementById(selectId).addEventListener('change', () => {
        const element = document.getElementById(resultId);
        element.textContent = message;
        element.className = '';
    });
}

// Movies shown in every dropdown: the first copy of each title (no duplicateOf),
// sorted by the original title, labelled with the displayTitle.
function dropdownMovies() {
    return distinctMovies.slice().sort((a, b) => a.title.localeCompare(b.title));
}

// Fill a <select> element with movie options, keeping its placeholder (first option).
function fillMovieSelect(selectElement) {
    while (selectElement.options.length > 1) {
        selectElement.remove(1);
    }
    dropdownMovies().forEach(movie => {
        const option = document.createElement('option');
        option.value = movie.id;
        option.textContent = movie.displayTitle;
        selectElement.appendChild(option);
    });
}

function populateMoviesDropdown() {
    fillMovieSelect(document.getElementById('movie-select'));
}

function populateProfileDropdowns() {
    ['watched-select-1', 'watched-select-2', 'watched-select-3'].forEach(id => {
        fillMovieSelect(document.getElementById(id));
    });
}

function formatGenres(genres) {
    return genres.length > 0 ? '[' + genres.join(', ') + ']' : '[]';
}

// Main item-to-item recommendation function
function getRecommendations() {
    const resultElement = document.getElementById('result');

    try {
        // Step 1: Get user input
        const selectElement = document.getElementById('movie-select');
        const selectedMovieId = parseInt(selectElement.value);

        if (isNaN(selectedMovieId)) {
            resultElement.textContent = "Please select a movie first.";
            resultElement.className = 'error';
            return;
        }

        // Step 2: Find the liked movie
        const likedMovie = movies.find(movie => movie.id === selectedMovieId);
        if (!likedMovie) {
            resultElement.textContent = "Error: Selected movie not found in database.";
            resultElement.className = 'error';
            return;
        }

        // Step 3: No genre information -> nothing to base recommendations on
        if (likedMovie.genreVector.every(v => v === 0)) {
            resultElement.textContent =
                `No genre information for "${likedMovie.displayTitle}". ` +
                "Cannot compute recommendations; please select another movie.";
            resultElement.className = 'error';
            return;
        }

        // Show loading message while processing
        resultElement.textContent = "Calculating recommendations...";
        resultElement.className = 'loading';

        // Use setTimeout to allow the UI to update before heavy computation
        setTimeout(() => {
            try {
                // Step 4: Rank candidates with the cosine score (Top-5, selected
                // movie excluded by titleKey, copies and zero vectors dropped inside)
                const { top, tiesAtK } = rankCandidates(
                    likedMovie.genreVector,
                    movies,
                    new Set([likedMovie.titleKey]),
                    5,
                    cosine
                );

                // Step 5: Build the result text
                const lines = [
                    `Because you liked "${likedMovie.displayTitle}" ${formatGenres(likedMovie.genres)}:`
                ];
                top.forEach((movie, index) => {
                    const score = movie.score === null ? 'n/a' : movie.score.toFixed(3);
                    lines.push(`${index + 1}. ${movie.displayTitle} ${formatGenres(movie.genres)} — cosine: ${score}`);
                });

                if (top.length > 0) {
                    const kth = top[top.length - 1].score;
                    const inTop = kth === null ? 0 : top.filter(m => m.score === kth).length;
                    const extra = Math.max(0, tiesAtK - inTop);
                    if (extra > 0) {
                        lines.push(`... and ${extra} more with the same score.`);
                    }
                }

                resultElement.textContent = lines.join('\n');
                resultElement.className = 'success';
            } catch (error) {
                console.error('Error in recommendation calculation:', error);
                resultElement.textContent = "An error occurred while calculating recommendations.";
                resultElement.className = 'error';
            }
        }, 10);
    } catch (error) {
        console.error('Error in getRecommendations:', error);
        resultElement.textContent = "An unexpected error occurred.";
        resultElement.className = 'error';
    }
}

// Main profile-based recommendation function
function getProfileRecommendations() {
    const resultElement = document.getElementById('profile-result');

    try {
        // Step 1: Collect the watched movies from the three dropdowns
        const watched = [];
        for (const id of ['watched-select-1', 'watched-select-2', 'watched-select-3']) {
            const selectedMovieId = parseInt(document.getElementById(id).value);
            if (isNaN(selectedMovieId)) continue;
            const movie = movies.find(m => m.id === selectedMovieId);
            if (movie) watched.push(movie);
        }

        // Step 2: Keep distinct titles (a title can exist twice, under a duplicate id)
        const distinct = [];
        const seenTitleKeys = new Set();
        for (const movie of watched) {
            if (!seenTitleKeys.has(movie.titleKey)) {
                seenTitleKeys.add(movie.titleKey);
                distinct.push(movie);
            }
        }

        if (distinct.length < 2) {
            resultElement.textContent = "Please select at least two different movies to build a profile.";
            resultElement.className = 'error';
            return;
        }

        // Step 3: Build the profile as the mean genre vector. Movies without genre
        // information are skipped by buildProfile but still excluded from the
        // recommendations by their titleKey.
        const profile = buildProfile(distinct.map(movie => movie.genreVector));
        if (profile.every(weight => weight === 0)) {
            resultElement.textContent = "None of the selected movies have genre information, cannot build a profile.";
            resultElement.className = 'error';
            return;
        }

        // Step 4: Rank all movies by cosine similarity to the profile
        const excludedTitleKeys = new Set(distinct.map(movie => movie.titleKey));
        const { top, tiesAtK } = rankCandidates(profile, movies, excludedTitleKeys, 5, cosine);

        // Step 5: Build the result text
        const lines = [];

        // Profile vector as "Genre: weight" for the non-zero components
        const weights = [];
        realGenreNames.forEach((name, index) => {
            if (profile[index] !== 0) {
                weights.push(`${name}: ${profile[index].toFixed(2)}`);
            }
        });
        lines.push('Profile vector: ' + (weights.length > 0 ? weights.join(', ') : '(empty)'));

        lines.push('Watched movies (cosine with profile):');
        distinct.forEach(movie => {
            const score = cosine(profile, movie.genreVector);
            lines.push(`  - ${movie.displayTitle}: ${score === null ? 'n/a' : score.toFixed(3)}`);
        });

        lines.push('Top-5 recommendations:');
        top.forEach((movie, index) => {
            const score = movie.score === null ? 'n/a' : movie.score.toFixed(3);
            lines.push(`${index + 1}. ${movie.displayTitle} ${formatGenres(movie.genres)} — cosine: ${score}`);
        });

        if (top.length > 0) {
            const kth = top[top.length - 1].score;
            const inTop = kth === null ? 0 : top.filter(m => m.score === kth).length;
            const extra = Math.max(0, tiesAtK - inTop);
            if (extra > 0) {
                lines.push(`... and ${extra} more with the same score.`);
            }
        }

        resultElement.textContent = lines.join('\n');
        resultElement.className = 'success';
    } catch (error) {
        console.error('Error in getProfileRecommendations:', error);
        resultElement.textContent = "An unexpected error occurred.";
        resultElement.className = 'error';
    }
}