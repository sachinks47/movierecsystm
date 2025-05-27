# movierecsystm
Movie Recommendation Project-Data Science

This project was done during my 2 months Data Science course offered by TechMaghi

This project is a hybrid **Movie Recommendation System** that provides two types of recommendations:

1. **Top-N Popular Movies by Genre**: Uses aggregated rating data to suggest top-rated movies in a specific genre.
2. **Content-Based Filtering**: Recommends movies based on genre similarity using TF-IDF and cosine similarity.

The system leverages interactive widgets to allow users to choose genres, rating thresholds, and other inputs dynamically in a Jupyter notebook.

### 📚 Features

* Genre-based filtering using pandas and `groupby`.
* Content-based filtering using TF-IDF vectorization of movie genres.
* Cosine similarity to measure content closeness.
* Interactive controls via `ipywidgets` for real-time recommendation generation.

### 🛠️ Libraries Used

* **pandas**: Data loading, manipulation, and aggregation.
* **numpy**: Numerical operations and array handling.
* **scikit-learn (sklearn)**:

  * `TfidfVectorizer`: To vectorize genre text for content-based recommendations.
  * `cosine_similarity`: To compute similarity between movies.
* **ipywidgets**: For dropdowns and sliders to make the notebook interactive.

### 📁 Dataset

* `movies.csv`: Contains movie metadata (title, genres, movieId).
* `ratings.csv`: User ratings for movies (userId, movieId, rating).

### 🧠 Functions

* `TopNPopularMovies(genre, threshold, topN)`: Lists top N popular movies based on average ratings and number of reviews.
* `recommendation_genre(movie_df, similarity_matrix, movie_title, topN)`: Generates content-based recommendations.

### 🚀 How to Run

1. Load the datasets `movies.csv` and `ratings.csv`.
2. Execute all cells in the Jupyter Notebook.
3. Use the dropdown widgets to interact with the recommender system.

