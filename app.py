import os
import re
import pandas as pd
import numpy as np
import urllib.parse
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash

# Import our custom OMDb API utility
from utils.omdb import fetch_movie_details as fetch_omdb

# ==========================================
# 1. LOAD ENV VARIABLES
# ==========================================
load_dotenv()

OMDB_API_KEY = os.getenv("OMDB_API_KEY")

if not OMDB_API_KEY:
    raise ValueError("CRITICAL ERROR: OMDB_API_KEY is missing from .env file! Please add it to start the server.")

# ==========================================
# 2. APP CONFIGURATION & SETUP
# ==========================================
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'portfolio-secret-key-xyz')

# Check for common Vercel Postgres variable names (Neon or Supabase)
db_url = os.environ.get('DATABASE_URL') or os.environ.get('POSTGRES_URL')

if db_url:
    # SQLAlchemy requires the URL to start with 'postgresql://' instead of 'postgres://'
    if db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql://", 1)
    app.config['SQLALCHEMY_DATABASE_URI'] = db_url
else:
    # Fallback to local SQLite for local testing
    # Vercel's filesystem is read-only. We must use /tmp if the database falls back to SQLite
    if os.environ.get('VERCEL'):
        app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:////tmp/database.db'
    else:
        app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(os.path.dirname(__file__), 'database.db')

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# ==========================================
# 3. DATABASE MODELS
# ==========================================
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)

class UserRating(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    movie_id = db.Column(db.Integer, nullable=False)
    rating = db.Column(db.Float, nullable=False)

class MovieCache(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    search_title = db.Column(db.String(250), unique=True, nullable=False)
    title = db.Column(db.String(250))
    year = db.Column(db.String(50))
    imdb_rating = db.Column(db.Float, default=0.0)
    imdb_votes = db.Column(db.Integer, default=0)
    poster_url = db.Column(db.String(500))
    plot = db.Column(db.Text)
    genre = db.Column(db.String(250))

@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))

# ==========================================
# 4. RECOMMENDATION ENGINE & OMDB LOGIC
# ==========================================
movies_df = pd.DataFrame()
ratings_df = pd.DataFrame()
all_genres_list = []
using_dummy_data = False
tfidf_matrix = None

def get_rich_movie_data(title):
    cached = MovieCache.query.filter_by(search_title=title).first()
    if cached:
        return {
            'title': cached.title or title, 'year': cached.year,
            'imdb_rating': cached.imdb_rating, 'imdb_votes': cached.imdb_votes,
            'poster': cached.poster_url, 'plot': cached.plot, 'genre': cached.genre
        }

    try:
        omdb_data = fetch_omdb(title)
    except TypeError:
        omdb_data = fetch_omdb(title, OMDB_API_KEY)
    
    if omdb_data:
        try:
            new_cache = MovieCache(
                search_title=title, title=omdb_data['title'], year=omdb_data['year'],
                imdb_rating=omdb_data['imdb_rating'], imdb_votes=omdb_data['imdb_votes'],
                poster_url=omdb_data['poster'], plot=omdb_data['plot'], genre=omdb_data['genre']
            )
            db.session.add(new_cache)
            db.session.commit()
        except Exception:
            db.session.rollback()
        return omdb_data

    return {
        'title': title, 'year': '', 'imdb_rating': 0.0, 'imdb_votes': 0,
        'poster': None, 'plot': 'No plot available.', 'genre': 'Unknown'
    }

def find_target_file(keyword, ext):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    curr_dir = os.getcwd()
    for d in [curr_dir, base_dir]:
        exact_path = os.path.join(d, f"{keyword}{ext}")
        if os.path.exists(exact_path): return exact_path
            
    for search_dir in [curr_dir, base_dir]:
        for root, dirs, files in os.walk(search_dir):
            if root[len(search_dir):].count(os.sep) >= 2: dirs[:] = []
            for file in files:
                if keyword.lower() in file.lower() and file.lower().endswith(ext):
                    return os.path.join(root, file)
    return None

def load_datasets():
    global movies_df, ratings_df, all_genres_list, using_dummy_data, tfidf_matrix
    m_path = find_target_file('movies', '.csv')
    r_path = find_target_file('ratings', '.csv')
    
    loaded = False
    if m_path and r_path:
        try:
            try: movies_df = pd.read_csv(m_path, encoding='utf-8-sig')
            except UnicodeDecodeError: movies_df = pd.read_csv(m_path, encoding='latin-1')
                
            try: ratings_df = pd.read_csv(r_path, encoding='utf-8-sig')
            except UnicodeDecodeError: ratings_df = pd.read_csv(r_path, encoding='latin-1')
                
            loaded = True
            using_dummy_data = False
        except Exception as e: print(f"Error reading CSV files: {e}")
            
    if not loaded:
        using_dummy_data = True
        movies_df = pd.DataFrame({
            'movieId': range(1, 101),
            'title': [f'Sample Movie {i} ({2000 + (i%24)})' for i in range(1, 101)],
            'genres': ['Action|Adventure', 'Comedy|Romance', 'Sci-Fi|Thriller', 'Drama', 'Horror|Mystery'] * 20
        })
        ratings_df = pd.DataFrame({
            'userId': np.random.randint(1, 20, 1000),
            'movieId': np.random.randint(1, 101, 1000),
            'rating': np.random.uniform(1.0, 5.0, 1000).round(1)
        })

    movies_df.columns = movies_df.columns.str.strip()
    ratings_df.columns = ratings_df.columns.str.strip()
    
    movies_df['movieId'] = pd.to_numeric(movies_df['movieId'], errors='coerce').fillna(-1).astype(int)
    ratings_df['movieId'] = pd.to_numeric(ratings_df['movieId'], errors='coerce').fillna(-1).astype(int)
    ratings_df['rating'] = pd.to_numeric(ratings_df['rating'], errors='coerce').fillna(0.0).astype(float)
    
    movies_df['year_num'] = movies_df['title'].str.extract(r'\((\d{4})\)').astype(float).fillna(2000)
    
    movies_df['genres'] = movies_df['genres'].str.replace('|', ', ')
    genres_set = set()
    for g_str in movies_df['genres'].dropna():
        for g in str(g_str).split(','):
            g_clean = g.strip()
            if g_clean and g_clean != '(no genres listed)':
                genres_set.add(g_clean)
    all_genres_list = sorted(list(genres_set))

    print("🧠 Building TF-IDF matrix for Content Matching...")
    movies_df['genres_clean'] = movies_df['genres'].str.replace(', ', ' ')
    tfidf = TfidfVectorizer(token_pattern=r'(?u)\b[\w-]+\b', stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies_df['genres_clean'].fillna(''))

    print("📈 Calculating internal popularity statistics...")
    if not ratings_df.empty:
        stats = ratings_df.groupby('movieId').agg(internal_rating=('rating', 'mean'), internal_votes=('rating', 'count')).reset_index()
        movies_df = pd.merge(movies_df, stats, on='movieId', how='left')
    else:
        movies_df['internal_rating'] = 0.0
        movies_df['internal_votes'] = 0

    movies_df['internal_rating'] = movies_df['internal_rating'].fillna(0.0).round(1)
    movies_df['internal_votes'] = movies_df['internal_votes'].fillna(0).astype(int)

def format_movies_for_frontend(df, limit=24):
    results = df.head(limit).to_dict('records')
    for m in results:
        details = get_rich_movie_data(m['title'])
        m['omdb_title'] = details['title']
        m['year'] = details['year']
        m['imdb_rating'] = details['imdb_rating']
        m['imdb_votes'] = details['imdb_votes']
        m['poster'] = details['poster']
        m['plot'] = details['plot']
        m['omdb_genre'] = details['genre']
    return results

def get_popularity_recs(genres, min_rating, min_reviews, year_min, year_max, sort_by, limit):
    filtered = filter_movies(movies_df, genres, min_rating, min_reviews, year_min, year_max)
    if sort_by == 'popularity': filtered = filtered.sort_values(by=['internal_votes', 'internal_rating'], ascending=[False, False])
    elif sort_by == 'rating': filtered = filtered.sort_values(by=['internal_rating', 'internal_votes'], ascending=[False, False])
    elif sort_by == 'latest': filtered = filtered.sort_values(by=['year_num', 'internal_votes'], ascending=[False, False])
    return format_movies_for_frontend(filtered, limit)

def filter_movies(df, genres, min_rating, min_reviews, year_min, year_max):
    res = df.copy()
    if genres and isinstance(genres, list) and len(genres) > 0 and genres[0] != '':
        pattern = '|'.join([re.escape(g) for g in genres])
        res = res[res['genres'].str.contains(pattern, case=False, na=False)]
        
    res = res[res['internal_rating'] >= min_rating]
    res = res[res['internal_votes'] >= min_reviews]
    res = res[(res['year_num'] >= year_min) & (res['year_num'] <= year_max)]
    return res

def get_content_recs(title, threshold, genres, min_rating, min_reviews, year_min, year_max, sort_by, limit):
    if not title: return []
    matches = movies_df[movies_df['title'].str.lower() == title.lower()]
    if matches.empty:
        matches = movies_df[movies_df['title'].str.lower().str.contains(title.lower(), na=False)]
        if matches.empty: return []
            
    target_idx = matches.index[0]
    sim_scores = cosine_similarity(tfidf_matrix[target_idx], tfidf_matrix).flatten()
    
    df_sim = movies_df.copy()
    df_sim['similarity'] = sim_scores
    df_sim = df_sim[(df_sim['similarity'] >= threshold) & (df_sim.index != target_idx)]
    
    filtered = filter_movies(df_sim, genres, min_rating, min_reviews, year_min, year_max)
    
    if sort_by == 'similarity': filtered = filtered.sort_values(by=['similarity', 'internal_rating'], ascending=[False, False])
    elif sort_by == 'popularity': filtered = filtered.sort_values(by=['internal_votes', 'similarity'], ascending=[False, False])
    elif sort_by == 'rating': filtered = filtered.sort_values(by=['internal_rating', 'similarity'], ascending=[False, False])
    elif sort_by == 'latest': filtered = filtered.sort_values(by=['year_num', 'similarity'], ascending=[False, False])

    return format_movies_for_frontend(filtered, limit)

# ==========================================
# 5. FLASK ROUTES
# ==========================================

@app.route('/')
def index():
    return render_template('index.html', 
                           all_genres=all_genres_list, 
                           using_dummy_data=using_dummy_data)

@app.route('/api/autocomplete')
def autocomplete():
    query = request.args.get('q', '').strip().lower()
    if not query or len(query) < 2: return jsonify([])
    matches = movies_df[movies_df['title'].str.lower().str.contains(query, na=False)].head(8)
    return jsonify([{'title': row['title']} for _, row in matches.iterrows()])

@app.route('/api/recommend', methods=['POST'])
def recommend_api():
    data = request.get_json() or {}
    active_tab = data.get('active_tab', 'content') # Defaults to AI Match
    genres = data.get('genres', [])
    
    def safe_float(key, default=0.0):
        try:
            val = data.get(key)
            return float(val) if val is not None and str(val).strip() != '' else default
        except (ValueError, TypeError):
            return default
            
    def safe_int(key, default=0):
        try:
            val = data.get(key)
            return int(val) if val is not None and str(val).strip() != '' else default
        except (ValueError, TypeError):
            return default

    min_rating = safe_float('min_rating', 0.0)
    min_reviews = safe_int('min_reviews', 0)
    year_min = safe_float('year_min', 1900)
    year_max = safe_float('year_max', 2026)
    sort_by = data.get('sort_by', 'similarity')
    limit = safe_int('limit', 24)
    
    try:
        movies = []
        if active_tab == 'popularity':
            movies = get_popularity_recs(genres, min_rating, min_reviews, year_min, year_max, sort_by, limit)
        elif active_tab == 'content':
            threshold = safe_float('similarity', 10.0) / 100.0
            movies = get_content_recs(data.get('target_title', ''), threshold, genres, min_rating, min_reviews, year_min, year_max, sort_by, limit)
            
        return render_template('movies_grid.html', movies=movies)
    except Exception as e:
        print(f"Error executing recommendation engine: {e}")
        return jsonify({'success': False, 'message': 'Internal Server Error'}), 500

@app.route('/movie/<int:movie_id>')
def movie_detail(movie_id):
    movie_row = movies_df[movies_df['movieId'] == movie_id]
    if movie_row.empty:
        flash("Movie not found in the database.", "error")
        return redirect(url_for('index'))
        
    movie = movie_row.iloc[0].to_dict()
    details = get_rich_movie_data(movie['title'])
    movie['omdb_title'] = details['title']
    movie['year'] = details['year']
    movie['imdb_rating'] = details['imdb_rating']
    movie['imdb_votes'] = details['imdb_votes']
    movie['poster'] = details['poster']
    movie['plot'] = details['plot']
    movie['omdb_genre'] = details['genre']
    
    user_rating = None
    if current_user.is_authenticated:
        rating_obj = UserRating.query.filter_by(user_id=current_user.id, movie_id=movie_id).first()
        if rating_obj: user_rating = int(rating_obj.rating)
            
    return render_template('movie_detail.html', movie=movie, user_rating=user_rating)

@app.route('/rate', methods=['POST'])
@login_required
def rate_movie():
    data = request.get_json()
    movie_id = data.get('movie_id')
    rating_val = data.get('rating')
    if not movie_id or not rating_val: return jsonify({'success': False, 'message': 'Invalid data'})
        
    existing_rating = UserRating.query.filter_by(user_id=current_user.id, movie_id=movie_id).first()
    if existing_rating: existing_rating.rating = float(rating_val)
    else: db.session.add(UserRating(user_id=current_user.id, movie_id=movie_id, rating=float(rating_val)))
        
    db.session.commit()
    return jsonify({'success': True})

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated: return redirect(url_for('index'))
    if request.method == 'POST':
        username = request.form.get('username')
        if User.query.filter_by(username=username).first():
            flash('Username already exists. Please choose another.', 'error')
            return redirect(url_for('register'))
        new_user = User(username=username, password_hash=generate_password_hash(request.form.get('password'), method='pbkdf2:sha256'))
        db.session.add(new_user)
        db.session.commit()
        login_user(new_user)
        flash('Registration successful! Welcome to RecFlix.', 'success')
        return redirect(url_for('index'))
    return render_template('auth.html', title="Register")

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated: return redirect(url_for('index'))
    if request.method == 'POST':
        user = User.query.filter_by(username=request.form.get('username')).first()
        if user and check_password_hash(user.password_hash, request.form.get('password')):
            login_user(user)
            flash('Logged in successfully!', 'success')
            return redirect(url_for('index'))
        flash('Invalid username or password.', 'error')
    return render_template('auth.html', title="Login")

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out successfully.', 'success')
    return redirect(url_for('index'))

# --- SERVERLESS INITIALIZATION ---
# Vercel imports the app, it doesn't run it via `python app.py`. 
# Therefore, we must initialize the DB and datasets outside the __main__ block.
with app.app_context(): 
    db.create_all()

load_datasets()

if __name__ == '__main__':
    app.run(debug=True, port=5000)
