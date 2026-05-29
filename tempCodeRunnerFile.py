import os
import re
import pandas as pd
import numpy as np
import urllib.parse
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, render_template_string, request, redirect, url_for, flash, jsonify
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

# Hard halt if the API key is not provided as requested
if not OMDB_API_KEY:
    raise ValueError("CRITICAL ERROR: OMDB_API_KEY is missing from .env file! Please add it to start the server.")

# ==========================================
# 2. APP CONFIGURATION & SETUP
# ==========================================
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'portfolio-secret-key-xyz')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
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
    """Caches OMDb API responses to optimize performance."""
    id = db.Column(db.Integer, primary_key=True)
    search_title = db.Column(db.String(250), unique=True, nullable=False) # Original local CSV title
    title = db.Column(db.String(250)) # Official OMDb title
    year = db.Column(db.String(50))
    imdb_rating = db.Column(db.Float, default=0.0)
    imdb_votes = db.Column(db.Integer, default=0)
    poster_url = db.Column(db.String(500))
    plot = db.Column(db.Text)
    genre = db.Column(db.String(250))

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# ==========================================
# 4. RECOMMENDATION ENGINE & OMDB LOGIC
# ==========================================
movies_df = pd.DataFrame()
ratings_df = pd.DataFrame()
all_genres_list = []
using_dummy_data = False
tfidf_matrix = None

def get_rich_movie_data(title):
    """Checks SQLite cache first, then fetches from OMDb if necessary."""
    # 1. Check local cache
    cached = MovieCache.query.filter_by(search_title=title).first()
    if cached:
        return {
            'title': cached.title or title,
            'year': cached.year,
            'imdb_rating': cached.imdb_rating,
            'imdb_votes': cached.imdb_votes,
            'poster': cached.poster_url,
            'plot': cached.plot,
            'genre': cached.genre
        }

    # 2. Fetch from OMDb API
    omdb_data = fetch_omdb(title, OMDB_API_KEY)
    
    if omdb_data:
        # Cache the result to avoid future API calls
        try:
            new_cache = MovieCache(
                search_title=title,
                title=omdb_data['title'],
                year=omdb_data['year'],
                imdb_rating=omdb_data['imdb_rating'],
                imdb_votes=omdb_data['imdb_votes'],
                poster_url=omdb_data['poster'],
                plot=omdb_data['plot'],
                genre=omdb_data['genre']
            )
            db.session.add(new_cache)
            db.session.commit()
        except Exception:
            db.session.rollback()
            
        return omdb_data

    # 3. Fallback Data
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
    """Load local CSVs, configure memory dataframes, and build AI TF-IDF matrices."""
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
    """Formats dataframe and injects rich OMDb data into the final payload."""
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

def get_local_trending(limit=12):
    """Gets the most popular movies from the local dataset to replace TMDB API trends."""
    filtered = movies_df.sort_values(by=['internal_votes', 'internal_rating'], ascending=[False, False])
    return format_movies_for_frontend(filtered, limit)

def get_local_latest(limit=12):
    """Gets the newest movies from the local dataset to replace TMDB API latest."""
    filtered = movies_df.sort_values(by=['year_num', 'internal_votes'], ascending=[False, False])
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

def get_popularity_recs(genres, min_rating, min_reviews, year_min, year_max, sort_by, limit):
    filtered = filter_movies(movies_df, genres, min_rating, min_reviews, year_min, year_max)
    if sort_by == 'popularity': filtered = filtered.sort_values(by=['internal_votes', 'internal_rating'], ascending=[False, False])
    elif sort_by == 'rating': filtered = filtered.sort_values(by=['internal_rating', 'internal_votes'], ascending=[False, False])
    elif sort_by == 'latest': filtered = filtered.sort_values(by=['year_num', 'internal_votes'], ascending=[False, False])
    return format_movies_for_frontend(filtered, limit)

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
# 5. HTML TEMPLATES (Jinja2 + Tailwind CSS)
# ==========================================

BASE_HTML = """
<!DOCTYPE html>
<html lang="en" class="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RecFlix | IMDb Powered Discovery</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <script>
        tailwind.config = {
            darkMode: 'class',
            theme: { extend: { colors: { brand: '#e50914', darkbg: '#0f172a', cardbg: '#1e293b', imdb: '#f5c518' } } }
        }
    </script>
    <style>
        body { background-color: #0f172a; }
        .toast { transition: opacity 0.3s ease-in-out; }
        ::-webkit-scrollbar { width: 8px; height: 8px; }
        ::-webkit-scrollbar-track { background: #0f172a; }
        ::-webkit-scrollbar-thumb { background: #334155; border-radius: 4px; }
        ::-webkit-scrollbar-thumb:hover { background: #475569; }
        .loader { border: 3px solid #334155; border-bottom-color: #f5c518; border-radius: 50%; display: inline-block; box-sizing: border-box; animation: rotation 1s linear infinite; }
        @keyframes rotation { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        .scrollbar-hide::-webkit-scrollbar { display: none; }
        .scrollbar-hide { -ms-overflow-style: none; scrollbar-width: none; }
    </style>
</head>
<body class="text-gray-200 min-h-screen flex flex-col font-sans">
    <!-- Navbar -->
    <nav class="bg-cardbg/95 backdrop-blur-md border-b border-gray-800 shadow-xl sticky top-0 z-40">
        <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div class="flex items-center justify-between h-16">
                <a href="/" class="text-2xl font-black text-brand tracking-tighter flex items-center gap-2">
                    <i class="fa-solid fa-play"></i> RecFlix
                </a>
                <div class="flex items-center space-x-4">
                    {% if current_user.is_authenticated %}
                        <span class="text-sm text-gray-400 hidden sm:block">Hi, <span class="font-bold text-white">{{ current_user.username }}</span></span>
                        <a href="/logout" class="text-gray-300 hover:text-white px-3 py-2 text-sm font-medium transition">Logout</a>
                    {% else %}
                        <a href="/login" class="text-gray-300 hover:text-white px-3 py-2 text-sm font-medium transition">Login</a>
                        <a href="/register" class="bg-brand hover:bg-red-700 px-4 py-2 rounded-md text-sm font-bold text-white transition shadow-lg shadow-brand/30">Sign Up</a>
                    {% endif %}
                </div>
            </div>
        </div>
    </nav>

    {% if using_dummy_data %}
    <div class="bg-orange-600 text-white text-center py-2 px-4 font-bold text-sm shadow-md">
        <i class="fa-solid fa-triangle-exclamation mr-2"></i> Local dataset not found. Generating simulated library data.
    </div>
    {% endif %}

    <div id="toast-container" class="fixed bottom-5 right-5 z-50 flex flex-col gap-2"></div>

    <main class="flex-grow w-full pb-12">
        {% block content %}{% endblock %}
    </main>
    
    <footer class="bg-cardbg border-t border-gray-800 py-8 text-center text-gray-500 text-sm mt-auto">
        <p>&copy; 2026 RecFlix - Integrated with OMDb API.</p>
    </footer>

    <script>
        function showToast(message, type = 'success') {
            const container = document.getElementById('toast-container');
            const toast = document.createElement('div');
            const bgColor = type === 'error' ? 'bg-red-600' : 'bg-emerald-600';
            toast.className = `toast ${bgColor} text-white px-6 py-3 rounded shadow-lg flex items-center gap-3 opacity-0 translate-y-2 transform transition-all duration-300 z-50`;
            toast.innerHTML = `<i class="fa-solid ${type === 'error' ? 'fa-circle-exclamation' : 'fa-check-circle'}"></i> <span>${message}</span>`;
            container.appendChild(toast);
            setTimeout(() => { toast.classList.remove('opacity-0', 'translate-y-2'); }, 10);
            setTimeout(() => { toast.classList.add('opacity-0', 'translate-y-2'); setTimeout(() => toast.remove(), 300); }, 3000);
        }
    </script>
</body>
</html>
"""

INDEX_HTML = """
{% extends "base" %}
{% block content %}

<!-- Hero Banner -->
<div class="relative bg-black border-b border-gray-800 overflow-hidden shadow-2xl">
    <div class="absolute inset-0 bg-[url('https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?ixlib=rb-4.0.3&auto=format&fit=crop&w=2070&q=80')] bg-cover bg-center opacity-30 mix-blend-overlay"></div>
    <div class="absolute inset-0 bg-gradient-to-t from-darkbg via-transparent to-transparent"></div>
    
    <div class="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16 lg:py-24 text-center md:text-left">
        <h1 class="text-4xl md:text-6xl font-black text-white mb-4 tracking-tight drop-shadow-lg">Discover <span class="text-imdb">Masterpieces</span>.</h1>
        <p class="text-lg md:text-xl text-gray-300 mb-8 max-w-2xl font-light drop-shadow-md">Powered by OMDb API & Advanced Recommendation Engines.</p>
        
        <!-- Toggle Tabs -->
        <div class="inline-flex bg-cardbg/80 backdrop-blur-md rounded-lg p-1.5 border border-gray-700 shadow-2xl">
            <button onclick="switchTab('popularity')" id="tab-btn-popularity" class="tab-btn active px-6 py-2.5 rounded-md text-sm font-bold transition-all bg-imdb text-black shadow-lg flex items-center gap-2">
                <i class="fa-solid fa-fire"></i> Popular
            </button>
            <button onclick="switchTab('content')" id="tab-btn-content" class="tab-btn px-6 py-2.5 rounded-md text-sm font-bold transition-all text-gray-400 hover:text-white flex items-center gap-2">
                <i class="fa-solid fa-brain"></i> AI Match
            </button>
        </div>
    </div>
</div>

<div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 -mt-8 relative z-10">
    <!-- Advanced Filtering Panel -->
    <div class="bg-cardbg rounded-xl border border-gray-700 shadow-2xl mb-12 backdrop-blur-xl">
        <form id="filterForm" class="p-6">
            <input type="hidden" id="active_tab" name="active_tab" value="popularity">
            
            <div class="grid grid-cols-1 md:grid-cols-12 gap-8">
                <!-- Left: Mode Specific -->
                <div class="col-span-1 md:col-span-4 space-y-6">
                    <div id="content_target_container" class="hidden space-y-5">
                        <h3 class="text-xs font-black text-gray-500 uppercase tracking-widest border-b border-gray-800 pb-2">Target Movie Reference</h3>
                        <div class="relative">
                            <label class="block text-sm text-gray-300 mb-2 font-bold">Search Title <span class="text-imdb">*</span></label>
                            <div class="relative">
                                <i class="fa-solid fa-search absolute left-3.5 top-3 text-gray-500"></i>
                                <input type="text" id="target_title" name="target_title" placeholder="e.g. Inception..." class="w-full bg-darkbg border border-gray-700 rounded-lg py-2.5 pl-10 pr-3 text-white focus:outline-none focus:border-imdb focus:ring-1 focus:ring-imdb shadow-inner" autocomplete="off">
                            </div>
                            <div id="autocomplete-results" class="absolute z-50 w-full bg-cardbg border border-gray-700 rounded-lg mt-1 shadow-2xl hidden max-h-60 overflow-y-auto divide-y divide-gray-800"></div>
                        </div>
                        
                        <div>
                            <label class="block text-sm text-gray-300 mb-2 font-bold flex justify-between">
                                <span>Similarity Threshold</span> <span id="sim_val" class="text-imdb bg-imdb/10 px-2 py-0.5 rounded">10%</span>
                            </label>
                            <input type="range" id="similarity" name="similarity" min="0" max="100" value="10" class="w-full accent-imdb" oninput="document.getElementById('sim_val').innerText = this.value + '%'">
                        </div>
                    </div>

                    <div id="popularity_intro_container" class="space-y-4">
                        <h3 class="text-xs font-black text-gray-500 uppercase tracking-widest border-b border-gray-800 pb-2">Global Discovery</h3>
                        <p class="text-sm text-gray-400 leading-relaxed font-light">Explore the most beloved classics globally. Tweak the rating and year metrics below.</p>
                    </div>

                    <div class="space-y-5 pt-2">
                        <div>
                            <label class="block text-sm text-gray-300 mb-2 font-bold">Sort Order</label>
                            <select name="sort_by" id="sort_by" class="w-full bg-darkbg border border-gray-700 rounded-lg py-2.5 px-3 text-white focus:outline-none focus:border-imdb shadow-inner appearance-none">
                                <option value="popularity">Most Popular (Internal Votes)</option>
                                <option value="rating">Highest Rated (Internal)</option>
                                <option value="latest">Newest Releases</option>
                                <option value="similarity" id="opt_sim" class="hidden">AI Similarity Match</option>
                            </select>
                        </div>
                    </div>
                </div>

                <!-- Middle: Sliders -->
                <div class="col-span-1 md:col-span-4 space-y-6">
                    <h3 class="text-xs font-black text-gray-500 uppercase tracking-widest border-b border-gray-800 pb-2">Metrics</h3>
                    
                    <div>
                        <label class="block text-sm text-gray-300 mb-2 font-bold flex justify-between items-center">
                            <span>Min Rating</span> <span id="rating_val" class="text-imdb bg-imdb/10 px-2 py-0.5 rounded font-black flex items-center gap-1">3.0 <i class="fa-solid fa-star text-[10px]"></i></span>
                        </label>
                        <input type="range" name="min_rating" min="0" max="5" step="0.1" value="3.0" class="w-full accent-imdb" oninput="document.getElementById('rating_val').innerHTML = this.value + ' <i class=\\\'fa-solid fa-star text-[10px]\\\'></i>'">
                    </div>

                    <div>
                        <label class="block text-sm text-gray-300 mb-2 font-bold">Min Reviews</label>
                        <input type="number" name="min_reviews" id="min_reviews" value="10" min="0" class="w-full bg-darkbg border border-gray-700 rounded-lg py-2.5 px-3 text-white focus:outline-none focus:border-imdb shadow-inner">
                    </div>

                    <div>
                        <label class="block text-sm text-gray-300 mb-2 font-bold">Release Timeline</label>
                        <div class="flex items-center gap-3">
                            <input type="number" name="year_min" value="1980" placeholder="From" class="w-1/2 bg-darkbg border border-gray-700 rounded-lg py-2.5 px-3 text-center text-white focus:outline-none focus:border-imdb shadow-inner">
                            <span class="text-gray-600 font-black">-</span>
                            <input type="number" name="year_max" value="2026" placeholder="To" class="w-1/2 bg-darkbg border border-gray-700 rounded-lg py-2.5 px-3 text-center text-white focus:outline-none focus:border-imdb shadow-inner">
                        </div>
                    </div>
                </div>

                <!-- Right: Genres -->
                <div class="col-span-1 md:col-span-4 flex flex-col h-full">
                    <h3 class="text-xs font-black text-gray-500 uppercase tracking-widest border-b border-gray-800 pb-2">Filter By Genres</h3>
                    <div class="bg-darkbg border border-gray-700 rounded-lg p-4 flex-grow h-48 overflow-y-auto shadow-inner">
                        <div class="grid grid-cols-2 gap-y-3 gap-x-2">
                            {% for g in all_genres %}
                            <label class="flex items-center space-x-3 cursor-pointer group">
                                <input type="checkbox" name="genres" value="{{ g }}" class="form-checkbox h-4 w-4 text-imdb bg-cardbg border-gray-600 rounded focus:ring-imdb focus:ring-offset-darkbg transition">
                                <span class="text-sm text-gray-400 group-hover:text-white transition font-medium">{{ g }}</span>
                            </label>
                            {% endfor %}
                        </div>
                    </div>
                </div>
            </div>

            <!-- Actions -->
            <div class="mt-8 flex flex-col sm:flex-row justify-end items-center gap-4 border-t border-gray-800 pt-6">
                <button type="reset" class="w-full sm:w-auto px-6 py-3 rounded-lg text-sm font-bold text-gray-400 bg-transparent hover:bg-gray-800 border border-gray-700 transition">Clear All Filters</button>
                <button type="button" onclick="fetchRecommendations()" class="w-full sm:w-auto px-8 py-3 rounded-lg text-sm font-black text-black bg-imdb hover:bg-yellow-500 shadow-[0_0_15px_rgba(245,197,24,0.4)] transition transform hover:-translate-y-0.5 flex items-center justify-center gap-2">
                    <i class="fa-solid fa-play"></i> Generate Results
                </button>
            </div>
        </form>
    </div>

    <!-- Live DB Trending Section -->
    {% if trending_movies %}
    <div class="mb-14">
        <h2 class="text-xl font-bold text-white mb-4 px-2 border-l-4 border-imdb">Trending in Database</h2>
        <div class="flex overflow-x-auto gap-4 pb-6 pt-2 scrollbar-hide px-2">
            {% for movie in trending_movies %}
                <div class="flex-none w-64">
                    {% include 'movie_card' %}
                </div>
            {% endfor %}
        </div>
    </div>
    {% endif %}

    {% if latest_movies %}
    <div class="mb-14">
        <h2 class="text-xl font-bold text-white mb-4 px-2 border-l-4 border-blue-500">Latest Discoveries</h2>
        <div class="flex overflow-x-auto gap-4 pb-6 pt-2 scrollbar-hide px-2">
            {% for movie in latest_movies %}
                <div class="flex-none w-64">
                    {% include 'movie_card' %}
                </div>
            {% endfor %}
        </div>
    </div>
    {% endif %}

    <!-- Filtered Results -->
    <h2 class="text-2xl font-black text-white mb-6 border-b border-gray-800 pb-2">Filter Results</h2>
    <div id="results-section" class="relative min-h-[400px]">
        <div id="loading-spinner" class="absolute inset-0 bg-darkbg/90 backdrop-blur-md z-10 hidden flex flex-col items-center justify-center rounded-xl border border-gray-800">
            <span class="loader w-14 h-14 mb-4 shadow-[0_0_15px_rgba(245,197,24,0.5)] rounded-full"></span>
            <p class="text-imdb font-bold tracking-widest text-sm uppercase animate-pulse">Contacting OMDb API...</p>
        </div>
        
        <div id="movies-container" class="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-6">
            <!-- Populated via AJAX -->
        </div>
    </div>
</div>

<script>
    function switchTab(mode) {
        document.getElementById('active_tab').value = mode;
        const btnPop = document.getElementById('tab-btn-popularity');
        const btnCon = document.getElementById('tab-btn-content');
        
        if (mode === 'popularity') {
            btnPop.className = "tab-btn active px-6 py-2.5 rounded-md text-sm font-bold transition-all bg-imdb text-black shadow-lg flex items-center gap-2";
            btnCon.className = "tab-btn px-6 py-2.5 rounded-md text-sm font-bold transition-all text-gray-400 hover:text-white flex items-center gap-2";
            document.getElementById('content_target_container').classList.add('hidden');
            document.getElementById('popularity_intro_container').classList.remove('hidden');
            document.getElementById('opt_sim').classList.add('hidden');
            document.getElementById('sort_by').value = 'popularity';
            document.getElementById('min_reviews').value = "50";
        } else {
            btnCon.className = "tab-btn active px-6 py-2.5 rounded-md text-sm font-bold transition-all bg-imdb text-black shadow-lg flex items-center gap-2";
            btnPop.className = "tab-btn px-6 py-2.5 rounded-md text-sm font-bold transition-all text-gray-400 hover:text-white flex items-center gap-2";
            document.getElementById('content_target_container').classList.remove('hidden');
            document.getElementById('popularity_intro_container').classList.add('hidden');
            document.getElementById('opt_sim').classList.remove('hidden');
            document.getElementById('sort_by').value = 'similarity';
            document.getElementById('min_reviews').value = "0"; 
        }
    }

    const searchInput = document.getElementById('target_title');
    const resultsBox = document.getElementById('autocomplete-results');
    
    searchInput.addEventListener('input', function() {
        const query = this.value;
        if(query.length < 2) { resultsBox.classList.add('hidden'); return; }
        
        fetch(`/api/autocomplete?q=${encodeURIComponent(query)}`)
            .then(res => res.json())
            .then(data => {
                resultsBox.innerHTML = '';
                if(data.length > 0) {
                    data.forEach(item => {
                        const div = document.createElement('div');
                        div.className = "px-4 py-3 hover:bg-gray-800 cursor-pointer text-sm text-gray-300 font-medium transition flex items-center gap-3";
                        div.innerHTML = `<i class="fa-solid fa-film text-gray-600"></i> ${item.title}`;
                        div.onclick = () => {
                            searchInput.value = item.title;
                            resultsBox.classList.add('hidden');
                        };
                        resultsBox.appendChild(div);
                    });
                    resultsBox.classList.remove('hidden');
                } else {
                    resultsBox.classList.add('hidden');
                }
            });
    });
    document.addEventListener('click', (e) => { if(!searchInput.contains(e.target)) resultsBox.classList.add('hidden'); });

    function fetchRecommendations() {
        const form = document.getElementById('filterForm');
        const formData = new FormData(form);
        const data = Object.fromEntries(formData.entries());
        data.genres = formData.getAll('genres');

        if (data.active_tab === 'content' && !data.target_title.trim()) {
            showToast("Target movie title required for AI Content matching.", "error");
            searchInput.focus();
            return;
        }

        const spinner = document.getElementById('loading-spinner');
        const container = document.getElementById('movies-container');
        spinner.classList.remove('hidden');
        
        fetch('/api/recommend', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        })
        .then(response => response.text())
        .then(html => { container.innerHTML = html; })
        .catch(err => { showToast("Error connecting to OMDb Engine.", "error"); })
        .finally(() => { spinner.classList.add('hidden'); });
    }

    document.addEventListener("DOMContentLoaded", () => fetchRecommendations());
</script>
{% endblock %}
"""

MOVIE_CARD_HTML = """
<div class="flex flex-col h-full bg-cardbg rounded-xl overflow-hidden shadow-lg transition-all duration-500 hover:shadow-[0_10px_30px_rgba(0,0,0,0.8)] hover:-translate-y-2 border border-gray-800">
    <div class="aspect-[2/3] w-full bg-gray-900 relative overflow-hidden group cursor-pointer" onclick="window.location.href='/movie/{{ movie.movieId }}'">
        <img src="{{ movie.poster if movie.poster else 'https://placehold.co/400x600/1e293b/ffffff?text=' ~ movie.title|urlencode }}" alt="{{ movie.title }}" loading="lazy" class="w-full h-full object-cover transition-transform duration-700 group-hover:scale-110 opacity-90 group-hover:opacity-100">
        
        <!-- OMDb / IMDb Rating Badge -->
        <div class="absolute top-2 right-2 bg-black/80 backdrop-blur-md text-imdb text-xs font-bold px-2.5 py-1 rounded-md border border-imdb/20 shadow-lg flex items-center gap-1.5 z-20">
            <span class="bg-imdb text-black px-1 rounded-[3px] text-[9px] uppercase tracking-wider">IMDb</span> 
            <i class="fa-solid fa-star text-[10px]"></i> {{ "%.1f"|format(movie.imdb_rating) }}
        </div>
        
        <!-- Similarity Match Badge (Only visible in Content tab) -->
        {% if movie.similarity %}
        <div class="absolute top-2 left-2 bg-blue-600/90 backdrop-blur-md text-white text-[10px] font-black px-2 py-1 rounded-md shadow-lg z-20">
            {{ "%.0f"|format(movie.similarity * 100) }}% MATCH
        </div>
        {% endif %}
    </div>
    
    <!-- Rich Details Container -->
    <div class="p-5 bg-cardbg flex flex-col flex-grow">
        <h3 class="font-black text-lg text-white leading-tight mb-1 truncate" title="{{ movie.omdb_title if movie.omdb_title else movie.title }}">{{ movie.omdb_title if movie.omdb_title else movie.title }}</h3>
        
        <div class="flex items-center justify-between text-xs text-gray-400 font-medium mb-3">
            <span class="bg-gray-800 border border-gray-700 px-2 py-0.5 rounded">{{ movie.year }}</span>
            <span class="truncate ml-2 text-gray-500">{{ movie.omdb_genre.split(',')[0] if movie.omdb_genre else 'Movie' }}</span>
        </div>
        
        <!-- Truncated Plot -->
        <p class="text-sm text-gray-400 line-clamp-3 mb-4 flex-grow leading-relaxed">
            {{ movie.plot }}
        </p>
        
        <!-- Action Buttons -->
        <div class="flex items-center gap-2 mt-auto">
            <a href="/movie/{{ movie.movieId }}" class="flex-1 bg-gray-800 hover:bg-gray-700 text-white text-center py-2.5 rounded-lg text-xs font-bold transition border border-gray-700">
                Details
            </a>
            <!-- Generic YT Search for Trailer -->
            <a href="https://www.youtube.com/results?search_query={{ (movie.omdb_title if movie.omdb_title else movie.title)|urlencode }}+official+trailer" target="_blank" class="bg-gray-800 hover:bg-gray-700 text-red-500 hover:text-red-400 py-2.5 px-4 rounded-lg transition border border-gray-700 tooltip" title="Search YouTube for Trailer">
                <i class="fa-brands fa-youtube text-lg"></i>
            </a>
        </div>
    </div>
</div>
"""

MOVIES_GRID_HTML = """
{% if movies %}
    {% for movie in movies %}
        {% include 'movie_card' %}
    {% endfor %}
{% else %}
    <div class="col-span-full bg-cardbg p-16 rounded-2xl text-center border border-gray-800 shadow-inner flex flex-col items-center">
        <div class="w-20 h-20 bg-gray-800 rounded-full flex items-center justify-center mb-6">
            <i class="fa-solid fa-satellite-dish text-4xl text-gray-500"></i>
        </div>
        <h3 class="text-2xl text-white mb-2 font-black tracking-tight">No Signal Found</h3>
        <p class="text-gray-400 max-w-md">Our algorithm couldn't find a match. Try expanding your search radius.</p>
    </div>
{% endif %}
"""

MOVIE_DETAIL_HTML = """
{% extends "base" %}
{% block content %}

<!-- Blurred Backdrop Graphic -->
<div class="absolute top-0 left-0 w-full h-[60vh] z-0 overflow-hidden pointer-events-none">
    <div class="absolute inset-0 bg-darkbg/80 backdrop-blur-3xl z-10"></div>
    <div class="absolute inset-0 bg-gradient-to-b from-transparent via-darkbg to-darkbg z-20"></div>
    <img src="{{ movie.poster if movie.poster else 'https://placehold.co/400x600/1e293b/ffffff?text=' ~ movie.title|urlencode }}" class="w-full h-full object-cover opacity-30 mix-blend-luminosity transform scale-110 filter blur-xl">
</div>

<div class="relative z-30 pt-10">
    <div class="bg-cardbg/60 backdrop-blur-xl rounded-3xl overflow-hidden shadow-2xl border border-gray-700/50">
        <div class="flex flex-col lg:flex-row">
            <!-- Left: Poster -->
            <div class="w-full lg:w-1/3 p-6 md:p-8 flex flex-col items-center">
                <div class="relative rounded-xl overflow-hidden shadow-[0_20px_50px_rgba(0,0,0,0.5)] w-full max-w-sm border border-gray-800">
                    <img src="{{ movie.poster if movie.poster else 'https://placehold.co/600x900/1e293b/ffffff?text=' ~ movie.title|urlencode }}" alt="{{ movie.title }}" class="w-full h-auto object-cover">
                </div>
                
                <a href="https://www.youtube.com/results?search_query={{ movie.omdb_title|urlencode }}+official+trailer" target="_blank" class="w-full max-w-sm mt-6 bg-white hover:bg-gray-200 text-black py-3 rounded-xl text-sm font-black transition flex items-center justify-center gap-2 shadow-lg">
                    <i class="fa-brands fa-youtube text-red-600 text-lg"></i> Search YouTube Trailer
                </a>
            </div>
            
            <!-- Right: Rich Details -->
            <div class="w-full lg:w-2/3 p-6 md:p-10 lg:pl-0 flex flex-col justify-center">
                
                <div class="flex flex-wrap items-center gap-3 mb-4">
                    <span class="bg-imdb text-black px-3 py-1 rounded text-xs font-black uppercase tracking-wider shadow-sm">IMDb Record</span>
                    <span class="text-gray-300 font-mono text-sm bg-gray-800 px-2 py-1 rounded">{{ movie.year }}</span>
                </div>

                <h1 class="text-4xl md:text-6xl font-black text-white mb-6 leading-tight tracking-tight">{{ movie.omdb_title if movie.omdb_title else movie.title }}</h1>
                
                <div class="flex flex-wrap items-center gap-6 mb-8 bg-gray-900/50 p-4 rounded-2xl border border-gray-800/50 inline-flex">
                    <div class="flex items-center gap-3">
                        <div class="w-12 h-12 bg-imdb/10 rounded-full flex items-center justify-center border border-imdb/20">
                            <i class="fa-solid fa-star text-imdb text-xl"></i>
                        </div>
                        <div>
                            <p class="text-white font-black text-xl leading-none">
                                {{ "%.1f"|format(movie.imdb_rating) }}<span class="text-sm text-gray-500 font-normal">/10</span>
                            </p>
                            <p class="text-gray-500 text-[10px] uppercase font-bold tracking-wider mt-1">IMDb Rating</p>
                        </div>
                    </div>
                    
                    <div class="w-px h-10 bg-gray-700 mx-2 hidden sm:block"></div>
                    
                    <div class="flex items-center gap-3">
                        <div class="w-12 h-12 bg-gray-800 rounded-full flex items-center justify-center border border-gray-700">
                            <i class="fa-solid fa-users text-gray-400 text-lg"></i>
                        </div>
                        <div>
                            <p class="text-white font-black text-xl leading-none">{{ "{:,}".format(movie.imdb_votes) }}</p>
                            <p class="text-gray-500 text-[10px] uppercase font-bold tracking-wider mt-1">IMDb Votes</p>
                        </div>
                    </div>
                </div>

                <h3 class="text-white font-bold mb-2 text-lg">The Plot</h3>
                <p class="text-gray-300 text-base leading-relaxed max-w-3xl font-light mb-8">
                    {{ movie.plot }}
                </p>

                <h3 class="text-white font-bold mb-2 text-sm uppercase tracking-widest text-gray-500">Genres</h3>
                <div class="flex flex-wrap gap-2 mb-10">
                    {% for genre in movie.omdb_genre.split(', ') %}
                        <span class="px-4 py-1.5 bg-gray-800/80 text-gray-300 border border-gray-700/80 rounded-lg text-sm font-medium hover:bg-gray-700 transition cursor-default">{{ genre }}</span>
                    {% endfor %}
                </div>
                
                <!-- Interaction Zone -->
                <div class="bg-gradient-to-r from-gray-900 to-transparent p-6 rounded-2xl border border-gray-800 border-l-4 border-l-imdb max-w-lg">
                    <h3 class="text-xs font-black text-gray-400 uppercase tracking-widest mb-3">Your Rating</h3>
                    {% if current_user.is_authenticated %}
                        <div class="flex items-center gap-3 text-3xl text-gray-600 star-rating" id="star-container">
                            {% for i in range(1, 6) %}
                                <i class="fa-solid fa-star cursor-pointer hover:text-imdb hover:scale-110 transition-transform" 
                                   data-rating="{{ i }}" 
                                   onclick="submitRating({{ movie.movieId }}, {{ i }})"
                                   onmouseover="highlightStars({{ i }})"
                                   onmouseout="resetStars()"></i>
                            {% endfor %}
                        </div>
                        <p class="text-sm font-medium mt-3" id="rating-status">
                            {% if user_rating %} 
                                <span class="text-emerald-400 bg-emerald-400/10 px-3 py-1 rounded-full"><i class="fa-solid fa-check-circle mr-1"></i> Saved: {{ user_rating }} Stars</span>
                            {% else %} 
                                <span class="text-gray-500">Tap a star to log to your profile.</span>
                            {% endif %}
                        </p>
                    {% else %}
                        <p class="text-gray-400 text-sm"><a href="/login" class="text-black bg-imdb hover:bg-yellow-500 px-4 py-2 rounded-lg font-bold transition inline-flex items-center gap-2"><i class="fa-solid fa-lock text-black/60"></i> Login to Rate</a></p>
                    {% endif %}
                </div>
            </div>
        </div>
    </div>
</div>

<script>
    let currentRating = {{ user_rating if user_rating else 0 }};
    const stars = document.querySelectorAll('.star-rating i');

    function highlightStars(rating) {
        stars.forEach((star, index) => {
            if (index < rating) {
                star.classList.add('text-imdb', 'drop-shadow-[0_0_12px_rgba(245,197,24,0.6)]');
                star.classList.remove('text-gray-600');
            } else {
                star.classList.remove('text-imdb', 'drop-shadow-[0_0_12px_rgba(245,197,24,0.6)]');
                star.classList.add('text-gray-600');
            }
        });
    }

    function resetStars() { highlightStars(currentRating); }
    resetStars();

    function submitRating(movieId, rating) {
        fetch('/rate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ movie_id: movieId, rating: rating })
        })
        .then(response => response.json())
        .then(data => {
            if(data.success) {
                showToast(`Saved rating of ${rating} stars!`);
                currentRating = rating;
                document.getElementById('rating-status').innerHTML = `<span class="text-emerald-400 bg-emerald-400/10 px-3 py-1 rounded-full"><i class="fa-solid fa-check-circle mr-1"></i> Saved: ${rating} Stars</span>`;
                resetStars();
            } else {
                showToast(data.message, 'error');
            }
        });
    }
</script>
{% endblock %}
"""

AUTH_HTML = """
{% extends "base" %}
{% block content %}
<div class="min-h-[70vh] flex items-center justify-center">
    <div class="w-full max-w-md bg-cardbg/80 backdrop-blur-xl rounded-3xl shadow-2xl border border-gray-700 overflow-hidden">
        <div class="p-8 text-center border-b border-gray-800">
            <div class="w-16 h-16 bg-imdb/10 rounded-full flex items-center justify-center mx-auto mb-4 border border-imdb/20">
                <i class="fa-solid fa-user text-2xl text-imdb"></i>
            </div>
            <h2 class="text-3xl font-black text-white">{{ title }}</h2>
            <p class="text-sm text-gray-400 mt-2">Access your AI profile and saved ratings.</p>
        </div>
        <div class="p-8">
            <form method="POST" action="{{ request.path }}" class="space-y-6">
                <div>
                    <label class="block text-xs font-black text-gray-500 mb-2 uppercase tracking-wider">Username</label>
                    <div class="relative">
                        <i class="fa-solid fa-at absolute left-4 top-3.5 text-gray-500"></i>
                        <input type="text" name="username" required class="block w-full pl-11 pr-3 py-3 border border-gray-700 rounded-xl leading-5 bg-darkbg text-white placeholder-gray-600 focus:outline-none focus:border-imdb focus:ring-1 focus:ring-imdb transition shadow-inner">
                    </div>
                </div>
                <div>
                    <label class="block text-xs font-black text-gray-500 mb-2 uppercase tracking-wider">Password</label>
                    <div class="relative">
                        <i class="fa-solid fa-key absolute left-4 top-3.5 text-gray-500"></i>
                        <input type="password" name="password" required class="block w-full pl-11 pr-3 py-3 border border-gray-700 rounded-xl leading-5 bg-darkbg text-white placeholder-gray-600 focus:outline-none focus:border-imdb focus:ring-1 focus:ring-imdb transition shadow-inner">
                    </div>
                </div>
                <button type="submit" class="w-full flex justify-center py-3.5 px-4 rounded-xl shadow-lg text-sm font-black text-black bg-imdb hover:bg-yellow-500 transition shadow-[0_0_15px_rgba(245,197,24,0.4)] transform hover:-translate-y-0.5 mt-4">{{ title }} Securely</button>
            </form>
        </div>
    </div>
</div>
{% endblock %}
"""

# ==========================================
# 5. FLASK ROUTES
# ==========================================

@app.route('/')
def index():
    trending = get_local_trending(12)
    latest = get_local_latest(12)
    return render_template_string(INDEX_HTML, 
                                  all_genres=all_genres_list, 
                                  using_dummy_data=using_dummy_data,
                                  trending_movies=trending,
                                  latest_movies=latest)

@app.route('/api/autocomplete')
def autocomplete():
    query = request.args.get('q', '').strip().lower()
    if not query or len(query) < 2: return jsonify([])
    matches = movies_df[movies_df['title'].str.lower().str.contains(query, na=False)].head(8)
    return jsonify([{'title': row['title']} for _, row in matches.iterrows()])

@app.route('/api/recommend', methods=['POST'])
def recommend_api():
    data = request.get_json()
    active_tab = data.get('active_tab', 'popularity')
    genres = data.get('genres', [])
    min_rating = float(data.get('min_rating', 0.0))
    min_reviews = int(data.get('min_reviews', 0))
    year_min = float(data.get('year_min', 1900))
    year_max = float(data.get('year_max', 2026))
    sort_by = data.get('sort_by', 'popularity')
    limit = int(data.get('limit', 24))
    
    movies = []
    if active_tab == 'popularity':
        movies = get_popularity_recs(genres, min_rating, min_reviews, year_min, year_max, sort_by, limit)
    elif active_tab == 'content':
        threshold = float(data.get('similarity', 10)) / 100.0
        movies = get_content_recs(data.get('target_title', ''), threshold, genres, min_rating, min_reviews, year_min, year_max, sort_by, limit)
        
    return render_template_string(MOVIES_GRID_HTML, movies=movies)

@app.route('/movie/<int:movie_id>')
def movie_detail(movie_id):
    movie_row = movies_df[movies_df['movieId'] == movie_id]
    if movie_row.empty:
        flash("Movie not found.", "error")
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
            
    return render_template_string(MOVIE_DETAIL_HTML, movie=movie, user_rating=user_rating)

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
            flash('Username already exists.', 'error')
            return redirect(url_for('register'))
        new_user = User(username=username, password_hash=generate_password_hash(request.form.get('password'), method='pbkdf2:sha256'))
        db.session.add(new_user)
        db.session.commit()
        login_user(new_user)
        return redirect(url_for('index'))
    return render_template_string(AUTH_HTML, title="Register")

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated: return redirect(url_for('index'))
    if request.method == 'POST':
        user = User.query.filter_by(username=request.form.get('username')).first()
        if user and check_password_hash(user.password_hash, request.form.get('password')):
            login_user(user)
            return redirect(url_for('index'))
        flash('Invalid username or password.', 'error')
    return render_template_string(AUTH_HTML, title="Login")

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.context_processor
def utility_processor(): return dict()
class StringLoader:
    def __init__(self): self.templates = {'base': BASE_HTML, 'movies_grid': MOVIES_GRID_HTML, 'movie_card': MOVIE_CARD_HTML}
    def get_source(self, env, tmpl):
        if tmpl in self.templates: return self.templates[tmpl], tmpl, lambda: True
        raise Exception(f"Template {tmpl} not found")
app.jinja_loader = StringLoader()

if __name__ == '__main__':
    with app.app_context(): db.create_all()
    load_datasets()
    app.run(debug=True, port=5000)