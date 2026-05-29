import urllib.request
import urllib.parse
import json
import re
import os
from dotenv import load_dotenv

load_dotenv()

OMDB_API_KEY = os.getenv("OMDB_API_KEY")

def fetch_movie_details(title):
    """
    Fetches rich movie data from OMDb API.
    Returns None if API fails.
    """
    if not OMDB_API_KEY:
        print("OMDB_API_KEY not found in .env")
        return None
        
    try:
        clean_title = re.sub(r'\(\d{4}\)', '', title).strip()
        query = urllib.parse.quote(clean_title)
        
        url = f"http://www.omdbapi.com/?t={query}&apikey={OMDB_API_KEY}"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode())
            
        if data.get('Response') == 'True':
            votes_str = data.get('imdbVotes', '0').replace(',', '')
            votes = int(votes_str) if votes_str.isdigit() else 0
            
            rating_str = data.get('imdbRating', '0.0')
            rating = float(rating_str) if rating_str.replace('.', '', 1).isdigit() else 0.0
            
            poster = data.get('Poster')
            if poster == 'N/A':
                poster = None
                
            return {
                "title": data.get("Title", title),
                "year": data.get("Year", ""),
                "imdb_rating": rating,
                "imdb_votes": votes,
                "poster": poster,
                "plot": data.get("Plot", "No plot available."),
                "genre": data.get("Genre", "Unknown")
            }
            
    except Exception as e:
        print(f"OMDb API Error for '{title}': {e}")
        
    return None