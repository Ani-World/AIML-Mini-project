"""
TMDB API Helper Functions
Used to fetch movie data and posters from The Movie Database API
Reference: https://developer.themoviedb.org/docs
"""

import os
import requests
from typing import Optional, Dict

# TMDB API Configuration
TMDB_API_KEY = os.environ.get("TMDB_API_KEY", "")
TMDB_API_BASE = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p"


def build_tmdb_poster_url(poster_path: Optional[str], size: str = 'w500') -> Optional[str]:
    """
    Build TMDB poster URL from poster path.
    Reference: https://developer.themoviedb.org/docs/image-basics
    
    Args:
        poster_path: TMDB poster path (e.g., '/1E5baAaEse26fej7uHcjOgEE2t2.jpg') or full URL
        size: Image size - 'w500', 'w780', or 'original'
    
    Returns:
        Full TMDB image URL or original URL if already a full URL, None if empty
    """
    if not poster_path:
        return None
    
    # If it's already a full URL, return as is
    if poster_path.startswith('http://') or poster_path.startswith('https://'):
        return poster_path
    
    # If it's a TMDB path (starts with /), build the URL
    if poster_path.startswith('/'):
        return f"{TMDB_IMAGE_BASE}/{size}{poster_path}"
    
    # Otherwise, treat as relative path
    return f"{TMDB_IMAGE_BASE}/{size}/{poster_path}"


def search_movie_by_title(title: str, year: Optional[int] = None) -> Optional[Dict]:
    """
    Search for a movie by title using TMDB API.
    Requires TMDB_API_KEY to be set.
    
    Args:
        title: Movie title to search for
        year: Optional year to narrow search
    
    Returns:
        Movie data dict with poster_path, or None if not found/API key missing
    """
    if not TMDB_API_KEY:
        return None
    
    try:
        url = f"{TMDB_API_BASE}/search/movie"
        params = {
            "api_key": TMDB_API_KEY,
            "query": title,
            "language": "en-US"
        }
        if year:
            params["year"] = year
        
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        if data.get("results") and len(data["results"]) > 0:
            # Return first result (most relevant)
            return data["results"][0]
    except Exception as e:
        print(f"TMDB API error: {e}")
    
    return None


def get_movie_by_tmdb_id(tmdb_id: int) -> Optional[Dict]:
    """
    Get movie details by TMDB ID.
    
    Args:
        tmdb_id: The Movie Database ID
    
    Returns:
        Movie data dict or None
    """
    if not TMDB_API_KEY:
        return None
    
    try:
        url = f"{TMDB_API_BASE}/movie/{tmdb_id}"
        params = {
            "api_key": TMDB_API_KEY,
            "language": "en-US"
        }
        
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"TMDB API error: {e}")
    
    return None


def fetch_poster_for_movie(movie_name: str, movie_year: Optional[int] = None) -> Optional[str]:
    """
    Fetch poster path for a movie by searching TMDB.
    Useful for populating missing poster URLs in the database.
    
    Args:
        movie_name: Name of the movie
        movie_year: Optional year
    
    Returns:
        TMDB poster path (e.g., '/1E5baAaEse26fej7uHcjOgEE2t2.jpg') or None
    """
    movie_data = search_movie_by_title(movie_name, movie_year)
    if movie_data and movie_data.get("poster_path"):
        return movie_data["poster_path"]
    return None

