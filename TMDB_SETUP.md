# TMDB API Key Setup

## Overview

The Movie Database (TMDB) provides free access to movie posters and metadata. This application uses TMDB in two ways:

1. **Image URLs (No API Key Required)**: TMDB image URLs are publicly accessible and don't need an API key. The `build_tmdb_poster_url()` function can construct image URLs from poster paths stored in your database.

2. **API Calls (API Key Required)**: If you want to fetch movie data or search for posters dynamically, you'll need a TMDB API key.

## Getting Your TMDB API Key

1. **Sign up for a free account**: Go to [https://www.themoviedb.org/](https://www.themoviedb.org/)
2. **Get your API key**: 
   - Go to [https://www.themoviedb.org/settings/api](https://www.themoviedb.org/settings/api)
   - Click "Create" or "Request an API Key"
   - Choose "Developer" (free)
   - Fill out the application form
   - Copy your API key

## Setting the API Key

### Option 1: Environment Variable (Recommended)

**Windows (PowerShell):**
```powershell
$env:TMDB_API_KEY="your_api_key_here"
```

**Windows (Command Prompt):**
```cmd
set TMDB_API_KEY=your_api_key_here
```

**Linux/Mac:**
```bash
export TMDB_API_KEY="your_api_key_here"
```

### Option 2: Create a .env file

Create a `.env` file in the project root:

```env
TMDB_API_KEY=your_api_key_here
PORT=5000
```

Then load it in your Python code (requires `python-dotenv` package).

### Option 3: Direct in Code (Not Recommended for Production)

You can set it directly in `Backend/app.py`:

```python
TMDB_API_KEY = "your_api_key_here"  # Only for development!
```

## Current Implementation

Currently, the app **does not require an API key** because:

- It uses TMDB image URLs directly (which are public)
- Poster paths are stored in your database (`poster_url` column)
- The `build_tmdb_poster_url()` function constructs URLs from stored paths

The API key configuration is added for **future use** if you want to:
- Fetch missing posters from TMDB API
- Search for movies and auto-populate poster URLs
- Get additional movie metadata

## Testing if API Key Works

You can test your API key by running:

```python
from Backend.tmdb_helper import search_movie_by_title
result = search_movie_by_title("The Matrix", 1999)
if result:
    print(f"Found: {result['title']}")
    print(f"Poster: {result['poster_path']}")
else:
    print("API key not set or movie not found")
```

## Reference

- [TMDB API Documentation](https://developer.themoviedb.org/docs)
- [Image Basics Guide](https://developer.themoviedb.org/docs/image-basics)
- [Get API Key](https://www.themoviedb.org/settings/api)

