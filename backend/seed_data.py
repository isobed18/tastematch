import pandas as pd
from sqlalchemy.orm import Session
from app.database import SessionLocal, engine, Base
from app.models import Item, ItemType
import os
import sys
import ast

# Create tables
Base.metadata.create_all(bind=engine)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
STEAM_FILE = os.path.join(DATA_DIR, "steam_games.csv")
TMDB_FILE = os.path.join(DATA_DIR, "tmdb-movies.csv")

# Placeholder for movies (since TMDB dataset lacks images)
MOVIE_PLACEHOLDER = "https://placehold.co/600x900/e74c3c/ffffff?text=Movie+Poster"

def seed_steam_games(db: Session):
    if not os.path.exists(STEAM_FILE):
        print(f"File not found: {STEAM_FILE}")
        print("Please place 'games.csv' in the 'backend/data/' directory.")
        return

    print("Reading Steam games...")
    try:
        df = pd.read_csv(STEAM_FILE)
        # Filter for valid names and URLs
        df = df.dropna(subset=['name', 'url'])
        
        # Filter by popularity/quality if possible (e.g., recent_reviews)
        # For now, just take the top 200
        df = df.head(200)

        count = 0
        for _, row in df.iterrows():
            # Extract App ID from URL: https://store.steampowered.com/app/379720/DOOM/
            try:
                url_parts = row['url'].split('/')
                app_id_index = url_parts.index('app') + 1
                external_id = url_parts[app_id_index]
            except (ValueError, IndexError):
                continue
            
            if db.query(Item).filter(Item.external_id == external_id).first():
                continue

            # Construct image URL
            image_url = f"https://cdn.akamai.steamstatic.com/steam/apps/{external_id}/header.jpg"

            # Clean genres
            genres = row.get('genre', '')
            if pd.isna(genres):
                genres = "Unknown"
            
            item = Item(
                type=ItemType.game,
                external_id=external_id,
                title=row['name'],
                image_url=image_url,
                metadata_content={
                    "description": row.get('desc_snippet', ''),
                    "genres": genres,
                    "price": row.get('original_price', '0.0'),
                    "developers": row.get('developer', ''),
                    "mature_content": row.get('mature_content', '')
                }
            )
            db.add(item)
            count += 1
        
        db.commit()
        print(f"Seeded {count} Steam games!")
    except Exception as e:
        print(f"Error processing Steam file: {e}")

def seed_tmdb_movies(db: Session):
    if not os.path.exists(TMDB_FILE):
        print(f"File not found: {TMDB_FILE}")
        print("Please place 'tmdb-movies.csv' in the 'backend/data/' directory.")
        return

    print("Reading TMDB movies...")
    try:
        df = pd.read_csv(TMDB_FILE)
        df = df.dropna(subset=['original_title'])
        
        # Sort by popularity or vote count to get good movies
        if 'popularity' in df.columns:
            df = df.sort_values(by='popularity', ascending=False)
        
        df = df.head(200)

        count = 0
        for _, row in df.iterrows():
            external_id = str(row['imdb_id']) if pd.notna(row.get('imdb_id')) else f"tmdb_{row['id']}"
            
            if db.query(Item).filter(Item.external_id == external_id).first():
                continue

            # Genres are pipe separated in TMDB dataset: "Action|Adventure"
            genres = row.get('genres', '')
            if pd.notna(genres):
                genres = genres.replace('|', ', ')
            else:
                genres = "Unknown"

            # Construct image URL from backdrop_path if available
            image_url = MOVIE_PLACEHOLDER
            if pd.notna(row.get('backdrop_path')):
                image_url = f"https://image.tmdb.org/t/p/w780{row['backdrop_path']}"

            item = Item(
                type=ItemType.movie,
                external_id=external_id,
                title=row['original_title'],
                image_url=image_url,
                metadata_content={
                    "description": row.get('overview', ''),
                    "genres": genres,
                    "vote_average": row.get('vote_average', 0.0),
                    "year": row.get('release_year', ''),
                    "adult": row.get('adult', False)
                }
            )
            db.add(item)
            count += 1
            
        db.commit()
        print(f"Seeded {count} TMDB movies!")
    except Exception as e:
        print(f"Error processing TMDB file: {e}")

if __name__ == "__main__":
    db = SessionLocal()
    try:
        seed_steam_games(db)
        seed_tmdb_movies(db)
    finally:
        db.close()
