import pandas as pd
from app.database import SessionLocal
from app.models import Item, ItemType
import os
import json

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
STEAM_FILE = os.path.join(DATA_DIR, "steam_games.csv")
TMDB_FILE = os.path.join(DATA_DIR, "tmdb-movies.csv")

def update_metadata():
    db = SessionLocal()
    try:
        # Update Steam Games
        if os.path.exists(STEAM_FILE):
            print("Updating Steam games...")
            df = pd.read_csv(STEAM_FILE)
            df = df.dropna(subset=['name', 'url'])
            
            for _, row in df.iterrows():
                try:
                    url_parts = row['url'].split('/')
                    app_id_index = url_parts.index('app') + 1
                    external_id = url_parts[app_id_index]
                except:
                    continue
                
                item = db.query(Item).filter(Item.external_id == external_id).first()
                if item:
                    meta = dict(item.metadata_content)
                    meta['mature_content'] = row.get('mature_content', '')
                    item.metadata_content = meta
            
            db.commit()
            print("Steam games updated.")

        # Update TMDB Movies
        if os.path.exists(TMDB_FILE):
            print("Updating TMDB movies...")
            df = pd.read_csv(TMDB_FILE)
            df = df.dropna(subset=['original_title'])
            
            for _, row in df.iterrows():
                external_id = str(row['imdb_id']) if pd.notna(row.get('imdb_id')) else f"tmdb_{row['id']}"
                
                item = db.query(Item).filter(Item.external_id == external_id).first()
                if item:
                    meta = dict(item.metadata_content)
                    meta['adult'] = row.get('adult', False)
                    item.metadata_content = meta
            
            db.commit()
            print("TMDB movies updated.")
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    update_metadata()
