from app.database import SessionLocal
from app import models
from app.vector_db import vector_db

def migrate_items():
    db = SessionLocal()
    try:
        print("Starting migration to ChromaDB...")
        vector_db.initialize()
        
        # Fetch all items
        items = db.query(models.Item).all()
        print(f"Found {len(items)} items in SQLite.")
        
        batch_size = 50
        batch = []
        
        for i, item in enumerate(items):
            meta = item.metadata_content or {}
            genres = meta.get('genres', '')
            if isinstance(genres, list):
                genres = ", ".join(genres)
                
            description = meta.get('description') or meta.get('overview') or ""
            
            # Construct text for embedding
            # "Title: Matrix. Genres: Action, Sci-Fi. Description: A computer hacker..."
            text_to_embed = f"Title: {item.title}. Genres: {genres}. Description: {description}"
            
            # Handle item.type safely (it might be a string or Enum)
            item_type_str = item.type.value if hasattr(item.type, 'value') else str(item.type)

            batch.append({
                "id": item.id,
                "text": text_to_embed,
                "metadata": {
                    "type": item_type_str,
                    "title": item.title,
                    "genres": genres
                }
            })
            
            if len(batch) >= batch_size:
                print(f"Processing batch {i - batch_size + 1} to {i}...")
                vector_db.add_items(batch)
                batch = []
                
        # Process remaining
        if batch:
            print(f"Processing final batch of {len(batch)} items...")
            vector_db.add_items(batch)
            
        print("Migration complete!")
        
    except Exception as e:
        print(f"Error during migration: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    migrate_items()
