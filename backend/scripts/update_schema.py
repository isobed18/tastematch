
import sys
import os
from sqlalchemy import create_engine, text

# Add parent dir to path to import app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import SQLALCHEMY_DATABASE_URL, Base, engine
from app.models import User, Item, Interaction, ItemEmbedding

def update_schema():
    print("Updating database schema...")
    
    # 1. Create new tables (Interaction, ItemEmbedding)
    # create_all only creates tables that don't exist
    Base.metadata.create_all(bind=engine)
    print("New tables (interactions, item_embeddings) created (if not existed).")

    # 2. Add columns to existing tables (User, Item)
    # SQLite doesn't support IF NOT EXISTS in ALTER TABLE ADD COLUMN universally in safe ways, 
    # so we try and catch errors.
    
    with engine.connect() as conn:
        # User columns
        user_columns = [
            ("taste_vectors", "JSON"),
            ("embedding_updated_at", "DATETIME"),
            ("preferences", "JSON")
        ]
        
        for col_name, col_type in user_columns:
            try:
                # Check if column exists strictly if possible, or just try-catch
                conn.execute(text(f"ALTER TABLE users ADD COLUMN {col_name} {col_type}"))
                print(f"Added column {col_name} to users table.")
            except Exception as e:
                # Likely "duplicate column name" error
                if "duplicate column name" in str(e).lower():
                    print(f"Column {col_name} already exists in users table.")
                else:
                    print(f"Error adding {col_name} to users: {e}")

        # Item columns
        item_columns = [
            ("geo_location", "JSON")
        ]
        
        for col_name, col_type in item_columns:
            try:
                conn.execute(text(f"ALTER TABLE items ADD COLUMN {col_name} {col_type}"))
                print(f"Added column {col_name} to items table.")
            except Exception as e:
                if "duplicate column name" in str(e).lower():
                    print(f"Column {col_name} already exists in items table.")
                else:
                    print(f"Error adding {col_name} to items: {e}")

    print("Schema update complete.")

if __name__ == "__main__":
    update_schema()
