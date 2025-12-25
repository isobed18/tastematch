import sqlite3
import os
import sys
from sqlalchemy import create_engine
from app.database import Base, engine
# Import ALL models to ensure they are registered with Base.metadata
from app.models import Message, UserInteraction 

# Direct SQLite connection for ALTER TABLE
CONN_STR = "tastematch.db"

def add_column_if_not_exists(cursor, table, col_name, col_type):
    try:
        cursor.execute(f"SELECT {col_name} FROM {table} LIMIT 1")
    except sqlite3.OperationalError:
        print(f"Adding column {col_name} to {table}...")
        try:
            cursor.execute(f"ALTER TABLE {table} ADD COLUMN {col_name} {col_type}")
            print(f"Successfully added {col_name}.")
        except Exception as e:
            print(f"Error adding {col_name}: {e}")
    else:
        print(f"Column {col_name} already exists in {table}.")

def fix_schema():
    print("Migrating Database Schema...")
    
    # 1. Update Users Table (Raw SQL)
    conn = sqlite3.connect(CONN_STR)
    cursor = conn.cursor()
    
    add_column_if_not_exists(cursor, "users", "first_name", "VARCHAR")
    add_column_if_not_exists(cursor, "users", "last_name", "VARCHAR")
    add_column_if_not_exists(cursor, "users", "tags", "TEXT") # JSON is stored as TEXT in SQLite
    
    conn.commit()
    conn.close()
    
    # 2. Create New Tables (Messages) - using SQLAlchemy
    print("Creating missing tables (Messages)...")
    Base.metadata.create_all(bind=engine)
    print("Database Schema Updated successfully.")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(current_dir)
    fix_schema()
