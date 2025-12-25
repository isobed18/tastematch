import sqlite3
import json

def add_columns():
    conn = sqlite3.connect("tastematch.db")
    cursor = conn.cursor()
    
    try:
        # Add movie_taste column
        try:
            cursor.execute("ALTER TABLE users ADD COLUMN movie_taste JSON DEFAULT '[]'")
            print("Added movie_taste column.")
        except sqlite3.OperationalError as e:
            print(f"movie_taste column might already exist: {e}")

        # Add game_taste column
        try:
            cursor.execute("ALTER TABLE users ADD COLUMN game_taste JSON DEFAULT '[]'")
            print("Added game_taste column.")
        except sqlite3.OperationalError as e:
            print(f"game_taste column might already exist: {e}")
            
        conn.commit()
        print("Schema update complete.")
        
    except Exception as e:
        print(f"Error updating schema: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    add_columns()
