import sqlite3
import os

# Path to database
# Assuming running from root or backend, let's try to locate it.
DB_PATH = "backend/tastematch.db"
if not os.path.exists(DB_PATH):
    DB_PATH = "tastematch.db" # If inside backend dir

if not os.path.exists(DB_PATH):
    print(f"Database not found at {DB_PATH}")
    # Try absolute path based on user cwd assumption
    DB_PATH = r"C:\Users\ishak\tastematch\backend\tastematch.db"

print(f"Connecting to database: {DB_PATH}")

try:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Check if column exists
    cursor.execute("PRAGMA table_info(swipes)")
    columns = [info[1] for info in cursor.fetchall()]
    
    if "rating" not in columns:
        print("Adding 'rating' column to 'swipes' table...")
        cursor.execute("ALTER TABLE swipes ADD COLUMN rating FLOAT")
        conn.commit()
        print("Migration successful.")
    else:
        print("Column 'rating' already exists.")
        
    conn.close()

except Exception as e:
    print(f"Error: {e}")
