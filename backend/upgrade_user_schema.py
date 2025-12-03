import sqlite3
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "backend", "tastematch.db")

if not os.path.exists(DB_PATH):
    # Try alternate location
    DB_PATH = os.path.join(BASE_DIR, "tastematch.db")

print(f"Connecting to database: {DB_PATH}")

try:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("PRAGMA table_info(users)")
    columns = [info[1] for info in cursor.fetchall()]
    
    print(f"Current User Columns: {columns}")
    
    if "embedding" not in columns:
        print("Adding 'embedding' column...")
        # SQLite doesn't natively enforce JSON, typically TEXT or JSON type (if enabled)
        cursor.execute("ALTER TABLE users ADD COLUMN embedding JSON")
    else:
        print("'embedding' column exists.")

    if "last_daily_feed" not in columns:
        print("Adding 'last_daily_feed' column...")
        cursor.execute("ALTER TABLE users ADD COLUMN last_daily_feed DATETIME")
    else:
        print("'last_daily_feed' column exists.")
        
    conn.commit()
    conn.close()
    print("Migration complete.")

except Exception as e:
    print(f"Error: {e}")
