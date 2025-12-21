
import sqlite3
import os

DB_PATH = "tastematch.db"

def migrate():
    print(f"Migrating {DB_PATH}...")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    columns = [
        ("embedding_version", "INTEGER DEFAULT 0"),
        ("birth_date", "TEXT"),
        ("gender", "TEXT"),
        ("interested_in", "TEXT"),
        ("location_city", "TEXT"),
        ("bio", "TEXT"),
        ("last_active", "TIMESTAMP")
    ]
    
    for col_name, col_type in columns:
        try:
            print(f"Adding column {col_name}...")
            cursor.execute(f"ALTER TABLE users ADD COLUMN {col_name} {col_type}")
            print("Done.")
        except sqlite3.OperationalError as e:
            if "duplicate column name" in str(e):
                print(f"Column {col_name} already exists. Skipping.")
            else:
                print(f"Error adding {col_name}: {e}")

    conn.commit()
    conn.close()
    print("Migration complete.")

if __name__ == "__main__":
    migrate()
