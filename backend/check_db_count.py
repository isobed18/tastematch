from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

DATABASE_URL = "sqlite:///./app/tastematch.db"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def check_db():
    db = SessionLocal()
    try:
        # Check Item Count
        result = db.execute(text("SELECT COUNT(*) FROM items"))
        count = result.scalar()
        print(f"Total Items in DB: {count}")
        
        if count == 0:
            print("DB IS EMPTY!")
            return

        # Check Modern Movies Count (Fallback relies on this)
        result = db.execute(text("SELECT COUNT(*) FROM items WHERE release_date >= '2000-01-01'"))
        modern_count = result.scalar()
        print(f"Modern Items (>= 2000): {modern_count}")
        
    except Exception as e:
        print(f"Error checking DB: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    check_db()
