import pickle
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from app.database import SessionLocal
from app.models import Item

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS_DIR = os.path.join(BASE_DIR, "runs_ncf", "run_20251216_154202")
MOVIE_MAPPING_PATH = os.path.join(RUNS_DIR, "movie_mapping.pkl")

def debug_mapping_overlap():
    print("Checking overlap between Model Candidates and DB Items...")
    
    # 1. Load Model Mapping
    with open(MOVIE_MAPPING_PATH, 'rb') as f:
        mapping = pickle.load(f)
    model_ids = set(mapping.keys())
    print(f"Model knows {len(model_ids)} movies (by ml_id).")
    print(f"Sample Model IDs: {list(model_ids)[:10]}")
    
    # 2. Load DB Items
    db = SessionLocal()
    db_items = db.query(Item.ml_id, Item.title).filter(Item.ml_id.isnot(None)).all()
    db_ids = set([i[0] for i in db_items])
    print(f"DB has {len(db_ids)} items with ml_id.")
    print(f"Sample DB IDs: {list(db_ids)[:10]}")
    
    # 3. Intersection
    common = model_ids.intersection(db_ids)
    print(f"INTERSECTION: {len(common)} items.")
    
    if len(common) == 0:
        print("CRITICAL: No overlap! The model predicts IDs that don't exist in your DB.")
    else:
        print("Overlap looks okay. Sample matches:")
        for ml_id in list(common)[:5]:
            item = next(i for i in db_items if i[0] == ml_id)
            print(f"  ml_id={ml_id} -> {item[1]}")

if __name__ == "__main__":
    debug_mapping_overlap()
