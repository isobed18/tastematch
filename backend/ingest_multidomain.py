
"""
Ingestion Script for Multi-Domain Data.
Supports: Goodreads (Books), Last.fm (Music), Yelp (Food), Steam (Games).
"""
import argparse
import os
import sys

# Add parent dir
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.database import SessionLocal
from app import models

def ingest_books(path):
    print(f"Ingesting Books from {path}...")
    # TODO: detailed logic
    pass

def ingest_music(path):
    print(f"Ingesting Music from {path}...")
    pass
    
def ingest_food(path):
    print(f"Ingesting Yelp Data from {path}...")
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", type=str, required=True, choices=["book", "music", "food", "game"])
    parser.add_argument("--path", type=str, required=True)
    
    args = parser.parse_args()
    
    if args.domain == "book":
        ingest_books(args.path)
    elif args.domain == "music":
        ingest_music(args.path)
    # ...
    
    print("Ingestion Task Completed.")
