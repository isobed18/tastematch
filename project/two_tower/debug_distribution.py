
import pickle
import sys
import os
import collections

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from project.two_tower import config

def check_distribution():
    print("Checking Interaction Weights...")
    if not os.path.exists(config.TRAIN_INTERACTIONS_PATH):
        print("Train interactions not found.")
        return

    with open(config.TRAIN_INTERACTIONS_PATH, 'rb') as f:
        data = pickle.load(f)
        
    weight_counts = collections.Counter()
    total_events = 0
    
    for user in data:
        for evt in user['events']:
            # event: (item, weight, ts)
            w = evt[1]
            weight_counts[w] += 1
            total_events += 1
            
    print(f"Total Events: {total_events}")
    print("Weight Distribution:")
    for w, c in weight_counts.items():
        print(f"  Weight {w}: {c} ({c/total_events*100:.2f}%)")

if __name__ == "__main__":
    check_distribution()
