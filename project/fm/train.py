import argparse
import os
import time
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from lightfm.cross_validation import random_train_test_split
from lightfm.evaluation import precision_at_k, auc_score

import config
from data_loader import FMDataLoader
from model import HybridLightFM

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def save_plot(history, metric, output_dir):
    plt.figure()
    plt.plot(history[metric])
    plt.title(f'Model {metric}')
    plt.ylabel(metric)
    plt.xlabel('Epoch')
    plt.savefig(os.path.join(output_dir, f'{metric}.png'))
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Train Hybrid FM Model (LightFM)")
    parser.add_argument('--epochs', type=int, default=config.EPOCHS)
    parser.add_argument('--lr', type=float, default=config.LEARNING_RATE)
    parser.add_argument('--components', type=int, default=config.NO_COMPONENTS)
    parser.add_argument('--loss', type=str, default=config.LOSS)
    parser.add_argument('--output_dir', type=str, default=config.DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    # Create Run Directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, f"run_{timestamp}")
    ensure_dir(run_dir)
    
    print(f"Starting Training Run: {timestamp}")
    print(f"Output Directory: {run_dir}")
    print(f"Config: {vars(args)}")

    # 1. Load Data
    loader = FMDataLoader()
    loader.load_and_process()
    interactions, weights, item_features, mappings = loader.get_matrices()

    # 2. Split Data
    print("Splitting data into Train/Test...")
    train_interactions, test_interactions = random_train_test_split(interactions, test_percentage=0.2, random_state=42)

    # 3. Initialize Model
    model = HybridLightFM(no_components=args.components, learning_rate=args.lr, loss=args.loss)
    
    # 4. Training Loop
    history = {'auc': [], 'precision': [], 'duration': []}
    best_auc = 0.0

    # Device/Thread Info
    import multiprocessing
    print(f"DEBUG: Numpy Version: {np.__version__}")
    
    # Force single thread if OpenMP is missing to avoid potential crashes
    force_single_thread = False
    num_threads = config.NUM_THREADS
    
    print(f"Device: CPU | Cores: {multiprocessing.cpu_count()} | Training Threads: {num_threads}")
    
    # CAST TO FLOAT32 explicitly to avoid C-level mismatches
    interactions = interactions.astype(np.float32)
    item_features = item_features.astype(np.float32)
    
    # Ensure CSR Format (Double check)
    import scipy.sparse as sp
    if not sp.isspmatrix_csr(interactions):
        print("WARNING: Interactions not CSR. Converting...")
        interactions = interactions.tocsr()
    if not sp.isspmatrix_csr(item_features):
        print("WARNING: Item Features not CSR. Converting...")
        item_features = item_features.tocsr()
        
    print("Starting Epochs...")
    for epoch in range(args.epochs):
        start_time = time.time()
        
        print(f"DEBUG: Entering Epoch {epoch+1}...")
        try:
            # Fit Partial (1 Epoch)
            print(f"DEBUG: Calling fit_partial for Epoch {epoch+1}...")
            
            # --- DIAGNOSTICS START ---
            print(f"DIAGNOSTICS:")
            print(f"  Train Interactions Shape: {train_interactions.shape}, Dtype: {train_interactions.dtype}, nnz: {train_interactions.nnz}")
            print(f"  Item Features Shape: {item_features.shape}, Dtype: {item_features.dtype}, nnz: {item_features.nnz}")
            
            # Verify Dimensions
            if train_interactions.shape[1] != item_features.shape[0]:
                print(f"CRITICAL WARNING: Mismatch! Interactions Cols ({train_interactions.shape[1]}) != Item Features Rows ({item_features.shape[0]})")
            else:
                print("  Dimension Check: OK (Interactions Columns match Item Features Rows)")
            # --- DIAGNOSTICS END ---

            model.fit_partial(train_interactions, item_features=item_features, epochs=1, verbose=True, num_threads=num_threads)
            print(f"DEBUG: Finished fit_partial for Epoch {epoch+1}")
            
            # --- EMERGENCY SAVE: Save immediately after training, before evaluation ---
            save_path = os.path.join(run_dir, f"model_epoch_{epoch+1}.pkl")
            model.save(save_path)
            
            # Also save mappings if not exists
            if not os.path.exists(os.path.join(run_dir, "mappings.pkl")):
                with open(os.path.join(run_dir, "mappings.pkl"), 'wb') as f:
                    pickle.dump(mappings, f)
            
            print(f"SAFEGUARD: Saved model to {save_path} (before evaluation)")
            # -------------------------------------------------------------------------
        except Exception as e:
            print(f"CRITICAL ERROR in fit_partial: {e}")
            import traceback
            traceback.print_exc()
            break
        
        duration = time.time() - start_time
        
        # Evaluate
        print("DEBUG: Starting Evaluation (Test Set)...")
        # Optimization: Skip Train AUC (too slow on CPU). Only calc Test AUC.
        # train_auc = auc_score(model.model, train_interactions, item_features=item_features).mean()
        train_auc = 0.0 # Placeholder
        
        test_auc = auc_score(model.model, test_interactions, item_features=item_features, num_threads=num_threads).mean()
        print(f"DEBUG: Test AUC Calculated: {test_auc}")
        
        test_precision = precision_at_k(model.model, test_interactions, item_features=item_features, k=5, num_threads=num_threads).mean()
        print(f"DEBUG: Precision Calculated: {test_precision}")
        
        history['auc'].append(test_auc)
        history['precision'].append(test_precision)
        history['duration'].append(duration)
        
        print(f"Epoch {epoch+1}/{args.epochs} | Time: {duration:.2f}s | Train AUC: {train_auc:.4f} | Test AUC: {test_auc:.4f} | Precision@5: {test_precision:.4f}")
        
        # Checkpoint Best Model
        if test_auc > best_auc:
            best_auc = test_auc
            model.save(os.path.join(run_dir, "best_model.pkl"))
            print(f"  --> New Best Model Saved (AUC: {best_auc:.4f})")
            
        # Save Regular Checkpoint
        if (epoch + 1) % 5 == 0:
             model.save(os.path.join(run_dir, f"checkpoint_ep{epoch+1}.pkl"))

    # 5. Final Report
    print("Training Complete. Generating Report...")
    
    # Save Report
    report = {
        'config': vars(args),
        'best_auc': best_auc,
        'history': history,
        'final_metrics': {
            'auc': float(history['auc'][-1]),
            'precision': float(history['precision'][-1])
        }
    }
    
    with open(os.path.join(run_dir, "training_report.json"), 'w') as f:
        json.dump(report, f, indent=4)
        
    # Save Plots
    save_plot(history, 'auc', run_dir)
    save_plot(history, 'precision', run_dir)
    
    # Save Mappings for Inference
    with open(os.path.join(run_dir, "mappings.pkl"), 'wb') as f:
        pickle.dump(mappings, f)

    print(f"Run Complete. All artifacts saved to {run_dir}")

if __name__ == "__main__":
    main()
