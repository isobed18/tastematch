import argparse
import os
import time
import json
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
from tqdm import tqdm
from datetime import datetime

import config
from data_loader import NCFDataProcessor, HybridNCFDataset
from model import HybridNCF

def parse_args():
    parser = argparse.ArgumentParser(description="Train Hybrid NCF Model (Regression)")
    parser.add_argument('--epochs', type=int, default=config.EPOCHS, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=config.BATCH_SIZE, help='Batch size')
    parser.add_argument('--lr', type=float, default=config.LEARNING_RATE, help='Learning rate')
    parser.add_argument('--embedding_dim', type=int, default=config.EMBEDDING_DIM, help='Embedding dimension')
    parser.add_argument('--output_dir', type=str, default=config.OUTPUT_DIR, help='Output directory for artifacts')
    parser.add_argument('--device', type=str, default=config.DEVICE, help='Training device (cuda/cpu)')
    return parser.parse_args()

def evaluate(model, test_loader, device):
    """
    Evaluates the model on test data using RMSE and MAE.
    Note: For full-GPU mode, test_loader is not used here, but logic is similar.
    This function is kept for legacy or CPU mode potential.
    """
    pass # Replaced by inline evaluation in main loop for Full-GPU mode

def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)

def main():
    args = parse_args()
    
    # 0. Setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, f"run_{timestamp}")
    ensure_dir(run_dir)
    
    print(f"--- Hybrid NCF Training (Regression) [{timestamp}] ---")
    print(f"Device: {args.device}")
    print(f"Config: {vars(args)}")
    
    # 1. Load Data
    processor = NCFDataProcessor()
    # Note: load_data maps IDs and builds genome matrix
    ratings_df = processor.load_data()
    
    # Save Mappings
    with open(os.path.join(run_dir, 'user_mapping.pkl'), 'wb') as f:
        pickle.dump(processor.user_mapping, f)
    with open(os.path.join(run_dir, 'movie_mapping.pkl'), 'wb') as f:
        pickle.dump(processor.movie_mapping, f)
    np.save(os.path.join(run_dir, 'genome_matrix.npy'), processor.genome_matrix)
    print("Mappings and Genome Matrix saved.")

    # 2. Split
    train_df, test_df, _ = processor.split_train_test(ratings_df)
    
    # 3. Create Datasets
    # Important: No negatives needed for regression.
    
    print("Initializing Training Dataset...")
    train_dataset = HybridNCFDataset(
        train_df, 
        processor.genome_matrix
    )
    
    print("Initializing Test Dataset...")
    test_dataset = HybridNCFDataset(
        test_df, 
        processor.genome_matrix 
    )
    
    # 4. Initialize Model
    model = HybridNCF(
        num_users=processor.num_users,
        num_movies=processor.num_movies,
        genome_dim=config.GENOME_DIM,
        embedding_dim=args.embedding_dim
    )
    model.to(args.device)
    
    # --- GPU OPTIMIZATION: Move ALL Data to VRAM ---
    print(f"Moving ALL Training Data to {args.device} (VRAM)...")
    train_users = torch.tensor(train_dataset.users_list, dtype=torch.long).to(args.device)
    train_movies = torch.tensor(train_dataset.items_list, dtype=torch.long).to(args.device)
    train_labels = torch.tensor(train_dataset.labels_list, dtype=torch.float32).to(args.device)
    
    print(f"Moving Test Data to {args.device}...")
    test_users = torch.tensor(test_dataset.users_list, dtype=torch.long).to(args.device)
    test_movies = torch.tensor(test_dataset.items_list, dtype=torch.long).to(args.device)
    test_labels = torch.tensor(test_dataset.labels_list, dtype=torch.float32).to(args.device)
    
    print(f"Moving Genome Matrix to {args.device}...")
    genome_tensor = torch.tensor(processor.genome_matrix, dtype=torch.float32).to(args.device)
    
    total_train_samples = len(train_users)
    print(f"Data Loaded to VRAM. Train Size: {total_train_samples}")
    
    # CRITICAL: Use MSELoss for Regression
    criterion = nn.MSELoss() 
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # 5. Training Loop
    best_rmse = float('inf')
    patience_counter = 0 # Early Stopping
    history = []
    
    print("Starting Training Loop (Full GPU Mode)...")
    for epoch in range(args.epochs):
        start_time = time.time()
        # ... (Training Code similar)
        
        # To avoid duplicating the whole loop in the prompt, I will target the end of the loop 
        # But replace_file_content needs contiguous block. 
        # I'll just check "Checkpoints" section where 'best_rmse' is updated.
        
    # Wait, I need to insert the variable initialization BEFORE the loop and the check INSIDE.
    # I better use multi_replace or ensure I cover the right parts.
    # The tool call above targets the start of the loop.
    
    # Let me re-read train.py to make sure I get line numbers right or context right.
    pass
    for epoch in range(args.epochs):
        start_time = time.time()
        model.train()
        train_loss = 0.0
        
        # Shuffle indices
        indices = torch.randperm(total_train_samples, device=args.device)
        
        num_batches = (total_train_samples + args.batch_size - 1) // args.batch_size
        pbar = tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for i in pbar:
            start_idx = i * args.batch_size
            end_idx = min(start_idx + args.batch_size, total_train_samples)
            batch_idx = indices[start_idx:end_idx]
            
            # Direct VRAM Slicing
            user = train_users[batch_idx]
            movie = train_movies[batch_idx]
            label = train_labels[batch_idx]
            features = genome_tensor[movie] 
            
            optimizer.zero_grad()
            output = model(user, movie, features) # Output: Raw Rating Prediction
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'mse_loss': loss.item()})
            
        train_loss /= num_batches
        
        # Validation Step (Full GPU Batching)
        model.eval()
        with torch.no_grad():
            val_targets = []
            val_preds = []
            
            total_test = len(test_users)
            num_test_batches = (total_test + args.batch_size - 1) // args.batch_size
            
            for i in range(num_test_batches):
                s = i * args.batch_size
                e = min(s + args.batch_size, total_test)
                
                u_b = test_users[s:e]
                m_b = test_movies[s:e]
                l_b = test_labels[s:e]
                f_b = genome_tensor[m_b]
                
                out = model(u_b, m_b, f_b)
                # No sigmoid! Models predicts rating directly (e.g. 3.5)
                
                val_targets.extend(l_b.cpu().numpy())
                val_preds.extend(out.cpu().numpy())
                
            # Metrics for Regression
            rmse = np.sqrt(mean_squared_error(val_targets, val_preds))
            mae = mean_absolute_error(val_targets, val_preds)
        
        duration = time.time() - start_time
        print(f"Epoch {epoch+1} Summary: Train MSE: {train_loss:.4f} | Val RMSE: {rmse:.4f} | Val MAE: {mae:.4f} | Time: {duration:.1f}s")
        
        # Record history
        history.append({
            'epoch': epoch + 1,
            'train_mse': train_loss,
            'val_rmse': rmse,
            'val_mae': mae,
            'time': duration
        })
        
        # Checkpoint (Save if RMSE improved)
        if rmse < best_rmse:
            best_rmse = rmse
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(run_dir, "best_model.pth"))
            print(f"  --> Best Model Saved (RMSE: {rmse:.4f})")
        else:
            patience_counter += 1
            print(f"  --> No improvement. Patience: {patience_counter}/{config.PATIENCE}")
            
        if patience_counter >= config.PATIENCE:
            print("EARLY STOPPING TRIGGERED.")
            break
            
        # Regular save
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), os.path.join(run_dir, f"checkpoint_epoch_{epoch+1}.pth"))
            
    # 6. Final Report
    print("Training Complete. Saving Report...")
    report = {
        'config': vars(args),
        'best_rmse': best_rmse,
        'history': history
    }
    
    with open(os.path.join(run_dir, 'training_report.json'), 'w') as f:
        json.dump(report, f, indent=4)
        
    print(f"Done. Artifacts in {run_dir}")

if __name__ == "__main__":
    main()
