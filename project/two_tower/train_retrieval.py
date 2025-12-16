import torch

import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np
import os
import time
import json
from datetime import datetime
from tqdm import tqdm

try:
    from . import config, dataset, models
except ImportError:
    import config, dataset, models

def evaluate(model, val_loader, all_item_embeddings, k=50):
    model.eval()
    hits = 0
    total = 0
    
    # Move item embeddings to GPU for fast search
    all_items = all_item_embeddings.to(model.user_tower.id_embedding.weight.device)
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            user_idx = batch['user_idx'].to(all_items.device)
            hist_indices = batch['hist_indices'].to(all_items.device)
            hist_weights = batch['hist_weights'].to(all_items.device)
            hist_mask = batch['hist_mask'].to(all_items.device)
            target_item = batch['item_idx'].to(all_items.device)
            
            user_vec = model.user_tower(user_idx, hist_indices, hist_weights, hist_mask)
            scores = torch.matmul(user_vec, all_items.T)
            _, topk_indices = torch.topk(scores, k, dim=1)
            
            targets = target_item.view(-1, 1)
            is_hit = (topk_indices == targets).any(dim=1)
            
            hits += is_hit.sum().item()
            total += len(targets)
            
    recall = hits / total if total > 0 else 0
    return recall


import argparse

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    


    # Setup Reporting Directory (Project Root)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Go up two levels from project/two_tower to tastematch/
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    run_dir = os.path.join(project_root, "runs_tt", f"run_{timestamp}")
    
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)
    print(f"Run Directory: {run_dir}")

    # Update Config with Args if needed (or just use args)
    print(f"Hyperparameters: LR={args.lr}, Batch={args.batch_size}, Epochs={args.epochs}, Patience={args.patience}")
    
    print("Loading Dataset...")
    train_ds = dataset.TwoTowerDataset(config.TRAIN_INTERACTIONS_PATH, config.ITEM_MAP_PATH, is_train=True)
    val_ds = dataset.TwoTowerDataset(config.VAL_INTERACTIONS_PATH, config.ITEM_MAP_PATH, is_train=False)
    

    # Optimize DataLoader
    num_workers = 4 # Adjust based on CPU cores (usually 4-8 is sweet spot)
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=dataset.collate_fn, 
        num_workers=num_workers,
        pin_memory=True, # Faster transfer to GPU
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None
    )
    
    val_loader = DataLoader(
        val_ds, 
        batch_size=2048, # Larger batch for validation
        shuffle=False, 
        collate_fn=dataset.collate_fn, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Load Maps
    import pickle
    with open(config.ITEM_MAP_PATH, 'rb') as f:
        data = pickle.load(f)
        item_map = data['item_map']
        num_items = len(item_map)
        
    with open(config.USER_MAP_PATH, 'rb') as f:
        user_map = pickle.load(f)
        num_users = len(user_map)
        
    print(f"Users: {num_users}, Items: {num_items}")
    
    model = models.TwoTowerModel(
        num_users, num_items, 
        config.EMBEDDING_DIM, config.LATENT_DIM
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler()
    

    best_recall = 0.0
    patience_counter = 0
    history = []
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        steps = 0
        
        pos_mu = 0
        neg_mu = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            user_idx = batch['user_idx'].to(device)
            item_idx = batch['item_idx'].to(device)
            weights = batch['weight'].to(device)
            hist_indices = batch['hist_indices'].to(device)
            hist_weights = batch['hist_weights'].to(device)
            hist_mask = batch['hist_mask'].to(device)
            
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                model_batch = {
                    'user_idx': user_idx,
                    'item_idx': item_idx,
                    'hist_indices': hist_indices,
                    'hist_weights': hist_weights,
                    'hist_mask': hist_mask
                }
                user_vec, item_vec = model(model_batch)
                
                # Debugging Logits stats
                logits = torch.matmul(user_vec, item_vec.T) / model.temperature
                # Diagonal is positive
                pos_logits = torch.diag(logits)
                # Off-diagonal is negative (approx)
                eye = torch.eye(logits.shape[0], device=logits.device)
                neg_logits = logits[eye == 0]
                
                # Stats for logging
                pos_mu += pos_logits.mean().item()
                neg_mu += neg_logits.mean().item()

                # Compute Loss (Weighted)
                loss_weighted = model.compute_loss(user_vec, item_vec, weights)
                
                # Compute Unweighted Loss for logging (Raw CrossEntropy)
                loss_unweighted = F.cross_entropy(logits, torch.arange(logits.shape[0], device=logits.device))
                
            scaler.scale(loss_weighted).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss_unweighted.item() # Log the unweighted loss to see convergence
            steps += 1
            
            pbar.set_postfix({
                'loss': total_loss/steps, 
                'pos': f"{pos_mu/steps:.2f}",
                'neg': f"{neg_mu/steps:.2f}",
                'lr': f"{optimizer.param_groups[0]['lr']:.2e}"
            })
        
        scheduler.step()
        
        avg_loss = total_loss / steps
        avg_pos = pos_mu / steps
        avg_neg = neg_mu / steps
        
        print(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")
        
        # Validation
        print("Pre-computing Item Embeddings...")
        model.eval()
        all_item_vecs = []
        all_item_indices = torch.arange(num_items, device=device)
        batch_size_eval = 2048
        
        with torch.no_grad():
            for i in range(0, num_items, batch_size_eval):
                end = min(i + batch_size_eval, num_items)
                indices = all_item_indices[i:end]
                vecs = model.item_tower(indices)
                all_item_vecs.append(vecs.cpu())
                
        all_item_embeddings = torch.cat(all_item_vecs, dim=0)
        
        recall = evaluate(model, val_loader, all_item_embeddings, k=50)
        print(f"Val Recall@50: {recall:.4f}")
        
        # Log History
        history.append({
            'epoch': epoch + 1,
            'loss': avg_loss,
            'recall_50': recall,
            'pos_logits': avg_pos,
            'neg_logits': avg_neg,
            'lr': optimizer.param_groups[0]['lr']
        })
        
        # Early Stopping Logic
        if recall > best_recall:
            best_recall = recall
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(run_dir, "best_two_tower.pth"))
            print(f"Saved Best Model (Recall: {best_recall:.4f})")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{args.patience}")
            
        if patience_counter >= args.patience:
            print("Early Stopping Triggered.")
            break
            
        # Regular Save
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), os.path.join(run_dir, f"checkpoint_epoch_{epoch+1}.pth"))
            
    # Final Report Generation
    print("Training Complete. Generating Reports...")
    
    # 1. JSON Report
    final_report = {
        'timestamp': timestamp,
        'config': vars(args),
        'best_recall': best_recall,
        'final_epoch': len(history),
        'history': history,
        'system': {
            'device': str(device),
            'cuda_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
        }
    }
    with open(os.path.join(run_dir, 'training_report.json'), 'w') as f:
        json.dump(final_report, f, indent=4)
        
    # 2. Plotting (if matplotlib available)
    try:
        import matplotlib.pyplot as plt
        
        epochs = [entry['epoch'] for entry in history]
        losses = [entry['loss'] for entry in history]
        recalls = [entry['recall_50'] for entry in history]
        pos_logits = [entry['pos_logits'] for entry in history]
        neg_logits = [entry['neg_logits'] for entry in history]
        
        plt.figure(figsize=(12, 10))
        
        # Loss
        plt.subplot(2, 2, 1)
        plt.plot(epochs, losses, label='Loss', color='red')
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.grid(True)
        
        # Recall
        plt.subplot(2, 2, 2)
        plt.plot(epochs, recalls, label='Call@50', color='blue')
        plt.title('Validation Recall@50')
        plt.xlabel('Epoch')
        plt.grid(True)
        
        # Logits
        plt.subplot(2, 2, 3)
        plt.plot(epochs, pos_logits, label='Pos Logits', color='green')
        plt.plot(epochs, neg_logits, label='Neg Logits', color='orange')
        plt.title('Logit Separation')
        plt.xlabel('Epoch')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, 'training_curves.png'))
        print(f"Plots saved to {run_dir}/training_curves.png")
        
    except ImportError:
        print("Matplotlib not installed. Skipping plots.")
    except Exception as e:
        print(f"Error generating plots: {e}")

    print(f"\nAll artifacts saved in: {run_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Two Tower Model")
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=config.BATCH_SIZE, help='Batch size')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')
    
    args = parser.parse_args()
    
    train(args)
