import os
import torch
import torch.nn as nn
import torch.optim as optim
import config
from data_loader import MovieLensDataLoader
from models import NCFModel

def run_deep_training():
    # 1. Veriyi Yükle
    loader = MovieLensDataLoader()
    loader.load_data()
    
    train_loader, val_loader = loader.get_pytorch_dataloader()
    
    # 2. Modeli Başlat
    model = NCFModel(loader.num_users, loader.num_movies, embedding_dim=config.EMBEDDING_DIM)
    model.to(config.DEVICE)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LR)
    
    print(f"Derin Model ({config.DEVICE}) üzerinde eğitiliyor. Veri seti büyük, sabırlı ol...")
    
    # 3. Eğitim Döngüsü
    for epoch in range(config.EPOCHS):
        model.train()
        total_loss = 0
        
        for batch_idx, (user_idx, movie_idx, rating) in enumerate(train_loader):
            user_idx, movie_idx, rating = user_idx.to(config.DEVICE), movie_idx.to(config.DEVICE), rating.to(config.DEVICE)
            
            optimizer.zero_grad()
            prediction = model(user_idx, movie_idx)
            loss = criterion(prediction, rating)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 1000 == 0:
                print(f"Epoch {epoch+1} Batch {batch_idx} Loss: {loss.item():.4f}")
            
        print(f"--> Epoch {epoch+1} Tamamlandı. Ort. Loss: {total_loss / len(train_loader):.4f}")
        
    # 4. Kaydet
    torch.save(model.state_dict(), os.path.join(config.MODELS_DIR, "ncf_model.pth"))
    print("Derin Model Kaydedildi.")

if __name__ == "__main__":
    run_deep_training()