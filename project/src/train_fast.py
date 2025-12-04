import os
import pickle
import config
from data_loader import MovieLensDataLoader
from models import FastSVDModel

def run_fast_training():
    # 1. Veriyi Yükle
    loader = MovieLensDataLoader()
    loader.load_data()
    
    # 2. Sparse Matrix Al
    sparse_matrix = loader.get_sparse_matrix()
    
    # 3. Modeli Eğit
    svd_model = FastSVDModel(n_components=config.EMBEDDING_DIM)
    svd_model.fit(sparse_matrix)
    
    # 4. Modeli Kaydet (Pickle ile)
    save_path = os.path.join(config.MODELS_DIR, "fast_svd_model.pkl")
    with open(save_path, 'wb') as f:
        pickle.dump(svd_model, f)
        
    print(f"Hızlı Model kaydedildi: {save_path}")

    # Mappingleri de kaydetmemiz lazım ki sonra user 0 kimdi bilelim
    with open(os.path.join(config.MODELS_DIR, "mappings.pkl"), 'wb') as f:
        pickle.dump({'user2idx': loader.user2idx, 'idx2user': loader.idx2user, 
                     'movie2idx': loader.movie2idx, 'idx2movie': loader.idx2movie}, f)

if __name__ == "__main__":
    run_fast_training()