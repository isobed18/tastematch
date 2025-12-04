# TasteMatch 🎬

TasteMatch is a Tinder-like movie recommendation application that learns your taste as you swipe. It uses advanced machine learning models (SVD, Neural Collaborative Filtering) to provide personalized recommendations.

## 🚀 Features

*   **Swipe Interface:** Like/Dislike/Superlike movies to build your taste profile.
*   **Personalized Feed:** Recommendations update in real-time based on your interactions.
*   **Match Logic:**
    *   **Perfect Match 🎯:** High-confidence recommendations.
    *   **Reverse Match 🤪:** "Definitely not your taste" items for exploration.
*   **Dynamic Triggers:** Special matches appear after a random number of swipes.

## 📦 Data Sources

We use the following datasets for training and metadata:

1.  **TMDB Movies Dataset (Metadata):**
    *   Used for movie details (titles, posters, overviews).
    *   [Download from Kaggle](https://www.kaggle.com/datasets/asaniczka/tmdb-movies-dataset-2023-930k-movies)
    *   *Note: You need to download `tmdb-movies.csv` and place it in `backend/data/` (create the folder if needed).*

2.  **MovieLens 32M (Training Data):**
    *   Used to train the collaborative filtering models.
    *   **Description:** 32 million ratings and two million tag applications applied to 87,585 movies by 200,948 users. Collected 10/2023 Released 05/2024.
    *   [Download from GroupLens](https://grouplens.org/datasets/movielens/)

## 🛠️ Setup & Installation

### Prerequisites
*   Python 3.9+
*   Node.js & npm
*   Git LFS (Large File Storage)

### 1. Clone the Repository
This repository uses Git LFS for storing trained models. Make sure to install Git LFS first.

```bash
git lfs install
git clone https://github.com/isobed18/tastematch.git
cd tastematch
git lfs pull  # Download the actual model files
```

### 2. Backend Setup

```bash
cd backend
python -m venv venv
# Windows
.\venv\Scripts\activate
# Mac/Linux
source venv/bin/activate

pip install -r requirements.txt
```

**Database Setup:**
You need to create the SQLite database.
1.  Ensure you have `tmdb-movies.csv` in `backend/data/`.
2.  Run the ingestion script (this might take a while):
    ```bash
    python ingest_simple.py
    ```

**Run the Server:**
```bash
uvicorn app.main:app --reload
```

### 3. Frontend Setup

```bash
cd frontend2/tastematch
npm install
npx expo start
```

## 🧠 AI Models

Our recommendation engine is powered by models trained on the MovieLens 32M dataset.

*   **Location:** `project/models/`
*   **Models:**
    *   `fast_svd_model.pkl`: Singular Value Decomposition model for fast, latent-factor based recommendations.
    *   `ncf_model.pth`: Neural Collaborative Filtering model (Deep Learning) for capturing complex patterns.
*   **Training:**
    *   Training scripts are located in `project/src/`.
    *   `train_fast.py`: Trains the SVD model.
    *   `train_deep.py`: Trains the NCF model.

## 🤝 Contributing

1.  Fork the repository.
2.  Create a feature branch.
3.  Commit your changes.
4.  Push to the branch.
5.  Open a Pull Request.
