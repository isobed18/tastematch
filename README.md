# TasteMatch - AI Powered Multi-Domain Dating App

**TasteMatch is not just a swiping app; it's a "Taste Inference Engine" that connects people through shared passions in Films, Books, Music, and Food.**

By leveraging a **Shared Semantic Space**, TasteMatch understands that a user who loves "Dark Sci-Fi Movies" might also enjoy "Cyberpunk Literature" and "Underground Techno Clubs," and matches them with someone compatible across these diverse domains.

---

## 🚀 Features

*   **Multi-Domain Taste Profiling**: 
    *   **Movies**: Powered by MovieLens data + TMDB metadata.
    *   **Books, Music, Food, Games**: Architecture ready for cross-domain vectors.
*   **Intelligent Matching**:
    *   **Mirror Match**: You share the exact same taste.
    *   **Complement Match**: Similar "vibe" but different genres (e.g., You like Hard Sci-Fi, they like High Fantasy).
    *   **Contrast Match**: Opposites that create great conversation potential.
*   **Date Concierge**: 
    *   The app suggests *where* to go based on your combined food/venue tastes.
    *   "Authority Pitch" mode helps break the indecision loop.
*   **Shared Semantic Space**: A unified vector architecture where User and Item embeddings live in the same latent dimension (512D).

## 🏗 System Architecture

TasteMatch uses a **Two-Tower Architecture** enhanced with **Weighted Taste Fusion** and **Vector Retrieval**.

### 1. Shared Semantic Space
Instead of separate silos for each interest, we map all users and items into a shared 512-dimensional space.
*   **User Tower**: Aggregates interaction histories from all domains (Movie, Book, etc.) into a single `Composite User Vector`.
*   **Item Towers**: Domain-specific encoders (e.g., BERT for Text, ResNet for Images, separate embeddings for ID) project items into the same space.

### 2. Inference Pipeline
1.  **Retrieval (ANN)**: Uses **ChromaDB** and **Two-Tower** indices to instantly fetch top-k candidates even from millions of items.
2.  **Filter**: Removes seen items and applies hard constraints (Geo-location for dates).
3.  **Ranking**: A pairwise interaction model (NCF/MLP) re-ranks candidates to maximize personal relevance.
4.  **Mixing**: The `MultiDomainInferenceService` fuses vectors:
    ```python
    Composite = Normalize( w_movie*V_movie + w_book*V_book + ... )
    ```

### 3. Tech Stack
*   **Backend**: Python, FastAPI, SQLAlchemy, SQLite (Development).
*   **AI/ML**: PyTorch, Sentence-Transformers, ChromaDB (Vector Store).
*   **Frontend**: React Native (Expo).
*   **DevOps**: Docker, Git LFS.

---

## 📦 Model Management (Git LFS)

We strictly use **Git LFS** to manage large model assets. **Only the best performing Production Models are tracked.**

*   `production_models/two_tower_retrieval_v1.pth`: Core retrieval engine.
*   `production_models/two_tower_ranker_v1.pth`: Precision ranking model.

intermediate checkpoints (`epoch_*.pth`) are **ignored** to keep the repository light.

---

## 🛠️ Setup & Installation

### Prerequisites
*   Python 3.9+
*   Node.js & npm
*   Git LFS
*   (Optional) Docker

### 1. Backend Setup

```bash
cd backend
python -m venv venv
# Windows
.\venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

pip install -r requirements.txt
```

**Database & Migration:**
```bash
# 1. Update Schema for Multi-Domain support
python scripts/update_schema.py

# 2. Migrate legacy Movie Swipes to Unified Interaction Table
python scripts/migrate_swipes.py
```

**Run Server:**
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 2. Frontend Setup

```bash
cd frontend2/tastematch
npm install
npx expo start
```

---

## 📊 Data Ingestion

To populate the vector database with new domains:

```bash
# Example
python ingest_multidomain.py --domain book --path /path/to/goodreads.csv
python ingest_multidomain.py --domain food --path /path/to/yelp.json
```

---

## 🤝 Contributing

1.  **Feature Branch**: Create a branch for your specific domain or feature (`feature/book-domain`).
2.  **Commit**: Use meaningful commit messages.
3.  **LFS**: Ensure you are NOT committing 1GB+ checkpoints unless they are the final `production_models`.
4.  **PR**: Submit a Pull Request to `main`.
