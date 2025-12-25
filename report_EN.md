# TasteMatch Recommendation System Report 🎬

## 1. Dataset Analysis: MovieLens 32M

We utilize the **MovieLens 32M** dataset, a gold-standard benchmark in recommender systems.

*   **Scale:**
    *   **32 Million Ratings:** Dense interaction matrix (Users × Items).
    *   **200,948 Users:** Large user base for collaborative patterns.
    *   **87,585 Movies:** Extensive item catalog.
    *   **2 Million Tags:** User-generated metadata.
*   **Data Points:**
    *   **Ratings:** 1.0 to 5.0 stars (Implicit feedback when swiping: Superlike=5, Like=4, Dislike=1-2).
    *   **Genome Scores:** A dense matrix where every movie has a relevance score for 1,128 unique tags (e.g., "atmospheric", "sci-fi", "plot twist"). This acts as a powerful **Content Embedding**.
    *   **Metadata:** We augment this with **TMDB** data (Posters, Overviews) for visual appeal and Text Embeddings.

---

## 2. User Data Acquisition (The Swipe Mechanism)

Our application acts as a data collection engine that translates user actions into training signals:

| User Action | Weight | Signal Interpretation |
| :--- | :--- | :--- |
| **Superlike** ⭐️ | **2.0** | Strong positive preference. High priority for matching locally. |
| **Like** ❤️ | **1.0** | Positive preference. Standard training target. |
| **Dislike** ❌ | **0.5** | Negative preference. **Excluded** from positive history in Two-Tower (to keep user vector "pure"). Used as negative samples in Ranking. |
| **Skip/Ignore** | **0.0** | Implicit negative or neutral. |

---

## 3. Implemented Architectures

We have explored a progression of advanced techniques, moving from simple heuristics to Deep Learning.

### A. Collaborative Filtering (CF) - `project/src/train_fast.py`
*   **Method:** **SVD (Singular Value Decomposition)**.
*   **Logic:** Decomposes the Interaction Matrix into two lower-rank matrices (User Factors × Item Factors).
*   **Pros:** Extremely fast, good for "people who liked X also liked Y".
*   **Cons:** Cannot handle new items (Cold Start), linear relationship only.

### B. Content-Based Filtering (CBF) - `project/two_tower/preprocess_content.py`
*   **Method:** **Vector Similarity**.
*   **Logic:**
    *   **Text Embeddings:** SBERT (`all-MiniLM-L6-v2`) processes Movie Overviews into 384-dim vectors.
    *   **Genome Matrix:** Uses the 1128-dim Tag Genome from MovieLens.
*   **Usage:** Used to enrich the item representation in NCF and Two-Tower. Allows recommending "similar" movies even if no one has rated them yet.

### C. Neural Collaborative Filtering (NCF) - `project/ncf/`
*   **Method:** **Hybrid MLP (Multi-Layer Perceptron)**.
*   **Architecture:**
    *   **Input:** User ID + Movie ID + **Genome Vector**.
    *   **Layers:** Concatenates embeddings and features, passes them through dense layers (e.g., 256 -> 128 -> 64).
    *   **Output:** Predicted Rating (Sigmoid scaled to 0.5 - 5.0).
*   **Pros:** Captures non-linear interactions between users and items. Uses content (Genome) to improve accuracy.
*   **Cons:** Slow inference (must score every User-Item pair).

### D. Factorization Machines (FM) - `project/fm/`
*   **Method:** **LightFM**.
*   **Logic:** A hybrid between Matrix Factorization and Linear Regression. It learns embeddings for IDs *and* features (Genres, Tags).
*   **Bias Handling:** Explicitly models User Bias and Item Bias.
*   **Pros:** Very efficient at handling sparse data and side information (metadata).

### E. Two-Tower Architecture (Current State) - `project/two_tower/`
*   **Method:** **Bi-Encoder / Retrieval & Ranking**.
*   **Goal:** Efficiently retrieve top 100 candidates from 87k movies in milliseconds.
*   **Architecture:**
    1.  **User Tower:** Encoder that transforms User ID + **Interaction History** (Sequence of liked items) into a `User Vector`.
    2.  **Item Tower:** Encoder that transforms Item ID + **Text Embedding** + **Genome** + **Genres** into an `Item Vector`.
    3.  **Training:**
        *   **InfoNCE Loss:** Maximizes similarity between User and their Positive Target, while minimizing similarity to random negatives (in-batch negatives).
    4.  **Inference:**
        *   Pre-compute all Item Vectors -> FAISS Index.
        *   Compute User Vector -> Nearest Neighbor Search.
*   **Status:** Currently tuning. Added **Early Stopping**, **Dislike Filtering**, and **Unweighted Loss** logging to improve stability.

---

## 4. Future Directions

1.  **Temporal Dynamics:** Give higher weight to recent interactions (Time Decay) in the User Tower history pooling.
2.  **Hard Negative Mining:** Train the Ranker specifically on items the Retrieval model *incorrectly* retrieves (high score but user disliked).
3.  **Session-Based RNN/GRU:** Instead of pooling history, use a GRU (Gated Recurrent Unit) to model the *sequence* of swipes.
