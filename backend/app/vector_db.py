import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import os

# Initialize Sentence Transformer model
# Using a lightweight model for speed
model = SentenceTransformer('all-MiniLM-L6-v2')

class VectorDB:
    def __init__(self):
        self.client = None
        self.items_collection = None
        self.users_collection = None
        self.initialized = False

    def initialize(self):
        if self.initialized:
            return

        print("Initializing ChromaDB...")
        # Persist data to disk
        persist_directory = os.path.join(os.getcwd(), "chroma_db")
        self.client = chromadb.PersistentClient(path=persist_directory)

        # Create or get collections
        self.items_collection = self.client.get_or_create_collection(
            name="items",
            metadata={"hnsw:space": "cosine"} # Use cosine similarity
        )
        
        # We might not need a users collection if we calculate user vector on the fly
        # But storing it allows for faster retrieval and "reverse match" logic
        self.users_collection = self.client.get_or_create_collection(
            name="users",
            metadata={"hnsw:space": "cosine"}
        )

        self.initialized = True
        print("ChromaDB Initialized.")

    def generate_vector(self, text: str) -> List[float]:
        """Generate embedding for a single text string."""
        return model.encode(text).tolist()

    def add_items(self, items: List[Dict[str, Any]]):
        """
        Add items to the vector database.
        items: List of dicts with keys: id, text, metadata
        """
        if not items:
            return

        ids = [str(item['id']) for item in items]
        documents = [item['text'] for item in items]
        metadatas = [item['metadata'] for item in items]
        embeddings = [self.generate_vector(doc) for doc in documents]

        self.items_collection.upsert(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents
        )
        print(f"Added/Updated {len(items)} items in ChromaDB.")

    def update_user_vector(self, user_id: str, vector: List[float], metadata: Dict[str, Any] = None):
        """Update or create a user's taste vector."""
        self.users_collection.upsert(
            ids=[str(user_id)],
            embeddings=[vector],
            metadatas=[metadata or {}]
        )

    def search_items(self, query_vector: List[float], limit: int = 10, where: Dict[str, Any] = None, where_document: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Search for similar items."""
        results = self.items_collection.query(
            query_embeddings=[query_vector],
            n_results=limit,
            where=where,
            where_document=where_document
        )
        
        # Parse results into a cleaner format
        parsed_results = []
        if results['ids']:
            for i in range(len(results['ids'][0])):
                parsed_results.append({
                    'id': results['ids'][0][i],
                    'score': results['distances'][0][i] if 'distances' in results else 0,
                    'metadata': results['metadatas'][0][i] if 'metadatas' in results else {},
                    'document': results['documents'][0][i] if 'documents' in results else ""
                })
                
        return parsed_results

# Global instance
vector_db = VectorDB()
