from sqlalchemy import Column, Integer, String, ForeignKey, Float, DateTime, JSON
from sqlalchemy.orm import relationship
from .database import Base
import datetime
import enum

class SwipeAction(str, enum.Enum):
    like = "like"
    dislike = "dislike"
    superlike = "superlike"
    watchlist = "watchlist"

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    # Identifiers
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    
    # 1. Taste Vectors (The separate domain tastes)
    # Structure: { "movie": [...], "book": [...], "music": [...], "food": [...] }
    taste_vectors = Column(JSON, default={}) 
    
    # 2. Composite Embedding (The Shared Semantic Space Vector)
    embedding = Column(JSON, default=[]) 
    embedding_version = Column(Integer, default=0)
    embedding_updated_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    # 3. Preferences & Constraints
    # Structure: { "max_distance_km": 50, "budget": 2, "dietary": ["vegan"], "available_times": [...] }
    preferences = Column(JSON, default={})
    
    # Social Profile Fields
    birth_date = Column(String, nullable=True) # YYYY-MM-DD
    gender = Column(String, nullable=True) # male, female, other
    interested_in = Column(String, nullable=True) # male, female, both, etc.
    location_city = Column(String, nullable=True)
    first_name = Column(String, nullable=True)
    last_name = Column(String, nullable=True)
    bio = Column(String, nullable=True)
    last_active = Column(DateTime, default=datetime.datetime.utcnow)
    
    # Auto-Generated Persona Tags (e.g. ["Sci-Fi Lover", "Indie Fan"])
    tags = Column(JSON, default=[]) 

    last_daily_feed = Column(DateTime, nullable=True) # Son öneri tarihi
    daily_match_ml_id = Column(Integer, nullable=True) # Günün önerisi (Cache)
    
    swipes = relationship("Swipe", back_populates="user")

class Item(Base):
    __tablename__ = "items"

    id = Column(Integer, primary_key=True, index=True)
    
    # --- YENİ MODEL İÇİN GEREKLİ ALANLAR ---
    ml_id = Column(Integer, index=True, nullable=True) # Modelin bildiği ID
    tmdb_id = Column(String, index=True, nullable=True) # TMDB ID
    
    type = Column(String, default="movie")
    external_id = Column(String, unique=True, index=True) # tmdb_123
    title = Column(String)
    overview = Column(String)
    genres = Column(String)
    
    # Görseller
    poster_path = Column(String)   # Dikey Afiş (https://image.tmdb...) (Book Cover, Album Art)
    backdrop_path = Column(String) # Yatay Kapak
    
    # --- YENİ ALANLAR (Geo/Venue) ---
    # Format: {"lat": 41.0, "lng": 29.0, "address": "..."}
    geo_location = Column(JSON, nullable=True) 

    
    vote_average = Column(Float, default=0.0)
    vote_count = Column(Integer, default=0)
    popularity = Column(Float, default=0.0)
    release_date = Column(String, nullable=True)
    
    metadata_content = Column(JSON, nullable=True) # Ekstra veriler için

    swipes = relationship("Swipe", back_populates="item")

class Swipe(Base):
    __tablename__ = "swipes"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    item_id = Column(Integer, ForeignKey("items.id"))
    action = Column(String)
    rating = Column(Float, nullable=True) # Backend Entegrasyonu (NCF rating: 0.5 - 5.0)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

    user = relationship("User", back_populates="swipes")
    item = relationship("Item", back_populates="swipes")

class Interaction(Base):
    """
    Unified Interaction Table for Multi-Domain.
    Replaces 'Swipe' table.
    """
    __tablename__ = "interactions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    item_id = Column(Integer, ForeignKey("items.id"))
    
    item_type = Column(String) # movie, book, music, food
    action = Column(String)    # like, dislike, superlike, view, click, save
    weight = Column(Float, default=1.0) # Explicit rating or derived weight
    
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    metadata_info = Column(JSON, nullable=True) # Extra context (sessionId, etc)

    user = relationship("User")
    item = relationship("Item")

class ItemEmbedding(Base):
    """
    Separate table for item embeddings to allow re-indexing without touching Item metadata.
    """
    __tablename__ = "item_embeddings"

    id = Column(Integer, primary_key=True, index=True)
    item_id = Column(Integer, ForeignKey("items.id"))
    
    model_name = Column(String, index=True) # e.g. "params_v1", "bert_base"
    vector = Column(JSON) # The float array
    version = Column(Integer, default=1)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    item = relationship("Item")

class Message(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, index=True)
    sender_id = Column(Integer, ForeignKey("users.id"))
    receiver_id = Column(Integer, ForeignKey("users.id"))
    content = Column(String)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    status = Column(String, default="sent") # sent, delivered, read
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

    sender = relationship("User", foreign_keys=[sender_id])
    receiver = relationship("User", foreign_keys=[receiver_id])

class UserInteraction(Base):
    __tablename__ = "user_interactions"

    id = Column(Integer, primary_key=True, index=True)
    liker_id = Column(Integer, ForeignKey("users.id"), index=True)
    liked_id = Column(Integer, ForeignKey("users.id"), index=True)
    action = Column(String) # 'like', 'pass'
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    
    # Relationships not strictly needed for basic logic but good for future
    liker = relationship("User", foreign_keys=[liker_id])
    liker = relationship("User", foreign_keys=[liker_id])
    liked = relationship("User", foreign_keys=[liked_id])

class BlockedUser(Base):
    __tablename__ = "blocked_users"

    id = Column(Integer, primary_key=True, index=True)
    blocker_id = Column(Integer, ForeignKey("users.id"), index=True)
    blocked_id = Column(Integer, ForeignKey("users.id"), index=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

class Report(Base):
    __tablename__ = "reports"

    id = Column(Integer, primary_key=True, index=True)
    reporter_id = Column(Integer, ForeignKey("users.id"))
    reported_id = Column(Integer, ForeignKey("users.id"))
    reason = Column(String) # spam, harassment, inappropriate, other
    details = Column(String, nullable=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)