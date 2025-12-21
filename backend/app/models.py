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
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    
    # Modelden gelen embedding vektörünü önbelleklemek istersek diye (şimdilik boş durabilir)
    embedding = Column(JSON, default=[]) # Soul Vector (Taste Vector)
    embedding_version = Column(Integer, default=0)
    
    # Social Profile Fields
    birth_date = Column(String, nullable=True) # YYYY-MM-DD
    gender = Column(String, nullable=True) # male, female, other
    interested_in = Column(String, nullable=True) # male, female, both, etc.
    location_city = Column(String, nullable=True)
    bio = Column(String, nullable=True)
    last_active = Column(DateTime, default=datetime.datetime.utcnow)

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
    poster_path = Column(String)   # Dikey Afiş (https://image.tmdb...)
    backdrop_path = Column(String) # Yatay Kapak
    
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