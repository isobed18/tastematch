
from fastapi import FastAPI
from .database import engine, Base
from .routers import auth, feed, swipe, social
from .inference_service import InferenceService
import os

# Create tables
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Taste Match API")

# --- Global Inference Service Setup ---
@app.on_event("startup")
async def startup_event():
    print("Starting up... Initializing Global Inference Service.")
    
    # Path to the specific best model requested by user
    # tastematch\runs_tt\run_20251217_012150
    # We need to construct absolute path
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # tastematch/
    model_path = os.path.join(base_dir, "runs/two_tower", "run_20251217_012150", "best_two_tower.pth")
    
    if os.path.exists(model_path):
        print(f"Loading Model from: {model_path}")
        app.state.inference_service = InferenceService(model_path=model_path)
    else:
        print(f"WARNING: Model not found at {model_path}. Loading default/latest.")
        app.state.inference_service = InferenceService() # Fallback

app.include_router(auth.router)
app.include_router(feed.router)
app.include_router(swipe.router)
app.include_router(social.router)

@app.get("/")
def read_root():
    return {"message": "Welcome to Taste Match API"}
