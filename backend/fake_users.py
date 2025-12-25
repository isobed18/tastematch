import requests
import random
import string
import time

BASE_URL = "http://127.0.0.1:8000"

def random_string(length=8):
    return ''.join(random.choice(string.ascii_letters) for _ in range(length))

def create_and_run_user(idx):
    username = f"bot_user_{idx}_{random_string(4)}"
    password = "password123"
    
    # 1. Register
    try:
        resp = requests.post(f"{BASE_URL}/auth/register", json={"username": username, "password": password})
        if resp.status_code != 200:
            print(f"[{idx}] Register failed: {resp.text}")
            return
            
        # 2. Login
        # Endpoint is /auth/login, expects form data (username, password)
        resp = requests.post(f"{BASE_URL}/auth/login", data={"username": username, "password": password})
        if resp.status_code != 200:
            print(f"[{idx}] Login failed: {resp.status_code} - {resp.text}")
            return
            
        token = resp.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        
        # 3. Get Feed (Training)
        print(f"[{idx}] Fetching Feed...")
        feed_resp = requests.get(f"{BASE_URL}/feed/?limit=25", headers=headers)
        if feed_resp.status_code != 200:
             print(f"[{idx}] Feed failed: {feed_resp.text}")
             return
             
        items = feed_resp.json()
        print(f"[{idx}] Received {len(items)} items. Swiping...")
        
        # 4. Swipe Randomly
        actions = ["like", "dislike", "superlike"]
        for item in items:
            action = random.choice(actions)
            # 80% chance to swipe, 20% skip
            if random.random() < 0.8:
                s_resp = requests.post(
                    f"{BASE_URL}/swipe/", 
                    json={"item_id": item["id"], "action": action},
                    headers=headers
                )
                
        # 5. Get Daily Match
        print(f"[{idx}] Requesting Daily Match...")
        start_t = time.time()
        daily_resp = requests.get(f"{BASE_URL}/feed/daily", headers=headers)
        duration = time.time() - start_t
        
        if daily_resp.status_code == 200:
            match = daily_resp.json()
            print(f"[{idx}] SUCCESS! Match: {match['title']} (Score: {match.get('match_score', 0):.2f}) in {duration:.2f}s")
        else:
            print(f"[{idx}] Daily Match failed: {daily_resp.text}")
            
    except Exception as e:
        print(f"[{idx}] Exception: {e}")

if __name__ == "__main__":
    print("Starting Simulation of 50 Users...")
    for i in range(50):
        create_and_run_user(i)
        time.sleep(0.5) # Slight stagger
