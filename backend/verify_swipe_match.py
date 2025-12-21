import requests
import json
import random
import string

# Configuration
API_URL = "http://127.0.0.1:8000"

def get_token(username, password):
    url = f"{API_URL}/auth/login"
    data = {"username": username, "password": password}
    response = requests.post(url, data=data)
    if response.status_code == 200:
        return response.json()["access_token"]
    print(f"Login failed for {username}: {response.text}")
    return None

def swipe(token, target_id, action="like"):
    url = f"{API_URL}/social/swipe"
    headers = {"Authorization": f"Bearer {token}"}
    data = {"liked_user_id": target_id, "action": action}
    response = requests.post(url, json=data, headers=headers)
    return response.json()

def register(username, password):
    url = f"{API_URL}/auth/register"
    data = {"username": username, "password": password}
    response = requests.post(url, json=data)
    return response.status_code

def main():
    print("Testing Mutual Match Logic with Fresh Users...")
    
    rand_suffix = ''.join(random.choices(string.ascii_lowercase, k=4))
    user_a = f"TestA_{rand_suffix}"
    user_b = f"TestB_{rand_suffix}"
    pwd = "password123"
    
    print(f"Registering {user_a} and {user_b}...")
    register(user_a, pwd)
    register(user_b, pwd)
    
    # 1. Login
    token_a = get_token(user_a, pwd)
    token_b = get_token(user_b, pwd)
    
    if not token_a or not token_b:
        print("Login failed for fresh users.")
        return

    # Helper to get ID
    def get_id(token):
        headers = {"Authorization": f"Bearer {token}"}
        res = requests.get(f"{API_URL}/auth/profile", headers=headers)
        if res.status_code != 200:
            print(f"Profile fetch failed: {res.text}")
            return None
        return res.json()["id"]

    id_a = get_id(token_a)
    id_b = get_id(token_b)
    print(f"User IDs: {user_a}={id_a}, {user_b}={id_b}")

    # 2. User A likes User B
    print(f"\n{user_a} swipes RIGHT on {user_b}...")
    res_a = swipe(token_a, id_b, "like")
    print(f"Result: {res_a}")

    # 3. User B likes User A
    print(f"\n{user_b} swipes RIGHT on {user_a}...")
    res_b = swipe(token_b, id_a, "like")
    print(f"Result: {res_b}")
    
    if res_b.get("is_match"):
        print("\nSUCCESS: Mutual Match Detected!")
    else:
        print("\nFAIL: Match NOT detected.")

if __name__ == "__main__":
    main()
