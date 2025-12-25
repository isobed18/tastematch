import requests
import json
import sys

# Configuration
API_URL = "http://127.0.0.1:8000"

BOTS = ["ActionAli", "RomanceRuyam", "HorrorHasan", "SciFiSelin", "RomanceRifat", "ActionAyse"]
BOT_PASSWORD = "123456" # Hardcoded in create_bots.py

def get_token(username, password):
    url = f"{API_URL}/auth/login"
    data = {"username": username, "password": password}
    try:
        response = requests.post(url, data=data)
        if response.status_code == 200:
            return response.json()["access_token"]
        # print(f"Login failed for {username}: {response.text}") # Quiet fail
    except Exception as e:
        print(f"Error logging in {username}: {e}")
    return None

def swipe(token, target_id, bot_name):
    url = f"{API_URL}/social/swipe"
    headers = {"Authorization": f"Bearer {token}"}
    data = {"liked_user_id": target_id, "action": "like"}
    try:
        response = requests.post(url, json=data, headers=headers)
        res_json = response.json()
        match_status = "MATCH! ❤️" if res_json.get("is_match") else "Liked (Pending)"
        print(f"[{bot_name}] Swiped Right on You -> {match_status}")
    except Exception as e:
        print(f"[{bot_name}] Swipe Failed: {e}")

import sqlite3

# ... existing imports ...

def get_user_id_by_username(username):
    try:
        conn = sqlite3.connect('tastematch.db')
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM users WHERE username = ?", (username,))
        result = cursor.fetchone()
        conn.close()
        if result:
            return result[0]
    except Exception as e:
        print(f"Database lookup failed: {e}")
    return None

def main():
    if len(sys.argv) < 2:
        print("Usage: python make_bots_swipe_me.py <YOUR_USERNAME_OR_ID>")
        print("Example: python make_bots_swipe_me.py RomanceFirat")
        return

    arg = sys.argv[1]
    
    # Try as ID first
    if arg.isdigit():
        target_user_id = int(arg)
    else:
        # Lookup by username
        print(f"Looking up ID for username: {arg}...")
        target_user_id = get_user_id_by_username(arg)
        if not target_user_id:
            print(f"Error: User '{arg}' not found in database.")
            return

    print(f"Making all bots swipe RIGHT on User ID: {target_user_id} ({arg})...\n")
    
    for bot_username in BOTS:
        token = get_token(bot_username, BOT_PASSWORD)
        if token:
            swipe(token, target_user_id, bot_username)
        else:
            print(f"[{bot_username}] Could not login (Bot might not exist).")

if __name__ == "__main__":
    main()
