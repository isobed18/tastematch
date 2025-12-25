import requests
import asyncio
import websockets
import json
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Config
API_URL = "http://127.0.0.1:8000"
WS_URL = "ws://127.0.0.1:8000/chat/ws"

def get_token(username, password):
    url = f"{API_URL}/auth/login"
    data = {"username": username, "password": password}
    resp = requests.post(url, data=data)
    if resp.status_code == 200:
        return resp.json()["access_token"]
    print(f"Login failed for {username}: {resp.text}")
    return None

async def test_chat():
    print("--- STARTING CHAT TEST ---")
    
    # 1. Login
    token_a = get_token("ActionAli", "123456") 
    token_b = get_token("RomanceRuyam", "123456")
    
    if not token_a or not token_b:
        print("Could not login bots. Ensure they exist (run create_bots.py).")
        return

    # User IDs (Need to fetch profile to know IDs, or assume if known. Let's fetch)
    def get_id(token):
        headers = {"Authorization": f"Bearer {token}"}
        r = requests.get(f"{API_URL}/auth/profile", headers=headers)
        return r.json()['id']
        
    id_a = get_id(token_a)
    id_b = get_id(token_b)
    
    print(f"User A: ActionAli (ID {id_a})")
    print(f"User B: RomanceRuyam (ID {id_b})")
    
    # 2. Connect WebSockets
    uri_a = f"{WS_URL}/{id_a}?token={token_a}"
    uri_b = f"{WS_URL}/{id_b}?token={token_b}"
    
    print("Connecting WS for A...")
    async with websockets.connect(uri_a) as ws_a:
        print("Connecting WS for B...")
        async with websockets.connect(uri_b) as ws_b:
            print("Both Connected!")
            
            # 3. A sends to B
            msg_content = "Hello Ruyam, this is Ali!"
            payload = {
                "receiver_id": id_b,
                "content": msg_content
            }
            
            print(f"A sending: {msg_content}")
            await ws_a.send(json.dumps(payload))
            
            # 4. B receives?
            # B should receive it instantly
            print("B waiting for message...")
            try:
                response = await asyncio.wait_for(ws_b.recv(), timeout=5.0)
                data = json.loads(response)
                print(f"B Received: {data}")
                
                if data['content'] == msg_content and data['sender_id'] == id_a:
                    print("SUCCESS: Message delivery verified!")
                else:
                    print("FAILURE: Content mismatch.")
                    
            except asyncio.TimeoutError:
                print("FAILURE: B did not receive message in time.")

            # 5. Check Persistence (History API)
            print("Checking History API...")
            headers = {"Authorization": f"Bearer {token_b}"}
            hist_resp = requests.get(f"{API_URL}/chat/history/{id_a}", headers=headers)
            history = hist_resp.json()
            
            if len(history) > 0 and history[0]['content'] == msg_content:
                print("SUCCESS: Message persisted in DB history.")
            else:
                print(f"FAILURE: History check failed. History: {history}")

if __name__ == "__main__":
    asyncio.run(test_chat())
