from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect, Query
from sqlalchemy.orm import Session
from sqlalchemy import or_, and_
from typing import List, Dict
import json
import datetime
from .. import models, schemas, database
from .auth import get_current_user, get_current_user_ws

router = APIRouter(
    prefix="/chat",
    tags=["chat"]
)

# --- WebSocket Manager ---
class ConnectionManager:
    def __init__(self):
        # Map: user_id -> List[WebSocket] (Allow multiple devices)
        self.active_connections: Dict[int, List[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, user_id: int):
        await websocket.accept()
        if user_id not in self.active_connections:
            self.active_connections[user_id] = []
        self.active_connections[user_id].append(websocket)
        print(f"WS: User {user_id} connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket, user_id: int):
        if user_id in self.active_connections:
            if websocket in self.active_connections[user_id]:
                self.active_connections[user_id].remove(websocket)
            if not self.active_connections[user_id]:
                del self.active_connections[user_id]
        print(f"WS: User {user_id} disconnected.")

    async def send_personal_message(self, message: str, user_id: int):
        if user_id in self.active_connections:
            for connection in self.active_connections[user_id]:
                await connection.send_text(message)

manager = ConnectionManager()

# --- HTTP Endpoints ---

@router.get("/history/{other_user_id}", response_model=List[schemas.MessageOut])
def get_chat_history(
    other_user_id: int,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(database.get_db),
    limit: int = 50
):
    """
    Fetch chat history between current user and other_user.
    """
    messages = db.query(models.Message).filter(
        or_(
            and_(models.Message.sender_id == current_user.id, models.Message.receiver_id == other_user_id),
            and_(models.Message.sender_id == other_user_id, models.Message.receiver_id == current_user.id)
        )
    ).order_by(models.Message.timestamp.desc()).limit(limit).all()
    
    # Reverse to show oldest first in UI usually, but typically API returns sorted by time
    # Let's keep desc for pagination, frontend can reverse.
    return messages[::-1] # Return chronological order

# --- WebSocket Endpoint ---

@router.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: int, token: str = Query(...), db: Session = Depends(database.get_db)):
    """
    WebSocket for real-time chat.
    Client connects with: ws://localhost:8000/chat/ws/{my_id}?token=...
    """
    # 1. Verify Token (Custom logic needed since Headers aren't easy in WS)
    # We use a query param 'token'
    user = get_current_user_ws(token, db)
    if not user or user.id != client_id:
        print(f"WS Authentication Failed for {client_id}")
        await websocket.close(code=1008)
        return

    await manager.connect(websocket, user.id)
    
    # --- ON CONNECT: Mark 'sent' messages as 'delivered' ---
    try:
        # 1. Find all messages meant for this user that are currently just 'sent'
        pending_msgs = db.query(models.Message).filter(
            models.Message.receiver_id == user.id,
            models.Message.status == "sent"
        ).all()
        
        if pending_msgs:
            sender_updates = {}
            for msg in pending_msgs:
                msg.status = "delivered"
                msg.updated_at = datetime.datetime.utcnow()
                
                # Group by sender to batch updates (optimization)
                if msg.sender_id not in sender_updates:
                    sender_updates[msg.sender_id] = []
                sender_updates[msg.sender_id].append(msg.id)
            
            db.commit()
            
            # 2. Notify each sender that their messages were delivered
            for sender_id, msg_ids in sender_updates.items():
                payload = json.dumps({
                    "type": "status_update",
                    "status": "delivered",
                    "message_ids": msg_ids,
                    "receiver_id": user.id 
                })
                # Check if sender is online and send
                await manager.send_personal_message(payload, sender_id)
                print(f"WS: Notified sender {sender_id} of delivery for msgs {msg_ids}")
                
    except Exception as e:
        print(f"WS Error updating delivered status: {e}")

    
    try:
        while True:
            # Receive message from Client
            data = await websocket.receive_text()
            # Expected JSON: { "type": "message", "receiver_id": 123, "content": "Hello" }
            # OR { "type": "typing", "receiver_id": 123 }
            # OR { "type": "read", "message_ids": [1,2], "receiver_id": 123 } (receiver_id here is the original sender)
            
            try:
                msg_data = json.loads(data)
                msg_type = msg_data.get('type', 'message') # Default to message for back-compat
                receiver_id = int(msg_data['receiver_id'])
                
                if msg_type == 'typing':
                    # Forward typing status to receiver
                    payload = json.dumps({
                        "type": "typing",
                        "sender_id": user.id
                    })
                    await manager.send_personal_message(payload, receiver_id)
                    
                elif msg_type == 'read':
                    # Mark messages as read in DB
                    message_ids = msg_data.get('message_ids', [])
                    if message_ids:
                        db.query(models.Message).filter(models.Message.id.in_(message_ids)).update(
                            {"status": "read", "updated_at": datetime.datetime.utcnow()},
                            synchronize_session=False
                        )
                        db.commit()
                        
                        # Notify original sender that messages were read
                        payload = json.dumps({
                            "type": "status_update",
                            "status": "read",
                            "message_ids": message_ids,
                            "reader_id": user.id
                        })
                        await manager.send_personal_message(payload, receiver_id)
                
                elif msg_type == 'message':
                    content = msg_data['content']
                    client_message_id = msg_data.get('client_message_id')
                    
                    # 2. Save to DB
                    new_msg = models.Message(
                        sender_id=user.id,
                        receiver_id=receiver_id,
                        content=content,
                        timestamp=datetime.datetime.utcnow(),
                        status="sent"
                    )
                    db.add(new_msg)
                    db.commit()
                    db.refresh(new_msg)
                    
                    # 3. Construct Payload
                    response_payload = json.dumps({
                        "id": new_msg.id,
                        "sender_id": user.id,
                        "receiver_id": receiver_id,
                        "content": content,
                        "timestamp": new_msg.timestamp.isoformat(),
                        "type": "message",
                        "status": "sent",
                        "client_message_id": client_message_id
                    })
                    
                    # 4. Send to Receiver (Instant)
                    await manager.send_personal_message(response_payload, receiver_id)
                    
                    # 5. Echo back to Sender (Confirmation: Sent -> Delivered if socket open)
                    # For now just confirm it's sent/saved
                    await manager.send_personal_message(response_payload, user.id)
                
            except Exception as e:
                print(f"WS Error processing message: {e}")
                
    except WebSocketDisconnect:
        manager.disconnect(websocket, user.id)
