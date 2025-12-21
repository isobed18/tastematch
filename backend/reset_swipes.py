import sqlite3
import sys
import os

def get_user_id_by_username(cursor, username):
    cursor.execute("SELECT id FROM users WHERE username = ?", (username,))
    result = cursor.fetchone()
    if result:
        return result[0]
    return None

def main():
    if len(sys.argv) < 2:
        print("Usage: python reset_swipes.py <USERNAME_OR_ID> [ALL]")
        print("Examples:")
        print("  python reset_swipes.py RomanceFirat  -> Resets matches for RomanceFirat")
        print("  python reset_swipes.py ALL           -> Resets ALL matches and messages in DB")
        return

    arg = sys.argv[1]
    db_path = "tastematch.db"
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    if arg == "ALL":
        print("⚠ WARNING: reseting ALL user interactions and messages!")
        cursor.execute("DELETE FROM user_interactions")
        cursor.execute("DELETE FROM messages")
        conn.commit()
        print("Done. All matches and chat history wiped.")
        conn.close()
        return

    # Specific User
    user_id = None
    if arg.isdigit():
        user_id = int(arg)
    else:
        user_id = get_user_id_by_username(cursor, arg)
    
    if not user_id:
        print(f"Error: User '{arg}' not found.")
        conn.close()
        return

    print(f"Resetting interactions for User ID: {user_id}...")

    # 1. Delete interactions where I am the liker
    cursor.execute("DELETE FROM user_interactions WHERE liker_id = ?", (user_id,))
    
    # 2. Delete interactions where I am the liked
    cursor.execute("DELETE FROM user_interactions WHERE liked_id = ?", (user_id,))
    
    # 3. Delete messages involving me
    cursor.execute("DELETE FROM messages WHERE sender_id = ? OR receiver_id = ?", (user_id, user_id))

    conn.commit()
    rows_deleted = cursor.rowcount # This might not capture all, but ok.
    print(f"✅ Successfully reset matches & chat history for User {user_id}.")
    
    conn.close()

if __name__ == "__main__":
    main()
