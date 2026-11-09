import sqlite3
from pathlib import Path

from langchain_core.messages import HumanMessage, AIMessage

DB_PATH = Path(__file__).with_name("conversation_memory.db")


# 1. Create database and tables
def init_db():

    with sqlite3.connect(DB_PATH) as conn:

        conn.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                summary TEXT DEFAULT ''
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                FOREIGN KEY (conversation_id)
                    REFERENCES conversations(id)
            )
        """)


# 2. Save conversation
def save_memory(conversation_id, summary, chat_history):

    with sqlite3.connect(DB_PATH) as conn:

        conn.execute("""
            INSERT INTO conversations (id, summary)
            VALUES (?, ?)
            ON CONFLICT(id)
            DO UPDATE SET summary = excluded.summary
        """, (conversation_id, summary))

        # Replace the saved active history
        conn.execute("""
            DELETE FROM messages
            WHERE conversation_id = ?
        """, (conversation_id,))

        message_data = []

        for message in chat_history:

            role = (
                "human"
                if isinstance(message, HumanMessage)
                else "ai"
            )

            message_data.append((
                conversation_id,
                role,
                message.content
            ))

        conn.executemany("""
            INSERT INTO messages (conversation_id, role, content)
            VALUES (?, ?, ?)
        """, message_data)


# 3. Load conversation
def load_memory(conversation_id):

    with sqlite3.connect(DB_PATH) as conn:

        cursor = conn.cursor()

        cursor.execute("""
            SELECT summary
            FROM conversations
            WHERE id = ?
        """, (conversation_id,))

        result = cursor.fetchone()

        if result is None:
            return "", []

        summary = result[0]

        cursor.execute("""
            SELECT role, content
            FROM messages
            WHERE conversation_id = ?
            ORDER BY id
        """, (conversation_id,))

        rows = cursor.fetchall()

    chat_history = []

    for role, content in rows:

        if role == "human":
            chat_history.append(HumanMessage(content=content))

        elif role == "ai":
            chat_history.append(AIMessage(content=content))

    return summary, chat_history


# 4. Test the database
if __name__ == "__main__":

    init_db()

    conversation_id = "conv_001"

    history = [
        HumanMessage(content="My name is Aadesh."),
        AIMessage(content="Nice to meet you, Aadesh."),
        HumanMessage(content="I am learning AI Engineering."),
        AIMessage(content="That's great!")
    ]

    save_memory(
        conversation_id,
        "The user is Aadesh and is learning AI Engineering.",
        history
    )

    summary, loaded_history = load_memory(conversation_id)

    print("Summary:", summary)
    print("\nLoaded messages:")

    for message in loaded_history:
        print(type(message).__name__, ":", message.content)