
import sqlite3


def initialize_database():

    connection = sqlite3.connect("agent_memory.db")

    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS agent_memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            key TEXT NOT NULL,
            value TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(user_id, key)
        )
    """)

    connection.commit()
    connection.close()




def save_memory(user_id, key, value):

    connection = sqlite3.connect("agent_memory.db")

    connection.execute("""
        INSERT INTO agent_memories (user_id, key, value)
        VALUES (?, ?, ?)
        ON CONFLICT(user_id, key)
        DO UPDATE SET value = excluded.value
    """, (user_id, key, value))

    connection.commit()
    connection.close()




def get_memories(user_id):

    connection = sqlite3.connect("agent_memory.db")

    cursor = connection.cursor()

    cursor.execute("""
        SELECT key, value
        FROM agent_memories
        WHERE user_id = ?
    """, (user_id,))

    memories = cursor.fetchall()

    connection.close()

    return memories

def delete_memory(user_id, key):

    connection = sqlite3.connect("agent_memory.db")

    cursor = connection.cursor()

    cursor.execute("""
        DELETE FROM agent_memories
        WHERE user_id = ? AND key = ?
    """, (user_id, key))

    deleted_count = cursor.rowcount

    connection.commit()
    connection.close()

    return deleted_count




if __name__ == "__main__":

    initialize_database()

    print("\nUser 101 Memories:")

    memories = get_memories("user_101")

    for key, value in memories:
        print(f"{key}: {value}")
