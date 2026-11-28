
import os
import json
import uuid
import requests
import numpy as np
import faiss

from pathlib import Path
from datetime import datetime, timezone
from dotenv import load_dotenv


# ==========================================
# 1. CONFIGURATION
# ==========================================

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

JINA_API_KEY = os.getenv("JINA_API_KEY")

MODEL_NAME = "jina-embeddings-v3"
EMBEDDING_DIMENSIONS = 1024

MEMORY_DIR = Path(__file__).resolve().parent / "memory_data"

INDEX_PATH = MEMORY_DIR / "semantic_memory.faiss"
METADATA_PATH = MEMORY_DIR / "semantic_memory_metadata.json"

MEMORY_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. GENERATE EMBEDDINGS
# ==========================================

def generate_embedding(text, task):

    if not JINA_API_KEY:
        raise ValueError(
            "JINA_API_KEY not found in root .env"
        )

    if not text or not text.strip():
        raise ValueError("Text cannot be empty.")

    url = "https://api.jina.ai/v1/embeddings"

    headers = {
        "Authorization": f"Bearer {JINA_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": MODEL_NAME,
        "task": task,
        "dimensions": EMBEDDING_DIMENSIONS,
        "input": [text]
    }

    response = requests.post(
        url,
        headers=headers,
        json=payload,
        timeout=30
    )

    response.raise_for_status()

    result = response.json()

    embedding = np.array(
        result["data"][0]["embedding"],
        dtype=np.float32
    )

    if embedding.shape != (EMBEDDING_DIMENSIONS,):
        raise ValueError(
            f"Unexpected embedding dimensions: {embedding.shape}"
        )

    return embedding


# ==========================================
# 3. SEMANTIC MEMORY CLASS
# ==========================================

class SemanticMemory:

    def __init__(self):

        self.index = faiss.IndexFlatIP(
            EMBEDDING_DIMENSIONS
        )

        self.metadata = []

        self.load_memory()


    # ======================================
    # 4. LOAD PERSISTED MEMORY
    # ======================================

    def load_memory(self):

        index_exists = INDEX_PATH.exists()
        metadata_exists = METADATA_PATH.exists()

        if index_exists != metadata_exists:
            raise RuntimeError(
                "FAISS index and metadata files are incomplete."
            )

        if index_exists:

            self.index = faiss.read_index(
                str(INDEX_PATH)
            )

            with open(
                METADATA_PATH,
                "r",
                encoding="utf-8"
            ) as file:

                self.metadata = json.load(file)

            if self.index.d != EMBEDDING_DIMENSIONS:
                raise ValueError(
                    "Stored FAISS index has incorrect dimensions."
                )

            if self.index.ntotal != len(self.metadata):
                raise ValueError(
                    "FAISS index and metadata are inconsistent."
                )

            # Migrate older records if memory_id is missing.
            changed = False

            for record in self.metadata:

                if "memory_id" not in record:

                    record["memory_id"] = str(uuid.uuid4())
                    changed = True

            if changed:
                self.save_metadata()

            print("Existing semantic memory loaded.")

        else:

            print("Starting with an empty semantic memory.")


    # ======================================
    # 5. SAVE METADATA
    # ======================================

    def save_metadata(self):

        temp_path = METADATA_PATH.with_suffix(".tmp")

        with open(
            temp_path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                self.metadata,
                file,
                indent=4,
                ensure_ascii=False
            )

        temp_path.replace(METADATA_PATH)


    # ======================================
    # 6. SAVE INDEX AND METADATA
    # ======================================

    def save_memory(self):

        temp_path = INDEX_PATH.with_suffix(".tmp")

        faiss.write_index(
            self.index,
            str(temp_path)
        )

        temp_path.replace(INDEX_PATH)

        self.save_metadata()


    # ======================================
    # 7. REBUILD FAISS INDEX
    # ======================================

    def rebuild_index(self, records):

        new_index = faiss.IndexFlatIP(
            EMBEDDING_DIMENSIONS
        )

        for record in records:

            embedding = generate_embedding(
                record["text"],
                task="retrieval.passage"
            )

            vector = embedding.reshape(1, -1)

            faiss.normalize_L2(vector)

            new_index.add(vector)

        return new_index


    # ======================================
    # 8. ADD NEW MEMORY
    # ======================================

    def add_memory(
        self,
        user_id,
        text,
        category="general"
    ):

        if not user_id:
            raise ValueError("user_id cannot be empty.")

        embedding = generate_embedding(
            text,
            task="retrieval.passage"
        )

        vector = embedding.reshape(1, -1)

        # Normalize for cosine similarity.
        faiss.normalize_L2(vector)

        memory_record = {
            "memory_id": str(uuid.uuid4()),
            "user_id": user_id,
            "text": text,
            "category": category,
            "created_at": datetime.now(
                timezone.utc
            ).isoformat()
        }

        self.index.add(vector)

        self.metadata.append(memory_record)

        self.save_memory()

        print(f"Memory saved for {user_id}.")

        return memory_record["memory_id"]


    # ======================================
    # 9. SEARCH RELEVANT MEMORIES
    # ======================================

    def search_memories(
        self,
        user_id,
        query,
        top_k=3,
        similarity_threshold=None
    ):

        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        if self.index.ntotal == 0:
            return []

        query_embedding = generate_embedding(
            query,
            task="retrieval.query"
        )

        query_vector = query_embedding.reshape(1, -1)

        faiss.normalize_L2(query_vector)

        # Search the complete index.
        scores, indices = self.index.search(
            query_vector,
            self.index.ntotal
        )

        results = []

        for score, index_position in zip(
            scores[0],
            indices[0]
        ):

            if index_position == -1:
                continue

            memory = self.metadata[index_position]

            # Enforce user isolation.
            if memory["user_id"] != user_id:
                continue

            similarity = float(score)

            # Apply threshold if provided.
            if (
                similarity_threshold is not None
                and similarity < similarity_threshold
            ):
                continue

            results.append({
                "memory_id": memory["memory_id"],
                "text": memory["text"],
                "category": memory["category"],
                "similarity": similarity
            })

            if len(results) >= top_k:
                break

        return results


    # ======================================
    # 10. UPDATE EXISTING MEMORY
    # ======================================

    def update_memory(
        self,
        user_id,
        memory_id,
        new_text,
        new_category=None
    ):

        updated_records = [
            dict(record) for record in self.metadata
        ]

        target = None

        for record in updated_records:

            if (
                record["memory_id"] == memory_id
                and record["user_id"] == user_id
            ):

                target = record
                break

        if target is None:
            print("Memory not found or access denied.")
            return False

        if not new_text or not new_text.strip():
            raise ValueError("New memory text cannot be empty.")

        target["text"] = new_text

        target["updated_at"] = datetime.now(
            timezone.utc
        ).isoformat()

        if new_category is not None:
            target["category"] = new_category

        # Rebuild because the updated text has a new embedding.
        new_index = self.rebuild_index(
            updated_records
        )

        self.metadata = updated_records
        self.index = new_index

        self.save_memory()

        print("Memory updated successfully.")

        return True


    # ======================================
    # 11. DELETE EXISTING MEMORY
    # ======================================

    def delete_memory(
        self,
        user_id,
        memory_id
    ):

        remaining_records = [
            dict(record)
            for record in self.metadata
            if not (
                record["memory_id"] == memory_id
                and record["user_id"] == user_id
            )
        ]

        if len(remaining_records) == len(self.metadata):

            print("Memory not found or access denied.")
            return False

        # Rebuild without the deleted memory.
        new_index = self.rebuild_index(
            remaining_records
        )

        self.metadata = remaining_records
        self.index = new_index

        self.save_memory()

        print("Memory deleted successfully.")

        return True


# ==========================================
# 12. TESTING
# ==========================================

if __name__ == "__main__":

    memory = SemanticMemory()

    # Add sample memories only on the first run.
    if memory.index.ntotal == 0:

        memory.add_memory(
            "user_101",
            "I am developing an Enterprise RAG Assistant.",
            "project"
        )

        memory.add_memory(
            "user_101",
            "I am learning LangChain and AI agent development.",
            "learning"
        )

        memory.add_memory(
            "user_101",
            "I use Python for machine learning and AI projects.",
            "technical_preference"
        )

        memory.add_memory(
            "user_102",
            "I am developing a Flutter mobile application.",
            "project"
        )

    # Search memories.
    query = "What AI projects am I working on?"

    results = memory.search_memories(
        user_id="user_101",
        query=query,
        top_k=3,
        similarity_threshold=None
    )

    print("\n--- Retrieved Memories ---")

    if results:

        for result in results:

            print("\nMemory ID:", result["memory_id"])
            print("Memory:", result["text"])
            print("Category:", result["category"])

            print(
                "Similarity:",
                round(result["similarity"], 4)
            )

    else:

        print("No relevant memories found.")

    # --------------------------------------
    # UPDATE EXAMPLE
    # --------------------------------------

    # memory.update_memory(
    #     user_id="user_101",
    #     memory_id="YOUR_MEMORY_ID",
    #     new_text="I am developing an advanced AI-powered RAG application.",
    #     new_category="project"
    # )

    # --------------------------------------
    # DELETE EXAMPLE
    # --------------------------------------

    # memory.delete_memory(
    #     user_id="user_101",
    #     memory_id="YOUR_MEMORY_ID"
    # )
