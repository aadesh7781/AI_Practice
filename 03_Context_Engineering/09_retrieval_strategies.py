
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever


# 1. Initialize embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# 2. Prepare sample documents
documents = [
    Document(
        page_content="HR-POLICY-2026-015: Employees receive 20 paid leaves annually.",
        metadata={"department": "HR", "type": "leave"}
    ),
    Document(
        page_content="Employees can work remotely two days per week.",
        metadata={"department": "HR", "type": "remote"}
    ),
    Document(
        page_content="Health insurance is provided to all eligible employees.",
        metadata={"department": "HR", "type": "insurance"}
    ),
    Document(
        page_content="Engineering employees must follow secure coding practices.",
        metadata={"department": "Engineering", "type": "security"}
    ),
    Document(
        page_content="Finance employees must submit monthly expense reports.",
        metadata={"department": "Finance", "type": "expenses"}
    )
]


# 3. Create vector database
vector_store = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    collection_name="retrieval_experiments",
    persist_directory="./retrieval_db"
)


# 4. Similarity Search
similarity_retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

print("\n--- SIMILARITY SEARCH ---")

results = similarity_retriever.invoke(
    "What is the employee leave policy?"
)

for doc in results:
    print(doc.page_content)


# 5. MMR Search
mmr_retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 3,
        "fetch_k": 5,
        "lambda_mult": 0.5
    }
)

print("\n--- MMR SEARCH ---")

results = mmr_retriever.invoke(
    "What benefits do employees receive?"
)

for doc in results:
    print(doc.page_content)


# 6. Metadata Filtering
print("\n--- METADATA FILTERING ---")

results = vector_store.similarity_search(
    "employee policies",
    k=3,
    filter={"department": "HR"}
)

for doc in results:
    print(doc.page_content)


# 7. Keyword Search using BM25
keyword_retriever = BM25Retriever.from_documents(
    documents
)

keyword_retriever.k = 3

print("\n--- KEYWORD SEARCH ---")

results = keyword_retriever.invoke(
    "HR-POLICY-2026-015"
)

for doc in results:
    print(doc.page_content)


# 8. Hybrid Search using EnsembleRetriever
# Combines keyword retrieval and vector retrieval.
hybrid_retriever = EnsembleRetriever(
    retrievers=[
        keyword_retriever,
        similarity_retriever
    ],
    weights=[0.5, 0.5]
)

print("\n--- HYBRID SEARCH ---")

results = hybrid_retriever.invoke(
    "HR-POLICY-2026-015"
)

for doc in results:
    print(doc.page_content)
