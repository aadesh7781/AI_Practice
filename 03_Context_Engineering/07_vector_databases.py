
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document


# 1. Load the embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# 2. Create a persistent ChromaDB collection
# Data will be stored locally in the chroma_db directory.
vector_store = Chroma(
    collection_name="company_policies",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)


# 3. Create sample documents with metadata
documents = [
    Document(
        page_content="Employees receive 20 paid leaves annually.",
        metadata={"department": "HR", "source": "leave_policy"}
    ),
    Document(
        page_content="The company provides health insurance to employees.",
        metadata={"department": "HR", "source": "insurance_policy"}
    ),
    Document(
        page_content="Python developers should understand APIs and databases.",
        metadata={"department": "Engineering", "source": "job_description"}
    ),
    Document(
        page_content="Employees can work remotely two days per week.",
        metadata={"department": "HR", "source": "remote_policy"}
    )
]


# 4. Add documents to ChromaDB
# Chroma generates and stores embeddings using our embedding model.
document_ids = ["doc1", "doc2", "doc3", "doc4"]

vector_store.add_documents(
    documents=documents,
    ids=document_ids
)

print("Documents added successfully!")


# 5. Perform semantic similarity search
query = "How many holidays can an employee take?"

results = vector_store.similarity_search(
    query,
    k=2
)

print("\n--- Similarity Search ---")

for result in results:
    print("Content:", result.page_content)
    print("Metadata:", result.metadata)
    print()


# 6. Search with distance scores
scored_results = vector_store.similarity_search_with_score(
    query,
    k=2
)

print("\n--- Search With Distance ---")

for document, distance in scored_results:
    print("Content:", document.page_content)
    print("Distance:", distance)


# 7. Metadata filtering
hr_results = vector_store.similarity_search(
    "What benefits do employees receive?",
    k=3,
    filter={"department": "HR"}
)

print("\n--- HR Documents Only ---")

for result in hr_results:
    print(result.page_content)


# 8. Update an existing document
updated_document = Document(
    page_content="Employees receive 25 paid leaves annually.",
    metadata={"department": "HR", "source": "leave_policy"}
)

vector_store.update_documents(
    ids=["doc1"],
    documents=[updated_document]
)

print("\nDocument updated!")


# 9. Delete a document
vector_store.delete(ids=["doc4"])

print("Document deleted!")
