
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

# ------------------------------------------------------------
# 1. INITIALIZE EMBEDDING MODEL
# ------------------------------------------------------------

# all-MiniLM-L6-v2 converts text into 384-dimensional vectors.

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ------------------------------------------------------------
# 2. DOCUMENT EMBEDDINGS
# ------------------------------------------------------------

documents = [
    "I love playing cricket.",
    "I enjoy watching cricket matches.",
    "Python is a programming language.",
    "Machine learning uses data to learn patterns."
]

# Convert all documents into numerical vectors.

document_vectors = embeddings.embed_documents(documents)

print("\nDOCUMENT EMBEDDINGS")
print("Number of documents:", len(document_vectors))
print("Vector dimensions:", len(document_vectors[0]))

# ------------------------------------------------------------
# 3. QUERY EMBEDDING
# ------------------------------------------------------------

query = "I like watching sports."

# Convert user query into a vector.

query_vector = embeddings.embed_query(query)

print("\nQUERY EMBEDDING")
print("Query:", query)
print("Vector dimensions:", len(query_vector))

# ------------------------------------------------------------
# 4. COSINE SIMILARITY
# ------------------------------------------------------------

# Compare query vector against document vectors.

similarity_scores = cosine_similarity(
    [query_vector],
    document_vectors
)[0]

print("\nSIMILARITY RESULTS")

for document, score in zip(documents, similarity_scores):
    print(f"Score: {score:.4f} | {document}")

# ------------------------------------------------------------
# 5. RETRIEVE MOST SIMILAR DOCUMENT
# ------------------------------------------------------------

best_index = similarity_scores.argmax()

print("\nMOST SIMILAR DOCUMENT")
print(documents[best_index])
print("Similarity:", round(similarity_scores[best_index], 4))