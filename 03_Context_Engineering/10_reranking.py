
from sentence_transformers import CrossEncoder


# 1. Load a pretrained cross-encoder model
# It evaluates the relevance of a query-document pair.
reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


# 2. User query
query = "How can employees apply for medical insurance?"


# 3. Documents retrieved by the initial retriever
documents = [
    "The company provides various employee benefits.",
    "Employees can apply for medical insurance through the HR portal.",
    "Employees are entitled to 20 paid leaves annually.",
    "Medical insurance eligibility is based on employee status.",
    "Employees can work remotely two days per week."
]


# 4. Create query-document pairs
# Each pair will be evaluated by the cross-encoder.
pairs = [
    [query, document]
    for document in documents
]


# 5. Calculate relevance scores
scores = reranker.predict(pairs)


# 6. Combine documents and scores
ranked_results = sorted(
    zip(documents, scores),
    key=lambda item: item[1],
    reverse=True
)


# 7. Display reranked results
print("\n--- RERANKED DOCUMENTS ---")

for rank, (document, score) in enumerate(ranked_results, start=1):
    print(f"\nRank: {rank}")
    print(f"Score: {score:.4f}")
    print(f"Document: {document}")


# 8. Select top 3 documents for LLM context
top_documents = [
    document
    for document, score in ranked_results[:3]
]

print("\n--- FINAL CONTEXT ---")

for document in top_documents:
    print("-", document)
