
from sklearn.metrics import precision_score, recall_score

from datasets import Dataset

from ragas import evaluate
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    LLMContextPrecisionWithoutReference,
    LLMContextRecall
)

# Documents retrieved by our RAG system
retrieved_documents = [
    "doc1", "doc2", "doc3", "doc4", "doc5"
]

# Documents that are actually relevant
relevant_documents = [
    "doc1", "doc3"
]


# Convert retrieval results into binary labels
y_true = [
    1 if doc in relevant_documents else 0
    for doc in retrieved_documents
]

# All retrieved documents are predicted as relevant
y_pred = [1] * len(retrieved_documents)


# Calculate precision and recall
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)


print("Precision:", precision)
print("Recall:", recall)




# Example evaluation records
data = {
    "user_input": [
        "How many paid leaves do employees receive?"
    ],
    "retrieved_contexts": [
        ["Employees receive 20 paid leaves annually."]
    ],
    "response": [
        "Employees receive 20 paid leaves annually."
    ],
    "reference": [
        "Employees receive 20 paid leaves annually."
    ]
}

dataset = Dataset.from_dict(data)

# Evaluate the RAG system
result = evaluate(
    dataset,
    metrics=[
        Faithfulness(),
        AnswerRelevancy(),
        LLMContextPrecisionWithoutReference(),
        LLMContextRecall()
    ]
)

print(result)
