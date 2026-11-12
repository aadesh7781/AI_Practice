
# 03_context_assembly.py

def assemble_context(question, history, document):

    context = f"""
    RELEVANT CONVERSATION:
    {history}

    REFERENCE DOCUMENT:
    {document}

    USER QUESTION:
    {question}
    """

    return context


question = "What is the company's remote work policy?"

history = "I work in the engineering department."

document = "Employees can work remotely 2 days per week."

final_context = assemble_context(
    question,
    history,
    document
)

print(final_context)
