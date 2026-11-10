from pathlib import Path
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda
)

load_dotenv(Path(__file__).with_name(".env"))

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.7
)


# Create different prompts
python_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert Python tutor."),
    ("human", "{question}")
])

ml_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert Machine Learning tutor."),
    ("human", "{question}")
])

general_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful general assistant."),
    ("human", "{question}")
])


# Define conditions
def is_python(data):
    return "python" in data["question"].lower()


def is_ml(data):
    return "machine learning" in data["question"].lower()


# Define routes
branch = RunnableBranch(
    (is_python, python_prompt | llm),
    (is_ml, ml_prompt | llm),
    general_prompt | llm
)

response = branch.invoke({
    "question": "Explain Python decorators with examples."
})

print(response.content)