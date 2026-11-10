
from pathlib import Path
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv(Path(__file__).with_name(".env"))

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.7
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an AI Engineering tutor."),
    ("human", "Explain {topic} with a practical example.")
])

chain = prompt | llm

for chunk in chain.stream({
    "topic": "Retrieval Augmented Generation"
}):
    print(chunk.content, end="", flush=True)

print()
