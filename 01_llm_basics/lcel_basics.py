
from dotenv import load_dotenv
from pathlib import Path

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv(Path(__file__).with_name(".env"))

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.7
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert {subject} tutor."),
    ("human", "{question}")
])

chain = prompt | llm

response = chain.invoke({
    "subject": "Python",
    "question": "Explain object-oriented programming with an example."
})

print(response.content)
