
from pathlib import Path
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableLambda
)

load_dotenv(Path(__file__).with_name(".env"))

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.7
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an AI Engineering tutor."),
    ("human", "Explain this topic: {topic}")
])


def prepare_topic(topic):
    return topic.strip().title()


chain = (
    RunnablePassthrough()
    | RunnableLambda(prepare_topic)
    | prompt
    | llm
)

response = chain.invoke("  retrieval augmented generation  ")

print(response.content)
