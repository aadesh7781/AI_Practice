from pathlib import Path
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    RunnableParallel,
    RunnableLambda
)

load_dotenv(Path(__file__).with_name(".env"))

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.7
)


def extract_topic(text):
    return text.strip().title()


def count_characters(text):
    return len(text)


parallel_chain = RunnableParallel(
    topic=RunnableLambda(extract_topic),
    character_count=RunnableLambda(count_characters)
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an AI Engineering tutor."),
    (
        "human",
        """
        Topic: {topic}

        Original question length: {character_count} characters.

        Explain the topic in a beginner-friendly way.
        """
    )
])

chain = parallel_chain | prompt | llm

response = chain.invoke(
    "  Explain Retrieval Augmented Generation  "
)

print(response.content)