from pathlib import Path
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    RunnableParallel,
    RunnableLambda,
    RunnableAssign
)

load_dotenv(Path(__file__).with_name(".env"))

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.7
)


def clean_question(data):
    return data["question"].strip()


def count_words(data):
    return len(data["question"].split())


processing = RunnableParallel(
    cleaned_question=RunnableLambda(clean_question),
    word_count=RunnableLambda(count_words)
)

prepare_input = RunnableAssign(processing)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an AI Engineering tutor."),
    (
        "human",
        """
        User: {user}
        Question: {cleaned_question}
        Question word count: {word_count}

        Explain the question clearly.
        """
    )
])

chain = prepare_input | prompt | llm

response = chain.invoke({
    "question": "  Explain vector databases  ",
    "user": "Aadesh"
})

print(response.content)