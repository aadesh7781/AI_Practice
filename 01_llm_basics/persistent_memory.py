import json
from pathlib import Path

from dotenv import load_dotenv
from langchain_groq import ChatGroq

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.7
)

# Persistent storage location
MEMORY_FILE = Path("conversation_memory.json")


# Load saved memory
def load_memory():

    if MEMORY_FILE.exists():
        with open(MEMORY_FILE, "r", encoding="utf-8") as file:
            data = json.load(file)

        summary = data.get("summary", "")

        chat_history = []

        for message in data.get("chat_history", []):
            if message["role"] == "human":
                chat_history.append(
                    HumanMessage(content=message["content"])
                )
            elif message["role"] == "ai":
                chat_history.append(
                    AIMessage(content=message["content"])
                )

        return summary, chat_history

    return "", []


# Save memory
def save_memory(summary, chat_history):

    data = {
        "summary": summary,
        "chat_history": [
            {
                "role": (
                    "human" if isinstance(message, HumanMessage)
                    else "ai"
                ),
                "content": message.content
            }
            for message in chat_history
        ]
    }

    with open(MEMORY_FILE, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)


# Load previous conversation
summary, chat_history = load_memory()

print("Previous conversation loaded!")
print("Stored messages:", len(chat_history))


prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful AI tutor. Previous conversation summary: {summary}"
    ),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")
])


while True:

    question = input("\nYou: ")

    if question.lower() == "exit":
        break

    formatted_prompt = prompt.invoke({
        "summary": summary,
        "chat_history": chat_history,
        "question": question
    })

    response = llm.invoke(formatted_prompt)

    print("\nAI:", response.content)

    chat_history.append(HumanMessage(content=question))
    chat_history.append(AIMessage(content=response.content))

    # Persist the updated conversation
    save_memory(summary, chat_history)

    print("\nMemory saved successfully!")