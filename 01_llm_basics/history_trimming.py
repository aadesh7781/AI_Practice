from dotenv import load_dotenv
from langchain_groq import ChatGroq

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.7
)

MAX_TURNS = 3

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI tutor. Answer clearly."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")
])

chat_history = []

while True:
    question = input("\nYou: ")

    if question.lower() == "exit":
        break

    # Keep only the previous 2 complete turns
    chat_history = chat_history[-((MAX_TURNS - 1) * 2):]

    # Construct the prompt
    formatted_prompt = prompt.invoke({
        "chat_history": chat_history,
        "question": question
    })

    response = llm.invoke(formatted_prompt)

    print("\nAI:", response.content)

    # Store the new conversation turn
    chat_history.append(HumanMessage(content=question))
    chat_history.append(AIMessage(content=response.content))
    print(chat_history)

    print(f"\nStored messages: {len(chat_history)}")