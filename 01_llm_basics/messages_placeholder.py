from dotenv import load_dotenv
from langchain_groq import ChatGroq

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.7
)

# Create a reusable prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI tutor. Explain concepts clearly."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")
])

# Store conversation history
chat_history = []

while True:
    question = input("\nYou: ")

    if question.lower() == "exit":
        break

    # Insert history and current question into the prompt
    formatted_prompt = prompt.invoke({
        "chat_history": chat_history,
        "question": question
    })

    # Get AI response
    response = llm.invoke(formatted_prompt)

    print("\nAI:", response.content)

    # Update history
    chat_history.append(HumanMessage(content=question))
    chat_history.append(AIMessage(content=response.content))