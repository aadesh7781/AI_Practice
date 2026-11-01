from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.7
)

# Store the conversation
history = [
    SystemMessage(content="You are a helpful AI tutor. Answer clearly.")
]

while True:
    user_input = input("\nYou: ")

    if user_input.lower() == "exit":
        break

    # Add the user's message
    history.append(HumanMessage(content=user_input))

    # Send the entire conversation to the model
    response = llm.invoke(history)

    # Store the AI's response
    history.append(AIMessage(content=response.content))

    print("\nAI:", response.content)