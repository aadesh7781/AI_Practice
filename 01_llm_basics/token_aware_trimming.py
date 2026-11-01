from dotenv import load_dotenv
from langchain_groq import ChatGroq

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages.utils import trim_messages

load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.7
)

# Maximum token budget for previous conversation history
MAX_HISTORY_TOKENS = 5000

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

    # Trim history based on token budget
    trimmed_history = trim_messages(
        chat_history,
        max_tokens=MAX_HISTORY_TOKENS,
        token_counter="approximate",
        strategy="last",
        start_on="human",
        allow_partial=False
    )

    # Construct the prompt
    formatted_prompt = prompt.invoke({
        "chat_history": trimmed_history,
        "question": question
    })

    # Get response
    response = llm.invoke(formatted_prompt)

    print("\nAI:", response.content)

    # Store the complete conversation
    chat_history.append(HumanMessage(content=question))
    chat_history.append(AIMessage(content=response.content))

    print("\nTotal stored messages:", len(chat_history))
    print("Messages sent as history:", len(trimmed_history))

    print("\n--- TRIMMED HISTORY ---")

for message in trimmed_history:
    print(type(message).__name__, ":", message.content[:100])