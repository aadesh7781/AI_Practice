from dotenv import load_dotenv
from langchain_groq import ChatGroq

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.3
)

# Number of recent messages to retain
RECENT_MESSAGES = 4

# Summarisation threshold
SUMMARY_THRESHOLD = 6

# Store conversation summary
summary = ""

# Store recent conversation messages
chat_history = []


# Prompt for updating the conversation summary
summary_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a conversation summarisation assistant.

        Update the existing summary using the new conversation.
        Preserve important facts, names, preferences, goals
        and relevant context.

        Keep the summary concise."""
    ),
    (
        "human",
        """Existing summary:
        {summary}

        New conversation:
        {conversation}

        Create an updated summary."""
    )
])


# Main chatbot prompt
chat_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a helpful AI assistant.

        Use the following conversation summary for context:
        {summary}"""
    ),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")
])


while True:

    question = input("\nYou: ")

    if question.lower() == "exit":
        break

    # Check whether summarisation is needed
    if len(chat_history) >= SUMMARY_THRESHOLD:

        # Separate older messages from recent messages
        old_messages = chat_history[:-RECENT_MESSAGES]
        recent_messages = chat_history[-RECENT_MESSAGES:]

        # Convert older messages into readable text
        conversation_text = "\n".join(
            f"{type(message).__name__}: {message.content}"
            for message in old_messages
        )

        # Generate updated summary
        formatted_summary_prompt = summary_prompt.invoke({
            "summary": summary,
            "conversation": conversation_text
        })

        summary_response = llm.invoke(formatted_summary_prompt)

        summary = summary_response.content

        # Retain only recent messages
        chat_history = recent_messages

        print("\n[Conversation summary updated]")

    # Generate chatbot response
    formatted_prompt = chat_prompt.invoke({
        "summary": summary,
        "chat_history": chat_history,
        "question": question
    })

    response = llm.invoke(formatted_prompt)

    print("\nAI:", response.content)

    # Save the new conversation
    chat_history.append(HumanMessage(content=question))
    chat_history.append(AIMessage(content=response.content))

    print("\nRecent messages:", len(chat_history))
    print("Current summary:", summary)