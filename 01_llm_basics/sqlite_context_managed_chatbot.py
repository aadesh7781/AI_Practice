
from pathlib import Path

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages.utils import trim_messages

# Reuse our existing SQLite functions
from sqlite_conversation_memory import (
    init_db,
    save_memory,
    load_memory
)

# Load environment variables
load_dotenv(Path(__file__).with_name(".env"))

# Model
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.7
)

# Memory configuration
SUMMARY_THRESHOLD = 6
RECENT_MESSAGES = 4
MAX_HISTORY_TOKENS = 500


# 1. Prompt used to summarise older conversations
summary_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a conversation summarisation assistant. "
        "Update the existing summary using the new conversation. "
        "Preserve important facts, preferences and context."
    ),
    (
        "human",
        """
        Existing summary:
        {old_summary}

        New conversation:
        {conversation}

        Create an updated, concise summary.
        """
    )
])


# 2. Main chatbot prompt
chat_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are a helpful AI assistant.

        Use the following conversation summary to understand
        important information from older conversations.

        Conversation summary:
        {summary}

        Answer the user naturally and use previous context
        when relevant.
        """
    ),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")
])


# 3. Summarise older messages
def update_summary(summary, chat_history):

    old_messages = chat_history[:-RECENT_MESSAGES]
    recent_messages = chat_history[-RECENT_MESSAGES:]

    conversation_text = "\n".join(
        f"{type(message).__name__}: {message.content}"
        for message in old_messages
    )

    response = llm.invoke(
        summary_prompt.invoke({
            "old_summary": summary,
            "conversation": conversation_text
        })
    )

    updated_summary = response.content

    return updated_summary, recent_messages


# 4. Generate chatbot response
def generate_response(summary, chat_history, question):

    trimmed_history = trim_messages(
        chat_history,
        max_tokens=MAX_HISTORY_TOKENS,
        token_counter="approximate",
        strategy="last",
        start_on="human",
        allow_partial=False
    )

    formatted_prompt = chat_prompt.invoke({
        "summary": summary or "No previous conversation.",
        "chat_history": trimmed_history,
        "question": question
    })

    response = llm.invoke(formatted_prompt)

    return response, trimmed_history


# 5. Main application
def main():

    init_db()

    conversation_id = (
        input("Enter conversation ID (default: conv_001): ").strip()
        or "conv_001"
    )

    # Load existing conversation
    summary, chat_history = load_memory(conversation_id)

    print("\nChatbot started!")
    print(f"Conversation ID: {conversation_id}")
    print(f"Loaded messages: {len(chat_history)}")
    print("Type 'exit' to stop.\n")

    while True:

        question = input("You: ").strip()

        if question.lower() == "exit":
            print("Conversation saved. Goodbye!")
            break

        if not question:
            continue

        # Summarise older messages when threshold is reached
        if len(chat_history) >= SUMMARY_THRESHOLD:

            summary, chat_history = update_summary(
                summary,
                chat_history
            )

            print("\n[Older conversation summarised]")

        # Generate response using managed context
        response, trimmed_history = generate_response(
            summary,
            chat_history,
            question
        )

        # Update active conversation history
        chat_history.append(HumanMessage(content=question))
        chat_history.append(AIMessage(content=response.content))

        # Persist updated memory
        save_memory(
            conversation_id,
            summary,
            chat_history
        )

        print("\nAI:", response.content)
        print("\n[Context information]")
        print("Stored messages:", len(chat_history))
        print("Messages sent as history:", len(trimmed_history))
        print("Summary available:", bool(summary))
        print("-" * 50)


if __name__ == "__main__":
    main()
