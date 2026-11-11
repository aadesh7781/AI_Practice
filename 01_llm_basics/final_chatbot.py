import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import trim_messages
from langchain_core.messages.utils import count_tokens_approximately

from sqlite_conversation_memory import (
    init_db,
    load_memory,
    save_memory
)


load_dotenv()

MODEL_NAME = "openai/gpt-oss-20b"

SUMMARY_THRESHOLD = 6
RECENT_MESSAGES = 4
MAX_HISTORY_TOKENS = 500

if not os.getenv("GROQ_API_KEY"):
    raise ValueError(
        "GROQ_API_KEY not found. Check your .env file."
    )



llm = ChatGroq(
    model=MODEL_NAME,
    temperature=0.2,
    streaming=True
)



summary_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are a conversation summarisation assistant.

        Update the existing summary using the new conversation.

        Preserve important information such as:
        - User preferences
        - Personal details shared by the user
        - Previously discussed topics
        - Important decisions
        - Ongoing tasks and goals

        Keep the summary concise but useful for future conversations.
        Do not invent information.
        """
    ),
    (
        "human",
        """
        Existing summary:
        {existing_summary}

        New conversation:
        {conversation}

        Produce the updated summary.
        """
    )
])

summary_chain = summary_prompt | llm



chat_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are a helpful AI assistant.

        Use the conversation summary and recent messages
        to maintain continuity.

        Previous conversation summary:
        {summary}

        Answer the user's latest question clearly.
        """
    ),

    MessagesPlaceholder(variable_name="chat_history"),

    ("human", "{question}")
])

# LCEL: Prompt -> LLM
chat_chain = chat_prompt | llm


def format_messages(messages):

    formatted = []

    for message in messages:

        if isinstance(message, HumanMessage):
            role = "User"

        elif isinstance(message, AIMessage):
            role = "Assistant"

        else:
            role = "Other"

        formatted.append(
            f"{role}: {message.content}"
        )

    return "\n".join(formatted)


def extract_text(content):

    # Normal text response
    if isinstance(content, str):
        return content

    # Handle block-based content if returned
    if isinstance(content, list):

        text_parts = []

        for block in content:

            if isinstance(block, dict):
                if block.get("type") == "text":
                    text_parts.append(block.get("text", ""))

            elif isinstance(block, str):
                text_parts.append(block)

        return "".join(text_parts)

    return str(content)



def update_summary(summary, chat_history):

    # Keep the most recent messages active
    old_messages = chat_history[:-RECENT_MESSAGES]

    recent_messages = chat_history[-RECENT_MESSAGES:]

    if not old_messages:
        return summary, chat_history

    conversation_text = format_messages(old_messages)

    try:

        response = summary_chain.invoke({
            "existing_summary": summary or "No previous summary.",
            "conversation": conversation_text
        })

        updated_summary = extract_text(response.content)

        print("\n[Memory updated: older messages summarised]")

        return updated_summary, recent_messages

    except Exception as error:

        print("\n[Warning] Summarisation failed:", error)

        # Preserve the original history if summarisation fails
        return summary, chat_history


def trim_chat_history(chat_history):

    if not chat_history:
        return []

    try:

        trimmed_history = trim_messages(
            chat_history,
            max_tokens=MAX_HISTORY_TOKENS,
            token_counter=count_tokens_approximately,
            strategy="last",
            start_on="human",
            include_system=False,
            allow_partial=False
        )

        return trimmed_history

    except Exception as error:

        print("\n[Warning] Token trimming failed:", error)

        # Safe fallback: retain only the latest messages
        return chat_history[-RECENT_MESSAGES:]



def generate_response(summary, chat_history, question):

    trimmed_history = trim_chat_history(chat_history)

    response_text = ""

    print("\nAssistant: ", end="", flush=True)

    try:

        for chunk in chat_chain.stream({
            "summary": summary or "No previous conversation.",
            "chat_history": trimmed_history,
            "question": question
        }):

            chunk_text = extract_text(chunk.content)

            if chunk_text:

                print(chunk_text, end="", flush=True)

                response_text += chunk_text

        print()

        return response_text

    except Exception:

        print("\n\n[Error] Unable to generate a response.")

        raise


def main():

    # Initialize SQLite database and tables
    init_db()

    print("=" * 50)
    print("       AI ENGINEERING - FINAL CHATBOT")
    print("=" * 50)

    conversation_id = input(
        "\nEnter conversation ID (or press Enter for default): "
    ).strip()

    if not conversation_id:
        conversation_id = "default"

    # Load existing conversation
    summary, chat_history = load_memory(conversation_id)

    print(f"\nConversation ID: {conversation_id}")

    if chat_history or summary:
        print("[Previous conversation loaded]")
    else:
        print("[Starting a new conversation]")

    print("\nType 'exit' to quit.")
    print("-" * 50)

    while True:

        try:

            question = input("\nYou: ").strip()

            if not question:
                continue

            if question.lower() in ["exit", "quit"]:
                print("\nConversation saved. Goodbye!")
                break

            # Summarise older messages when threshold is reached
            if len(chat_history) >= SUMMARY_THRESHOLD:

                summary, chat_history = update_summary(
                    summary,
                    chat_history
                )

            # Generate the assistant response
            response = generate_response(
                summary,
                chat_history,
                question
            )

            # Only save complete responses
            if response.strip():

                chat_history.append(
                    HumanMessage(content=question)
                )

                chat_history.append(
                    AIMessage(content=response)
                )

                save_memory(
                    conversation_id,
                    summary,
                    chat_history
                )

                print("\n[Conversation saved to SQLite]")

        except KeyboardInterrupt:

            print("\n\nChatbot interrupted. Goodbye!")
            break

        except Exception as error:

            print(f"\n[Error] {error}")
            print("Your previous saved conversation is preserved.")



if __name__ == "__main__":
    main()