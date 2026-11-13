

import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage
)
from langchain_core.messages import trim_messages
from langchain_core.messages.utils import count_tokens_approximately

load_dotenv()

# Initialize LLM
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)


# ------------------------------------------------------------
# 1. CONTEXT TRIMMING
# ------------------------------------------------------------

# LangChain removes older messages to fit the token budget.
# The latest messages are retained.

messages = [
    SystemMessage(content="You are a Python tutor."),
    HumanMessage(content="What is Python?"),
    AIMessage(content="Python is a programming language."),
    HumanMessage(content="Explain variables."),
    AIMessage(content="Variables store data."),
    HumanMessage(content="Explain functions.")
]

trimmed_messages = trim_messages(
    messages,
    max_tokens=100,
    token_counter=count_tokens_approximately,
    strategy="last",
    include_system=True,
    start_on="human"
)

print("\n1. TRIMMED CONTEXT")

for message in trimmed_messages:
    print(message.type, ":", message.content)


# ------------------------------------------------------------
# 2. CONTEXT SUMMARISATION
# ------------------------------------------------------------

# Use the LLM to summarise older conversation history.
# Unlike a manually written summary, this calls the model.

old_conversation = """
User is learning Python.
User understands variables and loops.
User has started learning functions.
User wants to learn object-oriented programming next.
"""

summary = llm.invoke([
    SystemMessage(
        content="Summarise conversations into concise, useful facts. Do not invent information."
    ),
    HumanMessage(content=old_conversation)
])

print("\n2. CONVERSATION SUMMARY")
print(summary.content)


# ------------------------------------------------------------
# 3. CONTEXT PRIORITISATION
# ------------------------------------------------------------

# Prioritisation depends on application requirements.
# We assign relevance scores and sort the information.

context_items = [
    {"content": "Current user question", "priority": 10},
    {"content": "Relevant project information", "priority": 9},
    {"content": "Recent conversation", "priority": 7},
    {"content": "Unrelated old conversation", "priority": 2}
]

prioritised_context = sorted(
    context_items,
    key=lambda item: item["priority"],
    reverse=True
)

print("\n3. PRIORITISED CONTEXT")

for item in prioritised_context:
    print(item["content"], "-", item["priority"])


# ------------------------------------------------------------
# 4. CONTEXT UPDATING
# ------------------------------------------------------------

# Update existing application context when new information
# is received. In a real application, persist this in a
# database with appropriate user ownership and validation.

user_context = {
    "preferred_language": "Python",
    "learning_level": "Beginner"
}

print("\n4. CONTEXT BEFORE UPDATE")
print(user_context)

# Update the existing information.
user_context.update({
    "preferred_language": "Java"
})

print("\n4. CONTEXT AFTER UPDATE")
print(user_context)
