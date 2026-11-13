
# Context management controls what information an LLM
# receives over time.
#
# Techniques covered:
# 1. Context Trimming
# 2. Context Summarisation
# 3. Context Prioritisation
# 4. Context Updating


# ------------------------------------------------------------
# 1. CONTEXT TRIMMING
# ------------------------------------------------------------

# Remove older messages and retain only recent messages.
# This example keeps the latest 3 messages.

conversation_history = [
    "Hello",
    "What is Python?",
    "Explain variables",
    "Explain loops",
    "Explain functions",
    "Explain classes"
]

def trim_context(history, max_messages):
    return history[-max_messages:]


trimmed_history = trim_context(conversation_history, 3)

print("\n1. CONTEXT TRIMMING")
print(trimmed_history)


# ------------------------------------------------------------
# 2. CONTEXT SUMMARISATION
# ------------------------------------------------------------

# Summarisation compresses older conversation into
# a shorter representation.
#
# This is a MOCK summary for learning purposes.
# Real LLM summarisation would require calling an LLM.

old_conversation = [
    "I am learning Python.",
    "I understand variables.",
    "I understand loops.",
    "I want to learn OOP next."
]

# Example of a manually prepared summary.
conversation_summary = (
    "User is learning Python, understands variables "
    "and loops, and wants to learn OOP next."
)

print("\n2. CONTEXT SUMMARISATION")
print("Original messages:", len(old_conversation))
print("Summary:", conversation_summary)


# ------------------------------------------------------------
# 3. CONTEXT PRIORITISATION
# ------------------------------------------------------------

# When context capacity is limited, important information
# should receive priority over less relevant information.
#
# We assign a priority score to each piece of information.
# Higher score = higher priority.

context_items = [
    {"content": "Current question", "priority": 10},
    {"content": "Relevant project details", "priority": 9},
    {"content": "Recent conversation", "priority": 7},
    {"content": "Unrelated old conversation", "priority": 2}
]

def prioritise_context(items):
    return sorted(
        items,
        key=lambda item: item["priority"],
        reverse=True
    )


prioritised_context = prioritise_context(context_items)

print("\n3. CONTEXT PRIORITISATION")

for item in prioritised_context:
    print(item["priority"], "-", item["content"])


# ------------------------------------------------------------
# 4. CONTEXT UPDATING
# ------------------------------------------------------------

# Context can become outdated.
# When new information arrives, we may need to update
# the existing information.
#
# Example: A user changes their preferred programming
# language for a particular project.

user_context = {
    "preferred_language": "Python",
    "learning_level": "Beginner"
}

print("\n4. CONTEXT UPDATING")
print("Before update:", user_context)

# Update the programming language.
user_context["preferred_language"] = "Java"

print("After update:", user_context)

