
# 01_context_engineering_fundamentals.py

# Different sources of context

system_instruction = "You are a helpful Python tutor."

user_question = "Explain Python functions."

user_profile = {
    "level": "Beginner",
    "preferred_style": "Simple explanations"
}

previous_knowledge = "The user understands Python variables."

# Context construction
context = f"""
System Instruction:
{system_instruction}

User Level:
{user_profile['level']}

Preferred Style:
{user_profile['preferred_style']}

Previous Knowledge:
{previous_knowledge}

Current Question:
{user_question}
"""

print("ASSEMBLED CONTEXT")
print(context)
