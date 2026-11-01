from dotenv import load_dotenv
from langchain_groq import ChatGroq
import os

# Load environment variables
load_dotenv()

# Initialize the LLM
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.7
)

# Take user input
question = input("Ask something: ")

# Send question to LLM
response = llm.invoke(question)

# Display response
print("\nAI Response:")
print(response.content)