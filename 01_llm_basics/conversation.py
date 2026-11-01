from dotenv import load_dotenv
from langchain_groq import ChatGroq
import os

load_dotenv()

llm=ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.7
)

messages=[
    {
        "role":"system",
        "content":"You are a helpful assistant."
    }
]

response=llm.invoke(messages)

print("\nAI Response:")
print(response.content)
print(type(response))
print(response.content)
print(response.response_metadata)
print(response.usage_metadata)