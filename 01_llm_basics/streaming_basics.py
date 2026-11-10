
from pathlib import Path
from dotenv import load_dotenv

from langchain_groq import ChatGroq

load_dotenv(Path(__file__).with_name(".env"))

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.7
)

# Stream response chunks
for chunk in llm.stream("Explain neural networks in simple terms."):

    print(chunk.content, end="", flush=True)

print()
