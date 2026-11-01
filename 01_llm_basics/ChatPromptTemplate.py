from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# Initialize model
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.7
)

# Create prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an experienced {role}."),
    ("human", "Teach me about {topic} at {level} level.")
])

# Provide values
messages = prompt.format_messages(
    role="Python instructor",
    topic="Decorators",
    level="beginner"
)

# Send messages to LLM
response = llm.invoke(messages)

print(response.content)