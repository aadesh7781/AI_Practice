
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
from pathlib import Path
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")
# 1. Define a tool
@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers and return the result."""
    return a * b


# 2. Initialize the LLM
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)

# 3. Make the tool available to the LLM
llm_with_tools = llm.bind_tools([multiply])

messages = [
    HumanMessage(content="Calculate 25 multiplied by 16.")
]

# 4. Ask the LLM
response = llm_with_tools.invoke(messages)

messages.append(response)

# 5. Execute the requested tool
for tool_call in response.tool_calls:

    if tool_call["name"] == "multiply":

        result = multiply.invoke(tool_call["args"])

        messages.append(
            ToolMessage(
                content=str(result),
                tool_call_id=tool_call["id"]
            )
        )

# 6. Ask the LLM to generate the final answer
final_response = llm_with_tools.invoke(messages)

print("Final Answer:", final_response.content)
