
from pathlib import Path
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage


# Load environment variables from project root
ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")


# 1. Define tools

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b


@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


# 2. Create a tool registry

tools = {
    "multiply": multiply,
    "add": add
}


# 3. Initialize the LLM

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)

llm_with_tools = llm.bind_tools(list(tools.values()))


# 4. Initialize conversation history

messages = [
    HumanMessage(
        content="Multiply 25 by 16, then add 100 to the result."
    )
]


# 5. Configure execution limit

MAX_ITERATIONS = 5


# 6. Agent execution loop

for iteration in range(MAX_ITERATIONS):

    print(f"\n--- Agent iteration {iteration + 1} ---")

    response = llm_with_tools.invoke(messages)

    # Preserve the LLM response in conversation history
    messages.append(response)

    # Check whether the LLM requested any tools
    if not response.tool_calls:
        print("Agent completed the task.")
        print("Final answer:", response.content)
        break

    # Execute every requested tool call
    for tool_call in response.tool_calls:

        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_id = tool_call["id"]

        print(f"Executing tool: {tool_name}")
        print(f"Arguments: {tool_args}")

        try:
            selected_tool = tools.get(tool_name)

            if selected_tool is None:
                raise ValueError(f"Unknown tool: {tool_name}")

            result = selected_tool.invoke(tool_args)

        except Exception as error:
            result = f"Tool execution failed: {error}"

        print(f"Tool result: {result}")

        # Return the tool result to the LLM
        messages.append(
            ToolMessage(
                content=str(result),
                tool_call_id=tool_id
            )
        )

else:
    print("Maximum iterations reached. Agent stopped.")
