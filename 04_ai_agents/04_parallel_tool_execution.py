
import time
import json

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage


# --------------------------------
# 1. Load environment variables
# --------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")


# --------------------------------
# 2. Define independent tools
# --------------------------------

@tool
def get_weather(city: str) -> str:
    """Get weather information for a given city."""

    print(f"Fetching weather for {city}...")

    time.sleep(3)  # Simulate API latency

    return f"Weather in {city}: 30°C, Sunny"


@tool
def get_population(city: str) -> str:
    """Get population information for a given city."""

    print(f"Fetching population for {city}...")

    time.sleep(3)

    return f"Population of {city}: Sample population data"


@tool
def get_currency(city: str) -> str:
    """Get currency information for a given city."""

    print(f"Fetching currency for {city}...")

    time.sleep(3)

    return f"Currency information for {city}: INR"


# --------------------------------
# 3. Register tools
# --------------------------------

tools = {
    "get_weather": get_weather,
    "get_population": get_population,
    "get_currency": get_currency
}


# --------------------------------
# 4. Prepare independent tasks
# --------------------------------

cities = ["Mumbai", "Pune", "Delhi"]

tasks = []

for city in cities:

    tasks.append(("get_weather", {"city": city}))
    tasks.append(("get_population", {"city": city}))
    tasks.append(("get_currency", {"city": city}))

print(f"Total independent tasks: {len(tasks)}")


# --------------------------------
# 5. Execute tasks concurrently
# --------------------------------

def execute_tool(tool_name, tool_args):

    selected_tool = tools.get(tool_name)

    if selected_tool is None:
        return f"Unknown tool: {tool_name}"

    try:
        return selected_tool.invoke(tool_args)

    except Exception as error:
        return f"Tool execution failed: {error}"


results = []

start_time = time.perf_counter()

with ThreadPoolExecutor(max_workers=9) as executor:

    future_to_task = {
        executor.submit(execute_tool, tool_name, tool_args):
        (tool_name, tool_args)
        for tool_name, tool_args in tasks
    }

    for future in as_completed(future_to_task):

        tool_name, tool_args = future_to_task[future]

        try:
            result = future.result()

        except Exception as error:
            result = f"Execution error: {error}"

        print(f"\nCompleted: {tool_name}")
        print(f"Arguments: {tool_args}")
        print(f"Result: {result}")

        results.append({
            "tool": tool_name,
            "arguments": tool_args,
            "result": result
        })


execution_time = time.perf_counter() - start_time

print("\n--- Parallel Execution Summary ---")
print(f"Tasks completed: {len(results)}")
print(f"Execution time: {execution_time:.2f} seconds")


# --------------------------------
# 6. Configure LLM
# --------------------------------

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)


# --------------------------------
# 7. Let LLM summarize results
# --------------------------------

summary_message = HumanMessage(
    content=(
        "Summarize the following city information clearly. "
        "Organize it by city and do not invent missing information.\n\n"
        f"{json.dumps(results, indent=2)}"
    )
)

response = llm.invoke([summary_message])

print("\n--- Agent Final Summary ---")
print(response.content)
