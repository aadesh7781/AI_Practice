
# Phase 3 — AI Agents & Tool Calling

This phase focuses on understanding how Large Language Models (LLMs) can become AI agents capable of making decisions, calling tools, executing tasks, interacting with external services, and coordinating multiple operations.

The goal is to understand the fundamentals of agent engineering through practical Python implementations using LangChain and Groq.

## Learning Objectives

- Understand the architecture and working of AI agents.
- Implement LLM tool calling using LangChain.
- Understand agent execution loops and tool orchestration.
- Manage tool dependencies and execution state.
- Implement error handling and retry mechanisms.
- Execute independent tools concurrently using Python.
- Integrate external services such as Gmail SMTP.
- Understand agent memory and information retention.

## Tech Stack

- Python
- LangChain
- LangChain Groq
- Groq API
- Python `concurrent.futures`
- Gmail SMTP
- Python `smtplib`
- Python `python-dotenv`

## Folder Structure

```text
04_ai_agents/
│
├── 01_tool_calling.py
├── 02_agent_execution_loop.py
├── 03_tool_orchestration.py
├── 04_parallel_tool_execution.py
│
└── README.md
```

## 1. Tool Calling

**File:** `01_tool_calling.py`

Learned how an LLM can request external tools to perform operations.

Topics covered:
- Difference between an LLM and an AI agent.
- Creating tools using LangChain's `@tool` decorator.
- Tool descriptions and argument schemas.
- Binding tools to an LLM using `bind_tools()`.
- Understanding `AIMessage` and `ToolMessage`.
- Executing requested tools using Python.
- Returning tool results to the LLM.

**Key concept:**

The LLM decides which tool to call, while the Python application executes the function.

## 2. Agent Execution Loop

**File:** `02_agent_execution_loop.py`

Implemented an agent execution loop that repeatedly interacts with the LLM and executes requested tools.

Topics covered:
- Decision → Action → Observation cycle.
- Maintaining conversation messages.
- Executing multiple tool calls.
- Handling tool responses.
- Maximum iteration limits.
- Basic error handling.
- Understanding when an agent should terminate execution.

**Implementation example:**

An agent receives a mathematical task:

1. Multiply 25 by 16.
2. Add 100 to the result.
3. Return the final answer.

The agent selects and executes the required tools through its execution loop.

## 3. Tool Orchestration

**File:** `03_tool_orchestration.py`

Learned how an agent coordinates multiple tools to complete a task.

Topics covered:
- Tool selection and routing.
- Tool dependencies.
- Sequential tool execution.
- Managing shared execution state.
- Error handling and retries.
- Exponential backoff.
- Idempotency.
- Integrating external services.

### Gmail SMTP Integration

Implemented actual email sending using Gmail SMTP.

The agent workflow:

1. Retrieve sample order details.
2. Calculate the refund amount.
3. Send a confirmation email.
4. Generate a final response.

Technologies and concepts used:

- `smtplib`
- `email.message.EmailMessage`
- Gmail SMTP authentication.
- Gmail App Passwords.
- STARTTLS encryption.
- Email construction and delivery.
- Environment variables for credentials.

**Important:** This is a learning demonstration. It calculates a refund and sends a confirmation email; it does not issue an actual payment or refund.

## 4. Parallel Tool Execution

**File:** `04_parallel_tool_execution.py`

Learned how independent tool operations can execute concurrently instead of waiting for one another.

Topics covered:
- Sequential vs parallel execution.
- Python `concurrent.futures`.
- `ThreadPoolExecutor`.
- `executor.submit()`.
- Future objects.
- Retrieving results using `.result()`.
- Processing completed tasks using `as_completed()`.
- Returning results to the LLM for summarization.

### Practical Implementation

Created three tools:

- `get_weather()`
- `get_population()`
- `get_currency()`

Executed nine independent tasks across Mumbai, Pune, and Delhi using a thread pool.

**Expected execution comparison:**

| Execution method | Approximate time |
|---|---:|
| Sequential | 27 seconds |
| Parallel (9 workers) | 3 seconds |

The implementation uses simulated API latency to demonstrate concurrency.

**Key learning:** Parallel execution is useful when tasks are independent. Tasks with dependencies must respect their required execution order.

## Environment Configuration

Environment variables are stored in the root project `.env` file.

Example:

```env
GROQ_API_KEY=your_groq_api_key

GMAIL_ADDRESS=your_gmail_address
GMAIL_APP_PASSWORD=your_gmail_app_password
TEST_RECIPIENT=your_test_email
```

The Python implementations load environment variables from the project root using:

```python
ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")
```

### Security Notes

- Never commit `.env` to GitHub.
- Never expose Groq API keys or Gmail App Passwords.
- Use a test recipient when experimenting with email tools.
- Validate tool arguments before executing external actions.
- Use appropriate authorization and approval mechanisms for sensitive operations.

## Running the Implementations

Run commands from the project root:

```powershell
python .\04_ai_agents\01_tool_calling.py

python .\04_ai_agents\02_agent_execution_loop.py

python .\04_ai_agents\03_tool_orchestration.py

python .\04_ai_agents\04_parallel_tool_execution.py
```

The project uses the Python virtual environment located under `01_llm_basics/.venv`.

## Phase 3 Progress

| Topic | Status |
|---|---|
| Tool Calling | Completed |
| Agent Execution Loop | Completed |
| Tool Orchestration | Completed |
| Sequential Execution | Completed |
| Parallel Execution | Completed |
| Gmail SMTP Integration | Completed |
| Agent Memory | In Progress |

## Next Learning Topic — Agent Memory

The remaining topic in this phase is Agent Memory.

Planned concepts:

- Why AI agents need memory.
- Conversation history vs agent memory.
- Short-term memory.
- Long-term memory.
- Memory storage and retrieval.
- Maintaining information across conversations.

## Learning Approach

This repository follows a progressive, implementation-focused learning approach.

Each concept is first understood theoretically, followed by small code examples, practical implementations, testing, and GitHub documentation.

The objective is not simply to use frameworks but to understand how AI agents work internally and develop the engineering knowledge required to build reliable agentic systems.