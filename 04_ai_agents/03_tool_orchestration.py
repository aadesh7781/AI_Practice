
import os
import json
import time
import smtplib

from pathlib import Path
from email.message import EmailMessage

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import (
    HumanMessage,
    ToolMessage
)


# --------------------------------
# 1. Load environment variables
# --------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")


# --------------------------------
# 2. Sample order database
# --------------------------------

orders = {
    105: {
        "customer": "Rahul",
        "amount": 5000,
        "eligible_refund": True
    }
}

# Stores information retrieved and calculated by tools
agent_state = {
    "order": None,
    "refund_amount": None
}


# --------------------------------
# 3. Define agent tools
# --------------------------------

@tool
def get_order_details(order_id: int) -> str:
    """Retrieve customer order details using the order ID."""

    order = orders.get(order_id)

    if order is None:
        return "Error: Order not found."

    if not order["eligible_refund"]:
        return "Error: This order is not eligible for a refund."

    agent_state["order"] = order
    agent_state["refund_amount"] = None

    return json.dumps(order)


@tool
def calculate_refund(order_id: int) -> str:
    """Calculate the refund amount for a previously retrieved eligible order."""

    order = agent_state["order"]

    if order is None:
        return "Error: Retrieve the order details first."

    if order_id not in orders:
        return "Error: Invalid order ID."

    refund_amount = order["amount"]

    agent_state["refund_amount"] = refund_amount

    return f"Calculated refund amount: Rs. {refund_amount}"


@tool
def send_confirmation_email() -> str:
    """Send a refund calculation confirmation to the configured test email.
    Only call after retrieving order details and calculating the refund.
    """

    order = agent_state["order"]
    refund_amount = agent_state["refund_amount"]

    if order is None or refund_amount is None:
        return "Error: Retrieve the order and calculate the refund first."

    sender_email = os.getenv("GMAIL_ADDRESS")
    app_password = os.getenv("GMAIL_APP_PASSWORD")
    recipient_email = os.getenv("TEST_RECIPIENT")

    if not sender_email or not app_password or not recipient_email:
        return "Error: Email configuration is missing."

    message = EmailMessage()
    message["Subject"] = "Refund Calculation Confirmation"
    message["From"] = sender_email
    message["To"] = recipient_email

    message.set_content(
        f"""
Hello {order['customer']},

Your refund calculation has been completed.

Order ID: 105
Refund amount: Rs. {refund_amount}

This is a learning demonstration.
No actual payment or refund has been issued.

Thank you.
"""
    )

    # Limited retry with exponential backoff
    for attempt in range(2):
        try:
            with smtplib.SMTP("smtp.gmail.com", 587, timeout=20) as smtp:
                smtp.ehlo()
                smtp.starttls()
                smtp.ehlo()

                smtp.login(sender_email, app_password)
                smtp.send_message(message)

                return "Confirmation email sent successfully."

        except Exception as error:

            if attempt == 0:
                print(f"Email attempt failed: {error}")
                print("Retrying in 2 seconds...")
                time.sleep(2)
            else:
                return f"Email sending failed: {error}"

    return "Email could not be sent."


# --------------------------------
# 4. Configure LLM and tools
# --------------------------------

tools = {
    "get_order_details": get_order_details,
    "calculate_refund": calculate_refund,
    "send_confirmation_email": send_confirmation_email
}

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)

llm_with_tools = llm.bind_tools(list(tools.values()))


# --------------------------------
# 5. Agent execution loop
# --------------------------------

messages = [
    HumanMessage(
        content=(
            "Process the refund calculation for order 105. "
            "First retrieve the order details, then calculate "
            "the refund, and finally send a confirmation email. "
            "Do not claim that an actual payment was issued."
        )
    )
]

MAX_ITERATIONS = 5

for iteration in range(MAX_ITERATIONS):

    print(f"\n--- Agent iteration {iteration + 1} ---")

    response = llm_with_tools.invoke(messages)
    messages.append(response)

    # Check whether the LLM requested any tools
    if not response.tool_calls:

        print("\nAgent completed the task.")
        print("Final response:", response.content)
        break

    # Execute requested tools
    for tool_call in response.tool_calls:

        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_id = tool_call["id"]

        print(f"\nSelected tool: {tool_name}")
        print(f"Arguments: {tool_args}")

        selected_tool = tools.get(tool_name)

        if selected_tool is None:
            result = f"Error: Unknown tool {tool_name}."

        else:
            try:
                result = selected_tool.invoke(tool_args)

            except Exception as error:
                result = f"Tool execution failed: {error}"

        print("Tool result:", result)

        # Return tool observation to the LLM
        messages.append(
            ToolMessage(
                content=str(result),
                tool_call_id=tool_id
            )
        )

else:
    print("\nMaximum iterations reached. Agent stopped.")
