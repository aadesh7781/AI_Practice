
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_groq import ChatGroq

load_dotenv(Path(__file__).with_name(".env"))


# 1. Define output schema
class ComplaintAnalysis(BaseModel):

    category: str = Field(
        description="The type of civic complaint"
    )

    department: str = Field(
        description="Municipal department responsible"
    )

    priority: Literal["Low", "Medium", "High"]

    sentiment: Literal[
        "Positive",
        "Neutral",
        "Negative"
    ]


# 2. Initialise model
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)


# 3. Create structured model
structured_llm = llm.with_structured_output(
    ComplaintAnalysis
)


# 4. Send input
response = structured_llm.invoke(
    """
    There is garbage accumulating near my building.
    Nobody has collected it for the last five days
    and it is creating a terrible smell.
    """
)


# 5. Access structured fields
print("Category:", response.category)
print("Department:", response.department)
print("Priority:", response.priority)
print("Sentiment:", response.sentiment)
