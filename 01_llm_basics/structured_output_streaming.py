
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_groq import ChatGroq

load_dotenv(Path(__file__).with_name(".env"))


class ComplaintAnalysis(BaseModel):

    category: str

    department: str

    priority: Literal["Low", "Medium", "High"]

    summary: str = Field(min_length=10)


llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)

structured_llm = llm.with_structured_output(
    ComplaintAnalysis
)

try:
    for chunk in structured_llm.stream(
        """
        Garbage has accumulated near my building.
        It has not been collected for five days.
        """
    ):
        # Partial Pydantic objects may be emitted.
        print(chunk)

except Exception as e:
    print("Structured streaming failed:", e)
