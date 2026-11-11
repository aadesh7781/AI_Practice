
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

# Add retry behaviour
retry_chain = structured_llm.with_retry(
    stop_after_attempt=3,
    wait_exponential_jitter=True
)

try:
    response = retry_chain.invoke(
        """
        Garbage has accumulated near my building.
        It has not been collected for five days.
        """
    )

    print(response.model_dump())

except Exception as e:
    print("Request failed after retries:", e)
