
from pathlib import Path
from enum import Enum

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq

load_dotenv(Path(__file__).with_name(".env"))


class Priority(str, Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"


class Sentiment(str, Enum):
    POSITIVE = "Positive"
    NEUTRAL = "Neutral"
    NEGATIVE = "Negative"


class Location(BaseModel):
    area: str = Field(
        description="Area where the complaint occurred"
    )

    landmark: str | None = Field(
        default=None,
        description="Nearby landmark, if mentioned"
    )


class ComplaintAnalysis(BaseModel):

    category: str = Field(
        description="Category of the civic complaint"
    )

    department: str = Field(
        description="Municipal department responsible"
    )

    priority: Priority

    sentiment: Sentiment

    location: Location

    summary: str = Field(
        min_length=10,
        description="Short summary of the complaint"
    )


llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)


structured_llm = llm.with_structured_output(
    ComplaintAnalysis
)


response = structured_llm.invoke(
    """
    Garbage has been accumulating in Andheri near
    the railway station for five days.

    Nobody has collected it. The smell is terrible
    and residents are facing problems.
    """
)


print("Category:", response.category)
print("Department:", response.department)
print("Priority:", response.priority.value)
print("Sentiment:", response.sentiment.value)

print("Area:", response.location.area)
print("Landmark:", response.location.landmark)

print("Summary:", response.summary)

# Convert Pydantic object into dictionary
print("\nComplete structured data:")
print(response.model_dump())
