
from pydantic import BaseModel, Field, field_validator
from typing import Literal


class ComplaintAnalysis(BaseModel):

    category: str

    department: str

    priority: Literal["Low", "Medium", "High"]

    summary: str = Field(min_length=10)

    @field_validator("category", "department", "summary")
    @classmethod
    def validate_text(cls, value):
        value = value.strip()

        if not value:
            raise ValueError("Field cannot be empty")

        return value


# Valid data
complaint = ComplaintAnalysis(
    category="Garbage",
    department="Waste Management",
    priority="High",
    summary="Garbage accumulation near residential buildings."
)

print(complaint.model_dump())


# Invalid data
try:
    complaint = ComplaintAnalysis(
        category="   ",
        department="Waste Management",
        priority="High",
        summary="Garbage issue"
    )

except Exception as e:
    print("\nValidation error:")
    print(e)
