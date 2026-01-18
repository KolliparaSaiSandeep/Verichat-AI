from pydantic import BaseModel, Field

class AnswerAudit(BaseModel):
    """Binary score to assess if the answer is grounded in facts."""
    is_supported: bool = Field(description="Is the answer supported by the documents? True/False")
    reasoning: str = Field(description="Brief explanation of why it passed or failed.")