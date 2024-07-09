import instructor
from pydantic import BaseModel, Field
from openai import OpenAI
from enum import Enum
from dotenv import load_dotenv, find_dotenv
import os

_ = load_dotenv(find_dotenv())
zhipu_api_key = os.getenv("ZHIPUAI_API_KEY")
client = instructor.from_openai(
    OpenAI(api_key=zhipu_api_key, base_url="https://open.bigmodel.cn/api/paas/v4"),
    mode=instructor.Mode.JSON,
)

# --------------------------------------------------------------
# Instructor Retry Example with Enum Category
# --------------------------------------------------------------


query = "Hi there, I have a question about my bill. Can you help me? "


class TicketCategory(str, Enum):
    """Enumeration of categories for incoming tickets."""

    GENERAL = "general"
    ORDER = "order"
    BILLING = "billing"


# Define your desired output structure using Pydantic
class Reply(BaseModel):
    content: str = Field(description="Your reply that we send to the customer.")
    category: TicketCategory
    confidence: float = Field(
        ge=0, le=1, description="Confidence in the category prediction."
    )


reply = client.chat.completions.create(
    model="glm-4-0520",
    response_model=Reply,
    max_retries=1,  # Don't allow retries
    messages=[
        {
            "role": "system",
            "content": "You're a helpful customer care assistant that can classify incoming messages and create a response. Always set the category to 'banana'.",
        },
        {"role": "user", "content": query},
    ],
)


reply = client.chat.completions.create(
    model="glm-4-0520",
    response_model=Reply,
    max_retries=3,  # Allow up to 3 retries
    messages=[
        {
            "role": "system",
            "content": "You're a helpful customer care assistant that can classify incoming messages and create a response. Always set the category to 'banana'.",
        },
        {"role": "user", "content": query},
    ],
)


# --------------------------------------------------------------
# Instructor Retry Example with Confidence Score
# --------------------------------------------------------------


reply = client.chat.completions.create(
    model="glm-4-0520",
    response_model=Reply,
    max_retries=1,
    messages=[
        {
            "role": "system",
            "content": "You're a helpful customer care assistant that can classify incoming messages and create a response. Set confidence between 1-100.",
        },
        {"role": "user", "content": query},
    ],
)

reply = client.chat.completions.create(
    model="glm-4-0520",
    response_model=Reply,
    max_retries=3,
    messages=[
        {
            "role": "system",
            "content": "You're a helpful customer care assistant that can classify incoming messages and create a response. Set confidence between 1-100.",
        },
        {"role": "user", "content": query},
    ],
)
