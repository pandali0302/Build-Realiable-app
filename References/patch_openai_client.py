import instructor
from openai import OpenAI

# Patch the OpenAI client
client = instructor.from_openai(OpenAI())

# ----------------------------------------------------------------
# Patch the OpenAI client with a custom base URL and API key
# ----------------------------------------------------------------

# local OpenAI server -- ollama
client = instructor.from_openai(
    OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama",  # required, but unused
    ),
    mode=instructor.Mode.JSON,
)



# online OpenAI server -- glm-4-0520
from dotenv import load_dotenv, find_dotenv
import os

_ = load_dotenv(find_dotenv())
zhipu_api_key = os.getenv("ZHIPUAI_API_KEY")
client = instructor.from_openai(
    OpenAI(api_key=zhipu_api_key, base_url="https://open.bigmodel.cn/api/paas/v4"),
    mode=instructor.Mode.JSON,
)
