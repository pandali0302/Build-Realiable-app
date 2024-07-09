from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
import os

# --------------------------------------------------------------
# Load API Token From the .env File
# --------------------------------------------------------------

_ = load_dotenv(find_dotenv())
zhipu_api_key = os.getenv("ZHIPUAI_API_KEY")


client = OpenAI(api_key=zhipu_api_key, base_url="https://open.bigmodel.cn/api/paas/v4")


def simple_chat(use_stream=False):
    messages = [
        {
            "role": "system",
            "content": "请在你输出的时候都带上“喵喵喵”三个字，放在开头。",
        },
        {"role": "user", "content": "你是谁"},
    ]
    response = client.chat.completions.create(
        model="glm-4-0520",
        messages=messages,
        stream=use_stream,
        max_tokens=256,
        temperature=0.4,
        presence_penalty=1.2,
        top_p=0.8,
    )
    if response:
        if use_stream:
            for chunk in response:
                print(chunk.choices[0].delta.content, end="")
        else:
            print(response)
    else:
        print("Error:", response.status_code)


if __name__ == "__main__":
    simple_chat(use_stream=False)
