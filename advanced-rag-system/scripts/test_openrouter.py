import os
import time
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
)

#meta-llama/llama-3.3-70b-instruct:free
#qwen/qwen3-next-80b-a3b-instruct:free
#openrouter/free
def call_with_retry(messages, model="openrouter/free", max_retries=3):
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(model=model, messages=messages)
        except RateLimitError as e:
            wait = 25 * (attempt + 1)
            print(f"Rate limited, retrying in {wait}s... (attempt {attempt+1}/{max_retries})")
            time.sleep(wait)
    raise RuntimeError("Max retries exceeded")

response = call_with_retry([{"role": "user", "content": "Say hello in one sentence."}])
print(response.choices[0].message.content)