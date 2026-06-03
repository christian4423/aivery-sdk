"""
Drop-in memory for an OpenAI chat agent.

This is the pattern developers compare against mem0's quickstart:
https://docs.mem0.ai/quickstart
"""

import os
from openai import OpenAI
from aivery import Memory

openai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
memory = Memory()

USER_ID = "alice"


def chat(message: str) -> str:
    # 1. Retrieve relevant memories
    ctx = memory.context(message, user_id=USER_ID, top_k=10)

    # 2. Build the prompt
    system = "You are a helpful personal assistant."
    if ctx:
        system += f"\n\nRelevant memories about the user:\n{ctx}"

    # 3. Call the LLM
    response = openai.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": message},
        ],
    )
    answer = response.choices[0].message.content

    # 4. Persist the conversation
    memory.add(
        [
            {"role": "user", "content": message},
            {"role": "assistant", "content": answer},
        ],
        user_id=USER_ID,
    )

    return answer


if __name__ == "__main__":
    print(chat("Hi! I'm training for my first marathon next spring."))
    print(chat("What should I focus on in my training right now?"))
    print(chat("What do you remember about my fitness goals?"))
