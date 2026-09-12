from ollama import chat

response = chat(
    model="qwen3",
    messages=[
        {
            "role": "user",
            "content": "Explain what an AI agent is in 2 sentences."
        }
    ]
)

print(response.message.content)