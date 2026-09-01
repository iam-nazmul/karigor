from openai import OpenAI

client = OpenAI(
    api_key="ollama",  # required by the SDK, but ignored by Ollama
    base_url="http://localhost:11434/v1",
)

resp = client.chat.completions.create(
    model="gemma3:4b",
    messages=[{"role": "user", "content": "Hello"}],
)
print(resp.choices[0].message.content)