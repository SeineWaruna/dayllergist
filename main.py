from openai import OpenAI

client = OpenAI(
    base_url = 'http://localhost:11434/v1',
    api_key='ollama', # required, but unused
)

response = client.chat.completions.create(
  model="gemma3:1b-it-qat",
  messages=[
    # {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What are common treatments for seasonal allergies?"},
  ]
)
print(response.choices[0].message.content)