import os
from groq import Groq
import gradio as gr

client = Groq(api_key=os.environ.get("GROQ_API_KEY"), )

system_prompt = {
    "role": "system",
    "content":
    "You are a helpful assistant. You reply with very short answers."
}

async def chat_groq(message, history):

  messages = [system_prompt]
  for msg in history:
    messages.append({"role": "user", "content": str(msg[0])})
    messages.append({"role": "assistant", "content": str(msg[1])})

  messages.append({"role": "user", "content": str(message)})

  response_content = ""

  stream = client.chat.completions.create(model="llama3-8b-8192",
                                          messages=messages,
                                          max_tokens=1024,
                                          temperature=1.2,
                                          stream=True)

  for chunk in stream:
    content = chunk.choices[0].delta.content
    if content:
      response_content += chunk.choices[0].delta.content
    yield response_content

with gr.Blocks(fill_height=True) as demo:
  gr.ChatInterface(chat_groq,
                   clear_btn=None,
                   undo_btn=None,
                   retry_btn=None,
                   fill_height=True,)
demo.launch()