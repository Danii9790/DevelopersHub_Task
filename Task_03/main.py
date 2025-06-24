# gemini_health_chatbot.py

import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

# Set your Gemini API Key
genai.configure(api_key= os.getenv("GEMINI_API_KEY"))

# Gemini model
model = genai.GenerativeModel("gemini-2.0-flash")

# Instructions
system_prompt = """
You are a helpful, friendly medical assistant. You only provide general health information in simple language.
⚠️ You should **never** give specific medical advice, prescribe medication, or diagnose conditions. 
Always recommend consulting a licensed healthcare provider for serious issues.
"""

# Interactive chat loop
chat = model.start_chat(history=[])
print("💬 Ask your health-related questions (type 'exit' to quit):")

while True:
    user_input = input("\nYou: ")
    if user_input.lower() in ['exit', 'quit']:
        break

    prompt = f"{system_prompt}\n\nQuestion: {user_input}"
    response = chat.send_message(prompt)
    print("\n🤖 Gemini:", response.text.strip())
