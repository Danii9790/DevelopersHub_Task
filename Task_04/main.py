import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()

# Set your API key
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Load Gemini 2.0 Flash model
model = genai.GenerativeModel(model_name="gemini-2.0-flash")

# Create a chat with memory
chat = model.start_chat(history=[])

# Instructions
SYSTEM_PROMPT = """
You are a friendly and empathetic mental health support assistant.
You help users feel emotionally supported during anxiety, stress, or sadness.
You must avoid giving medical or clinical advice. Always encourage users to talk to a trusted adult or a licensed therapist if needed.
Keep your responses supportive, gentle, and emotionally intelligent.
"""

print("🧠 Mental Health Chatbot (type 'exit' to quit)\n")

# Chat loop
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break

    response = chat.send_message(SYSTEM_PROMPT + "\n\nUser: " + user_input)
    print("\n🤖 Bot:", response.text.strip())
