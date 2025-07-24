

import openai
import os

openai.api_key = "your_api_key_here"

tickets = [
    "My internet is not working since morning",
    "I want to reset my password",
    "The billing amount seems incorrect",
    "I need to cancel my subscription"
]

prompt_template = """
You are a helpful AI assistant that classifies support tickets into categories:
1. Technical Issue
2. Billing Problem
3. Account Request
4. Cancellation

Ticket: "{}"
Top 3 categories with confidence:
"""

for ticket in tickets:
    prompt = prompt_template.format(ticket)
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5
    )
    print(f"Ticket: {ticket}")
    print("Tags:", response.choices[0].message['content'])
    print("-" * 50)
