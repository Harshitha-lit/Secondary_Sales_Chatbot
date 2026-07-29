import asyncio
import sys
import os

# add backend path
sys.path.append(r"C:\Users\LohiyaGroup\Documents\Secondary_Agent\Agent_Chatbot\backend")
from main import chat_endpoint
from models import ChatRequest

async def run_test():
    req = ChatRequest(messages=[
        {"role": "user", "content": "List out the customers who will churn in next 1 month."},
        {"role": "assistant", "content": "Based on the churn risk model..."},
        {"role": "user", "content": "List the top 5 declining outlets by value at risk."}
    ])
    
    try:
        response = await chat_endpoint(req)
        print("Success:", type(response))
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    asyncio.run(run_test())
