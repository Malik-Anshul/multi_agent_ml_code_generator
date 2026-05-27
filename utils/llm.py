from dotenv import load_dotenv
from groq import Groq
import os

load_dotenv()

def call_llm(system_prompt, user_prompt, temperature=0.4):
    print("llm")
    api_key = os.getenv("GROQ_API_KEY")

    # Check API key exists
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in .env file")
    
    groq_client = Groq(api_key=api_key)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    try:
        agent = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=temperature
        )

        response = agent.choices[0].message.content.strip()
        # print("response",response)
        return response
        
    except Exception as e:
        return f"Failed to establish the connection to LLM {e}"
        

