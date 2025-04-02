from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
import google.generativeai as genai  # Google's Gemini SDK



load_dotenv()
# api_key = os.getenv("OPENAI_API_KEY")
api_key = os.getenv("GEMINI_API_KEY")  # Update the key name if needed
if not api_key:
    raise ValueError("GEMINI_API_KEY is not set in the .env file or environment variables.")

# Configure Gemini API
genai.configure(api_key=api_key)


# Check if the API key is loaded
if not api_key:
    raise ValueError("API_KEY is not set in the .env file or environment variables.")

# Initialize the ChatOpenAI instance 
# llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key)
llm_model = genai.GenerativeModel("gemini-pro")



# Test the setup 
# response = llm.invoke("Hello! Are you working?") 
response = llm_model.generate_content("Hello! Are you working?")

print(response.content)