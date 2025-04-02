from dotenv import load_dotenv
import os
import google.generativeai as genai
from typing import TypedDict, List
from langgraph.graph import StateGraph, END

# Load environment variables from .env
load_dotenv()

# Retrieve the API key
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY is not set in the .env file or environment variables.")

# Configure the Gemini API
genai.configure(api_key=api_key)

# Initialize the Gemini model with the correct model name
llm_model = genai.GenerativeModel("models/gemini-1.5-pro-latest")

# Define state structure
class State(TypedDict):
    text: str
    classification: str
    entities: List[str]
    summary: str

def classification_node(state: State):
    prompt = f"""Classify the following text into one of the categories: News, Blog, Research, or Other.

Text: {state['text']}

Category:"""

    response = llm_model.generate_content(prompt)
    classification = response.text.strip()
    return {"classification": classification}

def entity_extraction_node(state: State):
    prompt = f"""Extract all the entities (Person, Organization, Location) from the following text. Provide the result as a comma-separated list.

Text: {state['text']}

Entities:"""

    response = llm_model.generate_content(prompt)
    entities = response.text.strip().split(", ")
    return {"entities": entities}

def summarize_node(state: State):
    prompt = f"""Summarize the following text in one short sentence.

Text: {state['text']}

Summary:"""

    response = llm_model.generate_content(prompt)
    summary = response.text.strip()
    return {"summary": summary}

# Agent Workflow
workflow = StateGraph(State)
workflow.add_node("classification_node", classification_node)
workflow.add_node("entity_extraction", entity_extraction_node)
workflow.add_node("summarization", summarize_node)

workflow.set_entry_point("classification_node")
workflow.add_edge("classification_node", "entity_extraction")
workflow.add_edge("entity_extraction", "summarization")
workflow.add_edge("summarization", END)

# Compile the graph
app = workflow.compile()

# Sample input text
sample_text = """
Anthropic's MCP (Model Context Protocol) is an open-source powerhouse that lets your applications interact effortlessly with APIs across various systems.
"""

state_input = {"text": sample_text}
result = app.invoke(state_input)

# Output the results
print("Classification:", result["classification"])
print("Entities:", result["entities"])
print("Summary:", result["summary"])
