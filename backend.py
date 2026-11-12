from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, Any
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
import sqlite3
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_tavily import TavilySearch
from langchain_core.tools import tool
import math
import requests
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
from langgraph.types import interrupt, Command


load_dotenv()


# LLM via OpenRouter
llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    temperature=0.7
)


# Free local embeddings (no API key needed)
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)



def ingest_rag_document(file_path):
    DB_PATH = "faiss_db"
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(DB_PATH)



def get_retriever():
    DB_PATH = "faiss_db"
    vector_store = FAISS.load_local(
            folder_path=DB_PATH,
            embeddings=embeddings,
            allow_dangerous_deserialization=True
        )

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    return retriever




# rag tool

@tool
def rag_tool(query: str) -> str:
    """
    Retrieve relevant information from the PDF document.

    Use this tool when the user asks factual or conceptual questions
    that may be answered using the stored PDF documents.

    Args:
        query: The question or search query used to retrieve PDF content.
    """
    retriever = get_retriever()
    documents = retriever.invoke(query)

    if not documents:
        return "No relevant information was found in the PDF."

    formatted_documents = []

    for index, document in enumerate(documents, start=1):
        source = document.metadata.get("source", "Unknown source")
        page = document.metadata.get("page", "Unknown page")

        formatted_documents.append(
            f"Document {index}\n"
            f"Source: {source}\n"
            f"Page: {page}\n"
            f"Content: {document.page_content}"
        )

    return "\n\n".join(formatted_documents)




# Tools

search_tool = TavilySearch(
    max_results=5,
    topic="general",
    search_depth="advanced"
)


@tool
def calculator(expression: str) -> str:
    """
    Useful for simple math calculations.
    Input should be a valid math expression.
    Example: 2 + 2, math.sqrt(16), 10 * 5
    """

    try:
        allowed = {
            "math": math,
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
            "sum": sum
        }

        result = eval(expression, {"__builtins__": {}}, allowed)
        return str(result)

    except Exception as e:
        return f"Calculation error: {str(e)}"




@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA')
    using Alpha Vantage.
    """
    api_key = os.getenv("ALPHAVANTAGE_API_KEY")
    if not api_key:
        return {"error": "Set the ALPHAVANTAGE_API_KEY environment variable."}

    response = requests.get(
        "https://www.alphavantage.co/query",
        params={
            "function": "GLOBAL_QUOTE",
            "symbol": symbol,
            "apikey": api_key,
        },
        timeout=10,
    )
    response.raise_for_status()
    return response.json()



@tool
def purchase_stock(symbol: str, quantity: int) -> dict:
    """
    Simulate purchasing a given quantity of a stock symbol.

    HUMAN-IN-THE-LOOP:
    Before confirming the purchase, this tool will interrupt
    and wait for a human decision ("yes" / anything else).
    """
    # This pauses the graph and returns control to the caller
    decision = interrupt(f"Approve buying {quantity} shares of {symbol}? (yes/no)")

    if isinstance(decision, str) and decision.lower() == "yes":
        return {
            "status": "success",
            "message": f"Purchase order placed for {quantity} shares of {symbol}.",
            "symbol": symbol,
            "quantity": quantity,
        }

    else:
        return {
            "status": "cancelled",
            "message": f"Purchase of {quantity} shares of {symbol} was declined by human.",
            "symbol": symbol,
            "quantity": quantity,
        }




@tool
def get_current_weather(location: str) -> str:
    """
    Get the current real-time weather for a given city or location.

    Args:
        location: City or location name, for example:
                  "Dhaka", "London", or "New York".

    Returns:
        A formatted current weather report.
    """

    api_key = os.getenv("WEATHERSTACK_API_KEY")

    if not api_key:
        return (
            "Weather API key is missing. "
            "Set the WEATHERSTACK_API_KEY environment variable."
        )

    try:
        url = "http://api.weatherstack.com/current"
        params = {
            "access_key": api_key,
            "query": location,
            "units": "m"  # metric
        }

        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        if "error" in data:
            return f"Weather API error: {data['error'].get('info', 'Unknown error')}"

        current = data.get("current", {})
        location_info = data.get("location", {})

        display_location = location_info.get("name", location)
        country = location_info.get("country", "")
        if country:
            display_location += f", {country}"

        temperature = current.get("temperature", "N/A")
        feels_like = current.get("feelslike", "N/A")
        humidity = current.get("humidity", "N/A")
        wind_speed = current.get("wind_speed", "N/A")
        description = current.get("weather_descriptions", ["N/A"])[0]
        visibility = current.get("visibility", "N/A")
        pressure = current.get("pressure", "N/A")

        return (
            f"Current weather in {display_location}:\n"
            f"- Condition: {description}\n"
            f"- Temperature: {temperature}°C\n"
            f"- Feels like: {feels_like}°C\n"
            f"- Humidity: {humidity}%\n"
            f"- Pressure: {pressure} hPa\n"
            f"- Wind speed: {wind_speed} km/h\n"
            f"- Visibility: {visibility} km"
        )

    except requests.Timeout:
        return "The weather service request timed out. Please try again."

    except requests.HTTPError as error:
        status_code = error.response.status_code if error.response else "unknown"
        return f"Weather API returned an HTTP error: {status_code}"

    except requests.RequestException as error:
        return f"Could not connect to the weather service: {error}"

    except (KeyError, TypeError, ValueError) as error:
        return f"Unexpected weather API response: {error}"



# Make tool list
tools = [search_tool, calculator, get_stock_price, get_current_weather, rag_tool, purchase_stock]

# Make the LLM tool-aware
llm_with_tools = llm.bind_tools(tools)




# State
class ChatState(TypedDict):

    messages: Annotated[list[BaseMessage], add_messages]



# Nodes 1
def chat_node(state: ChatState):
    """LLM node that can answer directly or call an appropriate tool."""

    system_message = SystemMessage(
        content=(
            "You are a helpful Agentic Chatbot with access to several tools.\n\n"

            "Tool usage instructions:\n"
            "- Use `rag_tool` for questions about the uploaded PDF or document. "
            "Always retrieve relevant document content before answering PDF-related questions.\n"
            "- Use `search_tool` for current events, recent information, or information "
            "that requires an internet search.\n"
            "- Use `calculator` for mathematical calculations. Do not calculate complex "
            "expressions manually when the calculator is available.\n"
            "- Use `get_stock_price` when the user asks for the current price of a stock.\n"
            "- Use `purchase_stock` when the user wants to purchase a stock.\n"
            "- Use `get_current_weather` when the user asks about current weather for a location.\n\n"

            "Answer general questions directly when no tool is required. "
            "Do not invent information from the uploaded document. "
            "If the user asks about a PDF but no document is available, ask them to upload a PDF. "
            "After receiving a tool result, provide a clear and helpful final answer."
        )
    )

    messages = [
        system_message,
        *state["messages"]
    ]

    response = llm_with_tools.invoke(messages)

    return {"messages": [response]}




# Nodes 2 - tool node
tool_node = ToolNode(tools)



# Checkpointer
conn = sqlite3.connect(database="chatbot.db", check_same_thread=False)
checkpoint = SqliteSaver(conn)



# graph
graph = StateGraph(ChatState)

# add nodes
graph.add_node('chat_node', chat_node)
graph.add_node('tools', tool_node)

#add edges
graph.add_edge(START, 'chat_node')
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge('tools', 'chat_node')

chatbot = graph.compile(checkpointer=checkpoint)



# Helper functions for Streamlit frontend
def get_all_threads():
    all_threads = set()
    for ckpt in checkpoint.list(None):
        all_threads.add(ckpt.config['configurable']['thread_id'])

    return list(all_threads)
