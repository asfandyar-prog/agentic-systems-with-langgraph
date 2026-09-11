
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Enable LangSmith tracing
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "LangGraph-Docs"

from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_tool_calling_agent
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import AgentExecutor

# Tools
tools = [
    TavilySearchResults(max_results=1)
]

# Prompt (replacing hub.pull)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful AI assistant."),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}")
    ]
)

# LLM
llm = ChatGroq(
    model="llama3-70b-8192",
    temperature=0.7
)

# Agent
agent = create_tool_calling_agent(
    llm,
    tools,
    prompt
)

# Executor
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True
)

# Run agent
response = agent_executor.invoke(
    {"input": "What is the capital of Hungary?"}
)

print(response)
