from typing import List
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os

load_dotenv()

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

class Source(BaseModel):
    """Schema for a source used by the agent"""

    url: str = Field(description="The URL of the source")

class AgentResponse(BaseModel):
    """Schema for agent response with answer and sources"""

    answer: str = Field(description="Thr agent's answer to the query")
    sources: List[Source] = Field(
        default_factory=list, description="List of sources used to generate the answer"
    )    

PROVIDER = "openrouter"

if PROVIDER == "openrouter":
    llm = ChatOpenAI(
        temperature=0,
        model="openai/gpt-oss-20b",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
    )

elif PROVIDER == "groq":
    llm = ChatGroq(temperature=0, model="openai/gpt-oss-20b")

tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)


def main() -> None:
    prompt = "What is the weather in Kolkata today?"
    result = agent.invoke({"messages": HumanMessage(content=prompt)})
    print(result)


if __name__ == "__main__":
    main()
