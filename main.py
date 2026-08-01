from dotenv import load_dotenv
import os

load_dotenv()

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch


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
agent = create_agent(llm, tools=tools)


def main() -> None:
    prompt = "What is the weather in Kolkata today?"
    result = agent.invoke({"messages": HumanMessage(content=prompt)})
    print(result)


if __name__ == "__main__":
    main()
