from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
import openai   
import os
from dotenv import load_dotenv  
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

## Web search agent for financial information
web_search_agent = Agent(
    name="FinancialAgent",
    role="Search the web for the information about stocks, market trends, and financial news.",
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[DuckDuckGo()],
    instructions=["Always include sources"],
    show_tool_calls=True,
    markdown=True,
)

## Financial data agent for stock information
financial_data_agent = Agent(
    name="FinancialDataAgent",
    role="Fetch stock information and market trends.",
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True, company_news=True)],   
    instructions=["Use tables to display the data"],
    show_tool_calls=True,
    markdown=True,
)

multi_ai_agent = Agent(
    team=[web_search_agent, financial_data_agent],
    instructions=["Always include sources and use tables to display data when possible"],
    show_tool_calls=True,
    markdown=True,
)

multi_ai_agent.print_response("Summarise analyst recommendations and the information & latest news for NVIDIA", stream=True)


