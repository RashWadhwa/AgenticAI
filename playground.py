from phi.agent import Agent
import phi.api
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo

import os
import phi
from phi.playground import Playground, serve_playground_app

from dotenv import load_dotenv  
load_dotenv() # Loading environment variables from .env file

phi.api_key = os.getenv("PHI_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

if not phi.api_key:
    raise RuntimeError("PHI_API_KEY is missing. Add it to your .env file.")

if not groq_api_key:
    raise RuntimeError("GROQ_API_KEY is missing. Add it to your .env file.")

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

app = Playground(agents=[web_search_agent, financial_data_agent]).get_app()

if __name__=="__main__":
    serve_playground_app("playground:app", reload=True)