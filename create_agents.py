from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import create_react_agent

from agent_prompts import SYSTEM_PROMPT_SEARCH, SYSTEM_PROMPT_SQL, SYSTEM_PROMPT_IMG


def create_search_agent(llm):
    search_tools = [TavilySearchResults(max_results=2)]
    agent_executor_search = create_react_agent(
        llm,
        search_tools,
        state_modifier=SYSTEM_PROMPT_SEARCH
    )
    return agent_executor_search


def create_sql_agent(llm, db):
    sql_tools = SQLDatabaseToolkit(db=db, llm=llm).get_tools()
    agent_executor_sql = create_react_agent(
        llm,
        sql_tools,
        state_modifier=SYSTEM_PROMPT_SQL
    )
    return agent_executor_sql

def create_image_agent(llm):
    search_tools = [TavilySearchResults(max_results=2)]
    agent_executor_img = create_react_agent(
        llm,
        search_tools,
        state_modifier=SYSTEM_PROMPT_IMG
    )
    return agent_executor_img
