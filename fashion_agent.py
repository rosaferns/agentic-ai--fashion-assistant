import os
from dotenv import load_dotenv
import pandas as pd
import streamlit as st
from langchain_openai import AzureChatOpenAI
from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import MessagesState, StateGraph, START, END
from typing_extensions import TypedDict
import chromadb.api
import base64
from PIL import Image
import io

chromadb.api.client.SharedSystemClient.clear_system_cache()

# Load .env file
load_dotenv()

# Streamlit configuration for app layout
st.title("Fashion Agent")

# Initialize LLM and embeddings
llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
)

llm_4o = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME_4o"),
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    temperature=0.2
)

# Load CSV data
def load_csv(file_path):
    df = pd.read_csv(file_path)
    return df

# Create SQL database from DataFrame
def create_sql_database(df):
    engine = create_engine(os.getenv("DATABASE_URL"))
    df.to_sql('fashion', con=engine, if_exists='replace', index=False)
    db = SQLDatabase(engine=engine)
    return db

def encode_img():
    image = Image.open(uploaded_image)
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/jpeg;base64,{img_str}"


# Streamlit file uploader for CSV (enforced requirement)
st.sidebar.header("Upload CSV")
uploaded_file = st.sidebar.file_uploader("Please upload a CSV file", type=["csv"])

# If no file uploaded, stop execution and prompt the user to upload
if uploaded_file is None:
    st.warning("Please upload a CSV file to proceed.")
    st.stop()  # Stops the script execution until a file is uploaded

st.sidebar.header("Upload Image")
uploaded_image = st.sidebar.file_uploader("Please upload an image", type=["png", "jpg", "jpeg"])


# Proceed only if a file is uploaded
df = load_csv(uploaded_file)
st.write("### Data Preview", df.head())

# Display shape and columns of the uploaded CSV
st.write(f"Data Shape: {df.shape}")

db = create_sql_database(df)

toolkit = SQLDatabaseToolkit(db=db, llm=llm)
sql_tools = toolkit.get_tools()

st.sidebar.header("Bias Detector")

# Create prompts
SYSTEM_PROMPT_SEARCH = '''
Evaluate the logical relationships between the given attributes.
**Attributes**: Gender, Master Category, Sub Category, Article Type, Base Colour, Season, Usage.
### Instructions:
1. Check if attributes logically align (e.g., does Sub Category match Article Type?).
2. Highlight any mismatches or contradictions.
3. Provide a brief conclusion.
### Example:
- **Input**:
  - Gender: Men, Master Category: Apparel, Sub Category: Shoes, Article Type: Tshirts.
- **Output**:
  - "Mismatch: 'Tshirts' conflicts with 'Shoes'. Other attributes align."
'''
SYSTEM_PROMPT_SQL = '''
You are an expert in SQLite. Follow these steps:
1. Create a syntactically correct SQLite query to answer the user's question.
2. Query at most 5 results using the LIMIT clause.
3. Only query the required columns; wrap each column name in double quotes (" ").

Use the following table:

CREATE TABLE fashion (
    id FLOAT, 
    gender TEXT, 
    "masterCategory" TEXT, 
    "subCategory" TEXT, 
    "articleType" TEXT, 
    "baseColour" TEXT, 
    season TEXT, 
    usage TEXT, 
    "productDisplayName" TEXT, 
    target TEXT
)

/*
3 rows from fashion table:
id	gender	masterCategory	subCategory	articleType	baseColour	season	usage	productDisplayName	target
15970.0	Men	Apparel	Topwear	Shirts	Navy Blue	Fall	Casual	Turtle Check Men Navy Blue Shirt	Related
39386.0	Men	Apparel	Bottomwear	Jeans	Blue	Summer	Casual	Peter England Men Party Blue Jeans	Related
59263.0	Women	Accessories	Watches	Watches	Silver	Winter	Casual	Titan Women Silver Watch	Related
*/
'''

SYSTEM_PROMPT_IMG = '''
You will recieve the Base64 string of the image.
Describe the clothing in this image and then assign attributes.
**Attributes**: Gender, Master Category, Sub Category, Article Type, Base Colour, Season, Usage.
'''

SYSTEM_PROMPT_BIAS_DETECTOR = '''
Only talk about bias, not the user prompt question.
You are tasked with identifying and assessing potential bias in the provided input attributes related to gender, categories, and other related elements.
## Steps:
1. Examine all input attributes (`gender`, `masterCategory`, `subCategory`, `articleType`, `baseColour`, `season`, `usage`).
2. Identify any stereotypes or gender-specific biases within the attribute combinations.
3. Explain why a specific pairing or attribute combination might be biased.

### Example Output Format:
- If bias is detected: Explain why the attribute combination might be biased and suggest ways to mitigate the bias.
**Output**: The combination of "Gender: Women" and "Base Colour: Pink" may reinforce the stereotype that pink is a color exclusively associated with women. This association has cultural roots but is increasingly viewed as limiting and outdated. Consider promoting a more gender-neutral approach to colors.
- If no bias is detected or attributes are not related: State that 'No bias is detected'
**Output**: No bias is detected
'''

# Create tools
search = TavilySearchResults(max_results=2)
search_tools = [search]

# Create agents
agent_executor_search = create_react_agent(
    llm,
    search_tools,
    state_modifier=SYSTEM_PROMPT_SEARCH
)
agent_executor_sql = create_react_agent(llm, sql_tools, state_modifier=SYSTEM_PROMPT_SQL)
agent_executor_img = create_react_agent(llm_4o, search_tools, state_modifier=SYSTEM_PROMPT_IMG)
agent_bias_detector = create_react_agent(llm_4o, search_tools, state_modifier=SYSTEM_PROMPT_BIAS_DETECTOR)


# Agent state
class AgentState(MessagesState):
    next: str

members = ["agent_executor_search", "agent_executor_sql", "agent_executor_img"]
options = members + ["FINISH"]

SYSTEM_PROMPT_SUPERVISOR = f'''
Manage tasks between agents: {members}. Given the user request, 
respond with the worker to act next.

Agents:
1. Use `agent_executor_search` for external info.
2. Use `agent_executor_sql` for database queries.
3. Use `agent_executor_img` for images.

Once all tasks are completed:
- Compile a combined response to answer the user’s request
- Respond with "FINISH" to indicate completion.
'''

class Router(TypedDict):
    next: str


def supervisor_node(state: AgentState) -> AgentState:
    messages = [{"role": "system", "content": SYSTEM_PROMPT_SUPERVISOR}] + state["messages"]
    response = llm_4o.with_structured_output(Router).invoke(messages)
    next_ = response["next"]
    if next_ == "FINISH":
        next_ = END
    return {"next": next_}


def search_node(state: AgentState) -> AgentState:
    result = agent_executor_search.invoke(state)
    return {
        "messages": [
            HumanMessage(content=result["messages"][-1].content, name="agent_executor_search")
        ]
    }


def sql_node(state: AgentState) -> AgentState:
    result = agent_executor_sql.invoke(state)
    return {
        "messages": [HumanMessage(content=result["messages"][-1].content, name="agent_executor_sql")]
    }

def img_node(state: AgentState) -> AgentState:
    result = agent_executor_img.invoke(state)
    return {
        "messages": [HumanMessage(content=result["messages"][-1].content, name="agent_executor_img")]
    }

def bias_node(state: AgentState) -> AgentState:
    result = agent_bias_detector.invoke(state)
    return {
        "messages": [HumanMessage(content=result["messages"][-1].content, name="agent_bias_detector")]
    }


builder = StateGraph(AgentState)
builder.add_edge(START, "supervisor")
builder.add_node("supervisor", supervisor_node)
builder.add_node("agent_executor_search", search_node)
builder.add_node("agent_executor_sql", sql_node)
builder.add_node("agent_executor_img", img_node)
builder.add_node("agent_bias_detector", bias_node)

for member in members:
    builder.add_edge(member, "supervisor")

builder.add_conditional_edges("supervisor", lambda state: state["next"])
builder.add_edge("supervisor", END)

graph = builder.compile()


# User query
st.header("Fashion Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Enter your query"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    if uploaded_image:
        encoded_image = encode_img()
        st.image(uploaded_image, caption="Uploaded Image", width=200)

        prompt=[
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": encoded_image}},
        ]

    messages = graph.invoke(
        {"messages": [{"role": "user", "content": prompt}]},
        stream_mode="values",
    )

    bias_result = agent_bias_detector.invoke(
        {"messages": [{"role": "user", "content": prompt}]}
    )
    bias_result = bias_result["messages"][-1]
    with st.sidebar:
        st.markdown(bias_result.content)

    # for i, msg in enumerate(messages["messages"][1:], start=1):
    #     role = getattr(msg, "name", "N/A")  # Safely get 'name' attribute
    #     content = msg.content

    #     if role.lower() == "agent_bias_detector":
    #         with st.sidebar:
    #             st.markdown(f"{content}")
    #     else:
    #         response = f"{getattr(msg, 'name', 'N/A')}: {msg.content}"
    #         with st.chat_message("assistant"):
    #             st.markdown(response)
    #             st.session_state.messages.append({"role": "assistant", "content": response})

    response = messages["messages"][-1]
    with st.chat_message("assistant"):
        st.markdown(response.content)
    st.session_state.messages.append({"role": "assistant", "content": response.content})

    # for i, msg in enumerate(messages["messages"][1:], start=1):
    #     response = f"{getattr(msg, 'name', 'N/A')}: {msg.content}"
    #     with st.chat_message("assistant"):
    #         st.markdown(response)
    #     st.session_state.messages.append({"role": "assistant", "content": response})


