from langchain_experimental.tools import PythonREPLTool
from langchain_community.agent_toolkits import create_sql_agent
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler

from typing import Annotated, Sequence, TypedDict

import os
import operator
import functools

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "[token]"
os.environ["LANGCHAIN_PROJECT"] = "pr-teste-lang-grapth"

llm_code = ChatGroq(temperature=0, groq_api_key="[token]", model_name="llama3-70b-8192")
llm = ChatGroq(temperature=0, groq_api_key="[token]", model_name="llama-3.1-70b-versatile")

#Tools - for plotting the diagram
python_repl_tool = PythonREPLTool()
tools = [python_repl_tool]

# function that returns AgentExecutor with given tool and prompt
def create_agent(llm: ChatOpenAI, tools: list, system_prompt: str):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                system_prompt,
            ),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    agent = create_openai_tools_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools)
    return executor

# agent node, funtion that we will use to call agents in our graph
def agent_node(state, agent, name):
    result = agent.invoke(state)
    return {"messages": [HumanMessage(content=result["output"], name=name)]}

class SQLHandler(BaseCallbackHandler):
    def __init__(self):
        self.sql_result = []

    def on_agent_action(self, action, **kwargs):
        """Run on agent action. if the tool being used is sql_db_query,
         it means we're submitting the sql and we can
         record it as the final sql"""

        if action.tool in ["sql_db_query"]: # "sql_db_query_checker",
            self.sql_result.append(action.tool_input)

def agent_sql_node(state, agent, name):
    handler = SQLHandler()
    result = agent.invoke({'input': state["messages"][-1].content}, {'callbacks': [handler]})
    if len(handler.sql_result) > 0:
        sql_query = handler.sql_result[0]['query']
        return {"messages": [SystemMessage(content="This is the return data from the database: " +
                                          db.run(sql_query, fetch="all", include_columns=True), name=name)]}

from langchain_community.utilities import SQLDatabase

maria_uri = 'mysql+mysqlconnector://root:art_llama3@localhost:3306/mme'
db = SQLDatabase.from_uri(maria_uri)

# QueryBuild as a node
sql_agent = create_sql_agent(llm, db=db, agent_type="tool-calling", max_iterations=5, verbose=True)
sql_node = functools.partial(agent_sql_node, agent=sql_agent, name="QueryBuild")

# Coder as a node
code_agent = create_agent(llm_code, [python_repl_tool], "You generate charts using matplotlib and return the chart as a Base64 image.")
code_node = functools.partial(agent_node, agent=code_agent, name="Coder")

members = ["QueryBuild", "Coder"]
system_prompt = (
    "You are a supervisor tasked with managing a conversation between the"
    " following workers:  {members}. Given the following user request,"
    " respond with the worker to act next. Each worker will perform a"
    " task and respond with their results and status. When finished,"
    " respond with FINISH."
)
# It will use function calling to choose the next worker node OR finish processing.
options = ["FINISH"] + members
# openai function calling
function_def = {
    "name": "route",
    "description": "Select the next role.",
    "parameters": {
        "title": "routeSchema",
        "type": "object",
        "properties": {
            "next": {
                "title": "Next",
                "anyOf": [
                    {"enum": options},
                ],
            }
        },
        "required": ["next"],
    },
}
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        (
            "system",
            "Given the conversation above, who should act next?"
            " Or should we FINISH? Select one of: {options}",
        ),
    ]
).partial(options=str(options), members=", ".join(members))

# we create the chain with llm binded with routing function + system_prompt
supervisor_chain = (
    prompt
    | llm.bind_functions(functions=[function_def], function_call="route")
    | JsonOutputFunctionsParser()
)

# defining the AgentState that holds messages and where to go next
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    # The 'next' field indicates where to route to next
    next: str

# defining the StateGraph
workflow = StateGraph(AgentState)

# agents as a node, supervisor_chain as a node
workflow.add_node("QueryBuild", sql_node)
workflow.add_node("Coder", code_node)
workflow.add_node("supervisor", supervisor_chain)

# when agents are done with the task, next one should be supervisor ALWAYS
workflow.add_edge("QueryBuild", "supervisor")
workflow.add_edge("Coder", "supervisor")

# Supervisor decides the "next" field in the graph state,
# which routes to a node or finishes. (Remember the special node END above)
workflow.add_conditional_edges(
                    "supervisor",
                    lambda x: x["next"],
                    {
                       "QueryBuild": "QueryBuild",
                       "Coder": "Coder",
                       "FINISH": END
                    })

# starting point should be supervisor
workflow.set_entry_point("supervisor")

graph = workflow.compile()

final_state = graph.invoke(
    {"messages": [HumanMessage(content="Qual foi o top 4 estados que tiveram o maior consumo em MWh total? Gere um gráfico de barras com o estado e o total MWh")]}
)