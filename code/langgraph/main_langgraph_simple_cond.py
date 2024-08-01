from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_community.llms import Ollama
from typing import Annotated, Sequence, TypedDict

import re
import os
import operator
import functools

llm = Ollama(model="llama3")

# agent node, funtion that we will use to call agents in our graph
def generic_node(state, name):
    message = state['messages'][-1]
    print(f'Step: {name} - mensagem recibada: {message}')
    return state

# defining the AgentState that holds messages and where to go next
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    # The 'next' field indicates where to route to next
    next: str

def where_to_go(state):
    messages = state['messages']
    result = llm.invoke("based on user input:{}, classify whether the text intends for the return to be a chart, table or other if the internalization is not clear.Importantly, return only the text: [chart, table, other]".format(messages[-1]))

    if "chart" in result.lower():
        return "chart"
    if "table" in result.lower():
        return "table"
    else:
        return "write"

if __name__ == '__main__':

    # defining the StateGraph
    workflow = StateGraph(AgentState)

    # agents as a node
    workflow.add_node("First", functools.partial(generic_node, name="First"))
    workflow.add_node("Chart", functools.partial(generic_node, name="Geração de grafico"))
    workflow.add_node("Table", functools.partial(generic_node, name="Geração de tabela"))
    workflow.add_node("Write", functools.partial(generic_node, name="Escreve o retorno"))

    workflow.add_conditional_edges("First", where_to_go, {  # Based on the return from where_to_go
            # If return is "continue" then we call the tool node.
            "chart": "Chart",
            "table": "Table",
            "write": "Write",
            # Otherwise we finish. END is a special node marking that the graph should finish.
            "end": END
        }
    )

    # when agents are done with the task, next one should be supervisor ALWAYS
    workflow.add_edge("Chart", END)
    workflow.add_edge("Table", END)
    workflow.add_edge("Write", END)

    # starting point should be QueryBuild
    workflow.set_entry_point("First")

    graph = workflow.compile()

    final_state = graph.invoke(
        {"messages": [HumanMessage(content="Qual foi o top 4 estados que tiveram o maior consumo em MWh total? Gere um gráfico de barras com o estado e o total MWh")]}
    )

    final_state = graph.invoke(
        {"messages": [HumanMessage(content="Quais foram o consumo em MWh total nos anos 2004, 2005 e 2006 para os estados SP, RJ, GO e DF? Crei uma tabela com as colunas para cada ano e uma linha para cada estado.")]}
    )

    final_state = graph.invoke(
        {"messages": [HumanMessage(content="Qual foi o consumo em MWh total em 2004?")]}
    )