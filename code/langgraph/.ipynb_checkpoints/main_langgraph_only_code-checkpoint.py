from langchain_community.llms import Ollama
import re

llm = Ollama(model="llama3")

user = """based on the data returned from the user's request database, generate a code in python that creates a graph in matplotlib and stores the graph image in base64 variable name: base64_graph. Response just code.
This is the return data from the database: [{'nome_do_estado': 'São Paulo', 'SUM(T2.consumo_MWh)': 2247429685.375}, {'nome_do_estado': 'Minas Gerais', 'SUM(T2.consumo_MWh)': 934651225.28125}, {'nome_do_estado': 'Rio de Janeiro', 'SUM(T2.consumo_MWh)': 661774178.34375}, {'nome_do_estado': 'Paraná', 'SUM(T2.consumo_MWh)': 492105686.1875}]"""

result = llm.invoke(user)

pattern = re.compile(r'```(.*?)```', re.DOTALL)
python_code = pattern.findall(result)
running_content = {}
if len(python_code) > 0:
    code_running = python_code[0].replace('python', '').replace('Python', '')
    exec(code_running, running_content)
    print(f"imagem aqui: {running_content['base64_graph']}")
