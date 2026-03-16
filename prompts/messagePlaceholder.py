from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()


chat_template = ChatPromptTemplate([
    {"role": "system", "content": "You are a helpful customer support agent."},
    MessagesPlaceholder(variable_name="chat_history"),
    {"role": "user", "content": "{query}"}
])

model = ChatOpenAI(model="gpt-3.5-turbo")

chat_history = []
# load chat history from file
with open("prompts/chat_history.txt", "r") as f:
    chat_history.extend(f.read().splitlines())


query = "What is my name? "

prompt = chat_template.invoke({"chat_history": chat_history, "query": query})
print(prompt)
response = model.invoke(prompt)

print(response.content)