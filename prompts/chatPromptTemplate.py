from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv  

load_dotenv()

model = ChatOpenAI(model="gpt-3.5-turbo")

# template
template = ChatPromptTemplate([
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "human", "content": "{question}"}
])

prompt = template.invoke({"domain": "general knowledge", "question": "What is the capital of India?"})
print(prompt)
response = model.invoke(prompt)

