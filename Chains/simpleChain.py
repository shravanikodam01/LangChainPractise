from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv()

template = ChatPromptTemplate([("user", "Give 5 interesting facts about {topic}.")], input_variables=["topic"])

parser = StrOutputParser()

model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.9)

chain = template | model | parser

print(chain.invoke({"topic": "space exploration"}))