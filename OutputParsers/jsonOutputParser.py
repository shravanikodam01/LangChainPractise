from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv

load_dotenv()


parser = JsonOutputParser()
model = ChatOpenAI(model="gpt-3.5-turbo")

template  = ChatPromptTemplate([("system", "You are a helpful assistant."),("user", "Give me the name, age, city of fictional person\n {format_instruction}")], 
input_variables=[], 
partial_variables={"format_instruction": parser.get_format_instructions()}
)

prompt  = template.invoke({})

response = model.invoke(prompt)
print(response)

finalResult = parser.parse(response.content)
print(finalResult)