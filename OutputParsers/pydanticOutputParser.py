from pydantic import BaseModel, Field
from typing import Optional
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()   

class Person(BaseModel):
    name: str = Field(description="The name of the user")
    age: Optional[int] = Field(gt=18, description="The age of the user")
    city: str = Field(description="The city of the user")



parser = PydanticOutputParser(pydantic_object=Person)
model = ChatOpenAI(model="gpt-3.5-turbo")

template = ChatPromptTemplate([("system", "You are a helpful assistant."),("user", "Give me the name, age, city of fictional person of place {place}\n {format_instruction}")],
input_variables=["place"],
partial_variables={"format_instruction": parser.get_format_instructions()}
)

prompt = template.invoke({"place": "New York"})
response = model.invoke(prompt)

print(response)

finalResult = parser.parse(response.content)
print(finalResult.city)

