from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StructuredOutputParser, ResponseSchema

load_dotenv()

schema = [
    ResponseSchema(name="fact 1", description="Fact 1 for the given topic"),
    ResponseSchema(name="fact 2", description="Fact 2 for the given topic"),
    ResponseSchema(name="fact 3", description="Fact 3 for the given topic"),
]
parser = StructuredOutputParser.from_response_schemas(schema)

model = ChatOpenAI(model="gpt-3.5-turbo")

template  = ChatPromptTemplate([("system", "You are a helpful assistant."),("user", "Give me three facts about the given topic\n {format_instruction}")], 
input_variables=[], 
partial_variables={"format_instruction": parser.get_format_instructions()}
)

prompt  = template.invoke({})

response = model.invoke(prompt)
print(response)

finalResult = parser.parse(response.content)
print(finalResult)