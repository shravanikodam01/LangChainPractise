from typing import Literal

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.runnables import RunnableParallel, RunnableBranch, RunnableLambda
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

class Sentiment(BaseModel):
    sentiment: Literal['positive', 'negative'] = Field(description="The sentiment of the feedback, can be positive, negative")

model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.9)
model2 = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.5)
pydanticParser = PydanticOutputParser(pydantic_object=Sentiment)
template = ChatPromptTemplate([('user', 'Evaluate the sentiment of the feedback given - {feedback} /n {format_instruction}')], input_variables=['feedback'], partial_variables={"format_instruction": pydanticParser.get_format_instructions()})


parser = StrOutputParser()
classifierChain = template | model | pydanticParser
response = classifierChain.invoke({'feedback': 'I love this product!'})

result = response.sentiment


template2 = ChatPromptTemplate([('user', 'Write a response to the positive feedback - {feedback}')], input_variables=['feedback'])

template3 = ChatPromptTemplate([('user', 'Write a response to the negative feedback - {feedback}')], input_variables=['feedback'])

branch_chain = RunnableBranch(
    (lambda x: x.sentiment=='positive', template2|model2),
    (lambda x: x.sentiment=='negative', template3|model2),
    RunnableLambda(lambda x: "Sorry, I could not understand the sentiment of the feedback.")
)

chain = classifierChain | branch_chain | parser

print(chain.invoke({'feedback': 'I love this product!'}))
