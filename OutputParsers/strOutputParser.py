from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

model = ChatHuggingFace(llm = HuggingFacePipeline.from_model_id(model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0", task="text-generation"))

response = model.invoke("What is the capital of India?")


# Prompt template

template = ChatPromptTemplate([("system", "You are a helpful assistant."),("user", "Write a detailed report on topic - {topic}?")], input_variables=["topic"])

template2 = ChatPromptTemplate([("system", "You are a helpful summarizer."),("user", "Summarize the report in 5 sentences - {report}?")], input_variables=["report"])



# Without output parser
prompt1 = template.invoke({"topic": "Climate Change"})

response = model.invoke(prompt1)

prompt2 = template2.invoke({"report": response.content})

finalResponse = model.invoke(prompt2)

print(response.content)
print(finalResponse.content)


# With output parser
parser = StrOutputParser()

chain = template | model | parser | template2 | model | parser

result = chain.invoke({"topic": "Climate Change"})

print(result)
