from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()       


embeddings = OpenAIEmbeddings(model="text-embedding-3-large", 
dimensions=32)


documents = ["Delhi is capital of India?", 
             "Mumbai is financial capital of India?",
               "New Delhi is political capital of India?"]


result = embeddings.embed_documents(documents)

print(result)
print(len(result))
print(len(result[0]))
print(len(result[1]))
print(len(result[2]))   