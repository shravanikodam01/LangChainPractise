from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv 
from sklearn.metrics.pairwise import cosine_similarity    
import numpy as np

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=32)

documents = ["LLMs are large language models. They are trained on large amounts of data and can generate human-like text.", 
             "Capital of India is Delhi",
             "Dhurandhar Movie is releasing on 19th march"
            ]

query = "when can i see movie"

document_embeddings = embeddings.embed_documents(documents)
query_embedding = embeddings.embed_query(query) 

similarity_scores = cosine_similarity([query_embedding], document_embeddings)[0]

index, score = sorted(list(enumerate(similarity_scores)), key=lambda x: x[1])[-1]

print(f"Most similar document with query {query}: {documents[index]}")

