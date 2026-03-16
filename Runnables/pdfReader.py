from langchain_openai import ChatOpenAI
from langchain_core.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings    
from langchain_core.vectorstores import FAISS

loader = TextLoader("/Runnables/doc.txt")
documents = loader.load()

print(documents)