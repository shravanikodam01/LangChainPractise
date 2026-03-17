from langchain_community.document_loaders import PyPDFLoader


loader = PyPDFLoader("file.pdf")

docs = loader.load()

print(docs[0])