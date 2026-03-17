from langchain_community.document_loaders import DirectoryLoader, TextLoader

loader = DirectoryLoader("DocumentLoaders/", glob="*.txt", loader_cls=TextLoader)

docs = loader.load()

for doc in docs:
    print(doc)



# This should be used when we want to load many files lazy_load

docs1 = loader.lazy_load()

for doc in docs1:
    print(doc)
