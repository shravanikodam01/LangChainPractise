from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

load_dotenv()

#Loading the transcript from youtube api transcript api
video_id = "Gfr50f6ZBvo"

try:

    ytt_api = YouTubeTranscriptApi()
    transcript_list = ytt_api.fetch(video_id).to_raw_data()
    transcript = "".join(chunk["text"] for chunk in transcript_list)
except:
    print("No captions available")

#Splitting the transcript into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.create_documents([transcript])
print(len(chunks))

#Creating the vectorstore and retriever
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(chunks, embeddings)


retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})


#Creating the prompt template
prompt_template = ChatPromptTemplate(
    [
        ("system", "You are a helpful assistant that summarizes the content of the video."),
        ("human", "Answer the question based on the context, if context is not available, say 'I don't know.' Context: {context}. Question: {question}"),
    ], input_variables=["context", "question"]
)

#Creating the chat model
# chat = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.5)

# question = "What is the video about?"

# #Getting the relevant documents and generating the answer
# relDoc = retriever.invoke(question)

# context = "\n".join([doc.page_content for doc in relDoc])

# finalPrompt = prompt_template.invoke({ "context": context, "question": question })
# answer = chat.invoke(finalPrompt)

# print(answer.content)


# Using chains to implement the same thing

def formatDocs(docs):
    return "\n".join([doc.page_content for doc in docs])

parallelChain = RunnableParallel({
    "context": retriever | RunnablePassthrough(formatDocs),
    "question": RunnablePassthrough()
})



parser = StrOutputParser()

model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.5)

mainChain = parallelChain | prompt_template | model | parser

question = "What is the video about? Is it talking about Aliens"
answer = mainChain.invoke(question)

print(answer)