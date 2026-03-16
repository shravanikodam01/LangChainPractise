from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

load_dotenv()

model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.9)

model2 = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.5)

template = ChatPromptTemplate([('user', 'Generate simple and easy to understand notes for the text - {text}')], input_variables=['text'])

template2 = ChatPromptTemplate([('user', 'Generate quiz related to the text - {text}')], input_variables=['text'])

template3 = ChatPromptTemplate([('user', "Merge the provided notes and quiz into a single text. Notes: {notes} Quiz: {quiz}")], input_variables=['notes', 'quiz'])

parser = StrOutputParser()

parallelChain = RunnableParallel({
    "notes": template | model | parser,
    "quiz": template2 | model2 | parser
})

mergeChain = template3 | model | parser

chain = parallelChain | mergeChain

response = chain.invoke({"text": "The water cycle, also known as the hydrological cycle, describes the continuous movement of water on, above, and below the surface of the Earth. It involves processes such as evaporation, condensation, precipitation, and runoff. The sun's energy drives the water cycle by causing water to evaporate from oceans, lakes, and rivers. The evaporated water then condenses into clouds and eventually falls back to the Earth's surface as precipitation. This cycle is crucial for maintaining life on Earth and regulating climate."})

print(response)
