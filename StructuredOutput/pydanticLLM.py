from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

from pydantic import BaseModel, Field

load_dotenv()

model = ChatOpenAI(model="gpt-3.5-turbo")

# Schema for structured output
class Review(BaseModel):
    summary: str = Field(description="A brief summary of the review")
    sentiment: str = Field(description="The sentiment of the review, either 'positive' or 'negative'")

structured_model = model.with_structured_output(Review)

result = structured_model.invoke("The product dint work and had a bad experience. ")

print(result)

