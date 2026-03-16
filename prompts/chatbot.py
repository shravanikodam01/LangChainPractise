from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model="gpt-3.5-turbo")

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break
    response = model.invoke(user_input)
    print("AI: ", response.content)