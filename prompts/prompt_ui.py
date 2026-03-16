from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate

load_dotenv()

st.header("Research Tool")

paper_input = st.selectbox("Select a research paper", ["Attention is all you need", "BERT", "GPT-3"])

style_input = st.selectbox("Select a research style", ["Summary", "Key Findings", "Methodology"])

length_input = st.selectbox("Select a research length", ["Short", "Medium", "Long"])

#template prompt

template = PromptTemplate(template=f"""
Please summarize the research paper titled "{paper_input}" with the following
specifications:
Explanation Style: {style_input}
Explanation Length: {length_input}
1. Mathematical Details:
- Include relevant mathematical equations if present in the paper.
- Explain the mathematical concepts using simple, intuitive code snippets
where applicable.
2. Analogies:
- Use relatable analogies to simplify complex ideas.
If certain information is not available in the paper, respond with: "Insufficient
information available" instead of guessing.
Ensure the summary is clear, accurate, and aligned with the provided style and
length.


""",
input_variables=["paper_input", "style_input", "length_input"])


model = ChatOpenAI(model="gpt-3.5-turbo")

if st.button("Search"):
    prompt = template.invoke({ "paper_input": paper_input, "style_input": style_input, "length_input": length_input })
    response = model.invoke(prompt)
    st.write(response.content)
    
