## integrating code to openai

import os
from properties import openai_key
from langchain.llms import OpenAI
import streamlit as st

os.environ["OPENAI_API_KEY"]=openai_key


#Streamlit Framework

st.title('Langchain Demo')

input_text = st.text_input("Search the topic you want")


## OPENAI LLMS
llm=OpenAI(temperature=0.8)

if input_text:
    st.write(llm(input_text))
    