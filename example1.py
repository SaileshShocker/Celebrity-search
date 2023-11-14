## integrating code to openai

import os
from constants import openai_key
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
from langchain.memory import ConversationBufferMemory

import streamlit as st

os.environ["OPENAI_API_KEY"]=openai_key


#Streamlit Framework

st.title('Celebrity Search')

input_text = st.text_input("Search the topic you want")



## First Prompt Template

first_input_prompt=PromptTemplate(
    input_variables=['name'],
    template = "Tell me about {name}"
)

# Memory

person_memory = ConversationBufferMemory(input_key='name', memory_key='chat_history')
dob_memory = ConversationBufferMemory(input_key='person', memory_key='chat_history')
descr_memory = ConversationBufferMemory(input_key='dob', memory_key='description_history')




## OPENAI LLMS
llm=OpenAI(temperature=0.8)
chain = LLMChain(
    llm=llm,prompt=first_input_prompt,verbose=True,output_key='person',memory=person_memory)



## Second Prompt

second_input_prompt=PromptTemplate(
    input_variables=['person'],
    template="when was {person} born"
)

chain2 = LLMChain(
    llm=llm,prompt=second_input_prompt,verbose=True,output_key='dob',memory=dob_memory)



## Third Prompt

Third_input_prompt=PromptTemplate(
    input_variables=['dob'],
    template="Mention 5 major events happened around {dob} in the world"
)

chain3 = LLMChain(
    llm=llm,prompt=Third_input_prompt,verbose=True,output_key='description',memory=descr_memory)


##SequentialChain

parent_chain = SequentialChain(
    chains=[chain,chain2,chain3],input_variables=['name'],output_variables=['person','dob','description'],verbose=True)



if input_text:
    st.write(parent_chain({'name':input_text}))
    
    
    with st.expander('Person Name'): 
        st.info(person_memory.buffer)

    with st.expander('Major Events'): 
        st.info(descr_memory.buffer)