## integrating code to openai

import os
from properties import openai_key
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
from langchain.memory import ConversationBufferMemory

import streamlit as st

st.markdown("""
<style>

.css-1rs6os.edgvbvh3
{
    visibility: hidden;
}

.css-cio0dv.egzxvld1
{
    visibility: hidden;
}

.css-9s5bis.edgvbvh3
{
    visibility: hidden;
}

.css-h5rgaw.egzxvld1

{
    visibility: hidden;
}
</style>
""",unsafe_allow_html=True)

os.environ["OPENAI_API_KEY"]=openai_key

############################################################################################################################


# Streamlit Framework
# st.set_page_config(
#     page_title="Search Application",
#     page_icon="🔍",
#     layout="wide",
#     initial_sidebar_state="expanded",
# )



#Background
custom_css = """
<style>
.main {
    background-image: url('https://images.unsplash.com/photo-1546484396-fb3fc6f95f98?q=80&w=2070&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D');
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: fixed;
}
</style>
"""

st.markdown(custom_css, unsafe_allow_html=True)

#for background image
st.markdown('<div class="main">', unsafe_allow_html=True)


#Title
original_title = '<p style="font-family:cursive; color:lightskyblue; font-size: 60px;"><strong>Search Application</strong></p>'
st.markdown(original_title, unsafe_allow_html=True)


#Search Bar

st.title('Search about a person or a celebrity.')

input_text = st.text_input("Enter your search", key="search_input", value="", type="default")

# Apply custom CSS to the text input
st.markdown(
    """
    <style>
    div[data-baseweb="input"] input {
        background-color: black !important;
        border-radius: 5px !important; /* Adjust the radius as needed */
        padding: 10px; /* Adjust padding for better appearance */
    }
    </style>
    """,
    unsafe_allow_html=True,
)


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
        
#for background image
st.markdown('</div>', unsafe_allow_html=True)