import os
from apikey import apikey

import streamlit as st

from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain

os.environ['OPENAI_API_KEY'] = apikey

# setup streamlit
st.title('Beer Reviewer')
prompt = st.text_input('What beer do you want to review?')

# Templating the prompt -- TODO: need to update UI to reflect this
script_template = PromptTemplate(
    input_variables=['title'],
    template='What {topic} should I order?'
)

#LLMs
llm = OpenAI(temperature=0.9)
llm1 = OpenAI(temperature=0.1)
title_chain = LLMChain(llm=llm, prompt=title_template, verbose = True)

if prompt: 
    response = title_chain.run(topic=prompt)
    st.write(response)