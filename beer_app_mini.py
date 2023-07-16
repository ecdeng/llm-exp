# WIP beer app, the Beer Reviews from BeerAdvocate are in the .csv files (using the short one for faster iteration for now)
from apikey import apikey
import os
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
import streamlit as st

from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import Chroma

# Import VectoreStore dependencies 
from langchain.agents.agent_toolkits import (
    create_vectorstore_agent,
    VectorStoreToolkit,
    VectorStoreInfo
)

os.environ['OPENAI_API_KEY'] = apikey

llm = OpenAI(temperature=0.9, verbose=True)

# Load in Beer Reviews (TODO - cite)
loader = CSVLoader(file_path="beer_reviews_short.csv", encoding="utf-8", csv_args={'delimiter':','})
beer_data = loader.load()
embeddings = OpenAIEmbeddings()
store = Chroma.from_documents(beer_data, embeddings, collection_name='beerreviews')

vectorstore_info = VectorStoreInfo(
    name="beerreviews", 
    description="reviews of beers from NAME", 
    vectorstore=store
)

# Convert to Langchain toolkit
toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info)

agent_executor = create_vectorstore_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True
)
st.title('Beer Review App')
# Create a text input box for the user
prompt = st.text_input('Input your prompt here')

# Basic loop
if prompt:
    # Then pass the prompt to the LLM
    response = agent_executor.run(prompt)
    # ...and write it out to the screen
    st.write(response)

    # With a streamlit expander  
    with st.expander('Document Similarity Search'):
        # Find the relevant pages
        search = store.similarity_search_with_score(prompt) 
        # Write out the first 
        st.write(search[0][0].page_content) 

# TODO -- Let the agent decide if it should be looking for new beers vs providing a review