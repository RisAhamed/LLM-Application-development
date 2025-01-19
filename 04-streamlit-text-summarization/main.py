import streamlit as st
from langchain_groq import ChatGroq
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
import time
# Function to generate a summary
def generate_response(txt, groq_api_key):
    try:
        # Initialize the ChatGroq model
        llm = ChatGroq(temperature=0, groq_api_key=groq_api_key, model_name="mixtral-8x7b-32768")
        
        # Split the text into manageable chunks
        text_splitter = CharacterTextSplitter()
        texts = text_splitter.split_text(txt)
        docs = [Document(page_content=t) for t in texts]
        
        # Load the summarization chain
        chain = load_summarize_chain(llm, chain_type="map_reduce")
        
        # Generate the summary
        return chain.run(docs)
    except Exception as e:
        return f"An error occurred during summarization: {e}"

# Streamlit app setup
st.set_page_config(
    page_title="Writing Text Summarization"
)
st.title("Writing Text Summarization")

# Input field for the text to summarize
txt_input = st.text_area(
    "Enter your text",
    "",
    height=200
)

# List to store results
result = []

# Form for user input
with st.form("summarize_form", clear_on_submit=True):
    groq_api_key = st.text_input(
        "Groq API Key",
        type="password",
    )
    submitted = st.form_submit_button("Submit")
    
   
    with st.spinner("Summarizing the text... Please wait."):
                # Optional: Use progress bar to simulate loading
                progress = st.progress(0)
                for i in range(100):  # Simulating loading
                    time.sleep(0.02)  # Adjust for desired duration
                    progress.progress(i + 1)
                
                response = generate_response(txt_input, groq_api_key)
                
                # Clear the progress bar after processing
                progress.empty()

                st.markdown("### Summary:")
                st.info(response)
   

# Display the summarization result
if len(result):
    st.markdown("### Summary:")
    st.info(result[-1])
