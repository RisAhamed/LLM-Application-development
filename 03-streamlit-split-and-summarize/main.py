import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from io import StringIO

# Function to load the LLM
def load_LLM(api_key):
    return ChatGroq(temperature=0, groq_api_key=api_key, model_name="mixtral-8x7b-32768")

# Page title and header
st.set_page_config(page_title="AI Long Text Summarizer")
st.header("AI Long Text Summarizer")

# Intro and instructions
st.markdown("ChatGPT cannot summarize long texts. Now you can do it with this app.")

# Input for OpenAI API Key
st.markdown("## Enter Your Groq API Key")
def get_openai_api_key():
    return st.text_input(label="Groq API Key", placeholder="Ex: sk-2twmA8tfCb8un4...", key="openai_api_key_input", type="password")

openai_api_key = get_openai_api_key()

# Input for text file upload
st.markdown("## Upload the text file you want to summarize")
uploaded_file = st.file_uploader("Choose a file", type="txt")

# Output section
st.markdown("### Here is your Summary:")

if uploaded_file is not None:
    # Read the uploaded file
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    file_input = stringio.read()

    if len(file_input.split(" ")) > 20000:
        st.error("Please upload a shorter file. The maximum length is 20,000 words.")
        st.stop()

    if file_input:
        if not openai_api_key:
            st.warning(
                'Please insert your Groq API Key. '
                'Instructions [here](https://help.openai.com/en/articles/4936850-where-do-i-find-my-secret-api-key)', 
                icon="⚠️"
            )
            st.stop()

        # Split the input text
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n"], 
            chunk_size=5000, 
            chunk_overlap=350
        )
        splitted_documents = text_splitter.create_documents([file_input])

        # Load the LLM
        try:
            llm = load_LLM(api_key=openai_api_key)
        except Exception as e:
            st.error(f"Failed to initialize the LLM: {e}")
            st.stop()

        # Load the summarization chain
        summarize_chain = load_summarize_chain(
            llm=llm, 
            chain_type="map_reduce"
        )

        # Generate the summary
        try:
            summary_output = summarize_chain.run(splitted_documents)
            st.write(summary_output)
        except Exception as e:
            st.error(f"An error occurred during summarization: {e}")
else:
    st.info("Please upload a file to begin summarization.")
