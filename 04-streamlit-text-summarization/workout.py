import streamlit as st
from langchain_groq  import ChatGroq
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
import time


def generate_response(txt,groq_api_key):
    # Create a new chat groq instance with the provided API key
    # Create a new document with the provided text
    chat_groq = chat = ChatGroq(temperature=0, groq_api_key=groq_api_key, model_name="llama-3.3-70b-versatile")

    text_splitter = CharacterTextSplitter()
    texts = text_splitter.split_text(txt)
    docs = [Document(page_content=t) for t in texts]
    # Create a new summarize chain instance
    summarize_chain = load_summarize_chain(chat_groq,chain_type="stuff",verbose=True)
    # Summarize the text using the summarize chain
    start_time = time.time()
    summary = summarize_chain.run(docs)
    end_time = time.time()
    # Return the summary and the time it took to generate it
    return summary, end_time - start_time

st.set_page_config(
    page_title="Groq Summarizer",
    page_icon="😁"
)
st.title("Text summarizer")

txt_input =st.text_area(
    "enter the text"," ",
    height=100
)
result = []

# Form for user input
with st.form("summarize_form", clear_on_submit=False):
    groq_api_key = st.text_input(
        "Groq API Key",
        type="password",
    )
    submitted = st.form_submit_button("Submit")
    
    if submitted :
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
    