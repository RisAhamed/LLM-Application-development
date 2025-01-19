import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from io import StringIO


def load_llm(api_key):
    return ChatGroq(api_key=api_key,model="mixtral-8x7b-32768")


st.set_page_config(page_icon="****",page_title="AI summarizer")

st.expander(label="What is this ")
st.header("AI LONG TEXT summarizer")


# st.markdown("Enter the api Key")
def get_api_key():
    return st.text_input(label= "enter your groq api key" ,type="password",key="api_key")

api_key = get_api_key()

st.markdown("## Upload the file to summarize")

upload_file = st.file_uploader("Choose a file ",type ='txt')

if upload_file :
    stringio =StringIO(upload_file.getvalue().decode("utf-8"))
    file_input = stringio.read()

    if len(file_input.split(" "))> 2000: 
        st.error("Please upload a shorter file")
        st.stop()

    if file_input:
        st.write("## Summary")
        
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n"], 
            chunk_size=5000, 
            chunk_overlap=350
        )

        splitted_docs = text_splitter.create_documents([file_input])

        try:
            llm = load_llm(api_key)
        except Exception as e:
            st.error(f"{e}")
            st.stop()

        chain = load_summarize_chain(
            llm=llm,
            chain_type="map_reduce"
        )

        st.write(chain.run(splitted_docs))

else:
    st.info("Please upload a file to begin summarization.")