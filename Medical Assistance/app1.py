import streamlit as st
import os
from src.helper import load_pdf_file, text_split, download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from src.prompt import system_prompt
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def initialize_pinecone():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    return pc

def setup_qa_chain(docsearch):
    llm = ChatGroq(
        temperature=0.1,
        model_name="mixtral-8x7b-32768",
        groq_api_key=GROQ_API_KEY
    )
    
    prompt =system_prompt
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=docsearch.as_retriever(search_kwargs={"k": 3}),
        chain_type_kwargs={"prompt": prompt}
    )
    
    return qa_chain

def main():
    st.set_page_config(page_title="Medical Assistant", page_icon="🏥")
    st.title("Medical Assistant Chatbot 🏥")
    
    # Initialize session state
    if 'qa_chain' not in st.session_state:
        st.session_state.qa_chain = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Sidebar for PDF upload
    with st.sidebar:
        st.header("Upload Medical Document")
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        
        if uploaded_file is not None and st.button("Process Document"):
            with st.spinner("Processing document..."):
                # Save the uploaded file temporarily
                with open("temp.pdf", "wb") as f:
                    f.write(uploaded_file.getvalue())
                
                # Process the PDF
                documents = load_pdf_file("Data/")
                chunks = text_split(documents)
                embeddings = download_hugging_face_embeddings()
                
                # Initialize Pinecone and create vector store
                pc = initialize_pinecone()
                index_name = "medical-chatbot"
                
                docsearch = PineconeVectorStore.from_documents(
                    documents=chunks,
                    index_name=index_name,
                    embedding=embeddings
                )
                
                # Setup QA chain
                st.session_state.qa_chain = setup_qa_chain(docsearch)
                st.success("Document processed successfully!")
                
                # Clean up
                os.remove("temp.pdf")
    
    # Main chat interface
    st.header("Chat with Medical Assistant")
    
    # Display chat history
    for q, a in st.session_state.chat_history:
        st.text_area("Question:", value=q, height=100, disabled=True)
        st.text_area("Answer:", value=a, height=200, disabled=True)
        st.markdown("---")
    
    # Chat input
    user_question = st.text_area("Ask a medical question:", height=100)
    
    if st.button("Send") and user_question:
        if st.session_state.qa_chain is None:
            st.error("Please upload and process a medical document first!")
        else:
            with st.spinner("Generating response..."):
                try:
                    response = st.session_state.qa_chain.invoke(user_question)
                    answer = response['result']
                    
                    # Add to chat history
                    st.session_state.chat_history.append((user_question, answer))
                    
                    # Clear input
                    st.experimental_rerun()
                    
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
