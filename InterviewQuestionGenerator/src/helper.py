import os
from dotenv import load_dotenv
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
os.environ["GROQ_API_KEY"] = GROQ_API_KEY
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
from langchain.document_loaders import PyPDFLoader
from langchain.docstore.document import Document
from langchain.text_splitter import TokenTextSplitter,RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv
from src.prompt import *
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from typing import List, Tuple
from langchain_google_genai import GoogleGenerativeAIEmbeddings



def chunk_for_question_generation(
    text: str, chunk_size: int = 1000, chunk_overlap: int = 200, separators: List[str] = [".", "!", "?", "\n", ""]
) -> List[Document]:
    """Chunks text for question generation."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=separators)
    chunks = splitter.split_text(text)
    return [Document(page_content=t) for t in chunks]


def chunk_for_answer_generation(
    documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 100, separators: List[str] = [".", "!", "?", "\n", ""]
) -> List[Document]:
    """Chunks documents for answer generation."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=separators)
    return splitter.split_documents(documents)


def file_processing(file_path: str) -> Tuple[List[Document], List[Document]]:
    """Processes a PDF file by chunking it for question and answer generation."""
    loader = PyPDFLoader(file_path)
    data = loader.load()

    question_gen = ""
    for page in data:
        question_gen += page.page_content

    document_ques_gen = chunk_for_question_generation(question_gen, chunk_size=5000, chunk_overlap=200)  
    document_answer_gen = chunk_for_answer_generation(document_ques_gen, chunk_size=1000, chunk_overlap=100)

    return document_ques_gen, document_answer_gen


def llm_pipeline(file_path, vector_store_path):
    """
    Generates questions from a PDF and sets up a chain to answer them.

    Args:
        file_path (str): Path to the PDF file.
        vector_store_path (str): Path to store the vector store.
    Returns:
        Tuple[RetrievalQA, List[str]]: An answer generation chain and a list of generated questions.
    """
    try:
        document_ques_gen, document_answer_gen = file_processing(file_path)

        # Question Generation Chain
        llm_ques_gen_pipeline = ChatGroq(temperature=0.3)

        prompt_template = """
            Generate multiple questions about the text provided. 
            Each question should be detailed and should focus on different aspects of the text.
            {text}
        """

        refine_template = """
            Your job is to generate questions from a text, if you have already generated questions about a text, then you must refine and make the questions better.
            Here is the original questions: {existing_answer}
            Here is the new text you need to use to refine your answers : {text}
        """

        PROMPT_QUESTIONS = PromptTemplate(template=prompt_template, input_variables=["text"])

        REFINE_PROMPT_QUESTIONS = PromptTemplate(
            input_variables=["existing_answer", "text"], template=refine_template
        )

        ques_gen_chain = load_summarize_chain(
            llm=llm_ques_gen_pipeline,
            chain_type="refine",
            verbose=True,
            question_prompt=PROMPT_QUESTIONS,
            refine_prompt=REFINE_PROMPT_QUESTIONS,
        )

        ques = ques_gen_chain.run(document_ques_gen)

        # Answer Retrieval and Generation Chain
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

        #Check if the vector store exists
        if os.path.exists(vector_store_path):
            vector_store = FAISS.load_local(vector_store_path, embeddings)
        else:
            vector_store = FAISS.from_documents(document_answer_gen, embeddings)
            vector_store.save_local(vector_store_path)
    
        llm_answer_gen = ChatGroq(temperature=0.1)
        ques_list = ques.split("\n")
        filtered_ques_list = [
            element for element in ques_list if element.endswith("?") or element.endswith(".")
        ]
        answer_generation_chain = RetrievalQA.from_chain_type(
            llm=llm_answer_gen, chain_type="stuff", retriever=vector_store.as_retriever()
        )

        return answer_generation_chain, filtered_ques_list
    
    except Exception as e:
        print(f"Error in llm_pipeline : {e}")
        return None, None

def answer_questions(answer_generation_chain, filtered_ques_list):
    """Generates answers for a list of questions using an answer generation chain.

        Args:
            answer_generation_chain: RetrievalQA chain to generate answers
            filtered_ques_list : List of questions to be answered

        Returns:
            answers: List of questions and their corresponding answers.
    """
    answers = []
    for question in filtered_ques_list:
        try:
            answer = answer_generation_chain.run(question)
            answers.append({"question": question, "answer": answer})
        except Exception as e:
            answers.append({"question": question, "answer": f"Error generating answer: {e}"})
    return answers

# if __name__ == "__main__":
#     file_path = r"C:\Users\riswa\Desktop\AI\LLMS-development\InterviewQuestionGenerator\data\SDG.pdf" # Replace with your PDF file
#     vector_store_path = "vector_store"
#     answer_generation_chain, filtered_ques_list = llm_pipeline(file_path, vector_store_path)

#     if answer_generation_chain and filtered_ques_list:
#         answers = answer_questions(answer_generation_chain, filtered_ques_list)
#         for item in answers:
#            print(f"Question : {item['question']}")
#            print(f"Answer : {item['answer']}")
#     else:
#         print("Pipeline failed.")
# # Corrected file path
# file_processing(r"C:\Users\riswa\Desktop\AI\LLMS-development\InterviewQuestionGenerator\data\SDG.pdf")