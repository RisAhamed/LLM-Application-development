import streamlit as st
import os
import csv
import json
from src.helper import llm_pipeline, answer_questions
import base64

def get_binary_file_downloader_html(bin_file, file_label='File'):
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">{file_label}</a>'
    return href

def main():
    st.set_page_config(layout="wide") # <--- MOVE THIS LINE TO THE TOP
    st.title("PDF Question Answering")
    
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    vector_store_path = "vector_store" # path to save the vector store
    
    if uploaded_file is not None:
        # Save the uploaded file
        base_folder = "static/docs/"
        if not os.path.isdir(base_folder):
            os.makedirs(base_folder)
        pdf_filename = os.path.join(base_folder, uploaded_file.name)

        with open(pdf_filename, "wb") as f:
            f.write(uploaded_file.read())

        st.success(f"File '{uploaded_file.name}' uploaded and saved successfully!")

        if st.button("Analyze"):

            with st.spinner("Analyzing the PDF and generating answers..."):
                answer_generation_chain, filtered_ques_list = llm_pipeline(pdf_filename,vector_store_path)
                if answer_generation_chain and filtered_ques_list:
                    answers = answer_questions(answer_generation_chain, filtered_ques_list)

                    # Create and download CSV file
                    base_folder = "static/output/"
                    if not os.path.isdir(base_folder):
                        os.makedirs(base_folder)
                    output_file = base_folder + "QA.csv"
                    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
                        csv_writer = csv.writer(csvfile)
                        csv_writer.writerow(["Question", "Answer"])  # Writing the header row
                        for item in answers:
                            csv_writer.writerow([item["question"], item["answer"]])
                    st.markdown(get_binary_file_downloader_html(output_file, 'Download CSV File'), unsafe_allow_html=True)
                    
                    
                    st.success("Analysis complete! CSV file available for download.")

                    # Display questions and answers (Optional)
                    with st.expander("Show Question and Answers"):
                      for item in answers:
                          st.write(f"**Question:** {item['question']}")
                          st.write(f"**Answer:** {item['answer']}")
                          st.write("---")
                
                else:
                  st.error("Failed to generate questions and answers, please try again.")

if __name__ == "__main__":
    main()