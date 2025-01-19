import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

# Streamlit Page Configuration
st.set_page_config(
    page_title="Blog Post Generator"
)

st.title("Blog Post Generator")

# Sidebar for API Key Input
openai_api_key = st.sidebar.text_input(
    "OpenAI API Key",
    type="password"
)

# Function to Generate Response
def generate_response(topic):
    if not openai_api_key:
        st.warning("Please enter your Groq API Key in the sidebar.")
        return

    try:
        # Initialize ChatGroq with the provided API key
        llm = ChatGroq(
            temperature=0,
            groq_api_key=openai_api_key,  # Pass API key explicitly
            model_name="mixtral-8x7b-32768"
        )

        # Define the prompt template
        template = """
        As an experienced startup and venture capital writer, 
        generate a 400-word blog post about {topic}.
        
        Your response should be in this format:
        First, print the blog post.
        Then, sum the total number of words in it and print the result like this: This post has X words.
        """
        prompt = PromptTemplate(
            input_variables=["topic"],
            template=template
        )

        # Format the query
        query = prompt.format(topic=topic)

        # Generate the response
        response = llm(query, max_tokens=2048)
        st.write(response)

    except Exception as e:
        st.error(f"An error occurred: {e}")

# Input for Blog Topic
topic_text = st.text_input("Enter topic:")

# Generate Response Button
if st.button("Generate Blog Post"):
    if topic_text.strip():
        generate_response(topic_text)
    else:
        st.warning("Please enter a topic to generate the blog post.")
