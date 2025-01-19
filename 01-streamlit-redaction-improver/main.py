import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

# Prompt Template
template = """
    Below is a draft text that may be poorly worded.
    Your goal is to:
    - Properly redact the draft text
    - Convert the draft text to a specified tone
    - Convert the draft text to a specified dialect

    Here are some examples different Tones:
    - Formal: Greetings! OpenAI has announced that Sam Altman is rejoining the company as its Chief Executive Officer. After a period of five days of conversations, discussions, and deliberations, the decision to bring back Altman, who had been previously dismissed, has been made. We are delighted to welcome Sam back to OpenAI.
    - Informal: Hey everyone, it's been a wild week! We've got some exciting news to share - Sam Altman is back at OpenAI, taking up the role of chief executive. After a bunch of intense talks, debates, and convincing, Altman is making his triumphant return to the AI startup he co-founded.  

    Here are some examples of words in different dialects:
    - American: French Fries, cotton candy, apartment, garbage, \
        cookie, green thumb, parking lot, pants, windshield
    - British: chips, candyfloss, flag, rubbish, biscuit, green fingers, \
        car park, trousers, windscreen

    Example Sentences from each dialect:
    - American: Greetings! OpenAI has announced that Sam Altman is rejoining the company as its Chief Executive Officer. After a period of five days of conversations, discussions, and deliberations, the decision to bring back Altman, who had been previously dismissed, has been made. We are delighted to welcome Sam back to OpenAI.
    - British: On Wednesday, OpenAI, the esteemed artificial intelligence start-up, announced that Sam Altman would be returning as its Chief Executive Officer. This decisive move follows five days of deliberation, discourse and persuasion, after Altman's abrupt departure from the company which he had co-established.

    Please start the redaction with a warm introduction. Add the introduction \
        if you need to.
    If the input is insufficient, generate a response indicating missing content with a tone and dialect specified.
    Below is the draft text, tone, and dialect:
    DRAFT: {draft}
    TONE: {tone}
    DIALECT: {dialect}

    YOUR {dialect} RESPONSE:
"""

# PromptTemplate definition
prompt = PromptTemplate(
    input_variables=["tone", "dialect", "draft"],
    template=template,
)

# Load LLM
def load_LLM(groq_api_key):
    """Load the LLM model with the Groq API key."""
    llm = ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile", api_key=groq_api_key)
    return llm

# Streamlit Page Setup
st.set_page_config(page_title="Re-write Your Text")
st.header("Re-write Your Text")

# Instructions
st.markdown("Re-write your text in different styles.")

# Input: API Key
st.markdown("## Enter Your Groq API Key")
def get_groq_api_key():
    input_text = st.text_input(label="Groq API Key", placeholder="Ex: sk-2twmA8tfCb8un4...", key="groq_api_key_input", type="password")
    return input_text

groq_api_key = get_groq_api_key()

# Input: Text Area
st.markdown("## Enter the Text You Want to Re-write")
def get_draft():
    draft_text = st.text_area(label="Text", label_visibility="collapsed", placeholder="Your Text...", key="draft_input")
    return draft_text

draft_input = get_draft()

# Limit text length
if len(draft_input.split(" ")) > 700:
    st.write("Please enter a shorter text. The maximum length is 700 words.")
    st.stop()

# Tone and Dialect Selection
col1, col2 = st.columns(2)
with col1:
    option_tone = st.selectbox(
        "Which tone would you like your redaction to have?",
        ("Formal", "Informal"),
    )
with col2:
    option_dialect = st.selectbox(
        "Which English Dialect would you like?",
        ("American", "British"),
    )

# Output: Generated Text
st.markdown("### Your Re-written Text:")

if draft_input:
    if not groq_api_key:
        st.warning("Please insert your Groq API Key.", icon="⚠️")
        st.stop()

    llm = load_LLM(groq_api_key=groq_api_key)

    # Fallback handling for insufficient or invalid input
    fallback_text = "The input provided was insufficient or unclear. Please try providing a complete draft for better results."
    effective_draft = draft_input.strip() if draft_input.strip() else fallback_text

    # Generate prompt
    prompt_with_draft = prompt.format(
        tone=option_tone,
        dialect=option_dialect,
        draft=effective_draft,
    )

    # Generate output
    try:
        improved_redaction = llm(prompt_with_draft)
        st.write(improved_redaction)
    except Exception as e:
        # If LLM fails, fallback to a user-friendly message
        fallback_message = f"Unable to generate a proper redaction based on the input. Please ensure the API key and input are correct."
        st.write(fallback_message)
else:
    st.write("Please provide some text for re-writing.")
