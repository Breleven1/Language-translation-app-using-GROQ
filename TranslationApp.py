import streamlit as st
import groq
from langchain_community.llms import Ollama
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

import os
from dotenv import load_dotenv

load_dotenv()

# LangSmith Tracking
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Language Translation App"

# Prompt Template for Translation
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant that translates text into the specified language."),
        ("user", "Translate the following text into {language}: {text}")
    ]
)

def generate_translation(groq_api_key, text, language, engine, temperature):
    groq.api_key = groq_api_key
    llm = ChatGroq(model=engine, temperature=temperature)
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    response = chain.invoke({"text": text, "language": language})
    return response

# Title
st.title("🌍 Language Translation App using GROQ")

groq_api_key = st.sidebar.text_input("Enter GROQ API Key:", type = "password")

# Sidebar - Model Selection
llm = st.sidebar.selectbox("Select a GROQ model:", ["llama-3.1-8b-instant"])

# Sidebar - Parameters
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7)

# Language Selection
language = st.selectbox(
    "Select target language:",
    ["Hindi", "French", "Spanish", "German", "Chinese", "Japanese", "Arabic", "Bengali"]
)

# User Input
st.write("Enter text to translate:")
user_input = st.text_area("Your text:")

# Translate Button
if st.button("Translate"):
    if user_input and groq_api_key:
        translated_text = generate_translation(groq_api_key, user_input, language, llm, temperature)
        st.subheader("Translated Text:")
        st.write(translated_text)
    elif user_input:
        st.write("Please enter the Groq api key")
    else:
        st.warning("Please enter some text to translate.")