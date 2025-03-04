import streamlit as st
import fitz  # PyMuPDF for PDF handling
import faiss
import numpy as np
import ollama
from gtts import gTTS
import os

# Streamlit UI
st.title("📄 AI-Powered PDF Summarizer with TTS 🎤")
st.write("Upload a PDF file to generate a summary and listen to it!")

# File uploader
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = "\n".join(page.get_text("text") for page in doc)
    return text

# Function to split text into chunks
def split_text(text, chunk_size=500):
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

# Function to store embeddings using FAISS
def store_embeddings(text_chunks):
    dimension = 768  # Adjust based on the model (e.g., 768 for MiniLM)
    index = faiss.IndexFlatL2(dimension)
    stored_chunks = text_chunks
    return index, stored_chunks

# Function to retrieve top text chunks
def retrieve_top_chunks(query, stored_chunks, top_k=3):
    return " ".join(stored_chunks[:top_k])  # Simple retrieval (FAISS can be added)

# Function to summarize text using Ollama
def summarize_text(text):
    response = ollama.chat(
        model="mistral",  # Change to "llama2" or another local model
        messages=[{"role": "user", "content": f"Summarize this text: {text}"}]
    )
    return response['message']['content']

# Function to convert text to speech
def text_to_speech(text):
    tts = gTTS(text=text, lang="en")
    filename = "summary.mp3"
    tts.save(filename)
    return filename

# Process PDF file
if uploaded_file:
    with st.spinner("Processing PDF..."):
        pdf_text = extract_text_from_pdf(uploaded_file)
        text_chunks = split_text(pdf_text, 500)

        # Store embeddings (for future improvements)
        index, stored_chunks = store_embeddings(text_chunks)

        # Retrieve most relevant chunks
        retrieved_text = retrieve_top_chunks("Summarize this document", stored_chunks)

        # Summarize using Ollama
        summary = summarize_text(retrieved_text)

        # Convert summary to speech
        speech_file = text_to_speech(summary)

    # Display results
    st.subheader("📑 Summary:")
    st.write(summary)

    # Audio player for text-to-speech
    st.audio(speech_file)
