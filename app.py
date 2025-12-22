import os
import streamlit as st
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import FakeEmbeddings
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

# ------------------ STREAMLIT CONFIG ------------------
st.set_page_config(page_title="PDF RAG", layout="wide")
st.title("Semantic PDF Question Answering using RAG and Transformer-Based Language Models")
st.write("App loaded. Please upload a PDF from the sidebar.")

# ------------------ SIDEBAR ------------------
st.sidebar.header("Upload PDF")
uploaded_file = st.sidebar.file_uploader("Upload a PDF (100+ pages)", type="pdf")

# ------------------ LOAD & PROCESS PDF ------------------
@st.cache_resource
def process_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(documents)

    embeddings = FakeEmbeddings(size=384)

    vectorstore = DocArrayInMemorySearch.from_documents(
        chunks,
        embeddings
    )  

    return vectorstore

# ------------------ MAIN LOGIC ------------------
if uploaded_file:
    with st.spinner("Processing PDF and building vector database..."):
        pdf_path = "temp.pdf"
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.read())

        vectorstore = process_pdf(pdf_path)

    st.success("PDF processed successfully!")

    # ------------------ HUGGINGFACE LLM ------------------
    hf_pipeline = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        max_new_tokens=256,
        temperature=0.3
    )


    llm = HuggingFacePipeline(pipeline=hf_pipeline)

    # ------------------ MODERN RAG CHAIN ------------------
    prompt = PromptTemplate.from_template(
        """
        You are a helpful assistant.

        Use ONLY the information from the context below to answer the question.
        Give a clear, short, direct answer in 2–4 sentences.
        DO NOT repeat the context.
        DO NOT mention the word "context".

        Context:
        {context}

        Question:
        {question}

        Answer:
        """
   )


    rag_chain = (
        {
            "context": vectorstore.as_retriever(search_kwargs={"k": 5}),
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
    )

    # ------------------ UI: ASK QUESTION ------------------
    st.subheader("Ask Questions from the PDF")
    query = st.text_input("Enter your question:")

    if query:
        with st.spinner("Generating answer..."):
            response = rag_chain.invoke(query)

        st.markdown("### Answer")
        st.write(response.strip())


else:
    st.info("Upload a PDF from the sidebar to begin.")




