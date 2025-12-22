import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import FakeEmbeddings
from langchain_community.vectorstores import DocArrayInMemorySearch

st.set_page_config(page_title="Semantic PDF QA (RAG)", layout="wide")
st.title("Semantic PDF Question Answering using RAG")
st.write("Upload a PDF and ask questions based on its content.")

uploaded_file = st.sidebar.file_uploader("Upload a text-based PDF", type="pdf")

@st.cache_resource
def process_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)

    embeddings = FakeEmbeddings(size=384)
    return DocArrayInMemorySearch.from_documents(chunks, embeddings)

if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    vectorstore = process_pdf("temp.pdf")
    st.success("PDF processed successfully!")

    query = st.text_input("Enter your question:")

    if query:
        docs = vectorstore.as_retriever(search_kwargs={"k": 3}).get_relevant_documents(query)
        st.markdown("### Retrieved Answer")

        for i, doc in enumerate(docs, 1):
            st.markdown(f"**Result {i}:**")
            st.write(doc.page_content)
            st.caption(f"Page: {doc.metadata.get('page', 'N/A')}")
else:
    st.info("Upload a PDF to begin.")
