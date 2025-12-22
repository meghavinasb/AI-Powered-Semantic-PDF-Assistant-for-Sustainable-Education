import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import FakeEmbeddings
from langchain_community.vectorstores import DocArrayInMemorySearch

# ------------------ STREAMLIT CONFIG ------------------
st.set_page_config(page_title="PDF RAG", layout="wide")
st.title("Semantic PDF Question Answering using RAG")
st.write("Upload a PDF and ask questions based on its content.")

# ------------------ SIDEBAR ------------------
st.sidebar.header("Upload PDF")
uploaded_file = st.sidebar.file_uploader(
    "Upload a text-based PDF", type="pdf"
)

# ------------------ LOAD & PROCESS PDF ------------------
@st.cache_resource
def process_pdf(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(documents)

    # Lightweight, cloud-safe embeddings
    embeddings = FakeEmbeddings(size=384)

    vectorstore = DocArrayInMemorySearch.from_documents(
        chunks,
        embeddings
    )

    return vectorstore

# ------------------ MAIN LOGIC ------------------
if uploaded_file:
    with st.spinner("Processing PDF and building index..."):
        pdf_path = "temp.pdf"
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.read())

        vectorstore = process_pdf(pdf_path)

    st.success("PDF processed successfully!")

    # ------------------ UI: ASK QUESTION ------------------
    st.subheader("Ask Questions from the PDF")
    query = st.text_input("Enter your question:")

    if query:
        with st.spinner("Retrieving relevant content..."):
            docs = vectorstore.as_retriever(
                search_kwargs={"k": 3}
            ).get_relevant_documents(query)

        st.markdown("### Retrieved Answer")

        if not docs:
            st.write("No relevant information found.")
        else:
            for i, doc in enumerate(docs, start=1):
                st.markdown(f"**Result {i}:**")
                st.write(doc.page_content)
                st.caption(
                    f"Page: {doc.metadata.get('page', 'N/A')}"
                )

else:
    st.info("Upload a PDF from the sidebar to begin.")

