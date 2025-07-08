import os
import tempfile
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint
from langchain_huggingface.chat_models import ChatHuggingFace
from langchain.chains import RetrievalQA

# Cache filename for the FAISS vectorstore
VECTORSTORE_CACHE = "faiss_index"

@st.cache_resource(show_spinner=False)
def get_vectorstore(uploaded_file) -> FAISS:
    """
    Either load a persisted FAISS index or build it from the uploaded PDF.
    """
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # If we've previously saved the vectorstore, load it
    if os.path.exists(VECTORSTORE_CACHE):
        return FAISS.load_local(VECTORSTORE_CACHE, embeddings)

    # Otherwise, write the uploaded file to a temp path and build
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    loader = PyPDFLoader(tmp_path)
    docs = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(loader.load())

    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(VECTORSTORE_CACHE)
    return vectorstore

@st.cache_resource(show_spinner=False)
def get_qa_chain(uploaded_file, hf_token: str) -> RetrievalQA:
    """
    Initialize the RetrievalQA chain with the uploaded PDF and Hugging Face token.
    """
    endpoint = HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.3",
        huggingfacehub_api_token=hf_token,
        task="conversational",
        temperature=0.5,
        max_new_tokens=512,
    )
    llm = ChatHuggingFace(llm=endpoint)

    vectorstore = get_vectorstore(uploaded_file)
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        return_source_documents=False,
    )

# Streamlit App UI
st.set_page_config(page_title="Ambient Listening RAG Chatbot")
st.title("Ambient Listening System — RAG-Powered Chatbot")

st.sidebar.header("Settings")
pdf_file = st.sidebar.file_uploader("Upload PDF", type=["pdf"])
hf_token = st.sidebar.text_input("Hugging Face API Token", type="password")

if pdf_file and hf_token:
    qa_chain = get_qa_chain(pdf_file, hf_token)
    query = st.text_input("Enter your question:")
    if query:
        with st.spinner("Generating answer..."):
            answer = qa_chain.invoke(query)
        st.markdown("**Answer:**")
        st.write(answer)
else:
    st.info("Please upload a PDF and enter your Hugging Face API token in the sidebar to begin.")
