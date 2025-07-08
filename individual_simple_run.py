from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import ChatHuggingFace
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv
import os
load_dotenv(override=True)

HF_TOKEN = os.environ["HF_TOKEN"]

loader = PyPDFLoader("Ambient_Listening_System__AI_Powered_Medical_Transcription_and_Data_Extraction.pdf")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap = 50)
docs = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(docs, embeddings)

endpoint = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    huggingfacehub_api_token=HF_TOKEN,
    task="conversational",
    temperature=0.5,
    max_new_tokens=512,
)
llm = ChatHuggingFace(
    llm=endpoint
)

qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

query = "Give me the example discussed of medication codification"
result = qa_chain.invoke(query)
print(result)