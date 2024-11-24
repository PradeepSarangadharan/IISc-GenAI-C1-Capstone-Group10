# import libraries
import os
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain.sql_database import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain.agents import AgentExecutor, Tool

############# Agent 1: RAG approach ################
# ----- Data Indexing Process -----

# Step 1: Extract text from PDF
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

# Step 2: Split text into chunks
def split_text_into_chunks(text, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return text_splitter.split_text(text)

# Step 3: Filter out small or non-alphanumeric chunks
def filter_chunks(chunks, min_length=10):
    return [chunk for chunk in chunks if len(chunk.strip()) >= min_length]

# Function to load and split documents based on user roles
def process_documents(governancedocs):
    documents = []
    role_based_docs = {
        "governance": governancedocs,   # Governance-only documents\
    }
    # Load documents based on role access control
    for role, doc_files in role_based_docs.items():
        for doc_file in doc_files:
            text = extract_text_from_pdf(doc_file)
            chunks = split_text_into_chunks(text)
            # Filter steps
            chunks = filter_chunks(chunks)
            documents.extend(Document(page_content=chunk, metadata={"role":role}) for chunk in chunks)
    return documents

def create_db_vectorstore(documents, embeddings):
    index_directory = "faissdb/digilocker"
    # Step 1: Check if the vector store exists
    if os.path.exists(index_directory):
        print(f"Vector store found at {index_directory}. Loading from disk...")
        # get OpenAI Embedding model
        loaded_vector_store = FAISS.load_local(index_directory, embeddings, allow_dangerous_deserialization=True)
        print("Vector store loaded from 'faiss_index_directory'.")
        #loaded_vector_store = FAISS.from_documents(documents, embeddings)
        return loaded_vector_store
    else:
        os.makedirs(index_directory, exist_ok=True)
        db_faiss = FAISS.from_documents(documents, embeddings)
        try:
            db_faiss.save_local(index_directory)
        except Exception as e:
            print(f"Error loading vector store: {e}")
        print("Vector store saved to 'faiss_index_directory'.")
        return db_faiss

# ----- Retrieval and Generation Process -----
def retrieve_generate_response_digilocker_docs(llm_model, query, db_faiss):
    retriever = db_faiss.as_retriever(search_kwargs={"k": 3})
    print(retriever)
    # Set up the RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(llm=llm_model, retriever=retriever,return_source_documents=True )
    response = qa_chain.invoke(query)
    source_docs = response.get("source_documents", [])
    return response['result']
  
