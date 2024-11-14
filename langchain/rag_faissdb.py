# import libraries
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
# UI Interface
import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader

OPENAI_API_KEY = "" # add your OpenAI API Key
DIGI_DOC_PATH = "./DIGILOCKER_ASK_EXPERT.pdf"
UPI_DOC_PATHC = "./UPI_INFO.pdf"

# ----- Data Indexing Process ----
# Function to load and split documents based on user roles
@st.cache_resource
def load_documents():
    documents = []
    role_based_docs = {
        "Digi Locker(Admin)": ["./DIGILOCKER_ASK_EXPERT.pdf"],   # Admin-only documents
        "UPI (Admin)": ["./UPI_INFO.pdf"]      # User-only documents
    }
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    # Load documents based on role access control
    for role, doc_files in role_based_docs.items():
        for doc_file in doc_files:
            loader = PyPDFLoader(doc_file)
            raw_documents = loader.load()
            for doc in raw_documents:
                # Add metadata for access control
                doc.metadata["role"] = role
            documents.extend(splitter.split_documents(raw_documents))
    return documents
documents = load_documents()

# get OpenAI Embedding model
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
db_faiss = FAISS.from_documents(documents, embeddings)

# ----- Retrieval and Generation Process -----
def retrieve_generate_response(user_role, query):
    print(user_role)
    accessible_docs = [doc for doc in documents if doc.metadata["role"] == user_role]
    # retrieve context - top 5 most relevant (closests) chunks to the query vector 
    # (by default Langchain is using cosine distance metric)
    docs_faiss = db_faiss.similarity_search(query, k=5)
    print(docs_faiss)
    # generate an answer based on given user query and retrieved context information
    context_text = "\n\n".join([doc.page_content for doc in docs_faiss if doc.metadata["role"] == user_role])
    print(context_text)
    if context_text == "":
        return "You can only access restricted documents for your current roles"
    
    # you can use a prompt template
    PROMPT_TEMPLATE = """
    Answer the question based only on the following context:
    {context}
    Answer the question based on the above context: {question}.
    Provide a detailed answer.
    Don’t justify your answers.
    Don’t give information not mentioned in the CONTEXT INFORMATION.
    Do not say "according to the context" or "mentioned in the context" or similar.
    """

    # load retrieved context and user query in the prompt template
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query)

    # call LLM model to generate the answer based on the given context and query
    model = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY, temperature=0)
    response_text = model.invoke(prompt)
    return response_text.content


# Simple UI in Streamlit

with open('./user_creds.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

# Load authentication with predefined usernames, hashed passwords, and roles
authenticator = stauth.Authenticate(config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'])


st.title("RAG Chatbox with Access Control")
#user_role = st.selectbox("Select your role", ["user", "admin"])

# Login form
authenticator.login("main")
print( st.session_state)
if st.session_state['authentication_status']:
    # Display logout option
    authenticator.logout(location="main")
    # Display user role from the roles list
    user_role = st.session_state['roles']
    username = st.session_state['username']
    if username:
        st.write(f"Welcome {st.session_state['name']} ({user_role})!")
        # Initialize session state to store conversation history
        # Store LLM generated responses

        # Initialize session state for conversation history if not already present
        if "conversation_history" not in st.session_state:
            st.session_state.conversation_history = {}

         # Ensure each user has their own conversation history
        if username not in st.session_state.conversation_history:
            st.session_state.conversation_history[username] = [{"role": "assistant", "content": "How may I help you today?"}]

        # if "messages" not in st.session_state.keys():
        #     st.session_state.messages = [{"role": "assistant", "content": "How may I help you today?"}]

        # Display conversation history relevant to the logged-in user
        user_conversation = st.session_state.conversation_history[st.session_state['username']]
        for entry in user_conversation:
            if entry["role"] == user_role:
                st.write(f"**User ({user_role}):** {entry['content']}")
            else:
                st.markdown(f"**Assistant:** {entry['content']}")

        # Input box for user query
        user_input = st.chat_input("Type your message here")

        # When user submits a message, generate a response
        if user_input:
            # Add user message to conversation history
            user_conversation.append({"role": user_role, "content": user_input, "id": len(st.session_state.messages) })
            with st.chat_message("user"):
                st.write(user_input)

        if user_conversation[-1]["role"] != "assistant":
            # Generate response from the assistant
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = retrieve_generate_response(user_role, user_input)
                    st.write(response) 
            
            # Add assistant's response to conversation history
            user_conversation.append({"role": "assistant", "content": response,  "id": len(st.session_state.messages) })
            
            # # Clear the user input
            # st.text_input("Type your message here", value="", key="new_input")
    

elif st.session_state['authentication_status'] is False:
    st.error("Username or password is incorrect")

elif st.session_state['authentication_status'] is None:
    st.warning("Please enter your username and password")
    

# """
# Generated response:

# The top risks mentioned in the provided context are:
# 1. Decline in the value of investments
# 2. Lack of adoption of products and services
# 3. Interference or interruption from various factors such as modifications, terrorist attacks, natural disasters, etc.
# 4. Compromised trade secrets and legal and financial risks
# 5. Reputational, financial, and regulatory exposure
# 6. Abuse of platforms and misuse of user data
# 7. Errors or vulnerabilities leading to service interruptions or failure
# 8. Risks associated with international operations.
# """
