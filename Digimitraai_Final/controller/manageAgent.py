# import libraries
import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.sql_database import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain.agents import Tool
#### local imports #####
from controller.aadhaarAgent import create_db_vectorstore as adhaar_db_vectorstore
from controller.aadhaarAgent import process_documents as aadhaar_process_documents
from controller.aadhaarAgent import retrieve_generate_response_aadhaar_docs as aadhaar_query
from controller.digiLockerAgent import create_db_vectorstore as digilocker_db_vectorstore
from controller.digiLockerAgent import process_documents as digilocker_process_documents
from controller.digiLockerAgent import retrieve_generate_response_digilocker_docs as digilocker_query

import pandas as pd
import sqlite3

OPENAI_API_KEY = "" # add your OpenAI API Key# for this example I used Alphabet Inc 10-K Report 2022 

# get OpenAI Embedding model
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Initialize a LLM
llm_model = ChatOpenAI(model="gpt-4o-mini", openai_api_key=OPENAI_API_KEY, temperature=0)

# loading aadhaar docs for Aadhaar Agent
aadhaar_documents = aadhaar_process_documents(governancedocs=["../docs/aadhaar_qa.pdf"])
aadhaar_db_faiss = adhaar_db_vectorstore(aadhaar_documents, embeddings)

# loading digilocker docs for Digilocker Agent
digilocker_documents = digilocker_process_documents(governancedocs=["../docs/DIGILOCKER_ASK_EXPERT.pdf" ])
digilocker_db_faiss = digilocker_db_vectorstore(digilocker_documents,embeddings)

############# Agent : text-to-SQL approach ################

# Load the CSV into a Pandas DataFrame
grocery_csv_file = "../docs/blinkit_retail.csv"
df = pd.read_csv(grocery_csv_file)

#define db file path
db_file = "./sqlitedb/grocery_data.db"

# Connect to SQLite (creates a database file if it doesn't exist)
conn = sqlite3.connect(db_file)
# Save DataFrame to a new SQLite table
table_name = "grocery_data"
df.to_sql(table_name, conn, if_exists="replace", index=False)
conn.close()

# Load the SQLite table into a SQLDatabase
sqlite_db = SQLDatabase.from_uri(f"sqlite:///{db_file}")
# Create the SQLDatabaseChain
db_chain = SQLDatabaseChain(llm=llm_model, database=sqlite_db, verbose=True)

def retrieve_generate_response_sql_db(query, db_faiss):
    try:
        response = db_chain.invoke(query)
        print(response)
        return response['result']
    except:
        return "Final answer here: No results found."
########################################################################
    
# Define intent-based tools
tools = [
    Tool(name="aadhaar docs query", func=aadhaar_query, description="For questions about the aadhaar base."),
    Tool(name="digilocker docs query", func=digilocker_query, description="For questions about the digilocker base."),
    Tool(name="E-commerce query", func=retrieve_generate_response_sql_db, description="For querying E-commerce data.")
]

### Embedding-Based Intent Recognition ###
# Define intents and their embeddings
intents = {
    "aadhaar_base": "aadhaar realted questions",
    "digilocker_base": "digilocker related questions",
    "sql_query": "Queries related to tabular data, numbers, avaibility,, price, items, cost, groceries, cheap, varieties",
}

# Embed intent
intent_embeddings = {name: embeddings.embed_query(description) for name, description in intents.items()}

# Function to determine intent based on embeddings
def determine_intent(query, intent_embeddings, embedding_model):
    query_embedding = embedding_model.embed_query(query)
    similarities = {
        intent: sum(qe * ie for qe, ie in zip(query_embedding, emb))
        for intent, emb in intent_embeddings.items()
    }
    return max(similarities, key=similarities.get)

### Router Chain with Embeddings ###
class EmbeddingRouterChain:
    def __init__(self, tools, intent_embeddings, embedding_model):
        self.tools = {tool.name: tool for tool in tools}
        self.intent_embeddings = intent_embeddings
        self.embedding_model = embedding_model

    def route(self, query, aadhaar_db_faiss, digilocker_db_faiss):
        # Determine intent
        intent = determine_intent(query, self.intent_embeddings, self.embedding_model)
        db_faiss = None
        if intent == "aadhaar_base":
            tool_name = "aadhaar docs query"
            db_faiss = aadhaar_db_faiss
        elif intent == "digilocker_base":
            tool_name = "digilocker docs query"
            db_faiss = digilocker_db_faiss
        elif intent == "sql_query":
            tool_name = "E-commerce query"
        else:
            return "Sorry! this question is out my knowledge"
        print(tool_name)
        tool = self.tools[tool_name]
        return tool.func(llm_model,query, db_faiss)

# Initialize the router chain
router_chain = EmbeddingRouterChain(tools, intent_embeddings, embeddings)

def agent_retrive_generate_response(role, query):
    response = router_chain.route( query, aadhaar_db_faiss, digilocker_db_faiss)
    return response
