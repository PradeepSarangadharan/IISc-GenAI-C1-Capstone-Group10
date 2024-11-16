import json
import faiss
from sentence_transformers import SentenceTransformer
from agentlite.llm.agent_llms import get_llm_backend
from agentlite.llm.LLMConfig import LLMConfig

class SimpleSearchAgent:
    def __init__(self, llm_config_dict, knowledge_base_path="aadhar_faq.json"):
        # Initialize LLM and load knowledge base
        self.llm_config = LLMConfig(llm_config_dict)
        self.llm = get_llm_backend(self.llm_config)
        self.knowledge_base = self.load_knowledge_base(knowledge_base_path)
        
        # Load the pre-trained SentenceTransformer model
        self.embedder = SentenceTransformer('paraphrase-MiniLM-L6-v2') 
        self.faq_embeddings, self.index = self.create_faiss_index()
        
    def load_knowledge_base(self, knowledge_base_path):
        """Loads the knowledge base from the JSON file."""
        with open(knowledge_base_path, "r") as file:
            return json.load(file)

    def create_faiss_index(self):
        """Creates a FAISS index with question-answer pairs."""
        faq_texts = [f"Q: {entry['question']} A: {entry['answer']}" for entry in self.knowledge_base]
        
        # Generate embeddings for question-answer pairs
        faq_embeddings = self.embedder.encode(faq_texts)
        
        # Initialize FAISS index
        embedding_dim = faq_embeddings.shape[1]
        index = faiss.IndexFlatIP(embedding_dim)  # Inner Product (Cosine similarity if normalized)
        
        # Add embeddings to the FAISS index
        faiss.normalize_L2(faq_embeddings)  # Normalize for cosine similarity
        index.add(faq_embeddings)
        
        return faq_embeddings, index

    def search(self, query, source="default"):
        """Searches the knowledge base for the most relevant answer."""
        relevant_answer, similarity_score, matched_question = self.find_relevant_answer(query)

        if similarity_score >= 0.7:
            # High confidence match, return the answer directly
            return relevant_answer
        elif similarity_score >= 0.5:
            # Medium confidence match, return an LLM-clarified response if LLM is allowed
            return relevant_answer
        else:
            # No relevant match found
            return "I'm here to help with Aadhar-related questions. For further details, please refer to the official UIDAI website or nearby enrollment center."

    def find_relevant_answer(self, query):
        """Finds the most relevant answer by embedding both questions and answers."""
        # Generate embedding for the query
        query_embedding = self.embedder.encode([query])
        faiss.normalize_L2(query_embedding)  # Normalize for cosine similarity
        
        # Search the index for the closest match
        distances, indices = self.index.search(query_embedding, k=1)
        
        # Retrieve the question and answer from the closest match
        most_similar_idx = indices[0][0]
        similarity_score = distances[0][0]
        best_entry = self.knowledge_base[most_similar_idx]
        matched_question = best_entry['question']
        best_answer = best_entry['answer']
        
        return best_answer, similarity_score, matched_question
