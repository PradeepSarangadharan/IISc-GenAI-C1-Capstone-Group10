import json
import faiss
from sentence_transformers import SentenceTransformer

class KnowledgeBaseAgent:
    def __init__(self, knowledge_base_path="aadhar_faq.json"):
        self.knowledge_base = self.load_knowledge_base(knowledge_base_path)
        self.embedder = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        self.faq_embeddings, self.index = self.create_faiss_index()
        
    def load_knowledge_base(self, knowledge_base_path):
        """Loads the knowledge base from the JSON file."""
        with open(knowledge_base_path, "r") as file:
            return json.load(file)

    def create_faiss_index(self):
        """Creates a FAISS index with question-answer pairs."""
        faq_texts = [f"Q: {entry['question']} A: {entry['answer']}" for entry in self.knowledge_base]
        faq_embeddings = self.embedder.encode(faq_texts)
        
        # Initialize FAISS index
        embedding_dim = faq_embeddings.shape[1]
        index = faiss.IndexFlatIP(embedding_dim)
        
        # Add embeddings to FAISS index
        faiss.normalize_L2(faq_embeddings)
        index.add(faq_embeddings)
        
        return faq_embeddings, index

    def find_relevant_answer(self, query):
        """Finds the most relevant answer by embedding the query and searching the FAISS index."""
        query_embedding = self.embedder.encode([query])
        faiss.normalize_L2(query_embedding)
        
        distances, indices = self.index.search(query_embedding, k=1)
        most_similar_idx = indices[0][0]
        similarity_score = distances[0][0]
        best_entry = self.knowledge_base[most_similar_idx]
        
        return best_entry['answer'], similarity_score, best_entry['question']
