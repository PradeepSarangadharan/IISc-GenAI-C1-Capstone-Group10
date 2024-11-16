import json
import torch
from sentence_transformers import SentenceTransformer, util
from agentlite.llm.agent_llms import get_llm_backend
from agentlite.llm.LLMConfig import LLMConfig

class SimpleSearchAgent:
    def __init__(self, llm_config_dict, knowledge_base_path="aadhar_faq.json"):
        # Create an instance of LLMConfig using the provided dictionary
        self.llm_config = LLMConfig(llm_config_dict)  # Create LLMConfig instance here
        self.llm = get_llm_backend(self.llm_config)  # Pass the instance of LLMConfig
        self.knowledge_base = self.load_knowledge_base(knowledge_base_path)
        
        # Load the pre-trained SentenceTransformer model
        self.embedder = SentenceTransformer('paraphrase-MiniLM-L6-v2')  # You can choose another model if needed
        self.faq_embeddings = self.compute_faq_embeddings()

    def load_knowledge_base(self, knowledge_base_path):
        """Loads the knowledge base from the JSON file."""
        with open(knowledge_base_path, "r") as file:
            return json.load(file)

    def compute_faq_embeddings(self):
        """Converts all FAQ answers into embeddings."""
        faq_answers = [entry['answer'] for entry in self.knowledge_base]
        return self.embedder.encode(faq_answers, convert_to_tensor=True)

    def get_llm_response(self, prompt):
        """Uses the LLM to generate a response."""
        return self.llm(prompt)

    def search(self, query, source="default"):
        """Searches the knowledge base for the most relevant answer."""
        relevant_answer = self.find_relevant_answer(query)

        if relevant_answer:
            # If relevant answer is found, use LLM for response
            prompt = f"Answer this query based on the Aadhar knowledge base: {query}. Relevant answer: {relevant_answer}"
            return self.get_llm_response(prompt)
        else:
            # If no relevant answer is found, return out of context response
            return "Sorry, this question is outside the scope of my knowledge."

    def find_relevant_answer(self, query):
        """Finds the most relevant answer from the knowledge base using semantic search."""
        query_embedding = self.embedder.encode(query, convert_to_tensor=True)
        
        # Compute cosine similarity between the query and all FAQ answers
        similarities = util.pytorch_cos_sim(query_embedding, self.faq_embeddings)[0]
        
        # Get the index of the most similar answer
        most_similar_idx = torch.argmax(similarities).item()
        best_answer = self.knowledge_base[most_similar_idx]['answer']

        # Set a higher threshold for similarity. If similarity is too low, consider it out of context.
        similarity_threshold = 0.7  # Adjusted to a higher value to ensure more accurate matches
        if similarities[most_similar_idx] > similarity_threshold:
            return best_answer
        else:
            return None
