import os
import faiss
from sentence_transformers import SentenceTransformer
import fitz  # PyMuPDF
import re
import torch
import gc

# Ensure only one thread is used in PyTorch to avoid conflicts
torch.set_num_threads(1)

class KnowledgeBaseAgent:
    def __init__(self, knowledge_base_folder="aadhar_pdfs"):
        """Initialize by loading all PDF segments in the specified folder."""
        self.knowledge_base = self.load_text_segments_from_pdfs(knowledge_base_folder)
        # Load a better-fit model (you can change this if you use a different one)
        self.embedder = SentenceTransformer('paraphrase-mpnet-base-v2', device='cpu')  # Use CPU to avoid GPU-related issues
        self.segment_embeddings, self.index = self.create_faiss_index()

    def load_text_segments_from_pdfs(self, knowledge_base_folder):
        """Loads text segments from multiple PDFs in a folder."""
        segments = []
        for filename in os.listdir(knowledge_base_folder):
            if filename.endswith(".pdf"):
                pdf_path = os.path.join(knowledge_base_folder, filename)
                pdf_segments = self.extract_text_segments_from_pdf(pdf_path)
                segments.extend(pdf_segments)
                print(f"Loaded segments from {filename}: {len(pdf_segments)} segments.")
        return segments

    def extract_text_segments_from_pdf(self, pdf_path):
        """Extracts text segments from a PDF."""
        segments = []
        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            print(f"Error opening PDF: {e}")
            return []  # Return empty segments if the PDF couldn't be opened

        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            text = page.get_text("text")
            paragraphs = self.clean_and_split_text(text)
            for paragraph in paragraphs:
                if paragraph.strip():  # Avoid empty paragraphs
                    segments.append({"text": paragraph.strip()})
        return segments

    def clean_and_split_text(self, text):
        """Clean the extracted text and split it into segments."""
        text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        text = text.strip()  # Remove leading and trailing spaces
        paragraphs = text.split("\n\n")  # Split into paragraphs based on double newlines
        return paragraphs

    def create_faiss_index(self):
        """Creates a FAISS index using the extracted text segments."""
        segment_texts = [entry["text"] for entry in self.knowledge_base]
        print("Total segments being embedded:", len(segment_texts))
        
        # Embed text segments using the model
        segment_embeddings = self.embedder.encode(segment_texts, convert_to_tensor=True)
        
        # Convert from PyTorch tensor to NumPy array
        segment_embeddings = segment_embeddings.cpu().detach().numpy()
        
        # Initialize FAISS index
        embedding_dim = segment_embeddings.shape[1]
        print(f"Embedding dimension: {embedding_dim}")
        index = faiss.IndexFlatIP(embedding_dim)
        
        # Add embeddings to FAISS index
        faiss.normalize_L2(segment_embeddings)  # Normalize L2 and add embeddings
        index.add(segment_embeddings)
        
        return segment_embeddings, index

    def find_relevant_segment(self, query):
        """Finds the most relevant text segment based on the query."""
        query_embedding = self.embedder.encode([query], convert_to_tensor=True, show_progress_bar=False)

        # Use batched embeddings for all segments
        segment_embeddings = self.embedder.encode([entry["text"] for entry in self.knowledge_base], convert_to_tensor=True, show_progress_bar=False)

        print(f"Query Embedding: {query_embedding}")
        print(f"Text Segment Embeddings: {segment_embeddings}")

        # Convert query embedding to NumPy array for FAISS
        query_embedding = query_embedding.cpu().detach().numpy()
        segment_embeddings = segment_embeddings.cpu().detach().numpy()

        faiss.normalize_L2(query_embedding)  # Normalize query embedding
        faiss.normalize_L2(segment_embeddings)  # Normalize text segment embeddings

        # Perform the search
        distances, indices = self.index.search(query_embedding, k=5)
        if distances[0][0] < 0.4:  # Adjust threshold here as needed
            print(f"No relevant match found for query: {query}")
            return "Sorry, no relevant information found.", 0

        most_similar_idx = indices[0][0]
        similarity_score = distances[0][0]
        best_entry = self.knowledge_base[most_similar_idx]
        
        print(f"Query: {query}")
        print(f"Matched Segment: '{best_entry['text']}'")
        print(f"Similarity Score: {similarity_score}")

        # Run garbage collection to avoid memory leaks
        gc.collect()

        return best_entry['text'], similarity_score
