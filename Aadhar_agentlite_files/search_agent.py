import os
import re

class SearchAgent:
    def __init__(self, knowledge_base_agent, llm_agent, confidence_threshold=0.4):
        self.knowledge_base_agent = knowledge_base_agent
        self.llm_agent = llm_agent
        self.confidence_threshold = confidence_threshold

    def is_aadhar_related(self, query):
        """Check if the query is related to Aadhar, including common misspellings and variations."""
        aadhar_keywords = [
            "Aadhar", "aadhar", "AADHAR", "adhaar", "adhar", "aadharr", "aadhaar", "aadhhar", 
            "aadharcard", "aadhaarcard", "aadhar id", "aadhar id", "aadhar number", "aadhaar number",
            "aadhaar uid", "uidai", "uid", "aadhar card", "adhar card", "uidai card", "uidai number", 
            "unique id", "unique identification", "adharcard", "aadhaar card", "aadhaarid", "uid", "adhar id"
        ]
        
        # Check if any keyword is found in the query (case insensitive)
        if any(keyword.lower() in query.lower() for keyword in aadhar_keywords):
            return True
        return False

    def reason(self, query):
        """Determine if the query should be answered from the knowledge base or by the LLM."""
        if not self.is_aadhar_related(query):
            return "not_aadhar_related", None

        relevant_segment, similarity_score = self.knowledge_base_agent.find_relevant_segment(query)
        print(f"Matched Segment: '{relevant_segment}', Similarity Score: {similarity_score}")

        if similarity_score >= self.confidence_threshold:
            return "found_answer", relevant_segment
        elif similarity_score >= 0.4:  # Adjusted lower bound to catch slightly lower confidence matches
            return "ask_llm", relevant_segment
        else:
            return "low_confidence", None

    def refine_answer_using_llm(self, query, segment):
        """Send the question and matched segment to LLM for refining the answer."""
        try:
            # Send both the query and the relevant segment to the LLM for refining the answer
            prompt = f"Question: {query}\nSegment: {segment}\nRefine the answer by removing any extraneous details, such as other answers or unrelated information, and focus only on the most accurate and relevant response."
            refined_answer = self.llm_agent.get_response(prompt)
            return refined_answer.strip()
        except Exception as e:
            print(f"Error in LLM refinement: {e}")
            return "Sorry, I couldn't refine the answer at this time."

    def perform_action(self, reasoning_result, query):
        """Respond to the query based on reasoning and, if low-confidence, await user feedback before updating the knowledge base."""
        action_type, relevant_answer = reasoning_result

        # Debugging: print action type and reasoning source
        print(f"Action Type: {action_type}")

        if action_type == "found_answer":
            # High confidence match in knowledge base: Send this to the LLM for refinement
            print("Source: PDF Knowledge Base (Refined by LLM)")
            refined_answer = self.refine_answer_using_llm(query, relevant_answer)
            return refined_answer, False, "LLM (Refined Answer from Knowledge Base)"  # Refined by LLM

        elif action_type == "ask_llm":
            # Medium confidence: Ask the LLM with context
            prompt = f"Question: {query}\nBased on known Aadhaar information: {relevant_answer}"
            try:
                llm_answer = self.llm_agent.get_response(prompt)
                print("Source: LLM (Medium Confidence with Context)")
                return llm_answer, False, "LLM (Medium Confidence with Context)"  # Indicate LLM source
            except Exception as e:
                print(f"Error fetching LLM response: {e}")
                return "Sorry, there was an issue with generating an answer.", False, "LLM (Error)"

        elif action_type == "low_confidence":
            # Low confidence: Generate answer from LLM with stronger context
            if not self.is_aadhar_related(query):
                # If the query is not Aadhaar-related, return a message and do not query LLM
                print("Source: N/A (Out of Scope)")
                return "Sorry, I can only answer questions related to Aadhaar.", False, "N/A (Out of Scope)"
            else:
                # If the question is Aadhaar-related, pass it to the LLM with stronger context
                prompt = f"Answer the following question based on Aadhaar-related information only: {query}"
                try:
                    llm_answer = self.llm_agent.get_response(prompt)
                    print("Source: LLM (Low Confidence)")
                    return llm_answer, True, "LLM (Low Confidence)"  # Indicate LLM source
                except Exception as e:
                    print(f"Error fetching LLM response: {e}")
                    return "Sorry, there was an issue with generating an answer.", True, "LLM (Error)"

        elif action_type == "not_aadhar_related":
            # If the question is not Aadhaar-related, return a message stating that only Aadhaar-related questions can be answered
            print("Source: N/A (Out of Scope)")
            return "Sorry, I can only answer questions related to Aadhaar.", False, "N/A (Out of Scope)"

    def search(self, query):
        """Execute the reasoning-action cycle and await feedback if needed."""
        reasoning_result = self.reason(query)
        response, await_feedback, source = self.perform_action(reasoning_result, query)

        # If the response is medium confidence, we refine the answer using LLM
        if await_feedback and source == "LLM (Medium Confidence with Context)":
            refined_answer = self.refine_answer_using_llm(query, response)
            return refined_answer, await_feedback, source  # Return refined response, await_feedback, and source

        return response, await_feedback, source  # Return response, await_feedback, and source

    def update_knowledge_base(self, query, response):
        new_entry = {"text": response}
        self.knowledge_base_agent.knowledge_base.append(new_entry)
        self.knowledge_base_agent.segment_embeddings, self.knowledge_base_agent.index = self.knowledge_base_agent.create_faiss_index()
