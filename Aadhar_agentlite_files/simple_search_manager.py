import gradio as gr
import json
import os
from agentlite.llm.LLMConfig import LLMConfig
from knowledge_base_agent import KnowledgeBaseAgent
from llm_agent import LLMAgent
from search_agent import SearchAgent

# Initialize LLMConfig and Agents
llm_config_dict = {
    "llm_name": "gpt-3.5-turbo",
    "temperature": 0,
    "max_tokens": 100
}

# Initialize agents
knowledge_base_agent = KnowledgeBaseAgent("aadhar_faq.json")
llm_agent = LLMAgent(llm_config_dict)
search_agent = SearchAgent(knowledge_base_agent, llm_agent)

# Function to handle search and feedback
def search_and_feedback(query, feedback):
    response, await_feedback = search_agent.search(query)  # search_agent is your SearchAgent instance

    if await_feedback and feedback is not None:
        if feedback == "Yes" or feedback == "👍":
            # Update the knowledge base with the LLM-generated answer
            search_agent.update_knowledge_base(query, response)
            message = "Thank you! The answer has been added to the knowledge base."
        else:
            message = "Thank you for your feedback. The answer was not added to the knowledge base."
        return response, message, gr.update(visible=True)  # Show feedback buttons
    else:
        # Return just the response if no feedback is required
        return response, "", gr.update(visible=False)  # Hide feedback buttons

# Gradio interface
def build_interface():
    with gr.Blocks() as demo:
        gr.Markdown("# Aadhar FAQ Search")

        with gr.Row():
            query_input = gr.Textbox(label="Ask a question related to Aadhar:", placeholder="Enter your question here...")

        response_output = gr.Textbox(label="Response", interactive=False)
        feedback_output = gr.Textbox(label="Feedback", interactive=False)

        # Feedback buttons (Radio or Thumbs Up/Down), hidden initially
        feedback_buttons = gr.Radio(["Yes", "No", "👍", "👎"], label="Was this answer helpful?", visible=False)

        # When the user submits a query
        query_input.submit(search_and_feedback, [query_input, feedback_buttons], [response_output, feedback_output, feedback_buttons])

        # Feedback handling (only if the answer needs feedback)
        feedback_buttons.change(search_and_feedback, [query_input, feedback_buttons], [response_output, feedback_output, feedback_buttons])

    return demo

# Launch the Gradio app with share=True for public link
demo = build_interface()
demo.launch(share=True)
