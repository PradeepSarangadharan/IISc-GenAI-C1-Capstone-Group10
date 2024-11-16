from agentlite.llm.agent_llms import get_llm_backend
from agentlite.llm.LLMConfig import LLMConfig

class LLMAgent:
    def __init__(self, llm_config_dict):
        self.llm_config = LLMConfig(llm_config_dict)
        self.llm = get_llm_backend(self.llm_config)

    def get_response(self, prompt):
        """Uses the LLM to generate a response based on the provided prompt."""
        return self.llm(prompt)
