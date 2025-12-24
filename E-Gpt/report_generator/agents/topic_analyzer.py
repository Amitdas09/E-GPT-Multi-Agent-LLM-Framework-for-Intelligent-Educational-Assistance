# agents/topic_analyzer.py
from langchain.prompts import PromptTemplate
from report_generator.utils.llm_loader import load_llm

class TopicAnalyzerAgent:
    def __init__(self, model_name: str = "mistral"):
        self.llm = load_llm(model=model_name)
        self.prompt = PromptTemplate(
            input_variables=["topic"],
            template="""
            You are a technical research assistant.
            Analyze the given research topic: "{topic}".

            Provide output in the following structured format:

            ### Refined Description
            - (2–3 sentences refining the topic scope)

            ### Key Sub-Topics
            - List of important dimensions

            ### Related Fields
            - Disciplines or domains connected

            ### Practical Applications
            - Real-world use cases
            """
        )

    def run(self, topic: str) -> str:
        """
        Analyze the topic using the LLM.

        Args:
            topic (str): Raw topic input from user.

        Returns:
            str: Structured analysis text.
        """
        final_prompt = self.prompt.format(topic=topic)
        response = self.llm.invoke(final_prompt)
        return response
