# agents/content_generator.py
from langchain.prompts import PromptTemplate
from report_generator.utils.llm_loader import load_llm

class ContentGeneratorAgent:
    def __init__(self, model_name: str = "mistral"):
        self.llm = load_llm(model=model_name)
        self.prompt = PromptTemplate(
            input_variables=["topic_analysis"],
            template="""
            You are a technical writer creating a detailed article based on topic analysis.
            
            Use the following topic analysis to create a comprehensive technical article (1000-1500 words):
            
            {topic_analysis}
            
            Your article should include:
            
            1. Introduction (150-200 words)
               - Brief overview of the topic
               - Why it's important/interesting
            
            2. Main Content (700-1100 words)
               - Detailed explanation of each sub-topic
               - Technical concepts with clear explanations
               - Practical applications and examples
            
            3. Conclusion (100-150 words)
               - Summary of key points
               - Future outlook or implications
            
            Format your response in Markdown with appropriate headers (# for main title, ## for sections, ### for subsections).
            """
        )

    def run(self, topic_analysis: str) -> str:
        """
        Generate content based on topic analysis.
        
        Args:
            topic_analysis (str): Analysis from the topic analyzer agent.
            
        Returns:
            str: Generated article content in Markdown format.
        """
        final_prompt = self.prompt.format(topic_analysis=topic_analysis)
        response = self.llm.invoke(final_prompt)
        return response