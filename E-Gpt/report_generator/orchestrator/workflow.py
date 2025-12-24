# orchestrator/workflow.py
from report_generator.agents.topic_analyzer import TopicAnalyzerAgent
from report_generator.agents.content_generator import ContentGeneratorAgent
from report_generator.agents.code_snippet import CodeSnippetAgent
from report_generator.agents.formatter import FormatterAgent
from report_generator.agents.exporter import ExporterAgent
from typing import Tuple, Optional

class OrchestratorAgent:
    def __init__(self, model_name: str = "mistral", code_model_name: str = "codellama:7b"):
        self.topic_analyzer = TopicAnalyzerAgent(model_name)
        self.content_generator = ContentGeneratorAgent(model_name)
        self.code_snippet_agent = CodeSnippetAgent(code_model_name)
        self.formatter = FormatterAgent()
        self.exporter = ExporterAgent()

    def run(self, topic: str, include_code: bool = True) -> Tuple[str, str, str]:
        """
        Run the complete workflow to generate and export an article.
        
        Args:
            topic (str): The topic to generate an article about.
            include_code (bool): Whether to include code examples.
            
        Returns:
            Tuple[str, str, str]: The final content, PDF path, and DOCX path.
        """
        print("Step 1: Analyzing topic...")
        topic_analysis = self.topic_analyzer.run(topic)
        print("Topic analysis completed.")
        
        print("Step 2: Generating content...")
        article_content = self.content_generator.run(topic_analysis)
        print("Content generation completed.")
        
        code_snippets = ""
        if include_code:
            print("Step 3: Generating code snippets...")
            code_snippets = self.code_snippet_agent.run(article_content)
            print("Code snippet generation completed.")
        
        print("Step 4: Formatting content...")
        formatted_content = self.formatter.run(article_content, code_snippets)
        title = self.formatter.extract_title(formatted_content)
        print("Formatting completed.")
        
        print("Step 5: Exporting to PDF and DOCX...")
        pdf_path, docx_path = self.exporter.run(formatted_content, title)
        print("Export completed.")
        
        return formatted_content, pdf_path, docx_path