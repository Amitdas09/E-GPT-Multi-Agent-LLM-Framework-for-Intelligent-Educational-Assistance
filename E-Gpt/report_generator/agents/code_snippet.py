# agents/code_snippet.py
from langchain.prompts import PromptTemplate
from report_generator.utils.llm_loader import load_llm

class CodeSnippetAgent:
    def __init__(self, model_name: str = "codellama:7b"):
        self.llm = load_llm(model=model_name)
        self.prompt = PromptTemplate(
            input_variables=["article_content"],
            template="""
            You are an expert programmer reviewing a technical article.
            
            Article content:
            {article_content}
            
            Your task is to enhance this article by adding relevant code examples where appropriate.
            Follow these guidelines:
            
            1. Identify 2-4 places where code examples would significantly improve understanding
            2. For each location:
               - Add a brief explanation of what the code demonstrates
               - Provide a relevant code example in the appropriate language (Python, JavaScript, etc.)
               - Add a short explanation of the code's key components
            
            Format your response as a list of code examples in Markdown format:
            
            ## Code Examples
            
            ### Example 1: [Brief Title]
            Brief explanation of what this code demonstrates.
            
            ```python
            # Sample code here
            def example_function():
                pass
            ```
            
            Explanation of key components in the code.
            
            (Repeat for each example)
            """
        )

    def run(self, article_content: str) -> str:
        """
        Generate code snippets for an article.
        
        Args:
            article_content (str): The article content to enhance with code examples.
            
        Returns:
            str: Generated code examples in Markdown format.
        """
        final_prompt = self.prompt.format(article_content=article_content)
        response = self.llm.invoke(final_prompt)
        return response