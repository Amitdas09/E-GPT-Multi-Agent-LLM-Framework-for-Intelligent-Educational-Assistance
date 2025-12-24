# agents/formatter.py
import re
from typing import Tuple

class FormatterAgent:
    def __init__(self):
        pass

    def run(self, content: str, code_snippets: str = "") -> str:
        """
        Format the article content with consistent styling and add code snippets if provided.
        
        Args:
            content (str): The main article content in Markdown format.
            code_snippets (str): Optional code snippets to include.
            
        Returns:
            str: Formatted article content in Markdown format.
        """
        # Clean up any extra whitespace
        formatted_content = re.sub(r'\n{3,}', '\n\n', content.strip())
        
        # Add code snippets if provided
        if code_snippets:
            # Insert code snippets before the conclusion section
            if "## Conclusion" in formatted_content:
                parts = formatted_content.split("## Conclusion")
                formatted_content = parts[0] + "\n\n" + code_snippets.strip() + "\n\n## Conclusion" + parts[1]
            else:
                # If no conclusion section, append at the end
                formatted_content += "\n\n" + code_snippets.strip()
        
        return formatted_content

    def extract_title(self, content: str) -> str:
        """
        Extract the main title from the content.
        
        Args:
            content (str): The article content.
            
        Returns:
            str: The main title.
        """
        # Find the first H1 header
        match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
        if match:
            return match.group(1).strip()
        return "Technical Article"