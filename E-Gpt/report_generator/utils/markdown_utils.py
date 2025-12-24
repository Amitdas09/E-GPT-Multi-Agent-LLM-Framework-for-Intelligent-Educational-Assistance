# utils/markdown_utils.py
import markdown2
import re

def convert_markdown_to_html(markdown_text: str) -> str:
    """
    Convert markdown text to HTML with syntax highlighting support.
    
    Args:
        markdown_text (str): The markdown text to convert.
        
    Returns:
        str: The converted HTML.
    """
    return markdown2.markdown(markdown_text, extras=['fenced-code-blocks', 'tables', 'code-friendly'])

def extract_headers(markdown_text: str) -> list:
    """
    Extract all headers from markdown text.
    
    Args:
        markdown_text (str): The markdown text to extract headers from.
        
    Returns:
        list: A list of tuples containing (header_level, header_text).
    """
    headers = []
    lines = markdown_text.split('\n')
    for line in lines:
        match = re.match(r'^(#{1,6})\s+(.+)$', line)
        if match:
            level = len(match.group(1))
            text = match.group(2).strip()
            headers.append((level, text))
    return headers

def add_syntax_highlighting(html_content: str) -> str:
    """
    Add syntax highlighting to code blocks in HTML content.
    
    Args:
        html_content (str): The HTML content with code blocks.
        
    Returns:
        str: The HTML content with syntax highlighting.
    """
    # This is a placeholder - in a real implementation, we would integrate with Pygments
    # For now, we'll just return the content as is
    return html_content