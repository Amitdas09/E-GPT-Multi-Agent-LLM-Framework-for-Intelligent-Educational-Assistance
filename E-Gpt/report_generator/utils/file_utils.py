# utils/file_utils.py
import os
from typing import Tuple

def create_output_directory(directory: str = "output") -> str:
    """
    Create an output directory if it doesn't exist.
    
    Args:
        directory (str): The name of the directory to create.
        
    Returns:
        str: The path to the created directory.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory

def get_file_paths(title: str, output_dir: str = "output") -> Tuple[str, str]:
    """
    Generate file paths for PDF and DOCX files based on the title.
    
    Args:
        title (str): The title of the document.
        output_dir (str): The output directory.
        
    Returns:
        Tuple[str, str]: The paths to the PDF and DOCX files.
    """
    # Sanitize title for filename
    filename = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).rstrip()
    filename = filename.replace(' ', '_')
    
    pdf_path = os.path.join(output_dir, f"{filename}.pdf")
    docx_path = os.path.join(output_dir, f"{filename}.docx")
    
    return pdf_path, docx_path

def save_text_file(content: str, filepath: str) -> None:
    """
    Save content to a text file.
    
    Args:
        content (str): The content to save.
        filepath (str): The path to the file.
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)