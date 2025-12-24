# Multi-Agent Technical Article Generator

This project implements a Multi-Agent System (MAS) designed to autonomously generate in-depth technical articles (1000–1500 words) with optional code examples, and export them in PDF and DOCX formats.

## Features

- **Topic Analysis**: Parses and validates topic; refines scope and outlines subtopics
- **Content Generation**: Creates detailed, structured text (intro, body, conclusion)
- **Code Snippets**: Generates or embeds code examples (Python, JS, etc.) when requested
- **Formatting**: Applies consistent heading styles, bullets, and syntax highlighting
- **Export**: Generates downloadable PDF and DOCX documents
- **Web Interface**: Streamlit-based UI for easy interaction

## Architecture

The system consists of 5 specialized agents coordinated by an orchestrator:

1. **Topic Analyzer Agent**: Analyzes and refines the input topic
2. **Content Generator Agent**: Creates the main article content
3. **Code Snippet Agent**: Generates relevant code examples
4. **Formatter Agent**: Ensures consistent formatting
5. **Exporter Agent**: Exports to PDF and DOCX formats

## Technical Stack

- **Backend Language**: Python
- **LLM Frameworks**: LangChain, LangGraph
- **Language Models**: Ollama (Mistral, Llama3, CodeLlama)
- **PDF Generation**: WeasyPrint (primary), ReportLab (fallback)
- **DOCX Generation**: python-docx
- **UI Framework**: Streamlit
- **Markdown Processing**: markdown2, Pygments

## Prerequisites

1. **Python 3.10+** installed
2. **Ollama** installed and running ([https://ollama.com](https://ollama.com))
3. Required models pulled:
   ```bash
   ollama pull mistral
   ollama pull llama3
   ollama pull codellama
   ```

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd multi-agent-article-generator
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Web Interface (Recommended)

To avoid import errors, run the Streamlit app as a module from the root of the project directory. This ensures Python can find all the project's packages (like `orchestrator`, `agents`, etc.).

```bash

python -m streamlit run ui/app.py
```

### Testing the System

```bash
python test_llm.py
```

## Project Structure

```
multi_agent_article_generator/
├── agents/
│   ├── topic_analyzer.py
│   ├── content_generator.py
│   ├── code_snippet.py
│   ├── formatter.py
│   └── exporter.py
├── orchestrator/
│   └── workflow.py
├── ui/
│   └── app.py
├── utils/
│   ├── llm_loader.py
│   ├── markdown_utils.py
│   └── file_utils.py
├── requirements.txt
└── README.md
```

## How It Works

1. **Topic Analysis**: The Topic Analyzer Agent refines the user's input topic and identifies key sub-topics
2. **Content Generation**: The Content Generator Agent creates a structured article based on the analysis
3. **Code Enhancement**: If requested, the Code Snippet Agent adds relevant code examples
4. **Formatting**: The Formatter Agent ensures consistent styling and structure
5. **Export**: The Exporter Agent converts the final content to PDF and DOCX formats

## Troubleshooting

### Tesseract OCR Conflict

If you encounter an error like:
```
cannot load library 'C:\Program Files\Tesseract-OCR\libgobject-2.0-0.dll': error 0x7e
```

This is a known issue on Windows systems where Tesseract OCR interferes with WeasyPrint's dependencies. 
Our system automatically falls back to ReportLab for PDF generation in this case, so functionality is preserved.

To fully resolve this issue:
1. Uninstall Tesseract OCR if you don't need it
2. Or adjust your system PATH to prioritize the correct GTK libraries
3. Or use a virtual environment with isolated dependencies

### Model Not Found Errors

If you encounter errors like:
```
Error calling Ollama: 404 Client Error: Not Found for url: http://localhost:11434/api/generate
```

Make sure you're using the correct model names with their tags (e.g., `codellama:7b` instead of just `codellama`).
Check available models with:
```bash
ollama list
```

## Customization

You can customize the behavior by modifying:
- Agent prompts in each agent file
- Output formatting in the formatter agent
- Export styling in the exporter agent
- UI elements in the Streamlit app

## License

This project is licensed under the MIT License - see the LICENSE file for details.