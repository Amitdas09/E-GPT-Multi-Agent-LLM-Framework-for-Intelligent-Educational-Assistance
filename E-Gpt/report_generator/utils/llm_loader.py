import requests
import json

class OllamaLLM:
    def __init__(self, model="mistral:latest", url="http://localhost:11434/api/generate"):
        self.model = model
        self.url = url

    def invoke(self, prompt):
        """
        Send prompt to Ollama and return the generated response.
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False
        }
        try:
            response = requests.post(self.url, json=payload)
            response.raise_for_status()
            return response.json().get("response", "")
        
        except Exception as e:
            print(f"Error calling Ollama: {e}")
            return ""

def load_llm(model="mistral"):
    """
    Load and return an Ollama LLM instance.
    
    Args:
        model (str): The name of the model to load.
        
    Returns:
        OllamaLLM: An instance of the Ollama LLM.
    """
    # If model already contains a tag, use it as is
    if ":" in model:
        return OllamaLLM(model=model)
    else:
        return OllamaLLM(model=f"{model}:latest")