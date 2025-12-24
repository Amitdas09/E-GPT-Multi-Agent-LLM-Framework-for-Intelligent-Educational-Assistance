from backend import chat
import requests

class RAGChatAgent:
    """Handles general conversation and knowledge retrieval."""

    def __init__(self):
        self.name = "RAGChatAgent"
        self.OLLAMA_URL = "http://localhost:11434/api/generate"
        self.MODEL = "llama3:8b"

    def handle(self, user_prompt: str, uploaded_files=None):
        try:
            # 1️⃣ Retrieve context + memory
            rag_context = chat.retrieve_relevant_context(user_prompt)
            need_search = chat.determine_search_need(user_prompt, rag_context)

            # 2️⃣ Optionally perform search
            internet_results = chat.search_internet(user_prompt) if need_search else []

            # 3️⃣ Build RAG prompt from backend logic
            prompt = chat.build_rag_prompt(user_prompt, rag_context, internet_results)

            # 4️⃣ Send prompt to local Ollama model
            payload = {
                "model": self.MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.7, "num_ctx": 8192}
            }

            try:
                res = requests.post(self.OLLAMA_URL, json=payload)
                data = res.json()
                response = data.get("response", "⚠️ No output from Ollama.")
            except Exception as e:
                response = f"⚠️ Ollama call failed: {e}"

            # 5️⃣ Return final model response
            return {"type": "text", "content": response}

        except Exception as e:
            return {"type": "text", "content": f"⚠️ RAG Chat failed: {e}"}
