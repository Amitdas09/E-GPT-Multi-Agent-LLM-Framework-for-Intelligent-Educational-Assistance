from backend.pdf_qa import DocumentQASystem
import streamlit as st
import tempfile
import hashlib
import traceback

class DocQAAgent:
    """Answers user questions from uploaded documents using LangChain + Ollama, with FAISS caching."""

    def __init__(self):
        self.name = "DocQAAgent"
        # Initialize system only once per session
        if "qa_system" not in st.session_state:
            st.session_state.qa_system = DocumentQASystem()
        self.qa_system = st.session_state.qa_system

        # Maintain cache for processed document indexes
        if "doc_cache" not in st.session_state:
            st.session_state.doc_cache = {}

    def _compute_file_hash(self, uploaded_files):
        """Create a unique hash for all uploaded files combined."""
        hash_input = "".join([f"{f.name}:{len(f.getvalue())}" for f in uploaded_files])
        return hashlib.md5(hash_input.encode()).hexdigest()

    def handle(self, user_prompt: str, uploaded_files=None):
        try:
            if not uploaded_files:
                return {"type": "text", "content": "Please upload at least one PDF or DOCX file."}

            # 1️⃣ Compute unique hash for uploaded files
            file_hash = self._compute_file_hash(uploaded_files)

            # 2️⃣ Check if we've already processed this document
            if file_hash in st.session_state.doc_cache:
                st.info("✅ Using cached FAISS index for this document.")
                self.qa_system.vector_store = st.session_state.doc_cache[file_hash]["vector_store"]
                self.qa_system.qa_chain = st.session_state.doc_cache[file_hash]["qa_chain"]

            else:
                st.info("📄 Extracting and indexing document (first-time setup)...")

                texts = []
                for f in uploaded_files:
                    try:
                        # Create a unique temporary file for each uploaded file
                        ext = f.name.split(".")[-1].lower()
                        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
                            tmp.write(f.getvalue())
                            tmp_path = tmp.name

                        # Extract text from this file
                        text = self.qa_system.extract_text_from_file(tmp_path)
                        if text.strip():
                            texts.append(text)
                        else:
                            st.warning(f"No readable text extracted from {f.name}")

                    except Exception as e:
                        st.warning(f"Failed to read {f.name}: {e}")

                combined_text = "\n".join(texts)
                if not combined_text.strip():
                    return {"type": "text", "content": "No readable text found in uploaded files."}

                st.info("🔍 Creating FAISS vector store...")
                chunks = self.qa_system.create_chunks(combined_text)
                self.qa_system.create_vector_store(chunks)
                self.qa_system.setup_qa_chain()

                # Cache the FAISS index for reuse
                st.session_state.doc_cache[file_hash] = {
                    "vector_store": self.qa_system.vector_store,
                    "qa_chain": self.qa_system.qa_chain,
                }

                st.success("✅ Document indexed and cached successfully!")

            # 3️⃣ Ask the question
            st.info("🧠 Generating answer...")
            result = self.qa_system.answer_question(user_prompt)

            # 4️⃣ Format result (cleaned output)
            if isinstance(result, dict):
                answer = result.get("answer", "").strip()
                sources = result.get("sources", [])
                corrected_q = result.get("corrected_question", "")

                formatted = f"**Answer:** {answer}" if answer else "No direct answer found."

                if sources:
                    formatted += "\n\n**Sources:**\n" + "\n".join(
                        [f"- {s.strip()}" for s in sources if s.strip()]
                    )

                if corrected_q and corrected_q.lower() != user_prompt.lower():
                    formatted += f"\n\n**Reinterpreted Question:** {corrected_q.strip()}"

                answer_text = formatted
            else:
                answer_text = str(result).strip()

            # 5️⃣ Return clean text
            return {"type": "text", "content": answer_text}

        except Exception as e:
            st.error("⚠️ Error during document QA.")
            st.error(traceback.format_exc())
            return {"type": "text", "content": f"⚠️ Document Q&A failed: {e}"}
