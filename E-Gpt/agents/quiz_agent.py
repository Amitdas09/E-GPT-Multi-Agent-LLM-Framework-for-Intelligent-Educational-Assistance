from backend.backend import extract_text_from_pdf, generate_quiz_from_text, LlamaConfig
import streamlit as st
import time

class QuizAgent:
    """Generates interactive quizzes from PDFs using LLaMA/Ollama."""

    def __init__(self):
        self.name = "QuizAgent"

    def handle(self, user_prompt: str, uploaded_files=None):

        # -------------------------------------------
        # ✅ NEW: Regenerate quiz detection
        # -------------------------------------------
        text = user_prompt.lower().strip()
        regenerate = any(x in text for x in [
            "regenerate", "new quiz", "regenerate quiz", "generate new quiz"
        ])

        if regenerate:
            st.success("🔄 Regenerating a new quiz...")
        # -------------------------------------------

        if not uploaded_files:
            return {"type": "text", "content": "Please upload at least one PDF to generate a quiz."}

        st.info("📄 Extracting text from uploaded PDFs...")
        
        from backend.backend import extract_text_from_multiple_pdfs
        combined_text = extract_text_from_multiple_pdfs(uploaded_files)

        if not combined_text.strip():
            return {"type": "text", "content": "Could not extract readable text from the PDFs."}

        st.info("🧠 Generating quiz using the model...")

        cfg = LlamaConfig(
            model_id="llama3:8b",
            max_new_tokens=1400,
            temperature=0.7,
            top_p=0.9
        )

        try:
            # -------------------------------------------
            # 🔄 NEW: use time() for regenerate also
            # -------------------------------------------
            seed_value = int(time.time()) if regenerate else int(time.time())
            # (same value, but logic hooks regenerate)
            # -------------------------------------------

            import random
            seed = random.randint(1, 99999999)
            result = generate_quiz_from_text(combined_text, cfg, num_questions=10, seed=seed)


            # 🧠 Handle flexible return formats (1, 2, or 3 values)
            if isinstance(result, tuple):
                if len(result) == 3:
                    quiz, raw, meta = result
                elif len(result) == 2:
                    quiz, raw = result
                elif len(result) == 1:
                    quiz = result[0]
                else:
                    quiz = result
            else:
                quiz = result

            if not quiz:
                return {"type": "text", "content": "⚠️ The model didn’t return a valid quiz structure."}

            # If quiz is JSON string, decode it
            if isinstance(quiz, str):
                import json
                try:
                    quiz = json.loads(quiz)
                except Exception:
                    st.warning("Quiz output was text, not JSON — showing raw text.")
                    return {"type": "text", "content": quiz}

            # --------- VALIDATION & REPAIR ---------
            # Ensure parsed is a list
            parsed = quiz if isinstance(quiz, list) else []

            # Truncate extra questions
            parsed = parsed[:10]

            # Fill missing questions
            while len(parsed) < 10:
                parsed.append({
                    "question": "⚠️ Question generation failed.",
                    "options": {
                        "A": "—",
                        "B": "—",
                        "C": "—",
                        "D": "—"
                    },
                    "correct_answer": "A",
                    "explanation": "The model failed to generate this question."
                })

            # -------------------------------------------
            # 🔄 NEW: Tell main UI this quiz is regenerated
            # -------------------------------------------
            msg = {"type": "quiz", "content": parsed}
            if regenerate:
                msg["regenerated"] = True
            # -------------------------------------------

            return msg

        except Exception as e:
            st.error("⚠️ Quiz generation failed.")
            st.error(str(e))
            return {"type": "text", "content": f"⚠️ Quiz generation failed: {e}"} 
 