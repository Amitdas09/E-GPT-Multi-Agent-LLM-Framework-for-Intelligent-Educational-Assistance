# agents/exam_agent.py
import re
import io
import streamlit as st
from backend.sid import generate_paper_and_answers

class ExamPaperAgent:
    """Agent to handle question paper and answer key generation inside chat UI."""

    def __init__(self):
        self.name = "Exam Paper Generator"
        self.description = "Generates question papers and answer keys from syllabus or uploaded files."

    # ------------------------------------------------------------
    # 🔹 Helper to render the entire SID UI inside chat container
    # ------------------------------------------------------------
    def render_exam_ui(self):
        st.markdown("## 🧾 AI Question Paper & Answer Key Generator")
        st.markdown("Provide your syllabus text or upload PDFs below, then set question distribution.")

        if "exam_inputs" not in st.session_state:
            st.session_state.exam_inputs = {
                "syllabus_text": "",
                "marks": {"1": 5, "2": 3, "4": 3, "6": 2, "8": 2, "10": 1},
            }

        # -------------------------------
        # 📘 Input Section
        # -------------------------------
        st.markdown("### 📘 Syllabus / Topics")
        st.session_state.exam_inputs["syllabus_text"] = st.text_area(
            "Enter syllabus or course content:",
            value=st.session_state.exam_inputs.get("syllabus_text", ""),
            height=200,
            placeholder="Paste syllabus or upload PDFs...",
        )

        uploaded_files = st.file_uploader("📂 Upload PDFs (optional)", type=["pdf"], accept_multiple_files=True)

        st.markdown("---")
        st.markdown("### ⚙️ Marks Distribution")
        marks = st.session_state.exam_inputs.get("marks", {"1": 5, "2": 3, "4": 3, "6": 2, "8": 2, "10": 1})
        marks = {int(k): v for k, v in marks.items()}


        # Render sliders
        cols = st.columns(3)
        marks[1] = cols[0].number_input("1-mark", 0, 20, marks[1])
        marks[2] = cols[1].number_input("2-mark", 0, 20, marks[2])
        marks[4] = cols[2].number_input("4-mark", 0, 20, marks[4])

        marks[6] = cols[0].number_input("6-mark", 0, 10, marks[6])
        marks[8] = cols[1].number_input("8-mark", 0, 10, marks[8])
        marks[10] = cols[2].number_input("10-mark", 0, 10, marks[10])

        st.session_state.exam_inputs["marks"] = marks

        # -------------------------------
        # 🚀 Generate Button
        # -------------------------------
        if st.button("🚀 Generate Question Paper & Answer Key"):
            syllabus_text = st.session_state.exam_inputs["syllabus_text"]

            if not syllabus_text.strip() and not uploaded_files:
                st.warning("⚠️ Please enter syllabus text or upload at least one file.")
                return

            with st.spinner("🧠 Generating question paper and answer key..."):
                try:
                    result = generate_paper_and_answers(syllabus_text, uploaded_files, marks)
                    
                    # Store the result in session state to persist across reruns
                    st.session_state.exam_result = result
                    st.session_state.exam_generated = True

                    st.success("✅ Paper & Answer Key generated successfully!")

                except Exception as e:
                    st.error(f"⚠️ Error while generating: {e}")

        # Display the results if they exist in session state
        if st.session_state.get("exam_generated", False) and "exam_result" in st.session_state:
            result = st.session_state.exam_result
            
            # Download buttons
            st.download_button(
                "📄 Download Question Paper (PDF)",
                result["paper_pdf"],
                "QuestionPaper.pdf",
                mime="application/pdf",
            )

            st.download_button(
                "📘 Download Question Paper (DOCX)",
                result["paper_docx"],
                "QuestionPaper.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )

            st.download_button(
                "🧩 Download Answer Key (PDF)",
                result["answer_pdf"],
                "AnswerKey.pdf",
                mime="application/pdf",
            )

            # Preview expanders
            st.subheader("📋 Preview")
            with st.expander("Question Paper Preview"):
                st.text(
                    result["paper_text"][:2500]
                    + "..." if len(result["paper_text"]) > 2500 else result["paper_text"]
                )

            with st.expander("Answer Key Preview"):
                st.text(
                    result["answer_text"][:2500]
                    + "..." if len(result["answer_text"]) > 2500 else result["answer_text"]
                )

    # ------------------------------------------------------------
    # 🔹 Core handler (called by mainui)
    # ------------------------------------------------------------
    def handle(self, user_input, uploaded_files=None):
        text = user_input.lower().strip()

        # If user asks to open or create paper — render UI
        if any(word in text for word in ["open", "create", "generate"]) and "paper" in text:
            return {
                "type": "exam_ui",
                "content": "📘 Use the controls below to generate your question paper and answer key.",
            }

        # Otherwise, provide simple instruction
        return {
            "type": "text",
            "content": "🧾 Type 'open question paper generator' to begin creating papers.",
        }