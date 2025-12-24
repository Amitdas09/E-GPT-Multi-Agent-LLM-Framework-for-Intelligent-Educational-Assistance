# agents/report_agent.py
import os
import streamlit as st
from report_generator.orchestrator.workflow import OrchestratorAgent
  # The MAS orchestrator

class ReportAgent:
    def __init__(self):
        self.name = "Report Generator"
        self.description = "Generates full structured technical reports using the MAS framework."

    def handle(self, user_input: str):
        try:
            topic = user_input.strip()
            if not topic:
                return {"type": "text", "content": "⚠️ Please provide a topic to generate the report."}

            with st.spinner("🧠 Generating report using multi-agent workflow..."):
                # Instantiate orchestrator and call run()
                orchestrator = OrchestratorAgent()
                formatted_content, pdf_path, docx_path = orchestrator.run(topic=topic)

            if not formatted_content:
                return {"type": "text", "content": "⚠️ No report generated. Please try again."}

            # Read files as bytes for download buttons
            with open(pdf_path, "rb") as f:
                pdf_bytes = f.read()
            with open(docx_path, "rb") as f:
                docx_bytes = f.read()

            text = formatted_content

            return {
                "type": "file_bundle",
                "files": [
                    {
                        "label": "📄 Download Report (PDF)",
                        "data": pdf_bytes,
                        "filename": "AI_Report.pdf",
                        "mime": "application/pdf"
                    },
                    {
                        "label": "📘 Download Report (DOCX)",
                        "data": docx_bytes,
                        "filename": "AI_Report.docx",
                        "mime": "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                    }
                ],
                # <-- Added explicit markdown field so caller/UI can access markdown directly
                "content": text,
                "markdown": formatted_content
            }

        except Exception as e:
            return {"type": "text", "content": f"⚠️ Report generation failed: {str(e)}"}
