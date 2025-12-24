# ui/app.py
import streamlit as st
import os
from report_generator.orchestrator.workflow import OrchestratorAgent

# ---------------------------
# Page Config
# ---------------------------
# Wrap in try-except to allow modular use
try:
    st.set_page_config(
        page_title="Multi-Agent Technical Article Generator",
        page_icon="📝",
        layout="wide",
        initial_sidebar_state="expanded"
    )
except:
    # Page config already set, skip
    pass

# ---------------------------
# Sidebar
# ---------------------------
st.sidebar.title("⚙️ Settings")

# Topic input
topic = st.sidebar.text_input(
    "📌 Enter Topic",
    placeholder="e.g., Machine Learning Optimization Techniques"
)

# Code inclusion
include_code = st.sidebar.toggle(
    "Include Code Examples", value=True
)

# Model selection
st.sidebar.subheader("🧠 Model Selection")

content_model = st.sidebar.selectbox(
    "Content Model",
    ["mistral:latest", "llama3.2:3b", "codellama:7b"],
    index=0
)

code_model = st.sidebar.selectbox(
    "Code Model",
    ["codellama:7b", "llama3.2:3b", "mistral:latest"],
    index=0
)

# Generate button in sidebar
generate_btn = st.sidebar.button("🚀 Generate Article", use_container_width=True)

# ---------------------------
# Custom Styling
# ---------------------------
st.markdown("""
<style>
    /* Headers */
    .main-header {
        color: #2c3e50;
        font-size: 2rem !important;
        font-weight: bold;
    }

    /* Card-like container */
    .stCard {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #e0e0e0;
        margin-bottom: 20px;
        color: Black;
    }

    /* Download buttons */
    .stDownloadButton > button {
        background-color: #3498db !important;
        color: white !important;
        border-radius: 8px;
        padding: 0.6rem 1.2rem;
        font-weight: bold;
        border: none;
    }
    .stDownloadButton > button:hover {
        background-color: #2980b9 !important;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Session State
# ---------------------------
if "generated" not in st.session_state:
    st.session_state.generated = False
    st.session_state.content = ""
    st.session_state.pdf_path = ""
    st.session_state.docx_path = ""

# ---------------------------
# Main Title
# ---------------------------
st.markdown("<h1 class='main-header'>📝 Multi-Agent Technical Article Generator</h1>", unsafe_allow_html=True)
st.write("Automatically generate structured technical articles with optional code snippets, and export them to PDF/DOCX.")

# ---------------------------
# Content Area
# ---------------------------
if generate_btn:
    if not topic:
        st.warning("⚠️ Please enter a topic first.")
    else:
        try:
            with st.spinner("🔄 Initializing multi-agent system..."):
                orchestrator = OrchestratorAgent(
                    model_name=content_model,
                    code_model_name=code_model
                )

            with st.spinner("✍️ Generating article... (this may take ~1-2 minutes)"):
                content, pdf_path, docx_path = orchestrator.run(topic, include_code)

            st.session_state.generated = True
            st.session_state.content = content
            st.session_state.pdf_path = pdf_path
            st.session_state.docx_path = docx_path

            st.success("✅ Article generated successfully!")

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.info("Make sure Ollama is running and the selected models are available.")

# Create tabs for preview and downloads
tab1, tab2 = st.tabs(["📄 Article Preview", "📥 Export & Downloads"])

with tab1:
    if st.session_state.generated:
        with st.container():
            st.markdown(f"<div class='stCard'>{st.session_state.content}</div>", unsafe_allow_html=True)
    else:
        st.info("Click 'Generate Article' in the sidebar to create an article.")

with tab2:
    if st.session_state.generated:
        if os.path.exists(st.session_state.pdf_path):
            with open(st.session_state.pdf_path, "rb") as f:
                st.download_button(
                    label=f"📄 Download {os.path.basename(st.session_state.pdf_path).upper()}",
                    data=f,
                    file_name=os.path.basename(st.session_state.pdf_path),
                    mime="application/octet-stream",
                    use_container_width=True
                )

        if os.path.exists(st.session_state.docx_path):
            with open(st.session_state.docx_path, "rb") as f:
                st.download_button(
                    label="📝 Download DOCX file",
                    data=f,
                    file_name=os.path.basename(st.session_state.docx_path),
                    mime="application/octet-stream",
                    use_container_width=True
                )

        st.info(f"Files saved at:\n- {st.session_state.pdf_path}\n- {st.session_state.docx_path}")
    else:
        st.info("Generate an article to enable downloads.")

# ---------------------------
# System Information (Expander)
# ---------------------------
with st.expander("ℹ️ System Information", expanded=False):
    st.markdown("""
    **Requirements:**
    - ✅ Ollama installed & running
    - ✅ Models available: `mistral`, `llama3`, `codellama`

    **Pull models with:**
    ```
    ollama pull mistral
    ollama pull llama3
    ollama pull codellama
    ```

    **Pipeline Agents:**
    - 🔍 Topic Analyzer
    - ✍️ Content Generator
    - 💻 Code Snippet Generator
    - 🎨 Formatter
    - 📦 Exporter
    """)

# Footer
st.markdown("---")
st.caption("Multi-Agent Technical Article Generator | Built with Streamlit, LangGraph, and Ollama 🚀")
