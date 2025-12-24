import streamlit as st
import time
import re
import json
import requests
from typing import Dict
from agents.quiz_agent import QuizAgent
from agents.doc_qa_agent import DocQAAgent
from agents.exam_agent import ExamPaperAgent
from agents.rag_chat_agent import RAGChatAgent
from agents.report_agent import ReportAgent

# ----------------------------------------------------------------------------------
# PAGE CONFIGURATION
# ----------------------------------------------------------------------------------
st.set_page_config(
    page_title="E-GPT - Multi-Agent AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------------------------------------------------------------
# CUSTOM CSS FOR MODERN UI
# ----------------------------------------------------------------------------------
def load_custom_css():
    st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Main Container */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        background-attachment: fixed;
    }
    
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Header Styles */
    .main-header {
        background: rgba(31, 41, 55, 0.95);
        backdrop-filter: blur(10px);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.3);
        margin-bottom: 2rem;
        text-align: center;
        border: 1px solid rgba(102, 126, 234, 0.3);
    }
    
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .main-subtitle {
        color: #d1d5db;
        font-size: 1rem;
        font-weight: 400;
    }
    
    /* Sidebar Styles */
    [data-testid="stSidebar"] {
        background: #1f2937;
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: white;
    }
    
    [data-testid="stSidebar"] h3 {
        color: white !important;
        font-weight: 600;
        margin-top: 1rem;
    }
    
    /* File Uploader */
    [data-testid="stSidebar"] .stFileUploader {
        background: #374151;
        border: 2px dashed #667eea;
        border-radius: 12px;
        padding: 1rem;
    }
    
    [data-testid="stSidebar"] .stFileUploader label {
        color: white !important;
    }
    
    /* Agent Cards in Sidebar */
    .agent-card {
        background: #374151;
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
        transition: all 0.3s ease;
    }
    
    .agent-card:hover {
        background: #4b5563;
        transform: translateX(4px);
    }
    
    .agent-card-active {
        background: #10b981;
        border-left-color: #10b981;
    }
    
    .agent-card-title {
        font-weight: 600;
        color: white;
        margin-bottom: 0.25rem;
        font-size: 0.95rem;
    }
    
    .agent-card-desc {
        font-size: 0.8rem;
        color: #9ca3af;
    }
    
    /* Status Cards */
    .status-card-success {
        background: #10b981;
        color: white;
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        font-weight: 600;
    }
    
    .status-card-error {
        background: #ef4444;
        color: white;
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        font-weight: 600;
    }
    
    /* Metric Cards in Sidebar */
    .sidebar-metric {
        background: #374151;
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        border-top: 3px solid #667eea;
    }
    
    .sidebar-metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #667eea;
    }
    
    .sidebar-metric-label {
        font-size: 0.8rem;
        color: #9ca3af;
        margin-top: 0.25rem;
    }
    
    /* Sidebar Buttons */
    [data-testid="stSidebar"] .stButton > button {
        background: #667eea;
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem;
        font-weight: 600;
        width: 100%;
        transition: all 0.3s ease;
    }
    
    [data-testid="stSidebar"] .stButton > button:hover {
        background: #5a67d8;
        transform: translateY(-2px);
    }
    
    [data-testid="stSidebar"] .stDownloadButton > button {
        background: #10b981;
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem;
        font-weight: 600;
        width: 100%;
    }
    
    /* Chat Messages */
    .stChatMessage {
        background: rgba(31, 41, 55, 0.95) !important;
        backdrop-filter: blur(10px);
        border-radius: 16px !important;
        padding: 1.5rem !important;
        margin: 1rem 0 !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2) !important;
        border: 1px solid rgba(102, 126, 234, 0.2) !important;
        color: white !important;
    }
    
    .stChatMessage p,
    .stChatMessage div,
    .stChatMessage span,
    .stChatMessage li,
    .stChatMessage strong,
    .stChatMessage em {
        color: white !important;
    }
    
    .stChatMessage[data-testid*="user"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
    }
    
    .stChatMessage[data-testid*="user"] p,
    .stChatMessage[data-testid*="user"] div {
        color: white !important;
    }
    
    .stChatMessage[data-testid*="assistant"] {
        background: rgba(31, 41, 55, 0.95) !important;
        border-left: 4px solid #667eea !important;
        color: white !important;
    }
    
    .stChatMessage[data-testid*="assistant"] p,
    .stChatMessage[data-testid*="assistant"] div,
    .stChatMessage[data-testid*="assistant"] span {
        color: white !important;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Download Buttons */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.3);
    }
    
    .stDownloadButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(16, 185, 129, 0.4);
    }
    
    /* Quiz Styles */
    .quiz-progress {
        background: rgba(55, 65, 81, 0.8);
        height: 8px;
        border-radius: 4px;
        margin-bottom: 1.5rem;
        overflow: hidden;
    }
    
    .quiz-progress-bar {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        height: 100%;
        transition: width 0.3s ease;
    }
    
    .quiz-question {
        font-size: 1.1rem;
        font-weight: 600;
        color: white;
        margin: 1.5rem 0 1rem 0;
        padding: 1rem;
        background: rgba(55, 65, 81, 0.8);
        backdrop-filter: blur(10px);
        border-radius: 12px;
        border-left: 4px solid #667eea;
    }
    
    /* Score Card */
    .score-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
        margin: 2rem 0;
    }
    
    .score-value {
        font-size: 4rem;
        font-weight: 700;
        margin: 1rem 0;
    }
    
    .score-label {
        font-size: 1.25rem;
        opacity: 0.9;
    }
    
    /* Info/Success/Error Cards */
    .info-card {
        background: rgba(59, 130, 246, 0.2);
        border: 1px solid rgba(59, 130, 246, 0.4);
        border-radius: 12px;
        padding: 1rem;
        margin: 1rem 0;
        color: #93c5fd;
        backdrop-filter: blur(10px);
    }
    
    .success-card {
        background: rgba(16, 185, 129, 0.2);
        border: 1px solid rgba(16, 185, 129, 0.4);
        border-radius: 12px;
        padding: 1rem;
        margin: 1rem 0;
        color: #6ee7b7;
        backdrop-filter: blur(10px);
    }
    
    .error-card {
        background: rgba(239, 68, 68, 0.2);
        border: 1px solid rgba(239, 68, 68, 0.4);
        border-radius: 12px;
        padding: 1rem;
        margin: 1rem 0;
        color: #fca5a5;
        backdrop-filter: blur(10px);
    }
    
    /* Radio Buttons */
    .stRadio > div {
        background: rgba(55, 65, 81, 0.9);
        padding: 0.75rem;
        border-radius: 12px;
        border: 2px solid rgba(102, 126, 234, 0.3);
        margin: 0.5rem 0;
        transition: all 0.3s ease;
    }
    
    .stRadio > div:hover {
        border-color: #667eea;
        background: rgba(75, 85, 99, 0.9);
    }
    
    .stRadio label {
        color: white !important;
    }
    
    /* Chat Input */
    .stChatInputContainer {
        background: rgba(31, 41, 55, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        padding: 0.5rem;
        border: 1px solid rgba(102, 126, 234, 0.3);
    }
    
    .stChatInput textarea {
        color: white !important;
    }
    
    /* Loading Animation */
    .loading-dots::after {
        content: '...';
        animation: dots 1.5s steps(4, end) infinite;
    }
    
    @keyframes dots {
        0%, 20% { content: '.'; }
        40% { content: '..'; }
        60%, 100% { content: '...'; }
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #1f2937;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #5a67d8 0%, #6b46c1 100%);
    }
    
    /* Global Text Colors */
    p, span, div, li, ul, ol, label {
        color: white !important;
    }
    
    /* Ensure all markdown content is white */
    .stMarkdown, .stMarkdown p, .stMarkdown div, .stMarkdown span {
        color: white !important;
    }
    
    /* Code blocks */
    code {
        background: rgba(55, 65, 81, 0.8);
        color: #a78bfa;
        padding: 0.2rem 0.4rem;
        border-radius: 4px;
    }
    
    pre {
        background: rgba(31, 41, 55, 0.9);
        border: 1px solid rgba(102, 126, 234, 0.3);
        border-radius: 8px;
        padding: 1rem;
    }
    
    pre code {
        color: #d1d5db;
    }
    
    /* Streamlit Info/Warning/Error boxes */
    .stAlert {
        background: rgba(31, 41, 55, 0.9);
        border-radius: 12px;
        color: white;
    }
    
    /* Text input and textarea */
    input, textarea {
        background: rgba(55, 65, 81, 0.8) !important;
        color: white !important;
        border: 1px solid rgba(102, 126, 234, 0.3) !important;
    }
    
    /* Select boxes */
    select {
        background: rgba(55, 65, 81, 0.8) !important;
        color: white !important;
    }
    
    /* Tables */
    table {
        background: rgba(31, 41, 55, 0.9);
        color: white;
    }
    
    th {
        background: rgba(55, 65, 81, 0.9);
        color: white;
    }
    
    td {
        color: white;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(55, 65, 81, 0.9) !important;
        color: white !important;
        border-radius: 12px;
        font-weight: 600;
        border: 1px solid rgba(102, 126, 234, 0.3);
    }
    
    .streamlit-expanderContent {
        background: rgba(31, 41, 55, 0.9);
        border-radius: 0 0 12px 12px;
        color: white;
    }
    
    .streamlit-expanderContent p,
    .streamlit-expanderContent div,
    .streamlit-expanderContent span {
        color: white !important;
    }
    
    [data-testid="stSidebar"] .streamlit-expanderHeader {
        background: #374151 !important;
    }
    
    [data-testid="stSidebar"] .streamlit-expanderContent {
        background: #2d3748;
        color: white;
    }
    
    /* Welcome Cards */
    .welcome-card {
        background: rgba(31, 41, 55, 0.95);
        backdrop-filter: blur(10px);
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.3);
        border-top: 4px solid #667eea;
        text-align: center;
        transition: all 0.3s ease;
        height: 100%;
    }
    
    .welcome-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.4);
        border-top-color: #764ba2;
    }
    
    .welcome-card div {
        color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)

load_custom_css()

# ----------------------------------------------------------------------------------
# SESSION STATE SETUP
# ----------------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

if "quiz_user_answers" not in st.session_state:
    st.session_state.quiz_user_answers = {}
if "quiz_results" not in st.session_state:
    st.session_state.quiz_results = None
if "quiz_submitted" not in st.session_state:
    st.session_state.quiz_submitted = False
if "quiz_active" not in st.session_state:
    st.session_state.quiz_active = False
if "current_quiz" not in st.session_state:
    st.session_state.current_quiz = None
if "QUIZ_RUN_ID" not in st.session_state:
    st.session_state.QUIZ_RUN_ID = f"{int(time.time())}"
if "exam_inputs" not in st.session_state:
    st.session_state.exam_inputs = {
        "syllabus_text": "", 
        "marks": {1: 5, 2: 3, 4: 3, 6: 2, 8: 2, 10: 1}
    }
if "active_agent" not in st.session_state:
    st.session_state.active_agent = None

# ----------------------------------------------------------------------------------
# AGENT INITIALIZATION
# ----------------------------------------------------------------------------------
quiz_agent = QuizAgent()
doc_qa_agent = DocQAAgent()
exam_agent = ExamPaperAgent()
rag_agent = RAGChatAgent()
report_agent = ReportAgent()

# ----------------------------------------------------------------------------------
# UTILITY FUNCTIONS
# ----------------------------------------------------------------------------------
def check_ollama_status():
    """Check if Ollama is running"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            models = response.json().get("models", [])
            return True, len(models)
        return False, 0
    except:
        return False, 0

def classify_query_with_llm(prompt: str, has_files: bool = False) -> str:
    """Intelligently classify user query"""
    p = prompt.lower().strip()
    
    if re.search(r"\b(generate|create|make).*(quiz|mcq)\b", p) or "quiz" in p:
        return "quiz"
    elif re.search(r"\b(question paper|exam|paper generator)\b", p):
        return "exam"
    elif "report" in p and re.search(r"\b(write|generate|create)\b", p):
        return "report"
    elif has_files and re.search(r"\b(pdf|document|context|uploaded|file|what|explain|summarize)\b", p):
        return "doc_qa"
    else:
        return "rag"

def route_intent(prompt: str) -> str:
    """Determine which agent should handle the request"""
    has_files = bool(st.session_state.get("uploaded_files", []))
    return classify_query_with_llm(prompt, has_files)

# ----------------------------------------------------------------------------------
# RENDER MESSAGE HELPER
# ----------------------------------------------------------------------------------
def render_message(msg):
    """Render a single message based on its type"""
    with st.chat_message(msg["role"]):
        msg_type = msg.get("type", "text")
        
        if msg_type == "text":
            st.markdown(f'<div style="color: white;">{msg["content"]}</div>', unsafe_allow_html=True)
        
        elif msg_type == "file_bundle":
            st.markdown(f'<div style="color: white;">{msg.get("content", "✅ Files generated successfully!")}</div>', unsafe_allow_html=True)
            st.markdown('<div style="color: white; font-weight: 600; margin-top: 1rem; font-size: 1.1rem;">📥 Downloads</div>', unsafe_allow_html=True)
            for idx, f in enumerate(msg.get("files", [])):
                st.download_button(
                    label=f.get("label", "📄 Download file"),
                    data=f["data"],
                    file_name=f["filename"],
                    mime=f["mime"],
                    key=f"download_{msg['id']}_{idx}"
                )
        
        elif msg_type == "quiz":
            quiz_data = msg.get("content", [])

            if isinstance(quiz_data, list) and quiz_data:
                total_questions = len(quiz_data)
                answered = sum(1 for i in range(1, total_questions + 1) 
                             if st.session_state.quiz_user_answers.get(i) is not None)
                progress = (answered / total_questions) * 100 if total_questions > 0 else 0
                
                st.markdown(f"""
                <div class="quiz-progress">
                    <div class="quiz-progress-bar" style="width: {progress}%"></div>
                </div>
                <p style="text-align: center; color: #d1d5db; margin-bottom: 1.5rem; font-weight: 500;">
                    Progress: {answered}/{total_questions} questions answered
                </p>
                """, unsafe_allow_html=True)
                
                for i, q in enumerate(quiz_data, start=1):
                    st.markdown(f"""
                    <div class="quiz-question">
                        <span style="color: #a78bfa; font-weight: 700;">Q{i}.</span> {q.get('question','')}
                    </div>
                    """, unsafe_allow_html=True)

                    opts = q.get("options", {})
                    option_labels = [
                        f"A. {opts.get('A','')}",
                        f"B. {opts.get('B','')}",
                        f"C. {opts.get('C','')}",
                        f"D. {opts.get('D','')}"
                    ]

                    if not st.session_state.quiz_submitted:
                        st.markdown('<p style="color: #d1d5db; font-weight: 500; margin-bottom: 0.5rem;">Select your answer:</p>', unsafe_allow_html=True)
                        selected = st.radio(
                            f"Select your answer:",
                            option_labels,
                            key=f"{msg['id']}_quiz_answer_{i}",
                            index=None,
                            label_visibility="collapsed"
                        )
                        if selected:
                            st.session_state.quiz_user_answers[i] = selected

                    else:
                        chosen = st.session_state.quiz_user_answers.get(i)
                        correct_letter = q.get("correct_answer","").upper().strip()
                        correct_label = f"{correct_letter}. {opts.get(correct_letter,'')}"

                        if chosen == correct_label:
                            st.markdown(f'<div class="success-card">✅ <strong>Correct!</strong> Your answer: {chosen}</div>', unsafe_allow_html=True)
                        else:
                            if chosen:
                                st.markdown(f'<div class="error-card">❌ <strong>Incorrect.</strong> Your answer: {chosen}</div>', unsafe_allow_html=True)
                            else:
                                st.markdown('<div class="error-card">⚠️ <strong>No answer selected</strong></div>', unsafe_allow_html=True)

                            st.markdown(f'<div class="info-card">✓ <strong>Correct answer:</strong> {correct_label}</div>', unsafe_allow_html=True)

                        if q.get("explanation"):
                            with st.expander("💡 View Explanation"):
                                st.markdown(f'<div style="color: white;"><strong>Explanation:</strong> {q["explanation"]}</div>', unsafe_allow_html=True)

                    st.markdown("---")

                if not st.session_state.quiz_submitted:
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        if st.button("✅ Submit Quiz", key=f"{msg['id']}_submit_quiz", use_container_width=True):
                            st.session_state.quiz_submitted = True

                            score = 0
                            total = len(quiz_data)

                            for i, q in enumerate(quiz_data, start=1):
                                chosen = st.session_state.quiz_user_answers.get(i)
                                correct = q.get("correct_answer","").upper().strip()
                                correct_label = f"{correct}. {q['options'].get(correct,'')}"

                                if chosen == correct_label:
                                    score += 1

                            st.session_state.quiz_results = {"score": score, "total": total}
                            st.rerun()

                else:
                    r = st.session_state.quiz_results
                    if r:
                        percentage = (r['score'] / r['total']) * 100
                        
                        if percentage >= 90:
                            grade, emoji = "A+", "🏆"
                        elif percentage >= 80:
                            grade, emoji = "A", "🎉"
                        elif percentage >= 70:
                            grade, emoji = "B", "👍"
                        elif percentage >= 60:
                            grade, emoji = "C", "📚"
                        else:
                            grade, emoji = "D", "💪"
                        
                        st.markdown(f"""
                        <div class="score-card">
                            <div style="font-size: 3rem;">{emoji}</div>
                            <div class="score-label">Your Score</div>
                            <div class="score-value">{r['score']}/{r['total']}</div>
                            <div style="font-size: 1.5rem; margin-top: 1rem;">
                                {percentage:.1f}% • Grade: {grade}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        col1, col2, col3 = st.columns([1, 2, 1])
                        with col2:
                            if st.button("🔄 Try Another Quiz", key=f"{msg['id']}_regenerate", use_container_width=True):
                                st.session_state.quiz_submitted = False
                                st.session_state.quiz_user_answers = {}
                                st.session_state.quiz_results = None
                                st.rerun()
            else:
                st.markdown('<div style="color: #d1d5db;">📋 Quiz was previously generated. Type "regenerate quiz" to create a new one.</div>', unsafe_allow_html=True)

        elif msg_type == "exam_ui":
            exam_agent.render_exam_ui()

# ----------------------------------------------------------------------------------
# HEADER
# ----------------------------------------------------------------------------------
st.markdown("""
<div class="main-header">
    <div class="main-title">🤖 E-GPT</div>
    <div class="main-subtitle">
        Your Multi-Agent AI Assistant • Quiz Generator • Document Analyzer • Report Writer
    </div>
</div>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 📁 File Management")
    
    uploaded_files = st.file_uploader(
        "Upload Documents",
        type=["pdf", "docx"],
        accept_multiple_files=True,
        help="Drag and drop files here\nLimit 200MB per file • PDF, DOCX"
    )
    
    if uploaded_files:
        st.session_state.uploaded_files = uploaded_files
        st.success(f"✅ {len(uploaded_files)} file(s) loaded")
        
        with st.expander("📄 View Files"):
            for idx, file in enumerate(uploaded_files, 1):
                st.markdown(f"**{idx}. {file.name}** ({file.size / 1024:.1f} KB)")
    
    st.markdown("---")
    
    st.markdown("### 🤖 Available Agents")
    
    agents = [
        {"name": "RAG Chat", "icon": "💬", "desc": "General conversation", "id": "rag"},
        {"name": "Document Q&A", "icon": "📄", "desc": "Analyze uploaded files", "id": "doc_qa"},
        {"name": "Quiz Generator", "icon": "🎯", "desc": "Create interactive quizzes", "id": "quiz"},
        {"name": "Exam Paper", "icon": "📝", "desc": "Generate question papers", "id": "exam"},
        {"name": "Report Writer", "icon": "📊", "desc": "Write technical reports", "id": "report"}
    ]
    
    for agent in agents:
        active = st.session_state.active_agent == agent["id"]
        card_class = "agent-card agent-card-active" if active else "agent-card"
        st.markdown(f"""
        <div class="{card_class}">
            <div class="agent-card-title">{agent['icon']} {agent['name']}</div>
            <div class="agent-card-desc">{agent['desc']}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### ⚙️ System Status")
    
    ollama_running, model_count = check_ollama_status()
    
    if ollama_running:
        st.markdown(f"""
        <div class="status-card-success">
            ✅ Ollama Online<br>
            <span style="font-size: 0.85rem; opacity: 0.9;">{model_count} model(s) available</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="status-card-error">
            ❌ Ollama Offline<br>
            <span style="font-size: 0.85rem; opacity: 0.9;">Start: ollama serve</span>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("### 📊 Session Stats")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="sidebar-metric">
            <div class="sidebar-metric-value">{len(st.session_state.history)}</div>
            <div class="sidebar-metric-label">Messages</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="sidebar-metric">
            <div class="sidebar-metric-value">{len(st.session_state.uploaded_files)}</div>
            <div class="sidebar-metric-label">Files</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### ⚡ Quick Actions")
    
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.history = []
        st.session_state.quiz_active = False
        st.session_state.quiz_user_answers = {}
        st.session_state.quiz_results = None
        st.session_state.quiz_submitted = False
        st.session_state.active_agent = None
        st.rerun()
    
    if st.button("📥 Export Chat", use_container_width=True):
        if st.session_state.history:
            chat_export = json.dumps({
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "messages": st.session_state.history
            }, indent=2)
            st.download_button(
                "💾 Download JSON",
                chat_export,
                f"chat_export_{int(time.time())}.json",
                "application/json",
                use_container_width=True
            )
    
    st.markdown("---")
    
    with st.expander("❓ Help & Examples"):
        st.markdown("""
        **📄 Document Q&A:**
        - "Summarize the PDF"
        - "What are the main points?"
        
        **🎯 Quiz Generator:**
        - "Generate a quiz"
        - "Create MCQ questions"
        
        **📝 Exam Paper:**
        - "Generate question paper"
        - "Create exam with answers"
        
        **📊 Report Writer:**
        - "Write a report on ML"
        - "Create an AI article"
        """)
    
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #9ca3af; font-size: 0.75rem;">
        Made with ❤️ by HKBK<br>
        Version 2.0
    </div>
    """, unsafe_allow_html=True)

# ----------------------------------------------------------------------------------
# MAIN CHAT AREA
# ----------------------------------------------------------------------------------

if not st.session_state.history:
    welcome_container = st.container()
    
    with welcome_container:
        col1, col2, col3 = st.columns([1, 3, 1])
        
        with col2:
            st.markdown("<div style='text-align: center; font-size: 4rem;'>👋</div>", unsafe_allow_html=True)
            st.markdown("<h2 style='text-align: center; color: white; text-shadow: 0 2px 10px rgba(0,0,0,0.2);'>Welcome to E-GPT!</h2>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center; color: rgba(255,255,255,0.9); font-size: 1.1rem; margin-bottom: 2rem;'>I'm your intelligent multi-agent AI assistant. Upload documents and start chatting!</p>", unsafe_allow_html=True)
            
            feat_cols = st.columns(5)
            
            features = [
                {"icon": "💬", "title": "Chat", "desc": "Ask anything"},
                {"icon": "📄", "title": "Analyze", "desc": "Upload PDFs"},
                {"icon": "🎯", "title": "Quiz", "desc": "Generate MCQs"},
                {"icon": "📝", "title": "Exams", "desc": "Question papers"},
                {"icon": "📊", "title": "Reports", "desc": "Technical docs"}
            ]
            
            for idx, feat in enumerate(features):
                with feat_cols[idx]:
                    st.markdown(f"""
                    <div class="welcome-card">
                        <div style="font-size: 2rem;">{feat['icon']}</div>
                        <div style="font-weight: 600; color: white; margin: 0.5rem 0 0.25rem 0;">{feat['title']}</div>
                        <div style="color: #9ca3af; font-size: 0.8rem;">{feat['desc']}</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            st.markdown("""
            <div style="padding: 1.5rem; background: rgba(31, 41, 55, 0.95); backdrop-filter: blur(10px); border-radius: 16px; margin-top: 2rem; border: 1px solid rgba(102, 126, 234, 0.3);">
                <div style="color: #a78bfa; font-weight: 600; margin-bottom: 1rem; font-size: 1.1rem;">💡 Quick Start:</div>
                <div style="color: #d1d5db; font-size: 0.95rem; line-height: 1.8;">
                    📤 <strong style="color: white;">Upload files</strong> using the sidebar<br>
                    🎯 Try: <em>"Generate a quiz from my PDF"</em><br>
                    📝 Ask: <em>"Create a question paper"</em><br>
                    📊 Say: <em>"Write a report about AI"</em>
                </div>
            </div>
            """, unsafe_allow_html=True)

for msg in st.session_state.history:
    render_message(msg)

# ----------------------------------------------------------------------------------
# CHAT INPUT
# ----------------------------------------------------------------------------------
user_input = st.chat_input("💬 Type your message here... (e.g., 'Generate quiz', 'Analyze PDF', 'Create exam')")

if user_input:
    st.session_state.history.append({
        "role": "user", 
        "type": "text", 
        "content": user_input,
        "id": f"user_{int(time.time())}"
    })
    
    with st.chat_message("user"):
        st.markdown(user_input)

    agent_type = route_intent(user_input)
    st.session_state.active_agent = agent_type
    
    with st.chat_message("assistant"):
        msg_placeholder = st.empty()
        
        agent_names = {
            "quiz": "Quiz Generator",
            "exam": "Exam Paper Generator",
            "doc_qa": "Document Analyzer",
            "report": "Report Writer",
            "rag": "AI Assistant"
        }
        
        msg_placeholder.markdown(f"""
        <div style="display: flex; align-items: center; gap: 0.5rem; padding: 1rem; background: rgba(55, 65, 81, 0.9); backdrop-filter: blur(10px); border-radius: 12px; border-left: 4px solid #667eea;">
            <div class="loading-dots" style="color: #a78bfa; font-weight: 600;">🤔 {agent_names.get(agent_type, 'AI Assistant')} is thinking</div>
        </div>
        """, unsafe_allow_html=True)
        
        time.sleep(0.5)

        try:
            if agent_type == "quiz":
                result = quiz_agent.handle(user_input, st.session_state.uploaded_files)
                st.session_state.quiz_active = True
                
            elif agent_type == "exam":
                result = exam_agent.handle(user_input, st.session_state.uploaded_files)
                
            elif agent_type == "doc_qa":
                result = doc_qa_agent.handle(user_input, st.session_state.uploaded_files)
                
            elif agent_type == "report":
                result = report_agent.handle(user_input)
                
            else:
                result = rag_agent.handle(user_input)

            msg_placeholder.empty()

            message_id = f"msg_{int(time.time())}"
            assistant_msg = {
                "role": "assistant",
                "type": result.get("type", "text"),
                "content": result.get("content", ""),
                "id": message_id
            }

            if result["type"] == "file":
                assistant_msg["filename"] = result.get("filename")
                assistant_msg["mime"] = result.get("mime")
                
            elif result["type"] == "file_bundle":
                assistant_msg["files"] = result.get("files", [])
                assistant_msg["preview"] = result.get("preview")
                
            elif result["type"] == "quiz":
                st.session_state.quiz_submitted = False
                st.session_state.quiz_user_answers = {}
                st.session_state.quiz_results = None
                st.session_state.current_quiz = result["content"]
                st.session_state.QUIZ_RUN_ID = message_id

            render_message(assistant_msg)
            st.session_state.history.append(assistant_msg)

        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            error_msg = f"⚠️ Error: {str(e)}"
            
            msg_placeholder.markdown(f"""
            <div class="error-card">
                <strong style="color: #fca5a5;">❌ Something went wrong</strong><br>
                <span style="font-size: 0.875rem; color: #fca5a5;">{error_msg}</span>
            </div>
            """, unsafe_allow_html=True)
            
            with st.expander("🔍 View Error Details"):
                st.code(error_trace, language="python")
            
            st.session_state.history.append({
                "role": "assistant",
                "type": "text",
                "content": error_msg,
                "id": f"error_{int(time.time())}"
            })