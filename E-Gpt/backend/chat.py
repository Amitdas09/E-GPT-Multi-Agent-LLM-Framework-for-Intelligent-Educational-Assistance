import streamlit as st
import subprocess
import time
import json
from typing import List, Dict, Generator, Optional
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import requests
from bs4 import BeautifulSoup
import re
from urllib.parse import quote_plus
import concurrent.futures
from datetime import datetime
import hashlib
import runpy

# Import app module functions only when needed to avoid Streamlit conflicts
# import app as pdf_quiz_app  # REMOVED to prevent set_page_config conflict
# import app_pdf  # REMOVED to prevent set_page_config conflict



# ---- CONFIG ----
OLLAMA_PATH = r"C:\\Users\\jason\\AppData\\Local\\Programs\\Ollama\\ollama.exe"
MONGO = MongoClient("mongodb://localhost:27017/")["vtu_gpt"]
DOCS = MONGO["docs"]  # collection for storing documents + embeddings
SEARCH_CACHE = MONGO["search_cache"]  # collection for caching search results
CONVERSATION_MEMORY = MONGO["conversation_memory"]  # NEW: for storing conversation context
EMB = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Search API configurations
SEARCH_ENGINES = {
    "duckduckgo": "https://api.duckduckgo.com/",
    "serp": "https://serpapi.com/search.json",  # You'll need API key
    "bing": "https://api.bing.microsoft.com/v7.0/search"  # You'll need API key
}

# Search settings
MAX_SEARCH_RESULTS = 5
SEARCH_TIMEOUT = 10
CACHE_DURATION = 3600  # 1 hour in seconds

# RAG settings
RAG_CONTEXT_WINDOW = 10  # Number of recent messages to include in context
MIN_SIMILARITY_THRESHOLD = 0.3  # Minimum similarity for relevant context

# ---- CUSTOM CSS ----
def load_css():
    st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    
    /* Root Variables */
    :root {
        --primary-color: #6366f1;
        --primary-dark: #4f46e5;
        --secondary-color: #8b5cf6;
        --accent-color: #06d6a0;
        --background-dark: #0f0f23;
        --background-light: #1a1a2e;
        --surface: #16213e;
        --text-primary: #ffffff;
        --text-secondary: #a1a1aa;
        --border-color: #374151;
        --success: #10b981;
        --warning: #f59e0b;
        --error: #ef4444;
        --gradient-1: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        --gradient-2: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        --gradient-3: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        --shadow-lg: 0 10px 25px -3px rgba(0, 0, 0, 0.3), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        --shadow-xl: 0 20px 25px -5px rgba(0, 0, 0, 0.4), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
    }
    
    /* Hide Streamlit Default Elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Main App Container */
    .stApp {
        background: var(--background-dark);
        font-family: 'Inter', sans-serif;
    }
    
    /* Animated Background */
    .main-container {
        position: relative;
        min-height: 100vh;
        background: var(--background-dark);
        overflow: hidden;
    }
    
    .main-container::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: 
            radial-gradient(circle at 20% 80%, rgba(120, 119, 198, 0.3) 0%, transparent 50%),
            radial-gradient(circle at 80% 20%, rgba(255, 119, 198, 0.3) 0%, transparent 50%),
            radial-gradient(circle at 40% 40%, rgba(120, 219, 255, 0.2) 0%, transparent 50%);
        z-index: -1;
        animation: backgroundShift 20s ease-in-out infinite;
    }
    
    @keyframes backgroundShift {
        0%, 100% { transform: translateX(0px) translateY(0px); }
        33% { transform: translateX(30px) translateY(-30px); }
        66% { transform: translateX(-20px) translateY(20px); }
    }
    
    /* Header Styling */
    .main-header {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.9) 0%, rgba(139, 92, 246, 0.9) 100%);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0 2rem 0;
        box-shadow: var(--shadow-xl);
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent);
        animation: shimmer 3s infinite;
    }
    
    @keyframes shimmer {
        0% { left: -100%; }
        100% { left: 100%; }
    }
    
    .main-title {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #ffffff 0%, #a1a1aa 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 0.5rem;
        animation: titleGlow 2s ease-in-out infinite alternate;
    }
    
    @keyframes titleGlow {
        from { filter: drop-shadow(0 0 5px rgba(255, 255, 255, 0.3)); }
        to { filter: drop-shadow(0 0 20px rgba(255, 255, 255, 0.6)); }
    }
    
    .subtitle {
        text-align: center;
        color: rgba(255, 255, 255, 0.8);
        font-size: 1.2rem;
        font-weight: 400;
    }
    
    /* Sidebar Styling */
    .css-1d391kg {
        background: linear-gradient(180deg, var(--surface) 0%, var(--background-light) 100%);
        border-right: 1px solid var(--border-color);
    }
    
    /* Card Components */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: var(--shadow-lg);
        transition: all 0.3s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-xl);
        border-color: rgba(99, 102, 241, 0.3);
    }
    
    /* Chat Message Styling */
    .stChatMessage {
        background: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        margin: 0.5rem 0;
        padding: 1rem;
        animation: messageSlide 0.5s ease-out;
    }
    
    @keyframes messageSlide {
        from { opacity: 0; transform: translateX(-20px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--primary-dark) 100%);
        border: none;
        border-radius: 12px;
        color: white;
        font-weight: 600;
        padding: 0.75rem 1.5rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.4);
        background: linear-gradient(135deg, var(--primary-dark) 0%, var(--secondary-color) 100%);
    }
    
    /* Input Styling */
    .stTextInput > div > div > input {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        color: white;
        padding: 1rem;
        font-size: 1rem;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: var(--primary-color);
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.2);
    }
    
    /* Chat Input Special Styling */
    .stChatInputContainer {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 0.5rem;
        margin: 1rem 0;
    }
    
    /* Metrics Styling */
    .metric-card {
        background: linear-gradient(135deg, rgba(6, 214, 160, 0.1) 0%, rgba(6, 214, 160, 0.05) 100%);
        border: 1px solid rgba(6, 214, 160, 0.2);
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        background: linear-gradient(135deg, rgba(6, 214, 160, 0.15) 0%, rgba(6, 214, 160, 0.08) 100%);
        transform: scale(1.02);
    }
    
    /* Loading Animation */
    .loading-spinner {
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 3px solid rgba(99, 102, 241, 0.3);
        border-radius: 50%;
        border-top-color: var(--primary-color);
        animation: spin 1s ease-in-out infinite;
    }
    
    @keyframes spin {
        to { transform: rotate(360deg); }
    }
    
    /* Status Indicators */
    .status-online {
        color: var(--success);
        animation: pulse 2s infinite;
    }
    
    .status-offline {
        color: var(--error);
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    /* RAG Context Indicator */
    .rag-context {
        background: linear-gradient(135deg, rgba(6, 214, 160, 0.1) 0%, rgba(6, 214, 160, 0.05) 100%);
        border: 1px solid rgba(6, 214, 160, 0.2);
        border-radius: 8px;
        padding: 0.5rem;
        margin: 0.5rem 0;
        font-size: 0.8rem;
        color: var(--accent-color);
    }
    
    /* Scrollbar Styling */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--background-light);
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, var(--primary-dark), var(--primary-color));
    }
    
    /* Expander Styling */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Chat History Item */
    .chat-history-item {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .chat-history-item:hover {
        background: rgba(255, 255, 255, 0.08);
        border-color: rgba(99, 102, 241, 0.3);
        transform: translateX(4px);
    }
    
    .chat-history-item.active {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.2) 0%, rgba(139, 92, 246, 0.1) 100%);
        border-color: var(--primary-color);
    }
    
    /* Floating Action Button */
    .fab {
        position: fixed;
        bottom: 2rem;
        right: 2rem;
        width: 60px;
        height: 60px;
        background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: var(--shadow-xl);
        z-index: 1000;
        animation: fabFloat 3s ease-in-out infinite;
    }
    
    @keyframes fabFloat {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }
    
    /* Success/Error Messages */
    .stSuccess, .stError, .stInfo, .stWarning {
        border-radius: 12px;
        border: none;
        backdrop-filter: blur(20px);
    }
    
    /* Download Button Special */
    .stDownloadButton > button {
        background: linear-gradient(135deg, var(--accent-color) 0%, #4ade80 100%);
        box-shadow: 0 4px 15px rgba(6, 214, 160, 0.3);
    }
    
    .stDownloadButton > button:hover {
        box-shadow: 0 6px 20px rgba(6, 214, 160, 0.4);
    }
    </style>
    """, unsafe_allow_html=True)

# ---- RAG SYSTEM FUNCTIONS ----

def extract_entities_and_facts(text: str) -> Dict[str, List[str]]:
    """Extract key entities and facts from text for better RAG retrieval"""
    entities = {
        'names': [],
        'facts': [],
        'topics': [],
        'keywords': []
    }
    
    # Simple entity extraction (can be enhanced with NER models)
    words = text.lower().split()
    
    # Extract potential names (capitalized words)
    name_patterns = re.findall(r'\b[A-Z][a-z]+\b', text)
    entities['names'] = list(set(name_patterns))
    
    # Extract facts (sentences with "is", "was", "are", etc.)
    fact_patterns = re.findall(r'[^.!?]*(?:is|was|are|were|am|will be|has|have)[^.!?]*[.!?]', text, re.IGNORECASE)
    entities['facts'] = [fact.strip() for fact in fact_patterns]
    
    # Extract topics/keywords (important nouns)
    important_words = [word for word in words if len(word) > 3 and word not in ['that', 'this', 'with', 'from', 'they', 'have', 'been', 'were', 'said']]
    entities['keywords'] = list(set(important_words))
    
    return entities

def store_conversation_context(chat_id: str, user_message: str, assistant_response: str, message_index: int):
    """Store conversation context in a structured way for better RAG retrieval"""
    
    # Extract entities from both user message and response
    user_entities = extract_entities_and_facts(user_message)
    assistant_entities = extract_entities_and_facts(assistant_response)
    
    # Create comprehensive context document
    context_doc = {
        "_id": f"{chat_id}_context_{message_index}",
        "chat_id": chat_id,
        "message_index": message_index,
        "user_message": user_message,
        "assistant_response": assistant_response,
        "timestamp": time.time(),
        "user_entities": user_entities,
        "assistant_entities": assistant_entities,
        "combined_text": f"User: {user_message}\nAssistant: {assistant_response}",
        "embedding": get_embedding(f"{user_message} {assistant_response}")
    }
    
    # Store in conversation memory
    CONVERSATION_MEMORY.update_one(
        {"_id": context_doc["_id"]},
        {"$set": context_doc},
        upsert=True
    )
    
    # Also store individual embeddings for specific retrieval
    user_doc_id = f"{chat_id}_user_{message_index}"
    assistant_doc_id = f"{chat_id}_assistant_{message_index}"
    
    add_doc_to_mongo(user_doc_id, user_message)
    add_doc_to_mongo(assistant_doc_id, assistant_response)

def retrieve_relevant_context(query: str, chat_id: str = None, k: int = 5) -> Dict[str, any]:
    """Enhanced RAG retrieval that finds relevant context from conversation history"""
    
    query_embedding = np.array(get_embedding(query))
    query_entities = extract_entities_and_facts(query)
    
    relevant_contexts = []
    
    # Search conversation memory
    search_filter = {}
    if chat_id:
        # Prioritize current chat context
        search_filter["chat_id"] = chat_id
    
    for context in CONVERSATION_MEMORY.find(search_filter):
        if "embedding" in context and isinstance(context["embedding"], list):
            context_embedding = np.array(context["embedding"])
            
            # Calculate semantic similarity
            norm_product = np.linalg.norm(query_embedding) * np.linalg.norm(context_embedding)
            semantic_similarity = float(np.dot(query_embedding, context_embedding) / norm_product) if norm_product != 0 else 0
            
            # Calculate entity overlap bonus
            entity_bonus = 0
            for entity_type in ['names', 'keywords']:
                query_entities_set = set(query_entities.get(entity_type, []))
                context_entities_set = set(context.get('user_entities', {}).get(entity_type, []))
                context_entities_set.update(set(context.get('assistant_entities', {}).get(entity_type, [])))
                
                if query_entities_set and context_entities_set:
                    overlap = len(query_entities_set.intersection(context_entities_set))
                    entity_bonus += overlap * 0.1  # Bonus for entity matches
            
            total_score = semantic_similarity + entity_bonus
            
            if total_score > MIN_SIMILARITY_THRESHOLD:
                relevant_contexts.append({
                    "content": context["combined_text"],
                    "user_message": context["user_message"],
                    "assistant_response": context["assistant_response"],
                    "score": total_score,
                    "chat_id": context["chat_id"],
                    "timestamp": context["timestamp"],
                    "entities": {
                        "user": context.get("user_entities", {}),
                        "assistant": context.get("assistant_entities", {})
                    }
                })
    
    # Sort by relevance score
    relevant_contexts.sort(key=lambda x: x["score"], reverse=True)
    
    # Also get recent conversation history from current chat
    recent_context = []
    if chat_id and st.session_state.chat_history:
        recent_messages = st.session_state.chat_history[-RAG_CONTEXT_WINDOW:]
        for i, msg in enumerate(recent_messages):
            recent_context.append({
                "role": msg["role"],
                "content": msg["content"],
                "timestamp": msg.get("timestamp", ""),
                "index": len(st.session_state.chat_history) - len(recent_messages) + i
            })
    
    return {
        "relevant_contexts": relevant_contexts[:k],
        "recent_context": recent_context,
        "query_entities": query_entities
    }

def build_rag_prompt(user_query: str, rag_context: Dict, internet_results: List[Dict] = None) -> str:
    """Build enhanced RAG prompt with all available context"""
    
    prompt_parts = []
    
    # Add system context
    prompt_parts.append("""You are E-GPT, an intelligent AI assistant. You have access to conversation history, stored knowledge, and current information. 
Provide helpful, accurate, and contextual responses based on the available information. Remember details from previous conversations and maintain continuity.""")
    
    # Add relevant conversation context
    if rag_context["relevant_contexts"]:
        prompt_parts.append("\n--- RELEVANT CONVERSATION HISTORY ---")
        for ctx in rag_context["relevant_contexts"]:
            relevance_score = f"(Relevance: {ctx['score']:.2f})"
            prompt_parts.append(f"{ctx['content']} {relevance_score}")
    
    # Add recent conversation context for immediate continuity
    if rag_context["recent_context"]:
        prompt_parts.append("\n--- RECENT CONVERSATION CONTEXT ---")
        for msg in rag_context["recent_context"]:
            prompt_parts.append(f"{msg['role'].title()}: {msg['content']}")
    
    # Add internet search results if available
    if internet_results:
        prompt_parts.append("\n--- CURRENT INFORMATION FROM INTERNET ---")
        for result in internet_results:
            prompt_parts.append(f"Source: {result['title']}\nContent: {result['snippet']}")
    
    # Add user query
    prompt_parts.append(f"\n--- CURRENT USER QUERY ---\nUser: {user_query}")
    
    # Add instruction for response
    prompt_parts.append("""
--- RESPONSE INSTRUCTIONS ---
Based on the conversation history and available information above, provide a helpful and contextual response. 
- Use information from previous conversations when relevant
- Remember user preferences, names, and facts mentioned earlier
- If this is a follow-up question, reference previous context naturally
- Be conversational and maintain continuity with the chat history
- If you find relevant information in the conversation history, use it naturally
- Provide a comprehensive answer using all available sources""")
    
    return "\n".join(prompt_parts)

# ---- INTERNET SEARCH FUNCTIONS ----
def get_search_cache_key(query: str) -> str:
    """Generate cache key for search query"""
    return hashlib.md5(query.lower().encode()).hexdigest()

def is_search_cached(query: str) -> Optional[Dict]:
    """Check if search results are cached and still valid"""
    cache_key = get_search_cache_key(query)
    cached = SEARCH_CACHE.find_one({"_id": cache_key})
    
    if cached and (time.time() - cached["timestamp"]) < CACHE_DURATION:
        return cached["results"]
    return None

def cache_search_results(query: str, results: List[Dict]):
    """Cache search results"""
    cache_key = get_search_cache_key(query)
    SEARCH_CACHE.update_one(
        {"_id": cache_key},
        {
            "$set": {
                "query": query,
                "results": results,
                "timestamp": time.time()
            }
        },
        upsert=True
    )

def search_duckduckgo(query: str) -> List[Dict]:
    """Search using DuckDuckGo Instant Answer API"""
    try:
        params = {
            'q': query,
            'format': 'json',
            'no_html': '1',
            'skip_disambig': '1'
        }
        
        response = requests.get(
            "https://api.duckduckgo.com/",
            params=params,
            timeout=SEARCH_TIMEOUT,
            headers={'User-Agent': 'VTU-GPT/1.0'}
        )
        
        if response.status_code == 200:
            data = response.json()
            results = []
            
            # Get instant answer
            if data.get('AbstractText'):
                results.append({
                    'title': data.get('Heading', 'DuckDuckGo Instant Answer'),
                    'snippet': data['AbstractText'],
                    'url': data.get('AbstractURL', ''),
                    'source': 'DuckDuckGo Instant Answer'
                })
            
            # Get related topics
            for topic in data.get('RelatedTopics', [])[:3]:
                if isinstance(topic, dict) and topic.get('Text'):
                    results.append({
                        'title': topic.get('FirstURL', '').split('/')[-1].replace('_', ' '),
                        'snippet': topic['Text'],
                        'url': topic.get('FirstURL', ''),
                        'source': 'DuckDuckGo'
                    })
            
            return results
            
    except Exception as e:
        st.warning(f"DuckDuckGo search failed: {str(e)}")
    
    return []

def search_web_scraping(query: str) -> List[Dict]:
    """Fallback web scraping search"""
    try:
        # Use DuckDuckGo HTML search as fallback
        search_url = f"https://duckduckgo.com/html/?q={quote_plus(query)}"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(search_url, headers=headers, timeout=SEARCH_TIMEOUT)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            results = []
            
            # Find search result elements
            result_elements = soup.find_all('div', class_='result')[:MAX_SEARCH_RESULTS]
            
            for element in result_elements:
                title_elem = element.find('a', class_='result__a')
                snippet_elem = element.find('a', class_='result__snippet')
                
                if title_elem and snippet_elem:
                    results.append({
                        'title': title_elem.get_text().strip(),
                        'snippet': snippet_elem.get_text().strip(),
                        'url': title_elem.get('href', ''),
                        'source': 'Web Search'
                    })
            
            return results
            
    except Exception as e:
        st.warning(f"Web scraping search failed: {str(e)}")
    
    return []

def search_internet(query: str) -> List[Dict]:
    """Main internet search function with multiple fallbacks"""
    
    # Check cache first
    cached_results = is_search_cached(query)
    if cached_results:
        return cached_results
    
    results = []
    
    # Try DuckDuckGo API first
    duckduckgo_results = search_duckduckgo(query)
    results.extend(duckduckgo_results)
    
    # If no results, try web scraping
    if not results:
        scraping_results = search_web_scraping(query)
        results.extend(scraping_results)
    
    # Cache results if we got any
    if results:
        cache_search_results(query, results)
    
    return results[:MAX_SEARCH_RESULTS]

def determine_search_need(query: str, rag_context: Dict) -> bool:
    """Determine if internet search is needed based on query and available context"""
    
    # Keywords that indicate current/recent information needed
    current_keywords = [
        'today', 'now', 'current', 'latest', 'recent', 'news', 'weather',
        'stock', 'price', 'update', '2024', '2025', 'this year', 'this month'
    ]
    
    query_lower = query.lower()
    
    # Check for current information needs
    needs_current_info = any(keyword in query_lower for keyword in current_keywords)
    
    # Check if we have sufficient context from RAG
    has_sufficient_context = (
        len(rag_context.get("relevant_contexts", [])) > 0 or
        len(rag_context.get("recent_context", [])) > 2
    )
    
    # Check if query seems to be asking for factual information we might not have
    factual_keywords = ['what is', 'who is', 'when did', 'where is', 'how does', 'explain']
    is_factual_query = any(keyword in query_lower for keyword in factual_keywords)
    
    # Search if: needs current info, OR (is factual AND no sufficient context)
    return needs_current_info or (is_factual_query and not has_sufficient_context)

# ---- ENHANCED RAG SYSTEM ----
def rag_generate_response(user_query: str, chat_id: str) -> str:
    """Main RAG function that retrieves context and generates response"""
    
    try:
        # Step 1: Retrieve relevant context using RAG
        rag_context = retrieve_relevant_context(user_query, chat_id, k=5)
        
        # Step 2: Determine if internet search is needed
        internet_results = []
        search_performed = False
        
        if st.session_state.get('enable_internet_search', True):
            if determine_search_need(user_query, rag_context):
                search_performed = True
                internet_results = search_internet(user_query)
        
        # Step 3: Build comprehensive RAG prompt
        rag_prompt = build_rag_prompt(user_query, rag_context, internet_results)
        
        # Step 4: Generate response using the model
        current_model = st.session_state.get('current_model', 'llama3.2:3b')
        response = get_ollama_response(rag_prompt, current_model)
        
        # Step 5: Add metadata about sources used
        sources_info = {
            "rag_contexts_found": len(rag_context.get("relevant_contexts", [])),
            "recent_context_used": len(rag_context.get("recent_context", [])),
            "internet_search_performed": search_performed,
            "internet_results_found": len(internet_results)
        }
        
        # Step 6: Store this conversation pair for future RAG retrieval
        if response and not response.startswith("Error"):
            store_conversation_context(
                chat_id, 
                user_query, 
                response, 
                len(st.session_state.chat_history)
            )
        
        return response, sources_info
        
    except Exception as e:
        return f"Error in RAG system: {str(e)}", {}

# ---- UTILS ----
def get_embedding(text: str) -> List[float]:
    """Generate embeddings for text"""
    return EMB.encode([text])[0].tolist()

def safe_rerun():
    """Safe rerun function with fallbacks"""
    try:
        st.rerun()
    except AttributeError:
        try:
            st.experimental_rerun()
        except AttributeError:
            st.session_state._rerun_requested = True

# ---- ENHANCED DOCUMENT STORAGE ----
def add_doc_to_mongo(doc_id: str, text: str):
    """Add document to MongoDB with embeddings"""
    vec = get_embedding(text)
    DOCS.update_one(
        {"_id": doc_id},
        {"$set": {"content": text, "vector": vec, "timestamp": time.time()}},
        upsert=True
    )

def search_similar_docs(query: str, k=3) -> List[Dict]:
    """Enhanced similarity search with better scoring"""
    q_vec = np.array(get_embedding(query))
    results = []
    
    for doc in DOCS.find({}, {"content": 1, "vector": 1, "timestamp": 1}):
        if "vector" in doc and isinstance(doc["vector"], list):
            d_vec = np.array(doc["vector"])
            norm_prod = np.linalg.norm(q_vec) * np.linalg.norm(d_vec)
            sim = float(np.dot(q_vec, d_vec) / norm_prod) if norm_prod != 0 else 0
            
            # Add recency bonus
            recency_bonus = 0
            if "timestamp" in doc:
                age_hours = (time.time() - doc["timestamp"]) / 3600
                recency_bonus = max(0, 0.1 * (1 - age_hours / 168))  # Bonus decreases over a week
            
            total_score = sim + recency_bonus
            
            results.append({
                "content": doc["content"], 
                "score": total_score,
                "semantic_similarity": sim,
                "recency_bonus": recency_bonus
            })
    
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:k]

# ---- Ollama Connection ----
def get_ollama_response(prompt: str, model: str = "llama3.2:3b") -> str:
    """Get response from Ollama model"""
    try:
        r = subprocess.run(
            [OLLAMA_PATH, 'run', model],
            input=prompt,
            text=True,
            encoding="utf-8",
            capture_output=True,
            timeout=120
        )
        return r.stdout.strip() if r.returncode == 0 else f"Error: {r.stderr.strip()}"
    except subprocess.TimeoutExpired:
        return "Error: Request timed out."
    except FileNotFoundError:
        return "Error: Ollama executable not found."
    except Exception as e:
        return f"Error: {str(e)}"

def check_model_availability(m: str) -> bool:
    """Check if model is available in Ollama"""
    try:
        list_out = subprocess.run(
            [OLLAMA_PATH, 'list'],
            capture_output=True,
            text=True,
            encoding="utf-8"
        )
        return m in list_out.stdout
    except Exception:
        return False

# ---- UI COMPONENTS ----
def render_header():
    st.markdown("""
    <div class="main-header">
        <div class="main-title">🤖 VTU-GPT RAG</div>
        <div class="subtitle">✨ Advanced RAG-Powered AI Assistant ✨</div>
    </div>
    """, unsafe_allow_html=True)

def render_status_indicator(is_online: bool = True):
    status_class = "status-online" if is_online else "status-offline"
    status_text = "🟢 Online" if is_online else "🔴 Offline"
    return f'<span class="{status_class}">{status_text}</span>'

def render_metric_card(title: str, value: str, icon: str = "📊"):
    st.markdown(f"""
    <div class="metric-card">
        <div style="font-size: 2rem; margin-bottom: 0.5rem;">{icon}</div>
        <div style="font-size: 1.5rem; font-weight: 600; color: white;">{value}</div>
        <div style="color: var(--text-secondary); font-size: 0.9rem;">{title}</div>
    </div>
    """, unsafe_allow_html=True)

def render_rag_context_indicator(sources_info: Dict):
    """Show which sources were used for the response"""
    indicators = []
    
    if sources_info.get("rag_contexts_found", 0) > 0:
        indicators.append(f"🧠 RAG: {sources_info['rag_contexts_found']} contexts")
    
    if sources_info.get("recent_context_used", 0) > 0:
        indicators.append(f"💬 Recent: {sources_info['recent_context_used']} messages")
    
    if sources_info.get("internet_search_performed", False):
        indicators.append(f"🌐 Web: {sources_info.get('internet_results_found', 0)} results")
    
    if indicators:
        st.markdown(f"""
        <div class="rag-context">
            📡 Sources: {' • '.join(indicators)}
        </div>
        """, unsafe_allow_html=True)

# ---- Chat Functions ----
def clear_chat_history():
    st.session_state.chat_history = []
    st.success("🧹 Chat cleared successfully!")
    time.sleep(1)
    safe_rerun()

def create_new_chat():
    if st.session_state.chat_history and st.session_state.current_chat_id:
        save_current_chat()
    st.session_state.chat_counter += 1
    st.session_state.current_chat_id = f"chat_{st.session_state.chat_counter}"
    st.session_state.chat_history = []
    st.success("✨ New chat created!")
    time.sleep(1)
    safe_rerun()

def save_current_chat():
    if st.session_state.chat_history and st.session_state.current_chat_id:
        t = "New Chat"
        for m in st.session_state.chat_history:
            if m["role"] == "user":
                t = m["content"][:50] + ("..." if len(m["content"]) > 50 else "")
                break
        doc = {
            "title": t,
            "messages": st.session_state.chat_history.copy(),
            "model": st.session_state.current_model,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "last_updated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "cid": st.session_state.current_chat_id,
        }
        st.session_state.all_chats[st.session_state.current_chat_id] = doc
        MONGO.chats.update_one({"cid": doc["cid"]}, {"$set": doc}, upsert=True)

def load_chat(cid: str):
    if cid in st.session_state.all_chats:
        if st.session_state.chat_history and st.session_state.current_chat_id:
            save_current_chat()
        d = st.session_state.all_chats[cid]
        st.session_state.current_chat_id = cid
        st.session_state.chat_history = d["messages"].copy()
        st.session_state.current_model = d["model"]
        safe_rerun()
    else:
        d = MONGO.chats.find_one({"cid": cid})
        if d:
            st.session_state.all_chats[cid] = d
            st.session_state.current_chat_id = cid
            st.session_state.chat_history = d["messages"].copy()
            st.session_state.current_model = d["model"]
            safe_rerun()

def delete_chat(cid: str):
    if cid in st.session_state.all_chats:
        del st.session_state.all_chats[cid]
    MONGO.chats.delete_one({"cid": cid})
    # Also clean up conversation memory for this chat
    CONVERSATION_MEMORY.delete_many({"chat_id": cid})
    if st.session_state.current_chat_id == cid:
        create_new_chat()
    else:
        st.success("🗑️ Chat deleted successfully!")
        time.sleep(1)
        safe_rerun()

def export_chat_history():
    if st.session_state.chat_history:
        chat_data = {
            "model": st.session_state.current_model,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "messages": st.session_state.chat_history
        }
        return json.dumps(chat_data, indent=2)

# ---- MAIN APP ----
def main():
    # Load Custom CSS - MOVED here to ensure set_page_config is first
    load_css()
    
    # Render Header
    render_header()

    with st.sidebar:
            st.markdown("### 🧭 Navigation")
            section = st.radio(
                "Choose Section:",
                ["💬 Chat Assistant", "📄 PDF Quiz Generator","❓ Question Generator","🧩PDF Analyzer Q&A", "📝 Report Generator"],
                index=0
            )
            st.markdown("---")

            if section == "💬 Chat Assistant":
                st.markdown("### ⚙️ RAG Control Panel")
                # keep your RAG control panel code here unchanged

            elif section == "📄 PDF Quiz Generator":
                st.markdown("### 📄 PDF Quiz Generator")
                st.info("Upload a PDF and generate quizzes using LLaMA.")

            elif section == "❓ Question Generator":
                st.markdown("### 🧠 Question Paper Generator")
                st.info("Use LLaMA to automatically generate question papers from syllabus.")
            
            elif section == "🧩PDF Analyzer Q&A":
                st.markdown("### 🧩 PDF Analyzer")
                st.info("Use LLaMA to automatically generate PDFs.")
                
            elif section == "📝 Report Generator":
                st.markdown("### 📝 Report Generator")
                st.info("Generate technical articles with code examples using multi-agent system.")

    if section == "📄 PDF Quiz Generator":
        import app
        app.run_app()
        return  # skip the chat UI when quiz generator is active
    elif section == "❓ Question Generator":        
        import sid
        sid.run_sid()
        return

    elif section == "🧩PDF Analyzer Q&A":
        runpy.run_module("app_pdf")
        return
        
    elif section == "📝 Report Generator":
        # Import and run the report generator wrapper
        from report_generator_wrapper import run_report_generator
        run_report_generator()
        return

    # Sidebar Controls
    with st.sidebar:
        #st.markdown("### ⚙️ RAG Control Panel")
        
        # Status indicator
        st.markdown(f"**Status:** {render_status_indicator(True)}", unsafe_allow_html=True)
        
        # RAG Settings
        st.markdown("### 🧠 RAG Settings")
        
        # Context window size
        context_window = st.slider(
            "📖 Context Window Size",
            min_value=5,
            max_value=20,
            value=RAG_CONTEXT_WINDOW,
            help="Number of recent messages to include in context"
        )
        
        # Similarity threshold
        similarity_threshold = st.slider(
            "🎯 Similarity Threshold",
            min_value=0.1,
            max_value=0.8,
            value=MIN_SIMILARITY_THRESHOLD,
            step=0.1,
            help="Minimum similarity score for relevant context"
        )
        
        # Internet search toggle
        enable_internet_search = st.checkbox(
            "🌐 Enable Internet Search",
            value=st.session_state.get('enable_internet_search', True),
            help="Automatically search internet for current information"
        )
        st.session_state.enable_internet_search = enable_internet_search
        
        # Show RAG debug info
        show_rag_debug = st.checkbox(
            "🔍 Show RAG Debug Info",
            value=st.session_state.get('show_rag_debug', False),
            help="Display detailed RAG retrieval information"
        )
        st.session_state.show_rag_debug = show_rag_debug

        st.markdown("---")
        
        # Chat Controls
        st.markdown("### 🎮 Chat Controls")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🧹 Clear", use_container_width=True, help="Clear current chat"):
                clear_chat_history()
        
        with col2:
            if st.button("✨ New", use_container_width=True, help="Start new chat"):
                create_new_chat()

        st.markdown("---")
        
        # Model Selection
        st.markdown("### 🤖 AI Model")
        available_models = ["llama3.2:3b", "llama2", "codellama", "mistral"]
        current_model = st.session_state.get('current_model', 'llama3.2:3b')
        
        selected_model = st.selectbox(
            "Choose Model:", 
            available_models, 
            index=available_models.index(current_model) if current_model in available_models else 0,
            help="Select AI model for conversation"
        )
        
        if selected_model != st.session_state.get('current_model'):
            st.session_state.current_model = selected_model
            st.success(f"🔄 Switched to {selected_model}")

        st.markdown("---")

        # RAG Analytics
        st.markdown("### 📊 RAG Analytics")
        
        # Memory statistics
        total_contexts = CONVERSATION_MEMORY.count_documents({})
        current_chat_contexts = CONVERSATION_MEMORY.count_documents({"chat_id": st.session_state.get('current_chat_id', '')}) if st.session_state.get('current_chat_id') else 0
        
        col1, col2 = st.columns(2)
        with col1:
            render_metric_card("Total Memory", str(total_contexts), "🧠")
        with col2:
            render_metric_card("Current Chat", str(current_chat_contexts), "💭")
        
        # Cache statistics
        cache_count = SEARCH_CACHE.count_documents({})
        recent_cache = SEARCH_CACHE.count_documents({
            "timestamp": {"$gt": time.time() - 3600}
        })
        
        col1, col2 = st.columns(2)
        with col1:
            render_metric_card("Cached Searches", str(cache_count), "💾")
        with col2:
            render_metric_card("Recent Cache", str(recent_cache), "⏰")

        st.markdown("---")

        # Enhanced Chat History
        st.markdown("### 💬 Chat History")
        if st.session_state.chat_history and st.session_state.current_chat_id:
            save_current_chat()
        
        if st.session_state.all_chats:
            # Search functionality
            search_query = st.text_input("🔍 Search chats:", placeholder="Search your chats...")
            
            filtered_chats = st.session_state.all_chats.items()
            if search_query:
                filtered_chats = [
                    (cid, chat) for cid, chat in st.session_state.all_chats.items()
                    if search_query.lower() in chat['title'].lower()
                ]
            
            for cid, d in sorted(filtered_chats, key=lambda x: x[1]["last_updated"], reverse=True):
                is_current = cid == st.session_state.current_chat_id
                
                with st.container():
                    col1, col2 = st.columns([4, 1])
                    
                    with col1:
                        if st.button(
                            f"{'🟢' if is_current else '⚪'} {d['title']}",
                            key=f"load_{cid}",
                            help=f"📅 {d['created_at']}\n🤖 {d['model']}\n💬 {len(d['messages'])} messages",
                            use_container_width=True,
                            disabled=is_current
                        ):
                            load_chat(cid)
                    
                    with col2:
                        if st.button("🗑️", key=f"delete_{cid}", help="Delete chat", use_container_width=True):
                            delete_chat(cid)
                    
                    st.caption(f"📅 {d['last_updated']} | 🤖 {d['model']} | 💬 {len(d['messages'])}")
                    st.markdown("---")
        else:
            st.info("💭 No previous chats available")

        if not st.session_state.current_chat_id:
            create_new_chat()

        st.markdown("---")

        # Export Options
        st.markdown("### 💾 Export Options")
        if st.session_state.chat_history:
            ex = export_chat_history()
            if ex:
                st.download_button(
                    "📄 Export Current Chat",
                    data=ex,
                    file_name=f"vtu_chat_{st.session_state.current_chat_id}_{int(time.time())}.json",
                    mime="application/json",
                    use_container_width=True
                )
        
        if st.session_state.all_chats:
            st.download_button(
                "📦 Export All Chats",
                data=json.dumps({
                    "export_date": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "total_chats": len(st.session_state.all_chats),
                    "chats": st.session_state.all_chats,
                }, indent=2),
                file_name=f"vtu_all_chats_{int(time.time())}.json",
                mime="application/json",
                use_container_width=True
            )

        st.markdown("---")

        # Help & Instructions
        with st.expander("📖 RAG System Guide"):
            st.markdown("""
            **🚀 RAG System Features:**
            
            **🧠 Memory & Context:**
            - Remembers your name, preferences, and facts you mention
            - Maintains conversation continuity across sessions
            - Retrieves relevant information from past conversations
            - Example: "My name is Leo" → Later: "What's my name?" → "Your name is Leo"
            
            **🔍 Multi-Source Intelligence:**
            1. **Conversation Memory**: Searches all your past interactions
            2. **Semantic Search**: Finds contextually relevant information
            3. **Internet Search**: Gets current information when needed
            4. **Smart Synthesis**: Combines all sources intelligently
            
            **⚡ How It Works:**
            1. You ask a question
            2. System searches conversation history for relevant context
            3. Checks if internet search is needed for current info
            4. Combines all information with AI model knowledge
            5. Generates contextual, personalized response
            6. Stores conversation for future reference
            
            **🎯 Example Conversations:**
            - User: "My favorite color is blue"
            - Assistant: "I'll remember that your favorite color is blue!"
            - User: "What's my favorite color?"
            - Assistant: "Your favorite color is blue, as you mentioned earlier."
            
            **💡 Tips:**
            - The system learns from every interaction
            - Ask follow-up questions for better context
            - Reference previous topics naturally
            - Use specific names and details for better memory
            """)

        st.markdown("---")
        st.markdown("### 💙 Made with ❤️ by HKBK")

    # Main Chat Interface
    st.markdown("## 💬 RAG Chat Interface")
    
    # Chat info bar with RAG indicators
    if st.session_state.current_chat_id:
        info = st.session_state.all_chats.get(st.session_state.current_chat_id)
        chat_title = 'New Chat' if not info else info['title']
        created_date = '' if not info else info['created_at']
        
        # Get context count for current chat
        current_chat_contexts = CONVERSATION_MEMORY.count_documents({"chat_id": st.session_state.current_chat_id})
        
        col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
        with col1:
            st.markdown(f"**📄 {chat_title}**")
        with col2:
            st.markdown(f"**🤖 {st.session_state.get('current_model', 'llama3.2:3b')}**")
        with col3:
            st.markdown(f"**🧠 {current_chat_contexts} contexts**")
        with col4:
            if st.session_state.get('is_generating', False):
                st.markdown('<div class="loading-spinner"></div>', unsafe_allow_html=True)
            else:
                st.markdown("**✅ Ready**")

    # Chat messages container
    chat_container = st.container()
    with chat_container:
        if st.session_state.chat_history:
            for i, message in enumerate(st.session_state.chat_history):
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    if message.get("timestamp"):
                        st.caption(f"*{message['timestamp']}*")
                    
                    # Show RAG source information if available and debug mode is on
                    if (st.session_state.get('show_rag_debug', False) and 
                        message["role"] == "assistant" and 
                        message.get("sources_info")):
                        render_rag_context_indicator(message["sources_info"])
        else:
            # Welcome message
            st.markdown("""
            <div class="glass-card" style="text-align: center; padding: 3rem;">
                <div style="font-size: 4rem; margin-bottom: 1rem;">🧠</div>
                <h2 style="color: white; margin-bottom: 1rem;">Welcome to VTU-GPT RAG!</h2>
                <p style="color: var(--text-secondary); font-size: 1.1rem; margin-bottom: 2rem;">
                    I'm your intelligent AI assistant with advanced memory capabilities.<br>
                    I remember our conversations and provide contextual, personalized responses.
                </p>
                <div style="display: flex; justify-content: center; gap: 2rem; flex-wrap: wrap;">
                    <div style="background: rgba(99, 102, 241, 0.1); padding: 1rem; border-radius: 12px; border: 1px solid rgba(99, 102, 241, 0.3);">
                        <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">🧠</div>
                        <div style="color: white; font-weight: 600;">Smart Memory</div>
                        <div style="color: var(--text-secondary); font-size: 0.9rem;">Remembers everything</div>
                    </div>
                    <div style="background: rgba(139, 92, 246, 0.1); padding: 1rem; border-radius: 12px; border: 1px solid rgba(139, 92, 246, 0.3);">
                        <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">🔍</div>
                        <div style="color: white; font-weight: 600;">Context Retrieval</div>
                        <div style="color: var(--text-secondary); font-size: 0.9rem;">Finds relevant info</div>
                    </div>
                    <div style="background: rgba(6, 214, 160, 0.1); padding: 1rem; border-radius: 12px; border: 1px solid rgba(6, 214, 160, 0.3);">
                        <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">🌐</div>
                        <div style="color: white; font-weight: 600;">Web Integration</div>
                        <div style="color: var(--text-secondary); font-size: 0.9rem;">Current information</div>
                    </div>
                </div>
                <div style="margin-top: 2rem; padding: 1rem; background: rgba(6, 214, 160, 0.05); border-radius: 12px; border: 1px solid rgba(6, 214, 160, 0.2);">
                    <div style="color: var(--accent-color); font-weight: 600; margin-bottom: 0.5rem;">💡 Try saying:</div>
                    <div style="color: var(--text-secondary);">"My name is Alex" → then later → "What's my name?"</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Chat input
    user_input = st.chat_input(
        "💬 Ask me anything - I'll remember our conversation!",
        disabled=st.session_state.get('is_generating', False)
    )

    # Process user input with enhanced RAG
    if user_input and not st.session_state.get('is_generating', False):
        if not st.session_state.current_chat_id:
            create_new_chat()

        # Add user message
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        user_message = {
            "role": "user",
            "content": user_input,
            "timestamp": timestamp
        }
        st.session_state.chat_history.append(user_message)

        # Display user message
        with st.chat_message("user"):
            st.markdown(user_input)

        # Generate RAG-enhanced response
        st.session_state.is_generating = True
        
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            status_placeholder = st.empty()
            
            # Show RAG processing status
            status_placeholder.markdown("🧠 **Retrieving context and generating response...**")
            
            with st.spinner("🔍 Searching conversation memory and generating contextual response..."):
                # Generate response using RAG system
                response, sources_info = rag_generate_response(user_input, st.session_state.current_chat_id)
            
            # Clear status and show final response
            status_placeholder.empty()
            message_placeholder.markdown(response)
            
            # Show RAG context if debug mode is on
            if st.session_state.get('show_rag_debug', False):
                render_rag_context_indicator(sources_info)
            
            # Add assistant message with RAG metadata
            assistant_message = {
                "role": "assistant",
                "content": response,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "sources_info": sources_info
            }
            st.session_state.chat_history.append(assistant_message)
            
            save_current_chat()

        st.session_state.is_generating = False
        safe_rerun()


    # Show RAG analytics if available
    if st.session_state.get('show_rag_debug', False) and st.session_state.all_chats:
        st.markdown("---")
        st.markdown("## 🧠 RAG System Analytics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Memory Usage")
            
            # Context retrieval statistics
            total_retrievals = 0
            successful_retrievals = 0
            
            for chat in st.session_state.all_chats.values():
                for message in chat['messages']:
                    if message['role'] == 'assistant' and message.get('sources_info'):
                        total_retrievals += 1
                        if message['sources_info'].get('rag_contexts_found', 0) > 0:
                            successful_retrievals += 1
            
            if total_retrievals > 0:
                success_rate = (successful_retrievals / total_retrievals) * 100
                st.metric("Context Retrieval Success Rate", f"{success_rate:.1f}%")
                
                col1_inner, col2_inner = st.columns(2)
                with col1_inner:
                    render_metric_card("Total Queries", str(total_retrievals), "🔍")
                with col2_inner:
                    render_metric_card("With Context", str(successful_retrievals), "🧠")
        
        with col2:
            st.markdown("### 🌐 Search Analytics")
            
            internet_searches = 0
            for chat in st.session_state.all_chats.values():
                for message in chat['messages']:
                    if (message['role'] == 'assistant' and 
                        message.get('sources_info', {}).get('internet_search_performed', False)):
                        internet_searches += 1
            
            col1_search, col2_search = st.columns(2)
            with col1_search:
                render_metric_card("Internet Searches", str(internet_searches), "🌐")
            with col2_search:
                cache_hits = SEARCH_CACHE.count_documents({})
                render_metric_card("Cache Hits", str(cache_hits), "💾")

    # Footer with RAG system info
    st.markdown("---")
    with st.expander("ℹ️ About RAG System", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **🧠 RAG Features:**
            - Semantic Memory Storage
            - Context-Aware Responses
            - Entity Recognition
            - Conversation Continuity
            - Multi-Chat Memory
            - Intelligent Retrieval
            - Fact Extraction
            - Personalized Responses
            """)
        
        with col2:
            st.markdown("""
            **🔧 RAG Technology:**
            - Sentence Transformers
            - Vector Embeddings
            - Cosine Similarity Search
            - Entity Extraction
            - Conversation Context
            - MongoDB Vector Store
            - Real-time Indexing
            - Smart Caching
            """)
        
        with col3:
            st.markdown("""
            **💡 Memory Examples:**
            - Names and personal info
            - Preferences and settings
            - Previous topics discussed
            - Facts you've shared
            - Project details
            - Learning progress
            - Custom instructions
            - Conversation patterns
            """)

    # Model availability check
    current_model = st.session_state.get('current_model', 'llama3.2:3b')
    if not check_model_availability(current_model):
        st.error(f"""
        ⚠️ **Model Not Available**
        
        The model `{current_model}` is not installed or available.
        
        Please install it using:
        ```bash
        ollama pull {current_model}
        ```
        
        Then refresh the page.
        """)

# Initialize session variables
for k, v in [
    ('chat_history', []),
    ('current_model', "llama3.2:3b"),
    ('is_generating', False),
    ('all_chats', {}),
    ('current_chat_id', None),
    ('chat_counter', 0),
    ('theme_mode', 'dark'),
    ('enable_internet_search', True),
    ('show_rag_debug', False)
]:
    if k not in st.session_state:
        st.session_state[k] = v

if __name__ == "__main__":
    main()
