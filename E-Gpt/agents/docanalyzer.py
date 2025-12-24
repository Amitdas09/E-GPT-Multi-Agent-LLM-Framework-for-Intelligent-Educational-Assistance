import streamlit as st
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
import io
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple
import requests
from duckduckgo_search import DDGS
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Initialize embedding model
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Llama 3.2 Integration
def query_llama(prompt: str, max_tokens: int = 1000, timeout: int = 30) -> str:
    """Query Llama 3.2 via Ollama API with shorter timeout"""
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3.2:3b",
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": 0.7
                }
            },
            timeout=timeout
        )
        if response.status_code == 200:
            return response.json().get("response", "")
        else:
            return f"Error: Unable to connect to Llama 3.2 (Status {response.status_code})"
    except Exception as e:
        return f"Error connecting to Llama 3.2: {str(e)}"

# Optimized Image Analysis - Batched and Cached
def analyze_image_with_llama(image: Image.Image, source_name: str = "image", quick_mode: bool = True) -> str:
    """Analyze image content using OCR (skip Llama in quick mode)"""
    try:
        # Quick OCR extraction
        ocr_text = pytesseract.image_to_string(image, config='--psm 6')  # Assume uniform text block
        
        if quick_mode:
            # Fast mode: Just return OCR text without Llama analysis
            if ocr_text.strip():
                return f"Image content: {ocr_text.strip()[:200]}"
            else:
                return "Image (no text detected)"
        
        # Full analysis mode (slower)
        width, height = image.size
        mode = image.mode
        ocr_section = f"Text found: {ocr_text.strip()}" if ocr_text.strip() else "No text detected."
        
        prompt = f"""Image from '{source_name}' ({width}x{height}px, {mode}). {ocr_section}
Brief description (1 sentence):"""
        
        description = query_llama(prompt, max_tokens=50, timeout=15)
        return description if description else f"Image: {ocr_text[:100]}"
        
    except Exception as e:
        return f"Image from {source_name}"

# OPTIMIZED PDF Extraction
def extract_text_from_pdf(file_bytes: bytes, filename: str, progress_callback=None) -> Tuple[str, List[Dict]]:
    """FAST PDF extraction with smart OCR and parallel processing"""
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        text_parts = []
        images_info = []
        total_pages = len(doc)
        
        # Progress tracking
        if progress_callback:
            progress_callback(0, total_pages)
        
        # Process pages in parallel batches
        def process_page(page_data):
            page_num, page = page_data
            page_text = page.get_text()
            page_result = {'text': '', 'images': []}
            
            # Smart OCR detection - only if page is truly empty or scanned
            if len(page_text.strip()) < 20:  # Reduced threshold for speed
                # Quick check: if page has no text but has images, likely scanned
                if page.get_images():
                    # Use lower resolution for speed (1.5x instead of 2x)
                    pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
                    img_data = pix.tobytes("png")
                    img = Image.open(io.BytesIO(img_data))
                    
                    # Fast OCR with optimized config
                    ocr_text = pytesseract.image_to_string(img, config='--psm 3 --oem 1')
                    page_result['text'] = f"\n[Page {page_num + 1} - OCR]\n{ocr_text}\n"
                else:
                    page_result['text'] = f"\n[Page {page_num + 1}]\n{page_text}\n"
            else:
                page_result['text'] = f"\n[Page {page_num + 1}]\n{page_text}\n"
            
            # Process images in quick mode (skip Llama analysis for speed)
            image_list = page.get_images()
            for img_index, img in enumerate(image_list[:3]):  # Limit to 3 images per page
                try:
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    
                    # Quick analysis without Llama
                    img_pil = Image.open(io.BytesIO(image_bytes))
                    img_description = analyze_image_with_llama(img_pil, f"{filename} p{page_num + 1}", quick_mode=True)
                    
                    page_result['images'].append({
                        'page': page_num + 1,
                        'index': img_index + 1,
                        'description': img_description
                    })
                    
                    page_result['text'] += f"\n[Image on Page {page_num + 1}, #{img_index + 1}]: {img_description}\n"
                except:
                    pass  # Skip problematic images silently for speed
            
            return page_num, page_result
        
        # Parallel processing with ThreadPoolExecutor
        page_data = [(i, page) for i, page in enumerate(doc)]
        
        # Process in batches to avoid memory issues
        batch_size = 5
        processed_pages = {}
        
        for batch_start in range(0, total_pages, batch_size):
            batch_end = min(batch_start + batch_size, total_pages)
            batch = page_data[batch_start:batch_end]
            
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = {executor.submit(process_page, pd): pd[0] for pd in batch}
                
                for future in as_completed(futures):
                    try:
                        page_num, result = future.result(timeout=30)
                        processed_pages[page_num] = result
                    except Exception as e:
                        page_num = futures[future]
                        processed_pages[page_num] = {'text': f"\n[Page {page_num + 1} - Error processing]\n", 'images': []}
            
            # Update progress
            if progress_callback:
                progress_callback(batch_end, total_pages)
        
        # Assemble results in order
        for page_num in range(total_pages):
            if page_num in processed_pages:
                result = processed_pages[page_num]
                text_parts.append(result['text'])
                images_info.extend(result['images'])
        
        doc.close()
        final_text = ''.join(text_parts).strip()
        
        return final_text, images_info
    except Exception as e:
        st.error(f"PDF extraction error: {str(e)}")
        return "", []

# Image File Processing
def extract_text_from_image(image_bytes: bytes, filename: str) -> Tuple[str, List[Dict]]:
    """Fast image extraction with OCR"""
    try:
        image = Image.open(io.BytesIO(image_bytes))
        
        # Fast OCR with optimized settings
        ocr_text = pytesseract.image_to_string(image, config='--psm 3 --oem 1')
        
        # Skip Llama analysis in quick mode
        text = f"[Image File: {filename}]\n\n"
        if ocr_text.strip():
            text += f"Extracted Text:\n{ocr_text}\n"
        else:
            text += "No text detected.\n"
        
        images_info = [{
            'page': 1,
            'index': 1,
            'description': f"Image file: {ocr_text[:100] if ocr_text.strip() else 'No text'}"
        }]
        
        return text.strip(), images_info
    except Exception as e:
        st.error(f"Image extraction error: {str(e)}")
        return "", []

def generate_summary(text: str, max_length: int = 200) -> str:
    """Generate summary using Llama 3.2"""
    prompt = f"""Provide a concise summary (max {max_length} words) of the following document:

{text[:3000]}

Summary:"""
    return query_llama(prompt, max_tokens=300, timeout=30)

# Text Chunking - Optimized
def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> List[str]:
    """Split text into overlapping chunks - optimized for speed"""
    if len(text) < chunk_size:
        return [text] if text.strip() else []
    
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end]
        
        if chunk.strip():
            chunks.append(chunk)
        
        start += chunk_size - overlap
        
        # Limit chunks for very large documents
        if len(chunks) >= 100:
            break
    
    return chunks

# Session-based FAISS Manager
class SessionFAISSManager:
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.id_map = {}
        self.doc_faiss_ids = {}
        self.next_id = 0
    
    def add_embeddings(self, doc_id: str, chunks: List[str], embeddings: np.ndarray):
        """Add embeddings to FAISS index"""
        start_id = self.next_id
        self.index.add(embeddings)
        
        faiss_ids = []
        for i, chunk in enumerate(chunks):
            faiss_id = start_id + i
            self.id_map[faiss_id] = {
                'doc_id': doc_id,
                'chunk_id': i,
                'chunk_text': chunk
            }
            faiss_ids.append(faiss_id)
        
        self.doc_faiss_ids[doc_id] = faiss_ids
        self.next_id += len(chunks)
    
    def search(self, query_embedding: np.ndarray, doc_id: str, k: int = 5) -> List[Dict]:
        """Search FAISS index for similar chunks in specific document"""
        if self.index.ntotal == 0 or doc_id not in self.doc_faiss_ids:
            return []
        
        doc_faiss_ids = self.doc_faiss_ids[doc_id]
        if not doc_faiss_ids:
            return []
        
        # Get embeddings for this document only
        doc_embeddings = []
        valid_ids = []
        for fid in doc_faiss_ids:
            if fid < self.index.ntotal:
                doc_embeddings.append(self.index.reconstruct(int(fid)))
                valid_ids.append(fid)
        
        if not doc_embeddings:
            return []
        
        # Create temporary index for this document
        temp_index = faiss.IndexFlatL2(self.dimension)
        temp_index.add(np.array(doc_embeddings).astype('float32'))
        
        k = min(k, len(doc_embeddings))
        distances, indices = temp_index.search(query_embedding, k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            actual_fid = valid_ids[idx]
            if actual_fid in self.id_map:
                result = self.id_map[actual_fid].copy()
                result['distance'] = float(dist)
                results.append(result)
        
        return results

# Web Search
def search_web(query: str, num_results: int = 3) -> List[Dict]:
    """Perform web search using DuckDuckGo"""
    try:
        with DDGS() as ddgs:
            results = []
            for r in ddgs.text(query, max_results=num_results):
                results.append({
                    'title': r.get('title', ''),
                    'snippet': r.get('body', ''),
                    'url': r.get('href', '')
                })
            return results
    except Exception as e:
        st.warning(f"Web search error: {e}")
        return []

def summarize_web_results(query: str, results: List[Dict]) -> str:
    """Summarize web search results using Llama 3.2"""
    if not results:
        return ""
    
    context = "\n\n".join([
        f"Source: {r['title']}\n{r['snippet']}"
        for r in results
    ])
    
    prompt = f"""Based on these web search results, provide a concise answer to: "{query}"

{context}

Answer:"""
    
    return query_llama(prompt, max_tokens=500, timeout=30)

# OPTIMIZED Document Processing
def process_document(file, filename: str) -> Dict:
    """Fast document processing with progress tracking"""
    
    # Read file bytes
    file_bytes = file.read()
    file_extension = filename.lower().split('.')[-1]
    
    # Progress tracking
    progress_bar = st.progress(0, text="Starting extraction...")
    start_time = time.time()
    
    def update_progress(current, total):
        progress = int((current / total) * 50)  # First 50% for extraction
        progress_bar.progress(progress, text=f"Extracting: {current}/{total} pages")
    
    # Extract text based on file type
    if file_extension == 'pdf':
        text, images_info = extract_text_from_pdf(file_bytes, filename, update_progress)
        doc_type = "PDF Document"
    elif file_extension in ['png', 'jpg', 'jpeg', 'tiff', 'bmp', 'gif']:
        progress_bar.progress(25, text="Processing image...")
        text, images_info = extract_text_from_image(file_bytes, filename)
        doc_type = "Image File"
    else:
        progress_bar.empty()
        st.error(f"Unsupported file type: {file_extension}")
        return None
    
    if not text:
        progress_bar.empty()
        st.error("Could not extract text from the document.")
        return None
    
    # Generate document ID
    doc_id = hashlib.md5(f"{filename}{datetime.now()}".encode()).hexdigest()
    
    # Generate summary
    progress_bar.progress(60, text="Generating summary...")
    summary = generate_summary(text)
    
    # Chunk text
    progress_bar.progress(70, text="Creating chunks...")
    chunks = chunk_text(text)
    
    # Generate embeddings
    progress_bar.progress(80, text="Creating embeddings...")
    model = load_embedding_model()
    embeddings = model.encode(chunks, show_progress_bar=False)
    
    progress_bar.progress(100, text="Complete!")
    elapsed = time.time() - start_time
    
    time.sleep(0.5)
    progress_bar.empty()
    
    st.success(f"✅ Processed in {elapsed:.1f}s - {len(chunks)} chunks, {len(images_info)} images")
    
    return {
        'id': doc_id,
        'filename': filename,
        'type': doc_type,
        'text': text,
        'summary': summary,
        'chunks': chunks,
        'embeddings': embeddings,
        'images': images_info,
        'uploaded_at': datetime.now(),
        'metadata': {
            'size': len(text),
            'num_chunks': len(chunks),
            'num_images': len(images_info),
            'processing_time': elapsed
        }
    }

# RAG Pipeline
def answer_question(question: str, use_web: bool, doc_data: Dict, faiss_manager: SessionFAISSManager) -> Tuple[str, List[Dict]]:
    """RAG pipeline for question answering"""
    
    model = load_embedding_model()
    doc_id = doc_data['id']
    
    # Generate query embedding
    query_embedding = model.encode([question]).astype('float32')
    
    # Search FAISS
    results = faiss_manager.search(query_embedding, doc_id, k=5)
    
    if not results:
        return "I couldn't find relevant information in the document to answer this question.", []
    
    # Build context from retrieved chunks
    context = "\n\n".join([
        f"[Excerpt {i+1}]:\n{r['chunk_text']}"
        for i, r in enumerate(results)
    ])
    
    # Check confidence
    low_confidence = len(results) < 3 or (results and results[0]['distance'] > 1.5)
    
    web_summary = ""
    if use_web and low_confidence:
        with st.spinner("Searching the web..."):
            web_results = search_web(question)
            if web_results:
                web_summary = summarize_web_results(question, web_results)
    
    # Construct prompt
    web_context_section = f"Additional Web Context:\n{web_summary}\n" if web_summary else ""
    
    prompt = f"""Based on the following document excerpts, answer the question precisely.

Document Context:
{context}

{web_context_section}

Question: {question}

Format your answer with:
- ## headings for key topics
- Bullet points (•) for lists
- **bold** for emphasis
- Short, scannable paragraphs

Answer:"""
    
    answer = query_llama(prompt, max_tokens=800, timeout=45)
    
    # Store query in session state
    if 'query_history' not in st.session_state:
        st.session_state.query_history = []
    
    st.session_state.query_history.append({
        'question': question,
        'answer': answer,
        'document_name': doc_data['filename'],
        'used_web': use_web and bool(web_summary),
        'timestamp': datetime.now()
    })
    
    return answer, results

# Streamlit UI
def main():
    st.set_page_config(
        page_title="DocSense AI - Fast Document Analysis",
        page_icon="📄",
        layout="wide"
    )

    # ✅ SAFE SESSION STATE INITIALIZATION
    for key, default in {
        'documents': {},
        'faiss_manager': None,
        'chat_history': [],
        'selected_doc_id': None,
        'query_history': []
    }.items():
        if key not in st.session_state:
            st.session_state[key] = default

    if st.session_state.faiss_manager is None:
        st.session_state.faiss_manager = SessionFAISSManager()

    # --- HEADER ---
    st.title("📄 DocSense AI - Fast Document Analysis")
    st.markdown("*⚡ Lightning-Fast Document Analysis powered by Llama 3.2*")
    st.caption("🚀 Optimized OCR | Parallel Processing | Multi-format Support")

    # --- SIDEBAR ---
    with st.sidebar:
        st.header("📤 Upload Document")

        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['pdf', 'png', 'jpg', 'jpeg', 'tiff', 'bmp', 'gif'],
            help="Upload PDF, scanned document, or image file"
        )

        if uploaded_file:
            filename = uploaded_file.name

            if filename not in st.session_state.documents:
                doc_data = process_document(uploaded_file, filename)
                if doc_data:
                    st.session_state.documents[filename] = doc_data
                    st.session_state.faiss_manager.add_embeddings(
                        doc_data['id'],
                        doc_data['chunks'],
                        doc_data['embeddings'].astype('float32')
                    )
                    st.session_state.selected_doc_id = doc_data['id']
                    st.rerun()
            else:
                st.info(f"📄 {filename} already loaded")

        st.divider()
        st.header("📚 Select Document")

        if st.session_state.documents:
            doc_names = list(st.session_state.documents.keys())
            selected_filename = st.selectbox("Choose a document", options=doc_names)
            selected_doc = st.session_state.documents[selected_filename]
            st.session_state.selected_doc_id = selected_doc['id']

            proc_time = selected_doc['metadata'].get('processing_time', 0)
            st.info(f"""
**Document:** {selected_doc['filename']}  
**Type:** {selected_doc['type']}  
**Processed:** {proc_time:.1f}s  
**Size:** {selected_doc['metadata']['size']:,} chars  
**Chunks:** {selected_doc['metadata']['num_chunks']}  
**Images:** {selected_doc['metadata']['num_images']}
            """)

            with st.expander("📝 View Summary"):
                st.write(selected_doc['summary'])

            with st.expander("🖼️ View Images Info"):
                if selected_doc['images']:
                    for img in selected_doc['images'][:10]:
                        st.markdown(f"**Page {img['page']}, Image {img['index']}:**")
                        st.write(img['description'])
                        st.divider()
                else:
                    st.write("No images found")

            with st.expander("📄 View Full Text"):
                st.text_area("Document Text", selected_doc['text'][:5000], height=300)
        else:
            st.warning("No documents uploaded yet")
            st.session_state.selected_doc_id = None

        st.divider()
        use_web = st.checkbox(
            "🌐 Use Internet Search",
            value=False,
            help="Enable web search when document info is insufficient"
        )

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()
        with col2:
            if st.button("🔄 Clear All", use_container_width=True):
                st.session_state.documents = {}
                st.session_state.faiss_manager = SessionFAISSManager()
                st.session_state.chat_history = []
                st.session_state.selected_doc_id = None
                st.session_state.query_history = []
                st.success("All data cleared!")
                st.rerun()

    # --- MAIN AREA ---
    if not st.session_state.selected_doc_id:
        st.warning("👆 Please upload a document from the sidebar to start!")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📚 Documents", len(st.session_state.documents))
        with col2:
            st.metric("📊 Total Chunks", st.session_state.faiss_manager.index.ntotal)
        with col3:
            st.metric("❓ Questions Asked", len(st.session_state.query_history))

        st.divider()
        st.info("💡 **Speed Optimizations:** Parallel processing, smart OCR, optimized embeddings")
        st.info("💡 **Tip:** Data is stored temporarily and cleared when you close the tab")
        return

    # --- SELECTED DOCUMENT HANDLING ---
    selected_doc = next(
        (doc for doc in st.session_state.documents.values()
         if doc['id'] == st.session_state.selected_doc_id),
        None
    )
    if not selected_doc:
        st.error("Selected document not found!")
        return

    # --- CHAT & STATS ---
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("💬 Ask Questions")

        # Display previous chat messages
        for chat in st.session_state.chat_history:
            with st.chat_message("user"):
                st.write(chat['question'])
            with st.chat_message("assistant"):
                st.write(chat['answer'])
                if chat.get('sources'):
                    with st.expander("🔎 View Sources"):
                        for i, source in enumerate(chat['sources'][:3]):
                            st.text(f"Source {i+1}:\n{source['chunk_text'][:300]}...")

    with col2:
        st.header("📊 Session Stats")
        st.metric("📄 Current Doc", selected_doc['filename'])
        st.metric("📋 Type", selected_doc['type'])
        st.metric("📊 Chunks", selected_doc['metadata']['num_chunks'])
        st.metric("💬 Messages", len(st.session_state.chat_history))
        st.divider()
        st.subheader("📚 Loaded Docs")
        st.metric("Total", len(st.session_state.documents))
        for doc_name in list(st.session_state.documents.keys())[:5]:
            st.caption(f"📄 {doc_name}")
        st.divider()
        st.subheader("🕐 Recent Questions")
        recent = list(reversed(st.session_state.chat_history[-5:]))
        if recent:
            for i, chat in enumerate(recent):
                st.caption(f"{i+1}. {chat['question'][:40]}...")
        else:
            st.caption("No questions yet")

    # ✅ st.chat_input() OUTSIDE OF COLUMNS
    question = st.chat_input("Ask a question about the document...")

    if question:
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                answer, sources = answer_question(
                    question,
                    use_web,
                    selected_doc,
                    st.session_state.faiss_manager
                )
                st.write(answer)
                if sources:
                    with st.expander("🔎 View Sources"):
                        for i, source in enumerate(sources[:3]):
                            st.text(f"Source {i+1}:\n{source['chunk_text'][:300]}...")

        st.session_state.chat_history.append({
            'question': question,
            'answer': answer,
            'sources': sources
        })
        st.rerun()

if __name__ == main():
    main()

#