# E-GPT: Multi-Agent LLM Framework for Intelligent Educational Assistance

E-GPT is a **locally deployable, multi-agent educational AI framework** designed to provide intelligent academic assistance using Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG).  
The system distributes educational tasks across **specialized autonomous agents**, improving contextual accuracy, scalability, and data privacy compared to traditional monolithic chatbot systems.

This project is based on our **research paper titled**  
**“E-GPT: A Multi-Agent LLM Framework for Intelligent Educational Assistance”**  
developed at **HKBK College of Engineering, Department of AI & ML**.

---

## 🚀 Key Features

- **Multi-Agent Architecture** for task-specific intelligence
- **Local Deployment** ensuring data privacy and offline capability
- **RAG-based Conversational Assistant** to reduce hallucinations
- **Automated Academic Content Generation**
- **Modular & Scalable Design**
- **Supports CPU & GPU execution**

---

## 🤖 Agents in E-GPT

Each educational task is handled by a dedicated agent:

- **Doubt Clarification Agent**  
  Uses RAG pipelines to answer academic queries contextually.

- **PDF-to-Quiz Generator**  
  Generates MCQs with difficulty distribution (Easy, Medium, Hard).

- **PDF-to-Technical Article Generator**  
  Converts academic PDFs into structured technical articles.

- **Question Paper Generator**  
  Automatically creates complete exam papers with sections.

- **OCR-based PDF Analyzer**  
  Extracts and processes scanned or handwritten documents.

---

## 🧠 Technologies Used

- **LLMs**: LLaMA 3 (3B, 8B), Mistral 7B  
- **Embeddings**: Sentence Transformers (MiniLM-L6-v2)
- **Vector Database**: FAISS + MongoDB
- **Backend**: Python
- **Frontend**: Streamlit (Multi-page UI)
- **OCR**: Tesseract + PyMuPDF

---

## 📁 Project Folder Structure
  
  E-GPT/
  │
  ├── agents/ # Task-specific intelligent agents
  ├── backend/ # Core processing logic
  ├── faiss_index/ # Vector embeddings & indexes
  ├── mainscreen/ # Streamlit UI screens
  ├── output/ # Generated results and reports
  ├── report_generator/ # Academic report generation module
  │
  ├── mainui.py # Application entry point
  ├── requirements.txt # Python dependencies

## ⚙️ Installation & Setup

```bash
git clone https://github.com/Amitdas09/E-GPT-Multi-Agent-LLM-Framework-for-Intelligent-Educational-Assistance.git
cd E-GPT-Multi-Agent-LLM-Framework-for-Intelligent-Educational-Assistance
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run mainui.py
```

---

## 📊 Performance Highlights

- 94% accuracy
- Reduced hallucinations using RAG + agents
- Better scalability than single-model chatbots

---

## 🔒 Privacy & Security

- Fully local execution
- No cloud APIs
- Academic data privacy preserved

---

## 📌 Limitations

- OCR struggles with low-quality PDFs
- CPU-only systems may have higher latency

---

## 🔮 Future Enhancements

- Reinforcement learning for agents
- Syllabus-to-dataset automation
- Model compression
- Personalized learning paths

---

## Output
- Chat UI
  ![chat ui](https://github.com/user-attachments/assets/cb0be064-7fd2-403d-8727-12bf06ec86e6)
- Q&A Output
![question paper](https://github.com/user-attachments/assets/88286315-bb49-4627-8cf6-81bb615447af)



## 👨‍💻 Authors

- Amit Ranjan Das  
- Tabassum Ara  
- Shreyansu Panda  
- Jason Samuel Das  
- Sidharth Vivek Prabhugoankar  

**HKBK College of Engineering**

---

## 📜 License

Academic and research use only.
