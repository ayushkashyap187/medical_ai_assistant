# 🩺 MediBot - Medical AI Assistant

MediBot is an AI-powered medical question-answering chatbot built using **Python, Streamlit, LangChain, FAISS, Hugging Face Embeddings, and Groq**.

The application uses a **Retrieval-Augmented Generation (RAG)** approach to retrieve relevant information from a medical knowledge base and generate answers based on that context.

> ⚠️ **Disclaimer:** MediBot is an educational/research project and should not be used as a replacement for professional medical advice, diagnosis, or treatment.

---

## ✨ Features

- 💬 Interactive medical chatbot interface
- 📚 Uses a medical encyclopedia as the knowledge base
- 🔎 Semantic search using FAISS vector database
- 🧠 Retrieval-Augmented Generation (RAG)
- 🤗 Hugging Face `all-MiniLM-L6-v2` embeddings
- ⚡ Groq-powered LLM responses
- 🖥️ Streamlit web interface
- 💾 Chat history maintained during the session
- 📖 Retrieves relevant source documents for answering questions

---

## 🛠️ Tech Stack

| Technology | Purpose |
|------------|---------|
| Python | Programming language |
| Streamlit | Web application interface |
| LangChain | RAG pipeline and LLM integration |
| FAISS | Vector database / similarity search |
| Hugging Face | Text embeddings |
| Groq | Large Language Model inference |
| PyPDF | Medical PDF document processing |

---

## 🧠 How MediBot Works

MediBot follows a Retrieval-Augmented Generation pipeline:

```text
Medical PDF
     ↓
PDF Document Loader
     ↓
Text Splitting
     ↓
Text Chunks
     ↓
Hugging Face Embeddings
     ↓
FAISS Vector Database
     ↓
User Question
     ↓
Similarity Search
     ↓
Relevant Medical Context
     ↓
Groq LLM
     ↓
Generated Answer
     ↓
Streamlit Chat Interface
