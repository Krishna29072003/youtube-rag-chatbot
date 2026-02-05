# 🎥 Video Transcript Q&A using RAG

A GenAI application that allows users to ask questions about a video transcript using
**Retrieval-Augmented Generation (RAG)**.  
The system answers questions **only from the transcript context**, reducing hallucinations
and ensuring accurate, grounded responses.

---

## 🚀 Live Demo
🔗 http://localhost:8501/ 

---

## ❓ Problem Statement
Large Language Models (LLMs) cannot reliably answer questions about **private or recent data**
(such as video transcripts) and may hallucinate information that does not exist in the source.

---

## 💡 Solution
This project uses **Retrieval-Augmented Generation (RAG)** to:
- Split the transcript into meaningful chunks
- Convert chunks into embeddings
- Store embeddings in a vector database (FAISS)
- Retrieve the most relevant chunks for a user question
- Generate answers **only using the retrieved transcript context**

This approach avoids fine-tuning, keeps data private, and supports up-to-date information.

---

## 🛠️ Tech Stack
- Python
- LangChain
- HuggingFace Embeddings (`all-MiniLM-L6-v2`)
- FAISS (Vector Store)
- Groq (LLaMA 3.1 – 8B Instant)
- Streamlit
- dotenv

