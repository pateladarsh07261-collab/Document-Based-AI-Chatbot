# 📚 Document-Based AI Chatbot (RAG)

An end-to-end Retrieval-Augmented Generation (RAG) chatbot built with Python, LangChain, ChromaDB, HuggingFace Embeddings, Ollama (`llama3.2`), and Streamlit.

## 🏗️ RAG Architecture & Workflow
1. **Document Ingestion & Cleaning:** Users upload `.pdf` or `.txt` files via Streamlit. Text is extracted using `PyPDFLoader` / `TextLoader`, normalized, and split using `RecursiveCharacterTextSplitter` (configurable `chunk_size` & `chunk_overlap`).
2. **Embeddings & Vector Store:** Cleaned chunks are embedded using `sentence-transformers/all-MiniLM-L6-v2` and persisted locally in `ChromaDB`.
3. **Query Rewriting & Retrieval:** Follow-up queries are contextualized using chat history, converted to embeddings, and matched via similarity search (`k=5`, distance threshold `<= 1.5`).
4. **Grounded Generation:** Retrieved context + conversation history are passed to `llama3.2` with strict grounding rules to prevent hallucination and display source document/page citations.

## 🚀 How to Run Locally
```bash
git clone [https://github.com/pateladarsh07261-collab/Document-Based-AI-Chatbot.git](https://github.com/pateladarsh07261-collab/Document-Based-AI-Chatbot.git)
cd Document-Based-AI-Chatbot
pip install -r requirements.txt
ollama pull llama3.2
Web UI (Recommended): streamlit run app.py
Terminal CLI Version: python ChatBot.py 