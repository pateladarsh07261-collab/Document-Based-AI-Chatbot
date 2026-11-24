# 🚀 Document-Based AI Assistant (Advanced RAG & Multi-Model LLM Workbench)

An upgraded local Retrieval-Augmented Generation (RAG) chatbot built with **Python, LangChain, ChromaDB, HuggingFace Embeddings, Ollama, and Streamlit**. Designed for grounded document question-answering, summarization, and concept explanation with zero external hallucinations.

---

## ✨ Key Features Implemented

### 1. Advanced RAG Pipeline
- **Query Transformation & Expansion:** Rewrites ambiguous follow-up questions using conversation history into standalone search queries and appends related keywords to improve vector retrieval.
- **MMR (Maximum Marginal Relevance) Retrieval:** Uses `max_marginal_relevance_search` (`k=4, fetch_k=12, lambda_mult=0.6`) to retrieve relevant yet diverse document chunks, eliminating redundant context.
- **Basic vs. Advanced RAG Mode Toggle:** Switch live between standard similarity search (`Basic RAG`) and `Query Expansion + MMR` (`Advanced RAG`) to compare retrieval behavior.
- **Dynamic Chunking & Score Filtering:** Configurable `chunk_size`, `chunk_overlap`, `top_k`, and similarity score thresholds.

### 2. Multi-Model Local LLM Workbench (Ollama)
- **Live Model Switching:** Compare multiple local Ollama models on the fly (`llama3.2:latest`, `qwen2.5:3b`, `qwen2.5:1.5b`).
- **Hyperparameter Controls:** Interactive controls for **Temperature** (randomness vs. determinism), **Top-P** (nucleus sampling), and **Max Tokens**.
- **Token-by-Token Streaming:** Real-time response streaming in both the Streamlit UI (`st.write_stream`) and CLI (`chain.stream`) along with inference latency tracking.

### 3. Prompt Engineering & Conversational Memory
- **Task-Specific System Prompts:** Dedicated prompt templates for **Question Answering**, **Summarization**, and **Document Explanation**.
- **Response Length Modes:** Control output verbosity with **Short** (1–2 sentences), **Medium** (1 paragraph), and **Detailed** (structured bullet points) modes.
- **Strict Grounding & Hallucination Control:** Restricts answers strictly to retrieved document context and chat history, safely replying `"Not found in the notes."` when information is unavailable.
- **Dual Memory Architecture:** Multi-session chat history (`chat_history.json`) combined with persistent Q&A long-term memory (`qa_memory.json` + ChromaDB).

---

## 🧠 Core Concepts Explained

- **Basic RAG vs. Advanced RAG:** Basic RAG embeds the raw user query and performs top-$k$ cosine similarity search, which can struggle with follow-up questions or return repetitive chunks. Advanced RAG expands the query with conversational context and applies MMR to balance relevance with diversity.
- **Tokenization, Context Window & Embeddings:** LLMs process text as sub-word **tokens** within a fixed **context window**. Dense vector **embeddings** (`all-MiniLM-L6-v2`) map text chunks into semantic vector space for meaning-based retrieval.
- **LLM Decoding Parameters:** Lower `Temperature` (`0.0–0.1`) ensures factual, deterministic RAG outputs, while higher values increase creativity. `Top-P` limits token selection to the smallest cumulative probability mass $P$.
- **Local Execution via Ollama:** Ollama runs quantized GGUF weights locally using `llama.cpp` on Apple Silicon unified memory, ensuring complete data privacy with zero external API costs.

---

## 📊 RAG Evaluation & Model Benchmark (20 Test Cases)

Evaluated using `evaluate_rag.py` across **15 in-document technical questions** and **5 out-of-scope hallucination checks** (full logs in `EVALUATION_REPORT.md` & `evaluation_results.json`):

| Model | RAG Mode | Answer Relevance (In-Doc) | Faithfulness (No Hallucination) | Avg Latency |
| :--- | :--- | :---: | :---: | :---: |
| `qwen2.5:3b` | **ADVANCED** | **14/15 (93.3%)** | **5/5 (100%)** | ~3.6s |
| `qwen2.5:1.5b` | **ADVANCED** | 12/15 (80.0%) | 4/5 (80%) | ~1.5s |
| `llama3.2:latest` | **ADVANCED** | 6/15 (40.0%) | **5/5 (100%)** | ~2.2s |

### Key Benchmark Takeaways
- **`qwen2.5:3b` (Best Overall):** Achieved **93.3%** grounded answer accuracy with **100%** hallucination prevention.
- **`qwen2.5:1.5b` (Fastest):** Delivered the lowest latency (~1.5s/query) but hallucinated on 1 out-of-scope question, illustrating the trade-off between model size and strict instruction adherence.

---

## 📁 Project Structure

```text
Document-Based-AI-Chatbot/
├── app.py                    # Streamlit Web App (Advanced RAG + Multi-Model + Streaming)
├── ChatBot.py                # CLI Terminal Chatbot with runtime slash commands
├── evaluate_rag.py           # Automated 20-question RAG & LLM benchmark script
├── EVALUATION_REPORT.md      # Markdown performance comparison report
├── evaluation_results.json   # Detailed per-question evaluation logs
├── requirements.txt