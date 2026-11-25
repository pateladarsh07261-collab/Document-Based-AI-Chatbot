# 🧠 Mental Wellness Educational AI Assistant (Advanced RAG)

An end-to-end **Retrieval-Augmented Generation (RAG)** chatbot built with **Python, LangChain, ChromaDB, BM25, HuggingFace Cross-Encoder, Ollama (`qwen2.5:3b`), and Streamlit**. Designed as a grounded retrieval foundation for an AI Mental Wellness Educational Assistant.

---

## 🏗️ Retrieval & Generation Architecture

1. **Knowledge Base & Metadata Indexing (`knowledge_base/`):**
   - Loads educational `.txt` and `.pdf` documents (`stress_management.txt`, `anxiety_awareness.txt`, `mindfulness_sleep.txt`, `emotional_wellbeing.txt`) using `RecursiveCharacterTextSplitter` (`chunk_size=500`, `chunk_overlap=50`).
   - Attaches structured metadata (`source`, `category`, `page`) to every chunk inside **ChromaDB**.
2. **Context-Aware Memory & Query Rewriting:**
   - Combines sliding short-term conversation turns (`chat_history`) with cosine-similarity retrieved Q&A memory (`qa_memory.json`).
   - Resolves follow-up pronouns (`it`, `that`, `more`) into standalone search queries before retrieval.
3. **Hybrid Search (BM25 + Semantic Vector RRF):**
   - **BM25 Retriever (`weight = 0.4`):** Captures exact domain keywords (e.g., `4-4-4-4`, `4-7-8`, `5-4-3-2-1`).
   - **Chroma Semantic Retriever (`weight = 0.6`):** Captures conceptual meaning using `sentence-transformers/all-MiniLM-L6-v2` with metadata `category` filtering.
   - Fuses candidates via **Weighted Reciprocal Rank Fusion (RRF)** into Top-8 candidate chunks.
4. **2-Stage Cross-Encoder Reranking:**
   - Scores `(query, chunk)` pairs jointly using `cross-encoder/ms-marco-MiniLM-L-6-v2` and passes only the **Top-3 highest-relevance chunks** to `qwen2.5:3b` via Ollama.
5. **Grounded Safety Guardrails:**
   - Strictly answers from retrieved context and refuses medical diagnoses or ungrounded claims (`"Not found in the knowledge base."`).

---

## 📊 4-Way Retrieval Evaluation Report (20 Test Cases)

| Pipeline Mode | Retrieval Relevance | Answer Accuracy | Avg Response Time |
| :--- | :---: | :---: | :---: |
| **Basic RAG** (Semantic `k=3`) | 95.0% (19/20) | 85.0% (17/20) | 2.03s |
| **Advanced RAG** (Rewrite + Semantic `k=3`) | 95.0% (19/20) | 85.0% (17/20) | 2.39s |
| **Hybrid Search** (BM25 + Semantic RRF) | **100.0% (20/20)** | 80.0% (16/20) | **1.92s** |
| **Hybrid + Cross-Encoder Rerank (Day 3)** | **95.0% (19/20)** | **95.0% (19/20)** | 2.22s |

### Key Observations & Improvements
- **Hybrid Search Achieved 100% Recall (20/20):** Combining BM25 keyword matching with Semantic embeddings retrieved the relevant document for every query, though raw RRF without reranking occasionally ranked secondary chunks above the exact answer passage (80.0% answer accuracy).
- **Cross-Encoder Reranking Boosted Answer Accuracy to 95.0% (19/20):** Re-scoring the Top-8 hybrid candidates with `ms-marco-MiniLM-L-6-v2` eliminated noisy chunks and promoted the exact answer passage to Rank 1 with minimal latency overhead (`2.22s`).
- **Hallucination Prevention:** On out-of-context queries (e.g., *"How many hours of sleep do adults need?"*), the assistant safely returned `"Not found in the knowledge base."` instead of fabricating numbers.

---

## 🚀 How to Run Locally

```bash
# 1. Pull local LLM via Ollama
ollama pull qwen2.5:3b

# 2. Run Terminal Chatbot (supports /cat <Category> metadata filter)
python ChatBot.py

# 3. Run Streamlit Web UI
streamlit run app.py

# 4. Run 20-Query 4-Way Evaluation Benchmark
python evaluate_rag.py