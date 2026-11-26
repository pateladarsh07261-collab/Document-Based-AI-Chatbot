# 🧠 Humaeli — Personalized Mental Wellness AI Assistant (RAG + Memory System)

An end-to-end **Context-Aware & Personalized AI Mental Wellness Educational Assistant** built with **Python, LangChain, ChromaDB, BM25, HuggingFace Cross-Encoder, SQLite, Ollama (`qwen2.5:3b`), and Streamlit**.

---

## 🚀 Key Features

### 1. Hybrid Retrieval & 2-Stage Reranking (Day 3)
- **Categorized Knowledge Base (`knowledge_base/`):** Indexes mental wellness guides (`stress_management.txt`, `anxiety_awareness.txt`, `mindfulness_sleep.txt`, `emotional_wellbeing.txt`) in **ChromaDB** with `source` and `category` metadata, plus live PDF/TXT upload support.
- **Hybrid Search (Weighted RRF):** Combines **BM25 Keyword Retrieval (0.4)** and **Semantic Vector Search (0.6)** via Reciprocal Rank Fusion into Top-8 candidate chunks.
- **Cross-Encoder Reranking:** Re-scores `(query, chunk)` pairs using `cross-encoder/ms-marco-MiniLM-L-6-v2` and selects the Top-3 chunks for generation.

### 2. Structured Memory, Multi-User Profiles & Privacy Controls (Day 4)
- **User Profiles & Account Creation (`user_profiles` table):** Supports creating new user accounts (`Create New User` in UI) and editing `age_group`, `preferred_language` (`Hinglish` / `English` / `Hindi`), `interests`, `communication_style`, `past_concerns`, and `conversation_summary`.
- **Short-Term Session Memory (`chat_messages` table):** Maintains strictly isolated chat histories per `(user_id, session_id)` and resolves follow-up pronouns (`it`, `that`, `more`).
- **Automatic Conversation Summarization:** Automatically condenses conversations of 8+ turns into the active user's profile summary.
- **Long-Term Cross-Session Memory (`long_term_memories` table):** Extracts user-shared facts with dense embeddings (`all-MiniLM-L6-v2`) and recalls relevant memories across new sessions via cosine similarity (`score >= 0.28`) with deduplication.
- **User Privacy Controls:** View saved memories, delete individual memories by ID, clear session history, toggle long-term memory ON/OFF, and enforce strict `user_id` data isolation.

---

## 📂 Project Structure

```text
Document-Based AI/
├── app.py                  # Streamlit Web UI (Profiles, Create User, Memory Panel & Hybrid RAG)
├── ChatBot.py              # Terminal CLI Chatbot (/user, /session, /memories, /delmem, /mem on|off)
├── memory_manager.py       # SQLite Backend (user_profiles, chat_messages, long_term_memories)
├── eveluate_mammory.py     # 20-Scenario Automated Evaluation Suite for Memory & Personalization
├── evaluate_rag.py         # 4-Way Retrieval Benchmark Script
├── knowledge_base/         # Mental Wellness Educational .txt Documents
└── requirements.txt        # Python Dependencies