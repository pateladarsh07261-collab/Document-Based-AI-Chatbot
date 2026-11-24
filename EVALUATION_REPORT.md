# 📊 RAG Evaluation & Model Performance Comparison Report

Evaluated across **20 test cases** (15 in-document technical questions + 5 out-of-scope hallucination checks).

## Performance Summary Table

| Model | RAG Mode | Retrieval Accuracy | Answer Relevance | Faithfulness (No Hallucination) | Avg Latency (sec) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `llama3.2:latest` | **BASIC** | 14/15 (93.3%) | 8/15 (53.3%) | 5/5 (100.0%) | 1.65s |
| `llama3.2:latest` | **ADVANCED** | 15/15 (100.0%) | 7/15 (46.7%) | 5/5 (100.0%) | 2.23s |
| `qwen2.5:3b` | **ADVANCED** | 15/15 (100.0%) | 14/15 (93.3%) | 5/5 (100.0%) | 3.46s |
| `qwen2.5:1.5b` | **ADVANCED** | 14/15 (93.3%) | 12/15 (80.0%) | 4/5 (80.0%) | 1.58s |
