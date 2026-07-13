# Hybrid Retrieval-Based YouTube Video Intelligence System

An AI-powered Retrieval-Augmented Generation (RAG) system that transforms YouTube videos into an intelligent, searchable knowledge base. The application automatically transcribes videos, performs timestamp-aware chunking, generates semantic embeddings, and enables natural language interaction through a local Large Language Model (LLM).

Unlike traditional chatbot implementations, this project focuses on retrieval quality by integrating hybrid search, reranking, citation-grounded responses, and evaluation methodologies to improve answer accuracy and reliability.

---

## Features

- Automatic YouTube video ingestion
- Speech-to-text transcription using Whisper/API
- Timestamp-aware transcript chunking
- Embedding generation and storage using ChromaDB
- Hybrid Retrieval (Semantic + BM25 Keyword Search)
- Reranking for improved context selection
- Local LLM-powered question answering
- Citation-grounded responses with video timestamps
- Multi-video knowledge retrieval
- Conversational query rewriting for follow-up questions
- Voice input and text-to-speech support
- RAG evaluation dashboard
- Local inference with complete data privacy

---

## Architecture

```
YouTube URL(s)
        │
        ▼
Transcript (Whisper / API)
        │
        ▼
Timestamp-Aware Chunking
        │
        ▼
Embedding Generation
        │
        ▼
ChromaDB + BM25 Index
        │
        ▼
Hybrid Retrieval
        │
        ▼
Reranker
        │
        ▼
Local LLM (Qwen)
        │
        ▼
Grounded Answer
        ├── Video Citations
        ├── Timestamps
        └── Voice Output
```

---

## Technology Stack

### AI / Machine Learning
- Retrieval-Augmented Generation (RAG)
- Local LLM (Qwen)
- Sentence Transformers
- Whisper Speech Recognition
- Hybrid Retrieval
- BM25
- Cross-Encoder Reranking

### Backend
- Python
- LangChain
- ChromaDB
- FastAPI / Streamlit

### Supporting Libraries
- yt-dlp
- FFmpeg
- PyTorch
- Hugging Face Transformers

---

## Evaluation

The system is evaluated using a benchmark dataset consisting of multiple YouTube videos and curated question-answer pairs.

Evaluation metrics include:

- Retrieval Hit Rate
- Answer Correctness
- Faithfulness
- Citation Accuracy
- Timestamp Accuracy
- Average Retrieval Latency
- End-to-End Response Time

---

## Key Highlights

- Converts YouTube videos into an interactive knowledge base.
- Supports natural language querying across one or multiple videos.
- Provides grounded answers with transcript-backed citations.
- Uses hybrid retrieval and reranking to improve retrieval accuracy.
- Optimized for fully local inference without external API dependency.

---

## Future Improvements

- Knowledge Graph Integration
- Multimodal Retrieval (OCR + Keyframes)
- Adaptive Chunking
- Agentic RAG Workflow
- Automatic Evaluation Dashboard
- Fine-Tuned Domain-Specific Embedding Models

---

## Project Goals

This project demonstrates the design, implementation, optimization, and evaluation of a production-inspired Retrieval-Augmented Generation system for educational and informational video understanding while maintaining privacy through local inference.
=======
# Hybrid-Retrieval-Based-YouTube-Video-Intelligence-System
