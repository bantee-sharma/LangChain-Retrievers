## 📚 LangChain-Retrievers

This repository contains implementations of various document retrieval techniques using LangChain and vector databases. 
These retrievers were built as part of my learning journey in information retrieval and RAG (Retrieval-Augmented Generation) systems.

*🔍 Included Files*
mmr.py

Implements Maximal Marginal Relevance (MMR) to re-rank search results by balancing relevance and diversity.

multi_ret.py
Implements multi-vector retrieval, enabling retrieval of multiple relevant document chunks for a given query using hybrid or ensemble methods.

vector_ret.py
Implements vector-based retrieval using dense embeddings and vector similarity search with tools like FAISS or ChromaDB.

wiki.py
Prepares Wikipedia data by downloading, chunking, and embedding it for retrieval tasks.

requirements.txt
Lists all required Python packages for setting up and running the retriever pipelines.
