# Agentic RAG System


# Project Setup

1. Use uv to manage dependencies
    * To intialize new project:
    ```
    uv init
    ```

    * To create virtual environment from pyproject.toml file
    ```
    uv sync
    ```


2. Use python 3.12

3. [ruff](https://docs.astral.sh/ruff/) as liner and code formatter.

    ```
    uv add --dev ruff
    ```


# Objective

* API that handle file uploads (.pdf to .txt), extract and chunk the text using either recursive, semtantic or optionally custom chunking logic.
* Generate embeddings and store them in a vector database (Pinecone, Qdrant, Weaviate, or Milvus- FAISS and Chroma are not allowed).\
    2.1 All metadata (e.g., file name, chunking method, embedding model used) should be saved in a relational or NoSQL database.

![Data Pipeline](assets/arch/image.png)

**Tech Stacks:** *ZenML, MongoDB, Qdrant* 

* The second API should implement a RAG-based agentic system using Langchain or Langgraph and must not use the RetrievalQA chain.

Architecture for Agentic RAG [Coming Soon.]

