# PDF RAG with DeepSeek

This project implements a Retrieval-Augmented Generation (RAG) system for PDF documents using locally hosted DeepSeek model through Ollama.

## Setup

1. Install Ollama:
```bash
# Install Ollama
curl https://ollama.ai/install.sh | sh

# Pull DeepSeek model
ollama pull deepseek
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start Ollama service:
```bash
ollama serve
```

2. Place your PDF file in the same directory as the script or provide the full path.

3. Run the script:
```bash
python pdf_rag.py
```

4. The script will:
   - Read the PDF
   - Split the text into chunks
   - Create embeddings using DeepSeek
   - Store them in a FAISS vector store
   - Allow you to ask questions about the PDF content

## Features

- Local processing of PDFs
- DeepSeek model served through Ollama
- FAISS for efficient vector storage
- Text chunking for better context handling
- File type validation
- Interactive chat interface

## Requirements

- Python 3.8+
- PyPDF2 for PDF reading
- FAISS for vector storage
- LangChain for RAG implementation
- Ollama for serving the DeepSeek model
