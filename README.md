# PDF RAG with LangChain

This project implements a Retrieval-Augmented Generation (RAG) system for PDF documents using LangChain. It supports both local DeepSeek models via Ollama and OpenAI's ChatGPT as language model backends. Chroma is used for efficient vector storage and retrieval, enabling you to query and chat with the content of your PDF files.

## Setup

1. (Only if using Ollama backend) Install Ollama:
   - Required only if you set `LLM_SERVICE=ollama`.
```bash
# Install Ollama
curl https://ollama.ai/install.sh | sh

# Pull DeepSeek model
ollama pull deepseek-r1
```
   - Skip this step if you are using ChatGPT (`LLM_SERVICE=chatgpt`).

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
   - Create a `.env` file in the project root
   - "OPENAI_API_KEY=your_openai_api_key_here"
   - Replace `your_openai_api_key_here` with your actual OpenAI API key

4. Configure PDF files:
   - Create or edit the `files.txt` file in the project root
   - Add the full paths to your PDF files, one per line
   - Example:
     ```
     /path/to/your/first.pdf
     /path/to/your/second.pdf
     ```
   - The application will process all PDFs listed in this file when it starts

## Choosing Your LLM Backend at Runtime

You can select which language model backend to use by setting the `LLM_SERVICE` variable in your `.env` file:

- `LLM_SERVICE=ollama` : Uses the local DeepSeek model served by Ollama. No API key or internet required. **Ollama must be installed and running in this mode.**
- `LLM_SERVICE=chatgpt`: Uses OpenAI's ChatGPT API. Requires a valid `OPENAI_API_KEY` and an internet connection. **Ollama is not required in this mode.**

> **Note:** The Ollama service is required only when using the `ollama` backend. For ChatGPT, you can skip all Ollama-related steps.

If you do not set `LLM_SERVICE`, the application will prompt you to choose one at runtime.

## Usage

1. (Only if using Ollama) Install and start Ollama service:
   - Required only if you set `LLM_SERVICE=ollama` (the default).
```bash
# Install Ollama
curl https://ollama.ai/install.sh | sh

# Pull DeepSeek model
ollama pull deepseek-r1

# Start Ollama service
ollama serve
```
   - Skip this step if you are using ChatGPT (`LLM_SERVICE=chatgpt`).

2. Place your PDF file in the same directory as the script or provide the full path.

3. Run the script:
```bash
python pdf_rag.py
```

4. The script will:
   - Read the PDF
   - Split the text into chunks
   - Create embeddings using a local sentence-transformers model (via HuggingFaceEmbeddings)
   - Store them in a Chroma vector store
   - Allow you to ask questions about the PDF content

**About Embeddings:**

Embeddings are numerical vector representations of text chunks, generated using a local model from the sentence-transformers library (accessed via HuggingFaceEmbeddings). These embeddings capture the semantic meaning of the text and are stored in Chroma for efficient similarity search and retrieval. The language model (DeepSeek or ChatGPT) is then used to answer your questions based on the most relevant retrieved chunks.

## Features

- Local processing of PDFs
- DeepSeek model served through Ollama
- Chroma for efficient vector storage
- Text chunking for better context handling
- File type validation
- Interactive chat interface
- Support for multiple PDFs via files.txt
- Environment variable configuration

## Requirements

- Python 3.9+
- PyPDF2 for PDF reading
- Chroma for vector storage (via langchain-community)
- LangChain for RAG implementation
- Ollama for serving the DeepSeek model
- sentence-transformers for embeddings
- python-dotenv for environment variable management
