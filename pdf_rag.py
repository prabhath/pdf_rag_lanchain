import os
from dotenv import load_dotenv
load_dotenv()
import PyPDF2
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
import magic

class PDFRAG:
    def __init__(self, model_name="deepseek-r1", llm_service="chatgpt"):
        self.model_name = model_name
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vector_store = None
        self.llm_service = llm_service
        if llm_service == "ollama":
            self.llm = OllamaLLM(model=model_name, streaming=True)
        elif llm_service == "chatgpt":
            self.llm = None  # Not used directly
        else:
            raise ValueError(f"Unknown LLM service: {llm_service}")

    def read_pdf(self, pdf_path):
        """Read text from PDF file"""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        mime = magic.Magic(mime=True)
        file_type = mime.from_file(pdf_path)
        if file_type != 'application/pdf':
            raise ValueError(f"File is not a PDF: {file_type}")

        text = ""
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text()
        return text

    @staticmethod
    def get_pdf_list(files_txt_path="files.txt"):
        """Read list of PDF paths from files.txt"""
        if not os.path.exists(files_txt_path):
            return []
        with open(files_txt_path, 'r') as f:
            return [line.strip() for line in f if line.strip()]

    @staticmethod
    def add_pdf_to_list(pdf_path, files_txt_path="files.txt"):
        """Add a PDF path to files.txt if not already present."""
        pdfs = PDFRAG.get_pdf_list(files_txt_path)
        if pdf_path not in pdfs:
            with open(files_txt_path, 'a') as f:
                f.write(pdf_path + '\n')

    def process_text(self, text):
        """Split text into chunks and create embeddings"""
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        return chunks

    def create_vector_store(self, chunks):
        """Create ChromaDB vector store from text chunks"""
        self.vector_store = Chroma.from_texts(
            chunks,
            self.embeddings,
            collection_name="pdf_chunks"
        )

    def query(self, question, stream=False):
        """Query the RAG system, optionally streaming reasoning."""
        if self.vector_store is None:
            raise ValueError("Vector store not initialized. Process text first.")

        if self.llm_service == "ollama":
            retriever = self.vector_store.as_retriever()
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=retriever
            )
            if stream:
                response_chunks = qa_chain.stream({"query": question})
                full = ""
                for chunk in response_chunks:
                    text = chunk.get('result', '')
                    print(text, end='', flush=True)
                    full += text
                print()
                return full
            else:
                return qa_chain.invoke({"query": question})
        elif self.llm_service == "chatgpt":
            # Use OpenAI API directly for RAG
            docs = self.vector_store.similarity_search(question, k=4)
            # Robustly extract text for context
            context = "\n".join(doc.page_content if hasattr(doc, 'page_content') else str(doc) for doc in docs)
            prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
            if stream:
                response = client.chat.completions.create(model="gpt-3.5-turbo",
                messages=[{"role": "system", "content": "You are a helpful assistant that answers questions about the provided context."},
                          {"role": "user", "content": prompt}],
                stream=True)
                full = ""
                for chunk in response:
                    delta = chunk.choices[0].delta
                    if hasattr(delta, 'content'):
                        content = delta.content
                        if content is not None:
                            print(content, end='', flush=True)
                            full += content
                print()
                return full
            else:
                response = client.chat.completions.create(model="gpt-3.5-turbo",
                messages=[{"role": "system", "content": "You are a helpful assistant that answers questions about the provided context."},
                          {"role": "user", "content": prompt}],
                stream=False)
                return response.choices[0].message.content
        else:
            raise ValueError(f"Unknown LLM service: {self.llm_service}")

if __name__ == "__main__":
    print("PDF RAG Chatbot")
    print("================")

    # Check for LLM_SERVICE in environment; if not set, ask the user
    llm_service = os.getenv("LLM_SERVICE")
    if llm_service is not None:
        llm_service = llm_service.strip().lower()
        if llm_service not in ("ollama", "chatgpt"):
            print("Invalid LLM_SERVICE in .env; must be 'ollama' or 'chatgpt'.")
            exit(1)
        print(f"Using LLM service from .env: {llm_service}")
    else:
        while llm_service not in ("ollama", "chatgpt"):
            llm_service = input("Choose LLM service ('ollama' or 'chatgpt'): ").strip().lower()
        print(f"Using LLM service: {llm_service}")

    # Always use files.txt for PDF paths
    pdf_paths = PDFRAG.get_pdf_list()
    if not pdf_paths:
        pdf_path = input("Enter path to a PDF to process: ").strip()
        PDFRAG.add_pdf_to_list(pdf_path)
        pdf_paths = [pdf_path]
    else:
        print(f"Processing PDFs from files.txt: {pdf_paths}")

    try:
        rag = PDFRAG(llm_service=llm_service)
        all_chunks = []
        for pdf_path in pdf_paths:
            print(f"Processing PDF: {pdf_path}")
            text = rag.read_pdf(pdf_path)
            chunks = rag.process_text(text)
            all_chunks.extend(chunks)
        rag.create_vector_store(all_chunks)
        print("All PDFs processed and vector store created!")
    except Exception as e:
        print(f"Error processing PDFs: {str(e)}")
        exit(1)

    while True:
        print("\nType 'exit' to quit")
        question = input("\nYour question: ")
        if question.lower() == 'exit':
            print("Goodbye!")
            break
        answer = ''
        try:
            print("Waiting for LLM to answer your question...")
            answer = rag.query(question, stream=False)
            # Handle both string and dict responses for Ollama/Deepseek
            if isinstance(answer, dict):
                result_text = answer.get('result', '')
                if isinstance(result_text, str) and result_text.strip():
                    print(f"\n{result_text}\n")
            elif isinstance(answer, str) and answer.strip():
                print(f"\n{answer}\n")
        except Exception as e:
            print(f"Error processing question: {str(e)}")
            continue

    # To add a new PDF interactively:
    # new_pdf = input("Enter path to another PDF to add (or leave blank to skip): ").strip()
    # if new_pdf:
    #     PDFRAG.add_pdf_to_list(new_pdf)
    #     # Optionally reprocess all PDFs here.
