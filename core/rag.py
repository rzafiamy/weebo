import fitz  # PyMuPDF
import os
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
import config

os.environ["CHROMA_TELEMETRY"] = "False"

# class Config:
#     CHROMA_DB = os.getenv("CHROMA_DB", "chroma_store")  # Use a directory instead of a file
#     EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
#     OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
#     CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
#     CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))
#     MODEL = os.getenv("MODEL", "llama3.2:1B")
# 
# config = Config()

class RAG:
    def __init__(self, chroma_path=config.CHROMA_DB, model=config.EMBEDDING_MODEL, base_url=config.OLLAMA_BASE_URL):
        """
        Initialize the RAG system with ChromaDB and Ollama embeddings.
        """
        self.base_url = base_url
        self.model = model
        self.chroma_path = chroma_path
        self.embedding_model = OllamaEmbeddings(base_url=base_url, model=model)
        self.vectorstore = None

    def extract_text_from_pdf(self, pdf_path):
        """
        Extracts text from a single PDF file.
        """
        doc = fitz.open(pdf_path)
        text = "\n".join([page.get_text("text") for page in doc])
        return text

    def ingest_pdfs(self, pdf_files):
        """
        Ingest one or multiple PDF files into ChromaDB.
        """
        if isinstance(pdf_files, str):
            pdf_files = [pdf_files]
        
        documents = []
        for pdf in pdf_files:
            text = self.extract_text_from_pdf(pdf)
            documents.append(text)
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)
        all_splits = text_splitter.create_documents(documents)
        
        # Store embeddings in ChromaDB
        self.vectorstore = Chroma.from_documents(
            documents=all_splits, 
            embedding=self.embedding_model, 
            persist_directory=self.chroma_path
        )
        print(f"Ingested {len(all_splits)} document chunks into ChromaDB.")
    
    def query(self, question, k=3):
        """
        Query the ChromaDB vector store using similarity search.
        """
        if not self.vectorstore:
            self.vectorstore = Chroma(persist_directory=self.chroma_path, embedding_function=self.embedding_model)
        
        docs = self.vectorstore.similarity_search(question, k=k)
        return [doc.page_content for doc in docs]

    def query_with_llm(self, question, llm_model=config.MODEL, k=3):
        """
        Query using an LLM integrated with the retriever, including similarity scores.
        """
        if not self.vectorstore:
            self.vectorstore = Chroma(persist_directory=self.chroma_path, embedding_function=self.embedding_model)

        # Retrieve documents with similarity scores
        retrieved_docs_with_scores = self.vectorstore.similarity_search_with_score(question, k=k)

        # Format retrieved documents for LLM input
        formatted_docs = [doc[0].page_content for doc in retrieved_docs_with_scores]
        
        llm = Ollama(model=llm_model)
        qa_chain = RetrievalQA.from_chain_type(llm, retriever=self.vectorstore.as_retriever())

        result = qa_chain.invoke({"query": question})

        # Include scores in the response
        return {
            "result": result["result"],
            "retrieved_docs": [
                {"content": doc[0].page_content, "score": doc[1]} for doc in retrieved_docs_with_scores
            ]
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RAG System for PDF Ingestion and Querying")
    parser.add_argument("--ingest", nargs='+', help="List of PDF files to ingest")
    parser.add_argument("--query", type=str, help="Query string for similarity search")
    parser.add_argument("--llm", action="store_true", help="Use LLM for query retrieval")
    
    args = parser.parse_args()
    rag = RAG()
    
    if args.ingest:
        rag.ingest_pdfs(args.ingest)
    
    if args.query:
        if args.llm:
            response = rag.query_with_llm(args.query)
        else:
            response = rag.query(args.query)
        print(response)