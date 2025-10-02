### Simple Rag pipeline using Chroma Vector Data base

import os
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
import docx
import ollama


class RAGPipeline:
    def __init__(self,
                 embedding_model="sentence-transformers/all-MiniLM-L6-v2",
                 generator_model="llama3",
                 chunk_size=300,
                 overlap=50,
                 persist_dir="chroma_db"):
        """
        RAG pipeline with ChromaDB for vector store and Ollama as generator
        """
        # 1. Embedding model
        self.embedder = SentenceTransformer(embedding_model)

        # 2. Chroma client (persistent)
        self.client = chromadb.PersistentClient(path=persist_dir)

        self.collection = self.client.get_or_create_collection("documents")

        # 3. Chunking
        self.chunk_size = chunk_size
        self.overlap = overlap

        # 4. Generator (Ollama model name)
        self.generator_model = generator_model

    # -------------------------
    # Document Readers
    # -------------------------
    def read_pdf(self, file_path):
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text

    def read_docx(self, file_path):
        doc = docx.Document(file_path)
        return "\n".join([p.text for p in doc.paragraphs])

    # -------------------------
    # Chunking
    # -------------------------
    def chunk_text(self, text):
        words = text.split()
        chunks = []
        for i in range(0, len(words), self.chunk_size - self.overlap):
            chunk = " ".join(words[i:i+self.chunk_size])
            chunks.append(chunk)
        return chunks

    # -------------------------
    # Ingest Documents
    # -------------------------
    def ingest(self, file_paths):
        texts, metadatas, ids = [], [], []

        for path in file_paths:
            if path.endswith(".pdf"):
                text = self.read_pdf(path)
            elif path.endswith(".docx"):
                text = self.read_docx(path)
            else:
                continue

            chunks = self.chunk_text(text)
            for idx, c in enumerate(chunks):
                texts.append(c)
                metadatas.append({"source": path})
                ids.append(f"{path}_{idx}")

        embeddings = self.embedder.encode(texts).tolist()

        self.collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )

    # -------------------------
    # Search
    # -------------------------
    def search(self, query, top_k=3):
        query_emb = self.embedder.encode([query]).tolist()
        results = self.collection.query(
            query_embeddings=query_emb,
            n_results=top_k,
            include=["documents", "metadatas"]
        )
        return results["documents"][0] if results["documents"] else []

    # -------------------------
    # Answer with Ollama
    # -------------------------
    def answer(self, query, top_k=3, max_tokens=500):
        results = self.search(query, top_k=top_k)
        context = "\n".join(results)
        prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"

        response = ollama.chat(
            model=self.generator_model,
            messages=[{"role": "user", "content": prompt}]
        )

        return response["message"]["content"]


# -------------------------
# Example Usage
# -------------------------
if __name__ == "__main__":
    rag = RAGPipeline(
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        generator_model="llama3.2:latest",
        persist_dir="chroma_store"
    )

    # Ingest documents
    rag.ingest(["/Users/apple/Documents/project/2312.10997v5.pdf"])

    # Ask a question
    query = "What are the key points in this document?"
    print(rag.answer(query))
