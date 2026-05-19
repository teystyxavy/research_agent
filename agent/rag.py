import io
from datetime import datetime
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


class RAGAgent:
    def __init__(self, model: str, name="knowledge_base", persist_dir="./chroma_db",
                 chunk_size=1000, chunk_overlap=200):
        self.embeddings = GoogleGenerativeAIEmbeddings(model=model)
        self.vector_store = Chroma(
            collection_name=name,
            embedding_function=self.embeddings,
            persist_directory=persist_dir,
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=True,
        )

    def retrieve(self, query: str, k: int = 5):
        return self.vector_store.similarity_search_with_score(query=query, k=k)

    def add_document(self, content: bytes, source: str):
        if source.endswith(".pdf"):
            import pypdf
            reader = pypdf.PdfReader(io.BytesIO(content))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
        else:
            text = content.decode("utf-8")

        splits = self.text_splitter.split_text(text)
        docs = [
            Document(
                page_content=split,
                metadata={"source": source, "timestamp": datetime.now().isoformat()},
            )
            for split in splits
        ]
        if docs:
            self.vector_store.add_documents(docs)
