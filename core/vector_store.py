import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict
from config import CHROMA_PERSIST_DIR, COLLECTION_NAME, EMBEDDING_MODEL


class VectorStore:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        self.ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self.ef,
        )

    def add_documents(self, documents: List[str], metadatas: List[Dict], ids: List[str]) -> None:
        self.collection.add(documents=documents, metadatas=metadatas, ids=ids)

    def query(self, query_text: str, n_results: int = 5) -> List[Dict]:
        results = self.collection.query(
            query_texts=[query_text],
            n_results=min(n_results, self.collection.count() or 1),
        )
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        return [{"text": d, "metadata": m} for d, m in zip(docs, metas)]

    def count(self) -> int:
        return self.collection.count()

    def reset(self) -> None:
        self.client.delete_collection(COLLECTION_NAME)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self.ef,
        )
