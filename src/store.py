from __future__ import annotations

from typing import Any, Callable

from .chunking import _dot
from .chunking import compute_similarity
from .embeddings import _mock_embed
from .models import Document


class EmbeddingStore:
    """
    A vector store for text chunks.

    Tries to use ChromaDB if available; falls back to an in-memory store.
    The embedding_fn parameter allows injection of mock embeddings for tests.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        embedding_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        self._embedding_fn = embedding_fn or _mock_embed
        self._collection_name = collection_name
        self._use_chroma = False
        self._store: list[dict[str, Any]] = []
        self._collection = None
        self._next_index = 0

        try:
            import chromadb  # noqa: F401

            self._client = chromadb.Client()
            self._collection = self._client.get_collection(name=collection_name, embedding_function=self._embedding_fn)
            self._use_chroma = True
        except Exception:
            self._use_chroma = False
            self._collection = None

    def _make_record(self, doc: Document) -> dict[str, Any]:
        return {
            "id": doc.id,
            "content": doc.content,
            "embedding": self._embedding_fn(doc.content),
            "metadata": doc.metadata or {},
        }


    def _search_records(self, query: str, records: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        query_embedding = self._embedding_fn(query)
        scored_records = []
        for record in records:
            score = compute_similarity(query_embedding, record["embedding"])
            scored_records.append({**record, "score": score})
        scored_records.sort(key=lambda r: r["score"], reverse=True)
        return scored_records[:top_k]

    def add_documents(self, docs: list[Document]) -> None:
        """
        Embed each document's content and store it.

        For ChromaDB: use collection.add(ids=[...], documents=[...], embeddings=[...])
        For in-memory: append dicts to self._store
        """
        records = [self._make_record(doc) for doc in docs]
        if self._use_chroma and self._collection is not None:
            self._collection.add(
                ids=[record["id"] for record in records],
                documents=[record["content"] for record in records],
                embeddings=[record["embedding"] for record in records],
                metadatas=[record["metadata"] for record in records],
            )
        else:
            self._store.extend(records)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Find the top_k most similar documents to query.

        For in-memory: compute dot product of query embedding vs all stored embeddings.
        """
        if self._use_chroma and self._collection is not None:
            results = self._collection.query(query_texts=[query], n_results=top_k)
            return [
                {"id": id_, "content": doc, "metadata": meta}
                for id_, doc, meta in zip(results["ids"][0], results["documents"][0], results["metadatas"][0])
            ]
        else:
            return self._search_records(query, self._store, top_k)
        

    def get_collection_size(self) -> int:
        """Return the total number of stored chunks."""
        if self._use_chroma and self._collection is not None:
            return self._collection.count()
        else:
            return len(self._store)

    def search_with_filter(self, query: str, top_k: int = 3, metadata_filter: dict = None) -> list[dict]:
        """
        Search with optional metadata pre-filtering.

        First filter stored chunks by metadata_filter, then run similarity search.
        """
        if self._use_chroma and self._collection is not None:
            # Apply metadata filter if provided
            if metadata_filter:
                filter_conditions = {f"metadata.{key}": value for key, value in metadata_filter.items()}
                results = self._collection.query(
                    query_texts=[query],
                    n_results=top_k,
                    where=filter_conditions,
                )
            else:
                results = self._collection.query(query_texts=[query], n_results=top_k)
            return [
                {"id": id_, "content": doc, "metadata": meta}
                for id_, doc, meta in zip(results["ids"][0], results["documents"][0], results["metadatas"][0])
            ]
        else:
            # Apply metadata filter if provided
            if metadata_filter:
                filtered_records = [
                    record for record in self._store
                    if all(record["metadata"].get(key) == value for key, value in metadata_filter.items())
                ]
            else:
                filtered_records = self._store

            return self._search_records(query, filtered_records, top_k)


    def delete_document(self, doc_id: str) -> bool:
            """
            Xoá tất cả các chunks của một document.
            Hỗ trợ xoá theo ID gốc (cho unit test) hoặc theo metadata 'doc_id' / prefix ID (cho code thực tế).
            """
            if self._use_chroma and self._collection is not None:
                try:
                    ids_to_delete = []
                    
                    # Bước 1: Thử tìm theo metadata (dùng cho luồng chạy thực tế trong main.py)
                    try:
                        res_meta = self._collection.get(where={"doc_id": doc_id})
                        if res_meta and res_meta.get("ids"):
                            ids_to_delete.extend(res_meta["ids"])
                    except Exception:
                        pass

                    # Bước 2: Thử tìm chính xác bằng Document ID (dùng cho Unit Test)
                    if not ids_to_delete:
                        res_id = self._collection.get(ids=[doc_id])
                        if res_id and res_id.get("ids"):
                            ids_to_delete.extend(res_id["ids"])

                    # Nếu không tìm thấy bằng cả 2 cách thì báo False
                    if not ids_to_delete:
                        return False

                    # Tiến hành xoá
                    self._collection.delete(ids=ids_to_delete)
                    return True

                except Exception as e:
                    print(f"Lỗi khi xóa từ ChromaDB: {e}")
                    return False

            else:
                # Fallback cho in-memory store
                initial_len = len(self._store)
                
                # Giữ lại những record KHÔNG khớp với bất kỳ điều kiện xoá nào
                self._store = [
                    record for record in self._store
                    if not (
                        record["id"] == doc_id or                      # Pass Unit Test (trùng khớp ID hoàn toàn)
                        record["id"].startswith(f"{doc_id}_") or       # Xử lý ID có chứa index của chunk (vd: nauan_0, nauan_1)
                        record["metadata"].get("doc_id") == doc_id or  # Pass Main App (có lưu trong metadata)
                        record["metadata"].get("source") == doc_id     # Hỗ trợ xoá theo đường dẫn file
                    )
                ]
                
                # Nếu số lượng phần tử giảm đi tức là đã xoá thành công
                return len(self._store) < initial_len