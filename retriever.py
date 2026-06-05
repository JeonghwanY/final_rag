"""retriever.py - 하이브리드 검색 + Parent-child 맥락 조립"""

import re
import chromadb
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv
from embed import embed_text

load_dotenv()


def tokenize(text: str) -> list[str]:
    # 1글자 조사는 BM25 점수를 낮추므로 2글자 이상만 추출
    return re.findall(r"[가-힣a-zA-Z0-9]{2,}", text)


class HybridRetriever:

    def __init__(self, chroma_path: str = "./chroma_db"):
        client = chromadb.PersistentClient(path=chroma_path)
        self.collection = client.get_collection("insurance_docs")

        leaf_data = self.collection.get(
            where={"is_leaf": 1},
            include=["documents"],
        )
        self.leaf_ids = leaf_data["ids"]
        self.leaf_docs = leaf_data["documents"]

        self.bm25 = BM25Okapi([tokenize(doc) for doc in self.leaf_docs])
        print(f"[검색기 준비] leaf {len(self.leaf_ids)}개 노드 로드")

    def vector_search(self, query: str, top_k: int = 20) -> list[tuple[str, float]]:
        results = self.collection.query(
            query_embeddings=[embed_text(query)],
            n_results=top_k,
            where={"is_leaf": 1},
            include=["distances"],
        )
        return list(zip(results["ids"][0], results["distances"][0]))

    def bm25_search(self, query: str, top_k: int = 20) -> list[tuple[str, float]]:
        scores = self.bm25.get_scores(tokenize(query))
        return sorted(zip(self.leaf_ids, scores), key=lambda x: x[1], reverse=True)[:top_k]

    def rrf_merge(
        self,
        vector_results: list[tuple[str, float]],
        bm25_results: list[tuple[str, float]],
        k: int = 60,
        top_n: int = 5,
    ) -> list[str]:
        # score = 1/(k+rank_vector) + 1/(k+rank_bm25), k=60은 원 논문 권장값
        scores: dict[str, float] = {}
        for rank, (doc_id, _) in enumerate(vector_results):
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
        for rank, (doc_id, _) in enumerate(bm25_results):
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
        return sorted(scores, key=lambda x: scores[x], reverse=True)[:top_n]

    def get_by_id(self, node_id: str) -> dict | None:
        result = self.collection.get(ids=[node_id], include=["documents", "metadatas"])
        if not result["ids"]:
            return None
        return {"id": result["ids"][0], "content": result["documents"][0], **result["metadatas"][0]}

    def get_ancestors(self, node: dict, seen_ids: set[str]) -> list[dict]:
        """hit 노드에서 L1 루트까지 조상을 [상위→하위] 순으로 반환"""
        ancestors = []
        current = node
        while True:
            parent_id = current.get("parent_id", "")
            if not parent_id or parent_id in seen_ids:
                break
            parent = self.get_by_id(parent_id)
            if not parent:
                break
            seen_ids.add(parent_id)
            ancestors.append(parent)
            current = parent
        return list(reversed(ancestors))

    def search(self, query: str, top_n: int = 5) -> list[dict]:
        top_ids = self.rrf_merge(
            self.vector_search(query, top_k=20),
            self.bm25_search(query, top_k=20),
            top_n=top_n,
        )

        results = []
        seen_ids: set[str] = set()

        for child_id in top_ids:
            child = self.get_by_id(child_id)
            if not child or child_id in seen_ids:
                continue
            seen_ids.add(child_id)

            for anc in self.get_ancestors(child, seen_ids):
                results.append({
                    "id": anc["id"],
                    "content": anc["content"],
                    "role": "parent",
                    "section_path": anc.get("section_path", ""),
                    "chunk_type": anc.get("chunk_type", "text"),
                })

            results.append({
                "id": child_id,
                "content": child["content"],
                "role": "child",
                "section_path": child.get("section_path", ""),
                "chunk_type": child.get("chunk_type", "text"),
            })

        return results
