"""
retriever.py - 하이브리드 검색 + Parent-child 맥락 조립

흐름:
1. 벡터 검색 (ChromaDB 코사인) + BM25 키워드 검색
2. RRF(Reciprocal Rank Fusion)로 두 결과 순위 통합
3. hit된 child의 parent를 자동으로 추가 (맥락 제공)
"""

import os
import re
import json
import boto3
import chromadb
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv

load_dotenv()


# ── Bedrock 설정 ────────────────────────────────────────────────────
REGION = os.getenv("bedrock_REGION", "ap-northeast-2")
EMBED_DIM = 1024

bedrock_client = boto3.client(
    service_name="bedrock-runtime",
    region_name=REGION,
)


def embed_text(text: str) -> list[float]:
    """Titan v2 임베딩 (embed.py와 동일한 설정 유지)"""
    response = bedrock_client.invoke_model(
        modelId="amazon.titan-embed-text-v2:0",
        body=json.dumps({
            "inputText": text,
            "dimensions": EMBED_DIM,
            "normalize": True,
        }),
        contentType="application/json",
        accept="application/json",
    )
    return json.loads(response["body"].read())["embedding"]


def tokenize(text: str) -> list[str]:
    """
    한국어+영문 2글자 이상 단어만 추출.
    "이", "의", "을" 같은 1글자 조사는 BM25 점수를 낮추므로 제외.
    """
    return re.findall(r"[가-힣a-zA-Z0-9]{2,}", text)


# ── 검색기 ──────────────────────────────────────────────────────────
class HybridRetriever:
    """
    벡터 검색(ChromaDB) + 키워드 검색(BM25) 하이브리드 검색기.

    검색 흐름:
        질문 → 벡터 검색 + BM25 검색
              → RRF로 순위 통합 (leaf 노드 대상)
              → 각 child의 parent 자동 조회 (맥락)
              → [parent 맥락 + child 내용] 반환
    """

    def __init__(self, chroma_path: str = "./chroma_db"):
        client = chromadb.PersistentClient(path=chroma_path)
        self.collection = client.get_collection("insurance_docs")

        # BM25는 leaf 노드만 대상 (검색 대상과 동일하게)
        leaf_data = self.collection.get(
            where={"is_leaf": 1},
            include=["documents", "metadatas"],
        )
        self.leaf_ids = leaf_data["ids"]
        self.leaf_docs = leaf_data["documents"]
        self.leaf_metas = leaf_data["metadatas"]

        tokenized = [tokenize(doc) for doc in self.leaf_docs]
        self.bm25 = BM25Okapi(tokenized)

        print(f"[검색기 준비] 전체 leaf {len(self.leaf_ids)}개 노드 로드")

    # ── 개별 검색 ─────────────────────────────────────────────────
    def vector_search(self, query: str, top_k: int = 20) -> list[tuple[str, float]]:
        """
        ChromaDB 코사인 유사도 검색 (leaf만).
        반환: [(id, distance)] — distance 낮을수록 유사
        """
        q_emb = embed_text(query)
        results = self.collection.query(
            query_embeddings=[q_emb],
            n_results=top_k,
            where={"is_leaf": 1},
            include=["distances"],
        )
        return list(zip(results["ids"][0], results["distances"][0]))

    def bm25_search(self, query: str, top_k: int = 20) -> list[tuple[str, float]]:
        """
        BM25 키워드 검색 (leaf만).
        "암사망특약N15" 같은 도메인 전문용어에 강함.
        반환: [(id, score)] — score 높을수록 유사
        """
        tokens = tokenize(query)
        scores = self.bm25.get_scores(tokens)
        ranked = sorted(
            zip(self.leaf_ids, scores),
            key=lambda x: x[1],
            reverse=True,
        )
        return ranked[:top_k]

    def rrf_merge(
        self,
        vector_results: list[tuple[str, float]],
        bm25_results: list[tuple[str, float]],
        k: int = 60,
        top_n: int = 5,
    ) -> list[str]:
        """
        Reciprocal Rank Fusion: 두 검색 순위 통합.

        score = 1/(k + rank_vector) + 1/(k + rank_bm25)

        k=60: 원 논문 권장값. 1위와 2위 점수 차이를 완만하게 만들어
              어느 한쪽에만 치우치지 않도록 함.
        """
        scores: dict[str, float] = {}
        for rank, (doc_id, _) in enumerate(vector_results):
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
        for rank, (doc_id, _) in enumerate(bm25_results):
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
        return sorted(scores, key=lambda x: scores[x], reverse=True)[:top_n]

    # ── ID 조회 ───────────────────────────────────────────────────
    def get_by_id(self, node_id: str) -> dict | None:
        """ChromaDB에서 ID로 노드 조회"""
        result = self.collection.get(
            ids=[node_id],
            include=["documents", "metadatas"],
        )
        if not result["ids"]:
            return None
        return {
            "id": result["ids"][0],
            "content": result["documents"][0],
            **result["metadatas"][0],  # level, parent_id, chunk_type 등
        }

    def get_ancestors(self, node: dict, seen_ids: set[str]) -> list[dict]:
        """
        hit 노드에서 L1 루트까지 모든 조상을 순서대로 반환.

        반환: [L1, L2, ...] (상위 → 하위 순)
        seen_ids에 방문한 id를 누적해서 중복 삽입 방지.
        """
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
        return list(reversed(ancestors))  # 상위 → 하위 순

    # ── 메인 검색 ─────────────────────────────────────────────────
    def search(self, query: str, top_n: int = 5) -> list[dict]:
        """
        하이브리드 검색 + parent-child 맥락 조립.

        반환 형식:
            [
                {"id": ..., "content": ..., "role": "parent", ...},
                {"id": ..., "content": ..., "role": "child",  ...},
                ...
            ]

        role 의미:
            "parent" → 맥락 제공용 (L1까지 전체 조상 체인)
            "child"  → 검색에서 직접 hit된 핵심 내용
        """
        vector_results = self.vector_search(query, top_k=20)
        bm25_results = self.bm25_search(query, top_k=20)
        top_ids = self.rrf_merge(vector_results, bm25_results, top_n=top_n)

        results = []
        seen_ids: set[str] = set()

        for child_id in top_ids:
            child = self.get_by_id(child_id)
            if not child or child_id in seen_ids:
                continue
            seen_ids.add(child_id)

            # 루트까지 모든 조상을 먼저 추가 (LLM에게 넓은 맥락을 먼저 제공)
            ancestors = self.get_ancestors(child, seen_ids)
            for anc in ancestors:
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


# ── 검색 테스트 ─────────────────────────────────────────────────────
if __name__ == "__main__":
    retriever = HybridRetriever()

    test_queries = [
        "갱신 못 하는 경우",
        "1종 만기지급형 남자 가입나이",
        "암사망특약 최대 한도",
        "보험료 선납 할인",
    ]

    for q in test_queries:
        print(f"\n{'='*50}")
        print(f"질문: {q}")
        results = retriever.search(q, top_n=3)
        for r in results:
            role_icon = "📁" if r["role"] == "parent" else "🔍"
            print(f"  {role_icon} [{r['role']}] {r['id']} | {r['section_path']}")
            print(f"     {r['content'][:80]}...")
