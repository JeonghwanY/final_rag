"""
embed.py - 구조 인식 청킹 + Parent-child 트리 + ChromaDB 저장

흐름:
1. parse.py로 PDF 파싱 (텍스트는 placeholder 포함, 표는 별도)
2. 텍스트를 조항 패턴으로 재귀 분할 (1. → 가. → ⑴ → ①)
3. 중복 L1 (footnote 등) → 첫 번째 섹션의 하위 노드로 처리
4. 표 노드를 placeholder가 있는 가장 깊은 섹션의 자식으로 연결
5. 모든 노드를 Titan v2로 임베딩하고 ChromaDB에 저장
"""

import os
import re
import json
import boto3
import chromadb
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv
from parse import parse_all

load_dotenv()


# ── Bedrock 설정 ────────────────────────────────────────────────────
REGION = os.getenv("bedrock_REGION", "ap-northeast-2")
EMBED_DIM = 1024  # 256/512/1024 중 선택. 클수록 정확도/용량 증가

bedrock_client = boto3.client(
    service_name="bedrock-runtime",
    region_name=REGION,
)


def embed_text(text: str) -> list[float]:
    """Titan v2로 텍스트를 1024차원 벡터로 변환 (코사인 정규화 적용)"""
    response = bedrock_client.invoke_model(
        modelId="amazon.titan-embed-text-v2:0",
        body=json.dumps({
            "inputText": text,
            "dimensions": EMBED_DIM,
            "normalize": True,  # 코사인 유사도에 적합한 단위벡터
        }),
        contentType="application/json",
        accept="application/json",
    )
    return json.loads(response["body"].read())["embedding"]


# ── 데이터 구조 ─────────────────────────────────────────────────────
@dataclass
class ChunkNode:
    """
    하나의 청크 = 트리의 한 노드.

    - is_leaf=1: 검색 대상 (가장 구체적인 정보 단위)
    - is_leaf=0: 맥락 제공용 (검색 후 부모로 함께 전달)
    """
    id: str
    content: str
    level: int              # 1=대조항(1.) 2=중조항(가.) 3=⑴ 4=① 99=표
    page: int
    parent_id: str = ""     # 빈 문자열 = 최상위
    section_path: str = ""
    chunk_type: str = "text"  # "text" | "table"
    is_leaf: int = 1        # bool 대신 int — ChromaDB 메타데이터 호환


# ── 계층 패턴 (순서가 깊이를 결정) ──────────────────────────────────
LEVEL_PATTERNS = [
    (1, re.compile(r"(?m)^(\d+)\.\s")),                                  # 1. 2. 13.
    (2, re.compile(r"(?m)^\s*([가나다라마바사아자차카타파하])\.\s")),  # 가. 나.
    (3, re.compile(r"(?m)^\s*([⑴⑵⑶⑷⑸⑹⑺⑻⑼⑽])\s")),               # ⑴ ⑵
    (4, re.compile(r"(?m)^\s*([①②③④⑤⑥⑦⑧⑨⑩])\s")),                # ① ②
]


# ── 구조 인식 분할 ──────────────────────────────────────────────────
def split_by_pattern(
    text: str,
    pattern: re.Pattern,
) -> Optional[tuple[str, list[tuple[str, str]]]]:
    """
    텍스트를 패턴 위치에서 분할.

    Returns:
        (preamble, [(marker, section_text), ...]) or None
    """
    matches = list(pattern.finditer(text))
    if not matches:
        return None

    preamble = text[:matches[0].start()].strip()
    sections = [
        (m.group(1), text[m.start(): (matches[i + 1].start() if i + 1 < len(matches) else len(text))].strip())
        for i, m in enumerate(matches)
    ]
    return preamble, sections


def build_subtree(
    text: str,
    page: int,
    current_id: str,
    current_path: str,
    level_idx: int,
    all_nodes: list[ChunkNode],
) -> bool:
    """
    text 안에서 자식 노드들을 재귀적으로 생성.

    Returns:
        True  → 자식 생성됨, 현재 노드는 is_leaf=0 으로 바꿔야 함
        False → 더 분할 불가, 현재 노드는 leaf 유지
    """
    if level_idx >= len(LEVEL_PATTERNS):
        return False

    _, pattern = LEVEL_PATTERNS[level_idx]
    result = split_by_pattern(text, pattern)

    # 현재 레벨 패턴 없으면 한 단계 더 깊은 레벨 시도
    if result is None:
        return build_subtree(text, page, current_id, current_path, level_idx + 1, all_nodes)

    _, sections = result
    level_num = LEVEL_PATTERNS[level_idx][0]

    for marker, section_text in sections:
        child_id = f"{current_id}_{marker}"
        child_path = f"{current_path}{marker}."

        child = ChunkNode(
            id=child_id,
            content=section_text,
            level=level_num,
            page=page,
            parent_id=current_id,
            section_path=child_path,
        )
        all_nodes.append(child)

        has_grandchildren = build_subtree(
            section_text, page,
            child_id, child_path,
            level_idx + 1,
            all_nodes,
        )
        if has_grandchildren:
            child.is_leaf = 0

    return True


# ── 텍스트 트리 구축 ────────────────────────────────────────────────
def build_text_tree(text_docs: list[tuple[int, str]]) -> list[ChunkNode]:
    """
    텍스트 페이지들을 합쳐서 계층 트리 생성.

    핵심 처리:
    - L1 중복 번호 (예: 페이지 3의 ㈜ 2.) → 첫 섹션의 하위 노드로 처리
      (PyMuPDF가 들여쓰기를 날려서 줄 시작에 오는 경우)
    """
    if not text_docs:
        return []

    combined = "\n\n".join(content for _, content in text_docs)
    default_page = text_docs[0][0]
    all_nodes: list[ChunkNode] = []

    l1_result = split_by_pattern(combined, LEVEL_PATTERNS[0][1])
    if not l1_result:
        return []

    _, l1_sections = l1_result

    # seen_markers: marker → ChunkNode 객체 직접 저장
    # (인덱스가 아닌 참조 → 나중에 is_leaf 변경 시 자동 반영)
    seen_markers: dict[str, ChunkNode] = {}

    for marker, section_text in l1_sections:

        if marker in seen_markers:
            # ── 중복 L1: 첫 번째 섹션의 하위 노드로 처리 ──────────
            # 예) ㈜ 주(注)의 "2. 단, 갱신일부터..." → sec_2 의 자식
            parent_node = seen_markers[marker]
            child_id = f"{parent_node.id}_추"   # 추(追): 추가 내용

            child = ChunkNode(
                id=child_id,
                content=section_text,
                level=2,                        # L1 바로 아래 L2로 처리
                page=default_page,
                parent_id=parent_node.id,
                section_path=f"{marker}.추.",
            )
            all_nodes.append(child)
            parent_node.is_leaf = 0             # 자식 생겼으니 parent 전환

            print(f"[중복 처리] L1 '{marker}' → {parent_node.id} 하위 노드로")

        else:
            # ── 정상 L1 노드 생성 ────────────────────────────────
            node_id = f"sec_{marker}"
            node_path = f"{marker}."

            l1_node = ChunkNode(
                id=node_id,
                content=section_text,
                level=1,
                page=default_page,
                section_path=node_path,
            )
            all_nodes.append(l1_node)
            seen_markers[marker] = l1_node      # 참조 저장

            has_children = build_subtree(
                section_text, default_page,
                node_id, node_path,
                level_idx=1,
                all_nodes=all_nodes,
            )
            if has_children:
                l1_node.is_leaf = 0

    return all_nodes


# ── 표 노드 연결 ────────────────────────────────────────────────────
def attach_table_nodes(
    text_nodes: list[ChunkNode],
    table_docs: list[dict],
) -> list[ChunkNode]:
    """
    표 노드를 생성하고 placeholder가 있는 가장 깊은 섹션의 자식으로 연결.

    예)
        sec_13_나_⑴.content 에 "[TABLE_p6_1]" 포함
        → table_TABLE_p6_1.parent_id = sec_13_나_⑴
        → sec_13_나_⑴.is_leaf = 0

    placeholder는 상위 노드 content에도 포함되므로 max(level)로 가장 구체적인 노드 선택.
    """
    table_nodes = []

    for tdoc in table_docs:
        table_id = tdoc["table_id"]           # 예: "TABLE_p6_1"
        placeholder = f"[{table_id}]"

        # placeholder가 있는 가장 깊은 텍스트 노드를 부모로
        candidates = [
            n for n in text_nodes
            if n.chunk_type == "text" and placeholder in n.content
        ]
        parent_node = max(candidates, key=lambda n: n.level) if candidates else None

        table_node = ChunkNode(
            id=f"table_{table_id}",
            content=f"[표]\n{tdoc['content']}",
            level=99,
            page=tdoc["page"],
            parent_id=parent_node.id if parent_node else "",
            section_path=(
                f"{parent_node.section_path}표" if parent_node
                else f"표_{tdoc['page']}페이지"
            ),
            chunk_type="table",
        )
        table_nodes.append(table_node)

        if parent_node:
            parent_node.is_leaf = 0

    return text_nodes + table_nodes


def build_all_nodes(documents: list[dict]) -> list[ChunkNode]:
    """parse_all() 결과 → ChunkNode 리스트"""
    text_docs = [(d["page"], d["content"]) for d in documents if d["type"] == "text"]
    table_docs = [d for d in documents if d["type"] == "table"]

    nodes = build_text_tree(text_docs)
    nodes = attach_table_nodes(nodes, table_docs)
    return nodes


# ── 디버그 출력 ─────────────────────────────────────────────────────
def print_tree(nodes: list[ChunkNode]) -> None:
    for n in nodes:
        indent = "  " * (n.level - 1) if n.level < 99 else "    "
        mark = "🔍" if n.is_leaf else "📁"
        parent = n.parent_id or "ROOT"
        print(f"   {indent}{mark} [{n.id}] L{n.level} parent={parent}")


# ── ChromaDB 저장 ───────────────────────────────────────────────────
def save_to_chromadb(
    nodes: list[ChunkNode],
    source: str,
    chroma_path: str = "./chroma_db",
    collection_name: str = "insurance_docs",
) -> None:
    """
    모든 노드를 임베딩해서 ChromaDB에 저장.

    ChromaDB 메타데이터 제약:
        값은 str / int / float / bool 만 가능 (list, dict 불가)
    """
    client = chromadb.PersistentClient(path=chroma_path)

    # 기존 컬렉션 초기화
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},  # 코사인 유사도 사용
    )

    for i, node in enumerate(nodes):
        embedding = embed_text(node.content)
        collection.add(
            ids=[node.id],
            embeddings=[embedding],
            documents=[node.content],
            metadatas=[{
                "level": node.level,
                "parent_id": node.parent_id,
                "section_path": node.section_path,
                "chunk_type": node.chunk_type,
                "page": node.page,
                "is_leaf": node.is_leaf,
                "source": source,
            }],
        )
        print(f"   [{i + 1}/{len(nodes)}] {node.id} 완료")


# ── 메인 파이프라인 ─────────────────────────────────────────────────
def embed_and_store(pdf_path: str, chroma_path: str = "./chroma_db") -> None:
    """파싱 → 청킹 → 임베딩 → ChromaDB 저장"""
    print("1. 파싱 중...")
    documents = parse_all(pdf_path)

    print("\n2. Structure-aware 청킹 중...")
    nodes = build_all_nodes(documents)

    leaves = sum(1 for n in nodes if n.is_leaf)
    parents = len(nodes) - leaves
    print(f"   전체 {len(nodes)}개 노드 | leaf(검색) {leaves}개 | parent(맥락) {parents}개")
    print_tree(nodes)

    print("\n3. 임베딩 + ChromaDB 저장 중...")
    save_to_chromadb(
        nodes,
        source=os.path.basename(pdf_path),
        chroma_path=chroma_path,
    )
    print(f"\n✅ '{chroma_path}'에 {len(nodes)}개 노드 저장 완료")


if __name__ == "__main__":
    embed_and_store("data/올인원암보험_사업방법서.pdf")
