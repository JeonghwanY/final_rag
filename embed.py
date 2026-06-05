import os
import re
import json
import boto3
import chromadb
from dataclasses import dataclass
from dotenv import load_dotenv
from parse import parse_all

load_dotenv()

REGION = os.getenv("bedrock_REGION", "ap-northeast-2")
EMBED_DIM = 1024  # 256/512/1024, 클수록 정확도↑ 용량↑

bedrock_client = boto3.client(service_name="bedrock-runtime", region_name=REGION)


def embed_text(text: str) -> list[float]:
    response = bedrock_client.invoke_model(
        modelId="amazon.titan-embed-text-v2:0",
        body=json.dumps({"inputText": text, "dimensions": EMBED_DIM, "normalize": True}),
        contentType="application/json",
        accept="application/json",
    )
    return json.loads(response["body"].read())["embedding"]


@dataclass
class ChunkNode:
    id: str
    content: str
    level: int              # 1=대조항(1.) 2=중조항(가.) 3=⑴ 4=① 99=표
    page: int
    parent_id: str = ""
    section_path: str = ""
    chunk_type: str = "text"
    is_leaf: int = 1        # bool 대신 int — ChromaDB 메타데이터 호환


LEVEL_PATTERNS = [
    (1, re.compile(r"(?m)^(\d+)\.\s")),
    (2, re.compile(r"(?m)^\s*([가나다라마바사아자차카타파하])\.\s")),
    (3, re.compile(r"(?m)^\s*([⑴⑵⑶⑷⑸⑹⑺⑻⑼⑽])\s")),
    (4, re.compile(r"(?m)^\s*([①②③④⑤⑥⑦⑧⑨⑩])\s")),
]


def split_by_pattern(text: str, pattern: re.Pattern) -> list[tuple[str, str]] | None:
    matches = list(pattern.finditer(text))
    if not matches:
        return None
    return [
        (m.group(1), text[m.start(): (matches[i + 1].start() if i + 1 < len(matches) else len(text))].strip())
        for i, m in enumerate(matches)
    ]


def build_subtree(
    text: str, page: int, current_id: str, current_path: str,
    level_idx: int, all_nodes: list[ChunkNode],
) -> bool:
    """True = 자식 생성됨 → 호출자가 is_leaf=0 처리, False = leaf 유지"""
    if level_idx >= len(LEVEL_PATTERNS):
        return False

    _, pattern = LEVEL_PATTERNS[level_idx]
    sections = split_by_pattern(text, pattern)
    if sections is None:
        return build_subtree(text, page, current_id, current_path, level_idx + 1, all_nodes)

    level_num = LEVEL_PATTERNS[level_idx][0]

    for marker, section_text in sections:
        child_id = f"{current_id}_{marker}"
        child_path = f"{current_path}{marker}."
        child = ChunkNode(
            id=child_id, content=section_text, level=level_num,
            page=page, parent_id=current_id, section_path=child_path,
        )
        all_nodes.append(child)
        if build_subtree(section_text, page, child_id, child_path, level_idx + 1, all_nodes):
            child.is_leaf = 0

    return True


def build_text_tree(text_docs: list[tuple[int, str]]) -> list[ChunkNode]:
    if not text_docs:
        return []

    combined = "\n\n".join(content for _, content in text_docs)
    default_page = text_docs[0][0]
    all_nodes: list[ChunkNode] = []

    l1_sections = split_by_pattern(combined, LEVEL_PATTERNS[0][1])
    if not l1_sections:
        return []
    seen_markers: dict[str, ChunkNode] = {}  # marker → 노드 참조 (is_leaf 변경 반영용)

    for marker, section_text in l1_sections:
        if marker in seen_markers:
            # 중복 L1 (주석의 "2. 단, ..." 등) → 첫 섹션의 하위 노드로 처리
            parent_node = seen_markers[marker]
            child = ChunkNode(
                id=f"{parent_node.id}_추",  # 추(追): 추가 내용
                content=section_text, level=2,
                page=default_page, parent_id=parent_node.id,
                section_path=f"{marker}.추.",
            )
            all_nodes.append(child)
            parent_node.is_leaf = 0
            print(f"[중복 처리] L1 '{marker}' → {parent_node.id} 하위 노드로")
        else:
            node_id = f"sec_{marker}"
            l1_node = ChunkNode(
                id=node_id, content=section_text, level=1,
                page=default_page, section_path=f"{marker}.",
            )
            all_nodes.append(l1_node)
            seen_markers[marker] = l1_node
            if build_subtree(section_text, default_page, node_id, f"{marker}.", 1, all_nodes):
                l1_node.is_leaf = 0

    return all_nodes


def attach_table_nodes(text_nodes: list[ChunkNode], table_docs: list[dict]) -> list[ChunkNode]:
    """placeholder가 있는 가장 깊은 텍스트 노드를 부모로 삼아 표 노드 연결"""
    table_nodes = []
    for tdoc in table_docs:
        table_id = tdoc["table_id"]
        placeholder = f"[{table_id}]"
        candidates = [n for n in text_nodes if n.chunk_type == "text" and placeholder in n.content]
        parent_node = max(candidates, key=lambda n: n.level) if candidates else None

        table_node = ChunkNode(
            id=f"table_{table_id}",
            content=f"[표]\n{tdoc['content']}",
            level=99, page=tdoc["page"],
            parent_id=parent_node.id if parent_node else "",
            section_path=f"{parent_node.section_path}표" if parent_node else f"표_{tdoc['page']}페이지",
            chunk_type="table",
        )
        table_nodes.append(table_node)
        if parent_node:
            parent_node.is_leaf = 0

    return text_nodes + table_nodes


def build_all_nodes(documents: list[dict]) -> list[ChunkNode]:
    text_docs = [(d["page"], d["content"]) for d in documents if d["type"] == "text"]
    table_docs = [d for d in documents if d["type"] == "table"]
    return attach_table_nodes(build_text_tree(text_docs), table_docs)


def print_tree(nodes: list[ChunkNode]) -> None:
    for n in nodes:
        indent = "  " * (n.level - 1) if n.level < 99 else "    "
        print(f"   {indent}{'🔍' if n.is_leaf else '📁'} [{n.id}] L{n.level} parent={n.parent_id or 'ROOT'}")


def save_to_chromadb(
    nodes: list[ChunkNode], source: str,
    chroma_path: str = "./chroma_db", collection_name: str = "insurance_docs",
) -> None:
    client = chromadb.PersistentClient(path=chroma_path)
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    collection = client.create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})

    for i, node in enumerate(nodes):
        collection.add(
            ids=[node.id],
            embeddings=[embed_text(node.content)],
            documents=[node.content],
            metadatas=[{
                "level": node.level, "parent_id": node.parent_id,
                "section_path": node.section_path, "chunk_type": node.chunk_type,
                "page": node.page, "is_leaf": node.is_leaf, "source": source,
            }],
        )
        print(f"   [{i + 1}/{len(nodes)}] {node.id} 완료")


def embed_and_store(pdf_path: str, chroma_path: str = "./chroma_db") -> None:
    print("1. 파싱")
    documents = parse_all(pdf_path)

    print("\n2. 청킹")
    nodes = build_all_nodes(documents)
    leaves = sum(1 for n in nodes if n.is_leaf)
    print(f"   전체 {len(nodes)}개 노드 | leaf(검색) {leaves}개 | parent(맥락) {len(nodes) - leaves}개")
    print_tree(nodes)

    print("\n3. 임베딩 + vectorDB 저장")
    save_to_chromadb(nodes, source=os.path.basename(pdf_path), chroma_path=chroma_path)
    print(f"\n'{chroma_path}'에 {len(nodes)}개 노드 저장 완료")


if __name__ == "__main__":
    embed_and_store("data/올인원암보험_사업방법서.pdf")
