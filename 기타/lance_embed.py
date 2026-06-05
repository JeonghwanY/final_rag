import os
import re
import json
import boto3
import lancedb
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv
from parse import parse_all
 
load_dotenv()
 
REGION = os.getenv("bedrock_REGION", "ap-northeast-2")
bedrock = boto3.client(
    service_name="bedrock-runtime",
    region_name=REGION
)
DIM = 1024 ## 차원이 클수록 정확도와 용량 올라감

def titan_embed(text: str) -> list:
    response = bedrock.invoke_model(
        modelId="amazon.titan-embed-text-v2:0",
        # normalize ->코사인 유사도 계산시 사용
        body=json.dumps({"inputText": text, "dimensions": DIM, "normalize": True}),
        contentType="application/json",
        accept="application/json"
    )
    return json.loads(response["body"].read())["embedding"]
 

@dataclass
class ChunkNode:
    id: str  #고유식별자
    content: str  #청크 본문
    level: int           # 계층 깊이 1=대조항 2=중조항 3=⑴ 4=① 99=표
    page: int
    parent_id: str = "" # 부모, 빈 문자열 = 최상위
    section_path: str = "" # 사람이 읽기 좋게
    chunk_type: str = "text"  # "text" | "table"
    is_leaf: int = 1     # 1=leaf(검색대상) 0=parent(맥락용)
    # where 에서 bollean호환성이 환경마다 다름

# ── 계층 패턴 (순서 중요) ────────────────────────────────────────────
LEVEL_PATTERNS = [
    (1, re.compile(r'(?m)^(\d+)\.\s')),                              # 1. 2. 13.
    (2, re.compile(r'(?m)^\s*([가나다라마바사아자차카타파하])\.\s')),   # 가. 나.
    (3, re.compile(r'(?m)^\s*([⑴⑵⑶⑷⑸⑹⑺⑻⑼⑽])\s')),            # ⑴ ⑵ ⑶
    (4, re.compile(r'(?m)^\s*([①②③④⑤⑥⑦⑧⑨⑩])\s')),            # ① ②
]
# (?m)        # MULTILINE 플래그 → ^ 가 줄 시작마다 매칭
# ^           # 줄 시작
# (\d+)       # 그룹 1: 숫자 (1+자리)
# \.          # 점 문자 그대로
# \s          # 공백 (스페이스/탭 등)

def split_by_pattern(text: str, pattern: re.Pattern) -> Optional[tuple]:
    """
    텍스트를 패턴 위치에서 분할.
    Returns (preamble, [(marker, section_text)]) or None if no match.
    """
    matches = list(pattern.finditer(text))
    if not matches:
        return None
 
    preamble = text[:matches[0].start()].strip()
    sections = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        marker = m.group(1)
        content = text[start:end].strip()
        sections.append((marker, content))
 
    return preamble, sections
 
# 재귀트리
def build_children(
    text: str,
    page: int,
    current_id: str,
    current_path: str,
    level_idx: int,
    all_nodes: list
) -> bool:
    """
    text에서 자식 노드를 재귀적으로 생성.
    current_id: 현재 노드 ID (자식들의 parent_id가 됨)
    Returns True if children were created (→ current node is NOT a leaf)
    """
    if level_idx >= len(LEVEL_PATTERNS):
        return False
 
    level_num, pattern = LEVEL_PATTERNS[level_idx]
    result = split_by_pattern(text, pattern)
 
    if result is None:
        # 현재 레벨 패턴 없음 → 다음 레벨 시도
        return build_children(text, page, current_id, current_path, level_idx + 1, all_nodes)
 
    _, sections = result
    if not sections:
        return False
 
    for marker, section_content in sections:
        child_id = f"{current_id}_{marker}"
        child_path = f"{current_path}{marker}."
 
        child = ChunkNode(
            id=child_id,
            content=section_content,
            level=level_num,
            page=page,
            parent_id=current_id,
            section_path=child_path,
            chunk_type="text",
            is_leaf=1
        )
        all_nodes.append(child)
 
        # 재귀: 이 자식의 자식 생성
        has_grandchildren = build_children(
            section_content, page,
            child_id, child_path,
            level_idx + 1,
            all_nodes
        )
 
        if has_grandchildren:
            child.is_leaf = 0  # 자식이 있으면 leaf 아님
 
    return True
 
 
def documents_to_nodes(documents: list[dict]) -> list[ChunkNode]:
    """
    parse_all() 결과 → ChunkNode 리스트 (계층 트리 구조)
    
    구조:
    sec_13 (L1, parent)
     └─ sec_13_가 (L2, parent)
         ├─ sec_13_가_⑴ (L3, leaf)  ← 검색 대상
         ├─ sec_13_가_⑵ (L3, leaf)
         └─ sec_13_가_⑶ (L3, parent)
             ├─ sec_13_가_⑶_① (L4, leaf)
             └─ sec_13_가_⑶_② (L4, leaf)
    """
    all_nodes = []
 
    # 텍스트/표 분리
    text_docs = [(d["page"], d["content"]) for d in documents if d["type"] == "text"]
    table_docs = [(d["page"], d["content"]) for d in documents if d["type"] == "table"]
 
    # 전체 텍스트 합치기 (섹션이 페이지에 걸쳐 있어서 합침)
    combined = "\n\n".join([content for _, content in text_docs])
    default_page = text_docs[0][0] if text_docs else 1
 
    # L1 분할 (1., 2., ..., 13.)
    l1_result = split_by_pattern(combined, LEVEL_PATTERNS[0][1])
    if l1_result:
        _, l1_sections = l1_result
        for marker, section_content in l1_sections:
            node_id = f"sec_{marker}"
            node_path = f"{marker}."
 
            l1_node = ChunkNode(
                id=node_id,
                content=section_content,
                level=1,
                page=default_page,
                parent_id="",
                section_path=node_path,
                chunk_type="text",
                is_leaf=1
            )
            all_nodes.append(l1_node)
 
            # L2 이하 재귀 생성
            has_children = build_children(
                section_content, default_page,
                node_id, node_path,
                level_idx=1,
                all_nodes=all_nodes
            )
            if has_children:
                l1_node.is_leaf = 0
 
    # 표 노드 (독립 leaf)
    for page, content in table_docs:
        table_node = ChunkNode(
            id=f"table_p{page}",
            content=f"[표 - {page}페이지]\n{content}",
            level=99,
            page=page,
            parent_id="",
            section_path=f"표_{page}페이지",
            chunk_type="table",
            is_leaf=1
        )
        all_nodes.append(table_node)
 
    return all_nodes
 
 
def embed_and_store(pdf_path: str, lance_path: str = "./lance_db"):
    """
    파싱 → structure-aware 청킹 → Titan v2 임베딩 → LanceDB 저장
    
    저장 스키마:
        id, vector, content, level, parent_id,
        section_path, chunk_type, page, is_leaf, source
    """
    print("1. 파싱 중...")
    documents = parse_all(pdf_path)
 
    print("2. Structure-aware 청킹 중...")
    nodes = documents_to_nodes(documents)
 
    leaves = [n for n in nodes if n.is_leaf == 1]
    parents = [n for n in nodes if n.is_leaf == 0]
    print(f"   전체 {len(nodes)}개 노드 | leaf(검색) {len(leaves)}개 | parent(맥락) {len(parents)}개")
 
    # 트리 구조 미리보기
    for n in nodes:
        indent = "  " * (n.level - 1) if n.level < 99 else 0
        leaf_mark = "🔍" if n.is_leaf else "📁"
        print(f"   {indent}{leaf_mark} [{n.id}] L{n.level} parent={n.parent_id or 'ROOT'}")
 
    print("3. LanceDB 초기화...")
    db = lancedb.connect(lance_path)
 
    print("4. 임베딩 + 저장 중...")
    source = os.path.basename(pdf_path)
    rows = []
 
    for i, node in enumerate(nodes):
        embedding = titan_embed(node.content)
        rows.append({
            "id": node.id,
            "vector": embedding,
            "content": node.content,
            "level": node.level,
            "parent_id": node.parent_id,
            "section_path": node.section_path,
            "chunk_type": node.chunk_type,
            "page": node.page,
            "is_leaf": node.is_leaf,
            "source": source,
        })
        print(f"   [{i+1}/{len(nodes)}] {node.id} 임베딩 완료")
 
    # 기존 테이블 덮어쓰기
    table = db.create_table("insurance_docs", data=rows, mode="overwrite")
    print(f"\n✅ LanceDB '{lance_path}'에 {len(rows)}개 노드 저장 완료")
    print(f"   검색 대상(leaf): {len(leaves)}개")
    print(f"   맥락 제공(parent): {len(parents)}개")
 
 
if __name__ == "__main__":
    embed_and_store("data/올인원암보험_사업방법서.pdf")