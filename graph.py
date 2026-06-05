"""
graph.py - LangGraph 기반 멀티홉 RAG 챗봇

흐름 (그래프):
    decompose → retrieve → generate → check → update_history → END
                              ↑           |
                              └───────────┘ (답변 부족 시 재검색)

특징:
    - 멀티홉: 복잡한 질문을 세부 질문으로 분해
    - 멀티턴: 이전 대화 기록 유지
    - Parent-child: 조항 맥락과 표 데이터 함께 제공
"""

import os
from typing import TypedDict, Annotated
from types import SimpleNamespace
import operator
from dotenv import load_dotenv
import boto3
from botocore.config import Config
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from retriever import HybridRetriever

load_dotenv()


# ── Bedrock Claude 클라이언트 ───────────────────────────────────────
REGION = "ap-northeast-2"
LLM_MODEL_ID = "anthropic.claude-3-5-sonnet-20240620-v1:0"


class BedrockClaude:
    """AWS Bedrock Claude 클라이언트 래퍼"""

    def __init__(self, model_id: str, region: str, temperature: float = 0.0):
        cfg = Config(
            connect_timeout=5,
            read_timeout=3600,  # Claude Sonnet 계열은 긴 read timeout 권장
            retries={"max_attempts": 1, "mode": "standard"},
        )
        self.model_id = model_id
        self.client = boto3.client(
            service_name="bedrock-runtime",
            region_name=region,
            config=cfg,
        )
        self.temperature = temperature

    def invoke(self, prompt: str) -> SimpleNamespace:
        response = self.client.converse(
            modelId=self.model_id,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"maxTokens": 1000, "temperature": self.temperature},
        )
        text = response["output"]["message"]["content"][0]["text"]
        return SimpleNamespace(content=text)


llm = BedrockClaude(model_id=LLM_MODEL_ID, region=REGION)
retriever = HybridRetriever()
MAX_HOPS = 3


# ── 상태 정의 ───────────────────────────────────────────────────────
class State(TypedDict):
    question: str
    chat_history: Annotated[list, operator.add]   # 멀티턴 대화 기록
    sub_questions: list[str]                       # 멀티홉 세부 질문들
    retrieved_docs: list[dict]                     # 검색된 청크
    answer: str                                    # 최종 답변
    hop_count: int                                 # 현재 홉 횟수
    need_more: bool                                # 재검색 필요 여부


# ── 노드: 질문 분해 ─────────────────────────────────────────────────
def decompose(state: State) -> State:
    """
    복잡한 질문을 세부 검색 질문 1~3개로 분해.
    단순 질문이면 그대로 1개만.
    """
    prompt = f"""다음 질문에 답하기 위해 필요한 검색 질문을 1~3개 만들어줘.
단순한 질문이면 그대로 1개만.
질문: {state['question']}
형식: 질문1 | 질문2 | 질문3"""

    response = llm.invoke(prompt)
    sub_qs = [q.strip() for q in response.content.split("|")]
    return {**state, "sub_questions": sub_qs, "hop_count": 0}


# ── 노드: 검색 ─────────────────────────────────────────────────────
def retrieve(state: State) -> State:
    """
    세부 질문마다 하이브리드 검색 실행.
    parent-child 구조로 조립된 결과를 그대로 사용.
    """
    all_docs = []
    seen_ids: set[str] = set()

    for q in state["sub_questions"]:
        for doc in retriever.search(q, top_n=5):
            if doc["id"] not in seen_ids:
                seen_ids.add(doc["id"])
                all_docs.append(doc)

    return {**state, "retrieved_docs": all_docs}


# ── 노드: 답변 생성 ─────────────────────────────────────────────────
def generate(state: State) -> State:
    """
    parent(맥락) + child(핵심 내용)를 구분해서 LLM에 전달.

    프롬프트 설계 포인트:
    - [상위 맥락]: 조항 간 관계 이해 (예: ⑴에도 불구하고...)
    - [핵심 조항]: 실제 답변 근거
    - 관계 설명 강제: "단,", "⑴에도 불구하고" 같은 조건을 반드시 설명
    """
    parent_docs = [d for d in state["retrieved_docs"] if d["role"] == "parent"]
    child_docs = [d for d in state["retrieved_docs"] if d["role"] == "child"]

    parent_context = "\n\n".join(
        f"[상위 맥락 - {d['section_path']}]\n{d['content']}"
        for d in parent_docs
    ) if parent_docs else "없음"

    child_context = "\n\n".join(
        f"[핵심 조항 - {d['section_path']}]\n{d['content']}"
        for d in child_docs
    )

    history = "\n".join(
        f"{'사용자' if m['role'] == 'user' else '봇'}: {m['content']}"
        for m in state["chat_history"][-6:]  # 최근 3턴만
    )

    prompt = f"""너는 삼성 New올인원 암보험 사업방법서 전문 챗봇이야.
아래 문서를 기반으로만 답변해. 문서에 없는 내용은 "문서에서 확인할 수 없습니다"라고 해.

규칙:
1. [상위 맥락]은 조항 간 관계를 이해하기 위한 배경이야.
   답변에서 원칙과 예외를 구분해서 설명해.
2. "⑴에도 불구하고", "단,", "이 경우" 같은 조건 관계를 반드시 설명해.
3. 번호 조항은 완전한 의미가 되도록 앞뒤 맥락을 포함해서 답변해.

[이전 대화]
{history}

[상위 맥락 - 조항 간 관계 이해용]
{parent_context}

[핵심 조항 - 답변 근거]
{child_context}

[현재 질문]
{state['question']}

답변:"""

    response = llm.invoke(prompt)
    return {**state, "answer": response.content}


# ── 노드: 답변 충분성 검토 ──────────────────────────────────────────
def check(state: State) -> State:
    """
    답변이 충분한지 LLM으로 판단.
    부족하면 보완 검색 질문을 생성해서 재검색.
    """
    if state["hop_count"] >= MAX_HOPS:
        return {**state, "need_more": False}

    prompt = f"""질문: {state['question']}
답변: {state['answer']}

답변이 질문에 충분히 답했으면 "충분", 더 검색이 필요하면 "부족"만 출력해."""

    response = llm.invoke(prompt)
    need_more = "부족" in response.content

    if need_more:
        followup = llm.invoke(
            f"""질문: {state['question']}
현재 답변: {state['answer']}
부족한 부분을 채울 추가 검색 질문 1~2개를 만들어줘.
형식: 질문1 | 질문2"""
        )
        new_qs = [q.strip() for q in followup.content.split("|")]
        return {
            **state,
            "need_more": True,
            "sub_questions": new_qs,
            "hop_count": state["hop_count"] + 1,
        }

    return {**state, "need_more": False}


# ── 노드: 대화 기록 업데이트 ────────────────────────────────────────
def update_history(state: State) -> State:
    """대화 기록에 현재 턴 추가"""
    new_history = state["chat_history"] + [
        {"role": "user", "content": state["question"]},
        {"role": "assistant", "content": state["answer"]},
    ]
    return {**state, "chat_history": new_history}


# ── 분기 조건 ──────────────────────────────────────────────────────
def should_retry(state: State) -> str:
    """답변 부족이면 retrieve로, 충분하면 update_history로"""
    return "retrieve" if state["need_more"] else "update_history"


# ── 그래프 빌드 ────────────────────────────────────────────────────
def build_graph():
    graph = StateGraph(State)

    graph.add_node("decompose",      decompose)
    graph.add_node("retrieve",       retrieve)
    graph.add_node("generate",       generate)
    graph.add_node("check",          check)
    graph.add_node("update_history", update_history)

    graph.set_entry_point("decompose")
    graph.add_edge("decompose",      "retrieve")
    graph.add_edge("retrieve",       "generate")
    graph.add_edge("generate",       "check")
    graph.add_conditional_edges("check", should_retry)
    graph.add_edge("update_history", END)

    memory = MemorySaver()  # 멀티턴 대화 메모리
    return graph.compile(checkpointer=memory)


# ── CLI 챗봇 ───────────────────────────────────────────────────────
def chat():
    app = build_graph()
    config = {"configurable": {"thread_id": "user_1"}}
    print("=== 암보험 챗봇 (종료: q) ===\n")

    while True:
        question = input("질문: ").strip()
        if question.lower() in ("q", "quit", "종료"):
            break
        if not question:
            continue

        result = app.invoke(
            {
                "question":       question,
                "chat_history":   [],
                "sub_questions":  [],
                "retrieved_docs": [],
                "answer":         "",
                "hop_count":      0,
                "need_more":      False,
            },
            config=config,
        )

        print(f"\n봇: {result['answer']}")
        print(f"(홉: {result['hop_count'] + 1}회)\n")


if __name__ == "__main__":
    chat()
