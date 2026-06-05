"""
graph.py - LangGraph 기반 멀티홉 RAG 챗봇

흐름:
    decompose → retrieve → generate → check → update_history → END
                              ↑           |
                              └───────────┘ (답변 부족 시 재검색)
"""

import os
from typing import TypedDict, Annotated
import operator
from dotenv import load_dotenv
from langchain_aws import ChatBedrockConverse
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from retriever import HybridRetriever

load_dotenv()

REGION = os.getenv("bedrock_REGION", "ap-northeast-2")
LLM_MODEL_ID = os.getenv("LLM_MODEL_ID", "anthropic.claude-3-5-sonnet-20240620-v1:0")

llm = ChatBedrockConverse(model_id=LLM_MODEL_ID, region_name=REGION, temperature=0.0)
retriever = HybridRetriever()
MAX_HOPS = 3


class State(TypedDict):
    question: str
    chat_history: Annotated[list, operator.add]
    sub_questions: list[str]
    retrieved_docs: list[dict]
    answer: str
    hop_count: int
    need_more: bool


def decompose(state: State) -> State:
    prompt = f"""다음 질문에 답하기 위해 필요한 검색 질문을 1~3개 만들어줘.
단순한 질문이면 그대로 1개만.
질문: {state['question']}
형식: 질문1 | 질문2 | 질문3"""

    sub_qs = [q.strip() for q in llm.invoke(prompt).content.split("|")]
    return {**state, "sub_questions": sub_qs, "hop_count": 0}


def retrieve(state: State) -> State:
    all_docs = []
    seen_ids: set[str] = set()
    for q in state["sub_questions"]:
        for doc in retriever.search(q, top_n=5):
            if doc["id"] not in seen_ids:
                seen_ids.add(doc["id"])
                all_docs.append(doc)
    return {**state, "retrieved_docs": all_docs}


def generate(state: State) -> State:
    parent_docs = [d for d in state["retrieved_docs"] if d["role"] == "parent"]
    child_docs  = [d for d in state["retrieved_docs"] if d["role"] == "child"]

    parent_context = "\n\n".join(
        f"[상위 맥락 - {d['section_path']}]\n{d['content']}" for d in parent_docs
    ) if parent_docs else "없음"

    child_context = "\n\n".join(
        f"[핵심 조항 - {d['section_path']}]\n{d['content']}" for d in child_docs
    )

    history = "\n".join(
        f"{'사용자' if m['role'] == 'user' else '봇'}: {m['content']}"
        for m in state["chat_history"][-6:]  # 최근 3턴
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

    return {**state, "answer": llm.invoke(prompt).content}


def check(state: State) -> State:
    if state["hop_count"] >= MAX_HOPS:
        return {**state, "need_more": False}

    need_more = "부족" in llm.invoke(f"""질문: {state['question']}
답변: {state['answer']}

답변이 질문에 충분히 답했으면 "충분", 더 검색이 필요하면 "부족"만 출력해.""").content

    if need_more:
        new_qs = [q.strip() for q in llm.invoke(
            f"""질문: {state['question']}
현재 답변: {state['answer']}
부족한 부분을 채울 추가 검색 질문 1~2개를 만들어줘.
형식: 질문1 | 질문2"""
        ).content.split("|")]
        return {**state, "need_more": True, "sub_questions": new_qs, "hop_count": state["hop_count"] + 1}

    return {**state, "need_more": False}


def update_history(state: State) -> State:
    return {**state, "chat_history": state["chat_history"] + [
        {"role": "user", "content": state["question"]},
        {"role": "assistant", "content": state["answer"]},
    ]}


def should_retry(state: State) -> str:
    return "retrieve" if state["need_more"] else "update_history"


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

    return graph.compile(checkpointer=MemorySaver())


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
                "question": question, "chat_history": [],
                "sub_questions": [], "retrieved_docs": [],
                "answer": "", "hop_count": 0, "need_more": False,
            },
            config=config,
        )
        print(f"\n봇: {result['answer']}")
        print(f"(홉: {result['hop_count'] + 1}회)\n")


if __name__ == "__main__":
    chat()
