import os
import fitz
import json
import boto3
import pdfplumber
from dotenv import load_dotenv

load_dotenv()

REGION = os.getenv("bedrock_REGION", "ap-northeast-2")
LLM_MODEL_ID = "anthropic.claude-3-5-sonnet-20240620-v1:0"

client = boto3.client(service_name="bedrock-runtime", region_name=REGION)


def table_page(pdf_path: str) -> list[int]:
    table_pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            if page.find_tables():
                table_pages.append(i + 1)
    return table_pages


def table_parse(img_bytes: bytes) -> list[dict]:
    """Claude Vision으로 병합셀 표를 행마다 펼쳐서 JSON으로 반환"""
    prompt = """이 PDF 페이지의 표를 분석해줘.
병합된 셀을 각 행마다 풀어서 JSON 배열로 변환해줘.

규칙:
1. 병합 셀 값은 해당되는 모든 행에 반복해서 채워줘
2. 병합 셀이 중첩되어 있으면(예: "최초계약" 안에 "1종 순수", "1종 만기", "2종 순수"가 각각 있는 경우),
   상위와 하위 값을 모두 포함해서 표시해줘.
   예) "구분": "최초계약 1종(일시지급형) [순수보장형]"
       "구분": "최초계약 1종(일시지급형) [만기지급형]"
       "구분": "최초계약 2종(생활자금형) [순수보장형]"
3. 표가 2개 이상이면 각각 파싱해서 하나의 배열에 담아줘
4. JSON만 출력, 설명 없이, 마크다운 코드블록 없이
5. 각 행은 객체(object) 형태로 출력해줘
6. 누락된 병합 셀 값은 위/왼쪽 맥락을 보고 반복 적용해줘
"""
    response = client.converse(
        modelId=LLM_MODEL_ID,
        messages=[{
            "role": "user",
            "content": [
                {"image": {"format": "png", "source": {"bytes": img_bytes}}},
                {"text": prompt},
            ],
        }],
        inferenceConfig={"maxTokens": 3000, "temperature": 0.0},
    )

    text = response["output"]["message"]["content"][0]["text"].strip()
    if text.startswith("```"):
        text = text.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        print("[경고] JSON 파싱 실패, raw 텍스트로 저장")
        return [{"raw": text}]


def table_rows_to_text(rows: list[dict]) -> str:
    """임베딩 품질을 위해 표 행을 자연어 문장으로 변환"""
    sentences = []
    for row in rows:
        parts = []
        for k, v in row.items():
            if not v or str(v) == "-":
                continue
            if isinstance(v, dict):
                for sub_k, sub_v in v.items():
                    if sub_v and str(sub_v) != "-":
                        parts.append(f"{k}_{sub_k}은(는) {sub_v}")
            else:
                parts.append(f"{k}은(는) {v}")
        if parts:
            sentences.append(". ".join(parts) + ".")
    return "\n".join(sentences)


def extract_text_with_placeholders(pdf_path: str, page_num: int) -> str:
    """페이지 텍스트에서 표 영역을 [TABLE_pN_i] placeholder로 치환"""
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_num - 1]
        tables = page.find_tables()

        if not tables:
            return page.extract_text() or ""

        table_items = []
        for i, t in enumerate(tables):
            table_items.append({
                "bbox": t.bbox,
                "y_top": t.bbox[1],
                "placeholder": f"[TABLE_p{page_num}_{i+1}]",
            })

        words = page.extract_words()
        non_table_words = []
        for w in words:
            in_table = any(
                w["x0"] >= t["bbox"][0] and w["x1"] <= t["bbox"][2] and
                w["top"] >= t["bbox"][1] and w["bottom"] <= t["bbox"][3]
                for t in table_items
            )
            if not in_table:
                non_table_words.append({"y": w["top"], "text": w["text"]})

        all_items = [(w["y"], w["text"]) for w in non_table_words]
        all_items += [(t["y_top"], t["placeholder"]) for t in table_items]
        all_items.sort(key=lambda x: x[0])

        # y좌표 기준으로 같은 줄 묶기 — 줄바꿈이 사라지면 (?m)^ 패턴 매칭 불가
        LINE_TOLERANCE = 3
        lines: list[list[str]] = []
        current_y: float | None = None
        current_line: list[str] = []
        for y, token in all_items:
            if current_y is None or abs(y - current_y) > LINE_TOLERANCE:
                if current_line:
                    lines.append(current_line)
                current_line = [token]
                current_y = y
            else:
                current_line.append(token)
        if current_line:
            lines.append(current_line)

        return "\n".join(" ".join(line) for line in lines)


def _rows_from_table(table) -> list[dict] | None:
    """
    pdfplumber Table 객체에서 행 추출.
    병합셀 의심(빈 셀 15% 초과)이면 None 반환 → Claude fallback.
    """
    rows = table.extract()
    if not rows or len(rows) < 2:
        return None

    total = sum(len(r) for r in rows)
    empty = sum(1 for r in rows for c in r if c is None or not c.strip())
    if empty / max(total, 1) > 0.15:
        return None

    headers = rows[0]
    return [{h: (v or "") for h, v in zip(headers, row)} for row in rows[1:]]


def render_table_region(pdf_path: str, page_num: int, bbox: tuple) -> bytes:
    """표 bbox 영역만 잘라서 PNG 렌더링 (Claude Vision 입력용)"""
    doc = fitz.open(pdf_path)
    pix = doc[page_num - 1].get_pixmap(dpi=200, clip=fitz.Rect(*bbox))
    return pix.tobytes("png")


def parse_all(pdf_path: str) -> list[dict]:
    documents = []
    table_pages = table_page(pdf_path)

    # 텍스트 추출 (표 페이지는 placeholder로 치환)
    doc = fitz.open(pdf_path)
    for i, page in enumerate(doc):
        page_num = i + 1
        if page_num in table_pages:
            text = extract_text_with_placeholders(pdf_path, page_num)
        else:
            text = page.get_text()

        if text.strip():
            documents.append({"page": page_num, "type": "text", "content": text.strip()})

    # 표 파싱 — pdfplumber를 페이지당 한 번만 open
    for page_num in table_pages:
        with pdfplumber.open(pdf_path) as pdf:
            page = pdf.pages[page_num - 1]
            tables = page.find_tables()

            for i, table in enumerate(tables):
                table_id = f"TABLE_p{page_num}_{i+1}"
                rows = _rows_from_table(table)
                if not rows:
                    img = render_table_region(pdf_path, page_num, table.bbox)
                    rows = table_parse(img)

                documents.append({
                    "page": page_num,
                    "type": "table",
                    "table_id": table_id,
                    "content": table_rows_to_text(rows),
                })

    documents.sort(key=lambda x: x["page"])
    return documents


if __name__ == "__main__":
    docs = parse_all("data/올인원암보험_사업방법서.pdf")
    for d in docs:
        print(f"\n=== p{d['page']} ({d['type']}) ===")
        print(d["content"])
