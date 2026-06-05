import os #파일경로 처리
import fitz # PyMuPDF pdf텍스트 추출
import json #claude 응답 json 파싱
import boto3 #aws bedrock호출
import pdfplumber # 표 자동감지
from dotenv import load_dotenv #환경변수 로드

load_dotenv() # .env 파일에서 환경변수 로드

REGION = "ap-northeast-2"
LLM_MODEL_ID = "anthropic.claude-3-5-sonnet-20240620-v1:0"

client = boto3.client(  #bedrock 모델 호출 통로
    service_name="bedrock-runtime",
    region_name=REGION
)

# 표 페이지 자동 감지
# 표가 발견되면 table객체 리스트를 반환
def table_page(pdf_path: str) -> list[int]:
    table_pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages): #여기 살짝 이해안됨
            if page.find_tables(): #pdf의 선과 좌표정보로 표 구조 추론.
                table_pages.append(i+1) # pdf는 1번부터
    return table_pages

def table_parse(img_bytes: bytes) -> list[dict]:
    """
    Claude Vision으로 복합셀(병합셀) 표 파싱.
    병합셀을 각 행마다 펼쳐서 JSON으로 반환.
    """
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
                {"text": prompt}
            ]
        }],
        inferenceConfig={"maxTokens": 3000, "temperature": 0.0},
    )
 
    text = response["output"]["message"]["content"][0]["text"].strip()
    if text.startswith("```"):
        text = text.replace("```json", "").replace("```", "").strip()
 
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        print(f"[경고] JSON 파싱 실패, raw 텍스트로 저장")
        return [{"raw": text}]


# 임베딩모델이 json보다 자연어를 넣었을 때 품질이 더 좋음
def table_rows_to_text(rows: list[dict]) -> str:
    sentences = []
    for row in rows:
        parts = []
        for k, v in row.items():
            if not v or str(v) == "-":
                continue
            
            # ⬇️ 추가: nested dict 풀어내기
            if isinstance(v, dict):
                for sub_k, sub_v in v.items():
                    if sub_v and str(sub_v) != "-":
                        parts.append(f"{k}_{sub_k}은(는) {sub_v}")
            else:
                parts.append(f"{k}은(는) {v}")
        
        if parts:
            sentences.append(". ".join(parts) + ".")
    return "\n".join(sentences)

def extract_text_with_placeholders(pdf_path: str, page_num: int) -> tuple[str, list[str]]:
    """
    페이지 텍스트에서 표 영역을 [TABLE_pN_i] placeholder로 치환.
    Returns (text_with_placeholders, list_of_table_ids)
    """
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_num - 1]
        tables = page.find_tables()
        
        if not tables:
            return page.extract_text() or "", []
        
        # 표 정보 수집
        table_ids = []
        table_items = []
        for i, t in enumerate(tables):
            tid = f"TABLE_p{page_num}_{i+1}"
            table_ids.append(tid)
            table_items.append({
                "bbox": t.bbox,
                "y_top": t.bbox[1],
                "placeholder": f"[{tid}]"
            })
        
        # 단어 추출 + 표 영역 제외
        words = page.extract_words()
        non_table_words = []
        for w in words:
            in_table = any(
                w['x0'] >= t['bbox'][0] and w['x1'] <= t['bbox'][2] and
                w['top'] >= t['bbox'][1] and w['bottom'] <= t['bbox'][3]
                for t in table_items
            )
            if not in_table:
                non_table_words.append({"y": w['top'], "text": w['text']})

        # 단어와 placeholder를 y좌표 순으로 정렬
        all_items = [(w["y"], w["text"]) for w in non_table_words]
        all_items += [(t["y_top"], t["placeholder"]) for t in table_items]
        all_items.sort(key=lambda x: x[0])

        # y좌표가 비슷한 항목끼리 같은 줄로 묶고 \n으로 구분
        # (줄바꿈이 사라지면 (?m)^ 패턴이 매칭 불가 → 계층 분할 실패)
        LINE_TOLERANCE = 3  # px 이내 차이는 같은 줄로 간주
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

        text = "\n".join(" ".join(line) for line in lines)
        return text, table_ids


def parse_single_table(pdf_path: str, page_num: int, table_idx: int) -> list[dict] | None:
    """
    페이지의 특정 인덱스 표 하나만 파싱.
    복합셀 의심되면 None 반환 (Claude fallback 신호).
    
    Args:
        pdf_path: PDF 경로
        page_num: 페이지 번호 (1-base)
        table_idx: 그 페이지 내 표 인덱스 (0-base)
    """
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_num - 1]
        tables = page.find_tables()
        
        if table_idx >= len(tables):
            return None
        
        table = tables[table_idx]
        rows = table.extract()
        
        if not rows or len(rows) < 2:
            return None
        
        # 병합셀 의심: None/빈 셀 비율
        total = sum(len(r) for r in rows)
        empty = sum(1 for r in rows for c in r if c is None or not c.strip())
        
        if empty / max(total, 1) > 0.15:
            return None  # Claude로 fallback
        
        # 깨끗한 표 → dict 리스트
        headers = rows[0]
        result = []
        for row in rows[1:]:
            result.append({
                h: (v or "") for h, v in zip(headers, row)
            })
        return result

def render_table_region(pdf_path: str, page_num: int, table_idx: int) -> bytes:
    """
    페이지에서 특정 표 영역만 잘라서 PNG 렌더링.
    """
    # 표 bbox 가져오기
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_num - 1]
        tables = page.find_tables()
        bbox = tables[table_idx].bbox   # (x0, top, x1, bottom)
    
    # PyMuPDF로 해당 영역만 렌더링
    doc = fitz.open(pdf_path)
    page = doc[page_num - 1]
    
    # bbox를 fitz의 Rect로 변환 (pdfplumber와 fitz는 좌표계 동일)
    clip = fitz.Rect(bbox[0], bbox[1], bbox[2], bbox[3])
    pix = page.get_pixmap(dpi=200, clip=clip)
    return pix.tobytes("png")

def parse_all(pdf_path: str) -> list[dict]:
    documents = []
    table_pages = table_page(pdf_path)
    
    # 텍스트 추출 (표 페이지는 placeholder로 치환)
    doc = fitz.open(pdf_path)
    for i, page in enumerate(doc):
        page_num = i + 1
        if page_num in table_pages:
            text, table_ids = extract_text_with_placeholders(pdf_path, page_num)
        else:
            text = page.get_text()
            table_ids = []
        
        if text.strip():
            documents.append({
                "page": page_num,
                "type": "text",
                "content": text.strip()
            })
    
    # 표 파싱 (각 표마다 고유 ID 부여)
    for page_num in table_pages:
        with pdfplumber.open(pdf_path) as pdf:
            page = pdf.pages[page_num - 1]
            tables_in_page = page.find_tables()
        
        for i, _ in enumerate(tables_in_page):
            table_id = f"TABLE_p{page_num}_{i+1}"
            
            # 이 표만 따로 파싱
            rows = parse_single_table(pdf_path, page_num, i)
            if not rows:
                # Claude fallback
                img = render_table_region(pdf_path, page_num, i)
                rows = table_parse(img)
            
            documents.append({
                "page": page_num,
                "type": "table",
                "table_id": table_id,     # ← placeholder와 매칭 키
                "content": table_rows_to_text(rows)
            })
    
    documents.sort(key=lambda x: x["page"])
    return documents
 
 
if __name__ == "__main__":
    docs = parse_all("data/올인원암보험_사업방법서.pdf")
    for d in docs:
        print(f"\n=== p{d['page']} ({d['type']}) ===")
        print(d["content"])

