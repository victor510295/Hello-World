import json
import os

import anthropic
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from models import AnalysisRequest, AnalysisResponse, MessageRequest, MessageResponse
from prompts import ANALYSIS_SYSTEM_PROMPT, MESSAGE_SYSTEM_PROMPT

app = FastAPI(
    title="旅遊合約 AI 審查系統",
    description="協助消費者理解旅遊合約條款、識別不利條款並生成溝通訊息",
    version="1.0.0",
)

app.mount("/static", StaticFiles(directory="static"), name="static")


def get_client() -> anthropic.Anthropic:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not configured")
    return anthropic.Anthropic(api_key=api_key)


def build_document_content(documents) -> str:
    parts = []
    if documents.contract.strip():
        parts.append(f"[CONTRACT]\n{documents.contract.strip()}")
    if documents.itinerary.strip():
        parts.append(f"[ITINERARY]\n{documents.itinerary.strip()}")
    if documents.chat.strip():
        parts.append(f"[CHAT]\n{documents.chat.strip()}")

    if not parts:
        raise HTTPException(status_code=400, detail="至少需要提供一份文件內容")

    return "\n\n---\n\n".join(parts)


@app.get("/")
async def index():
    return FileResponse("static/index.html")


@app.post("/api/analyze", response_model=AnalysisResponse)
async def analyze_contract(request: AnalysisRequest):
    """Prompt A：合約健檢分析，回傳六大維度風險評估"""
    client = get_client()
    document_content = build_document_content(request.documents)

    user_message = f"""請分析以下旅遊相關文件，依照六大審查維度進行風險評估：

{document_content}

請回傳符合規定格式的 JSON 分析結果。"""

    try:
        response = client.messages.create(
            model=request.model,
            max_tokens=4096,
            system=ANALYSIS_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )
    except anthropic.APIError as e:
        raise HTTPException(status_code=502, detail=f"Claude API 錯誤：{e}")

    raw = response.content[0].text.strip()
    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=502, detail=f"無法解析 AI 回應的 JSON：{e}\n原始回應：{raw[:500]}")

    return AnalysisResponse(**data)


@app.post("/api/generate-messages", response_model=MessageResponse)
async def generate_messages(request: MessageRequest):
    """Prompt B：根據審查結果生成溝通訊息"""
    client = get_client()

    target_findings = request.findings
    if request.finding_ids:
        target_findings = [f for f in request.findings if f.id in request.finding_ids]

    if not target_findings:
        raise HTTPException(status_code=400, detail="找不到指定的 finding IDs")

    findings_json = json.dumps(
        [f.model_dump() for f in target_findings],
        ensure_ascii=False,
        indent=2,
    )

    user_message = f"""請根據以下審查發現，以「{request.tone}」語氣生成對旅遊業者的溝通訊息：

{findings_json}

請回傳符合規定格式的 JSON 訊息結果。"""

    try:
        response = client.messages.create(
            model=request.model,
            max_tokens=4096,
            system=MESSAGE_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )
    except anthropic.APIError as e:
        raise HTTPException(status_code=502, detail=f"Claude API 錯誤：{e}")

    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=502, detail=f"無法解析 AI 回應的 JSON：{e}\n原始回應：{raw[:500]}")

    return MessageResponse(**data)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
