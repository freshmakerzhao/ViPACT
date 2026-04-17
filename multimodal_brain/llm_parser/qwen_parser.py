from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict
from interfaces import InstructionParseResult


@dataclass
class ParsedInstruction:
    text_prompt: str
    axis: str
    reverse: bool
    target_rank: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text_prompt": str(self.text_prompt),
            "axis": str(self.axis),
            "reverse": bool(self.reverse),
            "target_rank": int(self.target_rank),
        }


def _extract_json_block(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    if not text:
        raise ValueError("Empty LLM response")
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        raise ValueError(f"Response has no JSON object: {text[:300]}")
    return json.loads(m.group(0))


def _heuristic_parse(instruction: str) -> ParsedInstruction:
    s = str(instruction)
    sl = s.lower()

    # axis + direction
    if ("左" in s) or ("右" in s) or ("left" in sl) or ("right" in sl):
        axis = "x"
        reverse = ("右" in s) or ("right" in sl)
    elif ("上" in s) or ("下" in s) or ("top" in sl) or ("bottom" in sl):
        axis = "y"
        reverse = ("下" in s) or ("bottom" in sl)
    else:
        axis = "x"
        reverse = False

    rank = 1
    zh_map = {"第一": 1, "第二": 2, "第三": 3, "第四": 4, "第五": 5}
    for k, v in zh_map.items():
        if k in s:
            rank = v
            break
    if rank == 1:
        m = re.search(r"\b(\d+)(st|nd|rd|th)?\b", sl)
        if m:
            rank = max(1, int(m.group(1)))

    # text prompt
    if ("红" in s) or ("red" in sl):
        text_prompt = "red cube"
    elif ("绿" in s) or ("green" in sl):
        text_prompt = "green cube"
    elif ("蓝" in s) or ("blue" in sl):
        text_prompt = "blue cube"
    elif ("黄" in s) or ("yellow" in sl):
        text_prompt = "yellow cube"
    else:
        text_prompt = "cube"

    return ParsedInstruction(
        text_prompt=text_prompt,
        axis=axis,
        reverse=reverse,
        target_rank=rank,
    )


class QwenInstructionParser:
    """Parse natural language instruction into structured spatial rules."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        model: str = "qwen-plus",
        timeout_sec: float = 30.0,
    ):
        self.api_key = str(api_key or "").strip()
        self.base_url = str(base_url).rstrip("/")
        self.model = str(model)
        self.timeout_sec = float(timeout_sec)

    @staticmethod
    def _system_prompt() -> str:
        return (
            "你是机器人指令解析器。"
            "必须将用户抓取指令解析为严格 JSON，且只输出 JSON。"
            "字段固定为："
            "{\"text_prompt\": string, \"axis\": \"x\"|\"y\", \"reverse\": bool, \"target_rank\": int}。"
            "空间词按图像坐标系解释：左/右对应 x 轴；上/下对应 y 轴。"
            "左/上为 reverse=false；右/下为 reverse=true。"
            "第几个就是 target_rank 几；最左/最右/最上/最下都视为 target_rank=1。"
            "text_prompt 只保留可用于视觉检测的短语，如 red cube。"
        )

    @staticmethod
    def _normalize_result(obj: Dict[str, Any], instruction: str) -> ParsedInstruction:
        text_prompt = str(obj.get("text_prompt", "")).strip()
        axis = str(obj.get("axis", "x")).strip().lower()
        reverse = bool(obj.get("reverse", False))
        target_rank = int(obj.get("target_rank", 1))

        if axis not in ("x", "y"):
            axis = "x"
        if target_rank < 1:
            target_rank = 1
        if not text_prompt:
            text_prompt = _heuristic_parse(instruction).text_prompt

        return ParsedInstruction(
            text_prompt=text_prompt,
            axis=axis,
            reverse=reverse,
            target_rank=target_rank,
        )

    def _call_qwen(self, instruction: str) -> Dict[str, Any]:
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self._system_prompt()},
                {"role": "user", "content": str(instruction)},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.0,
        }
        req = urllib.request.Request(
            url=url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                body = resp.read().decode("utf-8", errors="ignore")
        except urllib.error.HTTPError as e:
            err = e.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"Qwen API HTTPError {e.code}: {err[:600]}") from e
        except Exception as e:
            raise RuntimeError(f"Qwen API request failed: {e}") from e

        obj = json.loads(body)
        choices = obj.get("choices", [])
        if not choices:
            raise RuntimeError("Qwen response has no choices")
        content = choices[0].get("message", {}).get("content", "")
        return _extract_json_block(content)

    def parse(self, instruction: str) -> ParsedInstruction:
        if not self.api_key:
            return _heuristic_parse(instruction)
        try:
            obj = self._call_qwen(instruction)
            return self._normalize_result(obj, instruction=instruction)
        except Exception:
            return _heuristic_parse(instruction)

    def parse_typed(self, instruction: str) -> InstructionParseResult:
        parsed = self.parse(instruction)
        return InstructionParseResult(
            text_prompt=str(parsed.text_prompt),
            axis=str(parsed.axis),
            reverse=bool(parsed.reverse),
            target_rank=int(parsed.target_rank),
            raw=parsed.to_dict(),
        )


def parse_instruction_with_qwen(
    instruction: str,
    *,
    api_key: str,
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
    model: str = "qwen-plus",
) -> Dict[str, Any]:
    parser = QwenInstructionParser(api_key=api_key, base_url=base_url, model=model)
    return parser.parse(instruction).to_dict()
