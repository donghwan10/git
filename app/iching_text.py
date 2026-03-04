from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"


def _load_json_like(path: Path) -> Dict[int, Dict[str, Any]]:
    """
    Supports:
      - JSON file: {"1": {...}, "2": {...}, ...}
      - JS file  : export default {...}  OR  module.exports = {...}
    Returns:
      - {1: {...}, 2: {...}, ...}
    """
    raw = path.read_text(encoding="utf-8", errors="ignore").strip()

    # JS wrapper 제거 (대충 안전하게)
    if raw.startswith("export default"):
        raw = raw[len("export default"):].strip()
    if raw.startswith("module.exports"):
        raw = raw.split("=", 1)[1].strip()

    # 끝의 세미콜론 제거
    if raw.endswith(";"):
        raw = raw[:-1].strip()

    data = json.loads(raw)

    # 키가 문자열("1")일 가능성이 높음 → int로 변환
    out: Dict[int, Dict[str, Any]] = {}
    for k, v in data.items():
        try:
            out[int(k)] = v
        except Exception:
            continue
    return out


def load_wilhelm_dataset() -> Dict[int, Dict[str, Any]]:
    """
    Load local file:
      app/data/iching_wilhelm_translation.json OR .js
    """
    json_path = DATA_DIR / "iching_wilhelm_translation.json"
    js_path = DATA_DIR / "iching_wilhelm_translation.js"

    if json_path.exists():
        return _load_json_like(json_path)
    if js_path.exists():
        return _load_json_like(js_path)
    return {}


def format_hex_text(
    dataset: Dict[int, Dict[str, Any]],
    hex_no: int,
    changing_idx: Optional[List[int]] = None,
    include_comments: bool = False,
    show_lines_if_changing: bool = True,
) -> str:
    """
    본괘: judgment + image + (동효 line text)
    변괘: judgment + image (changing_idx 무시 or show_lines_if_changing=False)
    """
    h = dataset.get(hex_no)
    if not h:
        return f"[주역 텍스트 없음] hex={hex_no}\n(데이터 파일이 없거나 로드 실패)\n"

    title = f"#{hex_no} {h.get('trad_chinese','')} ({h.get('pinyin','')}) - {h.get('english','')}"
    out = [title, "-" * len(title)]

    # 괘사 = wilhelm_judgment
    j = h.get("wilhelm_judgment", {})
    j_text = (j.get("text") or "").strip()
    if j_text:
        out.append("[괘사 / Judgment]")
        out.append(j_text)
        out.append("")
        if include_comments and j.get("comments"):
            out.append("[괘사 주석 / Comments]")
            out.append(str(j["comments"]).strip())
            out.append("")

    # 대상 = wilhelm_image
    im = h.get("wilhelm_image", {})
    im_text = (im.get("text") or "").strip()
    if im_text:
        out.append("[대상 / Image]")
        out.append(im_text)
        out.append("")
        if include_comments and im.get("comments"):
            out.append("[대상 주석 / Comments]")
            out.append(str(im["comments"]).strip())
            out.append("")

    # 동효 = wilhelm_lines
    if show_lines_if_changing and changing_idx:
        lines = h.get("wilhelm_lines", {})
        out.append("[동효 / Changing Lines]")
        for idx in changing_idx:
            node = lines.get(str(idx)) or lines.get(idx) or {}
            t = (node.get("text") or "").strip()
            c = (node.get("comments") or "").strip()
            out.append(f"• {idx}효:")
            out.append(t if t else "(해당 효 텍스트 없음)")
            if include_comments and c:
                out.append("  - 주석:")
                out.append("    " + c.replace("\n", "\n    "))
            out.append("")
    return "\n".join(out).strip() + "\n"
