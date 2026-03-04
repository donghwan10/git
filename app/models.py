from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List
from zoneinfo import ZoneInfo


KST = ZoneInfo("Asia/Seoul")

_BASE_RE = re.compile(
    r"BASE \(본괘\)\s*:\s*#(?P<no>\d+)\s+(?P<name>\S+)\s+\((?P<upper>\S+)\s+over\s+(?P<lower>[^)]+)\)"
)
_CHANGED_RE = re.compile(
    r"CHANGED\(변괘\)\s*:\s*#(?P<no>\d+)\s+(?P<name>\S+)\s+\((?P<upper>\S+)\s+over\s+(?P<lower>[^)]+)\)"
)
_CHANGING_RE = re.compile(r"Changing lines:\s*(?P<value>.+)")


def _parse_ts_to_iso(ts_raw: str) -> str:
    raw = (ts_raw or "").strip()
    if not raw:
        return ""

    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y.%m.%d %H:%M", "%Y.%m.%d %H:%M:%S"):
        try:
            dt = datetime.strptime(raw, fmt).replace(tzinfo=KST)
            return dt.isoformat(timespec="seconds")
        except ValueError:
            continue
    try:
        dt = datetime.fromisoformat(raw)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=KST)
        return dt.isoformat(timespec="seconds")
    except ValueError:
        return raw


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _parse_changing_idx_from_output(output_text: str) -> List[int]:
    m = _CHANGING_RE.search(output_text or "")
    if not m:
        return []
    raw = m.group("value").strip()
    if raw.lower() == "none":
        return []
    if raw.startswith("[") and raw.endswith("]"):
        inner = raw[1:-1].strip()
        if not inner:
            return []
        out: List[int] = []
        for tok in inner.split(","):
            tok = tok.strip()
            if tok.isdigit():
                out.append(int(tok))
        return out
    return []


def _parse_names_from_output(output_text: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "base_no": 0,
        "base_name": "",
        "base_upper_symbol": "",
        "base_lower_symbol": "",
        "changed_no": 0,
        "changed_name": "",
        "changed_upper_symbol": "",
        "changed_lower_symbol": "",
    }

    bm = _BASE_RE.search(output_text or "")
    if bm:
        out.update(
            {
                "base_no": _safe_int(bm.group("no")),
                "base_name": bm.group("name"),
                "base_upper_symbol": bm.group("upper"),
                "base_lower_symbol": bm.group("lower"),
            }
        )
    cm = _CHANGED_RE.search(output_text or "")
    if cm:
        out.update(
            {
                "changed_no": _safe_int(cm.group("no")),
                "changed_name": cm.group("name"),
                "changed_upper_symbol": cm.group("upper"),
                "changed_lower_symbol": cm.group("lower"),
            }
        )
    return out


@dataclass
class RunResult:
    ts_iso: str
    lines: List[int]
    changed_lines: List[int]
    changing_idx: List[int]
    base_no: int
    base_name: str
    base_upper_symbol: str
    base_lower_symbol: str
    changed_no: int
    changed_name: str
    changed_upper_symbol: str
    changed_lower_symbol: str
    output_text: str
    ww_rows: List[Dict[str, Any]]

    @staticmethod
    def from_dict(obj: Dict[str, Any]) -> "RunResult":
        return RunResult(
            ts_iso=obj.get("ts_iso", ""),
            lines=obj.get("lines", []),
            changed_lines=obj.get("changed_lines", []),
            changing_idx=obj.get("changing_idx", []),
            base_no=obj.get("base_no", 0),
            base_name=obj.get("base_name", ""),
            base_upper_symbol=obj.get("base_upper_symbol", ""),
            base_lower_symbol=obj.get("base_lower_symbol", ""),
            changed_no=obj.get("changed_no", 0),
            changed_name=obj.get("changed_name", ""),
            changed_upper_symbol=obj.get("changed_upper_symbol", ""),
            changed_lower_symbol=obj.get("changed_lower_symbol", ""),
            output_text=obj.get("output_text", ""),
            ww_rows=obj.get("ww_rows", []),
        )

    @staticmethod
    def from_legacy_dict(obj: Dict[str, Any]) -> "RunResult":
        output_text = str(obj.get("output", ""))
        parsed = _parse_names_from_output(output_text)

        ts_iso = _parse_ts_to_iso(str(obj.get("ts", "")).strip())
        base_no = _safe_int(obj.get("base_no"), parsed["base_no"])
        changed_no = _safe_int(obj.get("changed_no"), parsed["changed_no"])

        changing_idx = obj.get("changing_idx")
        if not isinstance(changing_idx, list):
            changing_idx = _parse_changing_idx_from_output(output_text)
        else:
            changing_idx = [x for x in changing_idx if isinstance(x, int)]

        return RunResult(
            ts_iso=ts_iso,
            lines=[],
            changed_lines=[],
            changing_idx=changing_idx,
            base_no=base_no,
            base_name=parsed["base_name"],
            base_upper_symbol=parsed["base_upper_symbol"],
            base_lower_symbol=parsed["base_lower_symbol"],
            changed_no=changed_no,
            changed_name=parsed["changed_name"],
            changed_upper_symbol=parsed["changed_upper_symbol"],
            changed_lower_symbol=parsed["changed_lower_symbol"],
            output_text=output_text,
            ww_rows=[],
        )


@dataclass
class UIState:
    selected_tab: int = 0
    summary_mode: bool = True
    history_filter_text: str = ""
    moving_only: bool = False


@dataclass
class MigrationResult:
    already_migrated: bool = False
    attempted: int = 0
    imported: int = 0
    skipped: int = 0
    failed: int = 0
    message: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "already_migrated": self.already_migrated,
            "attempted": self.attempted,
            "imported": self.imported,
            "skipped": self.skipped,
            "failed": self.failed,
            "message": self.message,
        }
