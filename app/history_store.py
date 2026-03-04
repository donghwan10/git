from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional

from models import MigrationResult, RunResult, UIState


def base_runtime_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


BASE_DIR = Path(".")
HISTORY_DIR = Path(".")
EXPORT_DIR = Path(".")
HISTORY_FILE = Path(".")
LEGACY_HISTORY_FILE = Path(".")
UI_STATE_FILE = Path(".")
MIGRATION_META_FILE = Path(".")


def configure_runtime(base_dir: Optional[Path] = None) -> None:
    global BASE_DIR
    global HISTORY_DIR
    global EXPORT_DIR
    global HISTORY_FILE
    global LEGACY_HISTORY_FILE
    global UI_STATE_FILE
    global MIGRATION_META_FILE

    BASE_DIR = (base_dir or base_runtime_dir()).resolve()
    HISTORY_DIR = BASE_DIR / "history"
    EXPORT_DIR = HISTORY_DIR / "exports"
    HISTORY_FILE = HISTORY_DIR / "history.jsonl"
    LEGACY_HISTORY_FILE = BASE_DIR / "history.json"
    UI_STATE_FILE = HISTORY_DIR / "ui_state.json"
    MIGRATION_META_FILE = HISTORY_DIR / "migration_meta.json"

    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)


def _record_key(rr: RunResult) -> tuple:
    return (
        rr.ts_iso,
        rr.base_no,
        rr.changed_no,
        tuple(rr.changing_idx),
        rr.output_text[:160],
    )


def _safe_int(text: str) -> Optional[int]:
    text = (text or "").strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def load_history() -> List[RunResult]:
    if not HISTORY_FILE.exists():
        return []
    out: List[RunResult] = []
    with HISTORY_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(RunResult.from_dict(json.loads(line)))
            except json.JSONDecodeError:
                continue
    return out


def append_history(rr: RunResult) -> None:
    with HISTORY_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rr.__dict__, ensure_ascii=False) + "\n")


def clear_history() -> None:
    if HISTORY_FILE.exists():
        HISTORY_FILE.unlink()


def query_history(
    records: Iterable[RunResult],
    text_query: str = "",
    moving_only: bool = False,
    base_no: Optional[int] = None,
    changed_no: Optional[int] = None,
) -> List[RunResult]:
    query = (text_query or "").strip().lower()
    out: List[RunResult] = []

    for rr in records:
        if moving_only and not rr.changing_idx:
            continue
        if base_no is not None and rr.base_no != base_no:
            continue
        if changed_no is not None and rr.changed_no != changed_no:
            continue
        if query:
            search_blob = " ".join(
                [
                    rr.ts_iso,
                    str(rr.base_no),
                    rr.base_name,
                    str(rr.changed_no),
                    rr.changed_name,
                    " ".join(str(x) for x in rr.changing_idx),
                ]
            ).lower()
            if query not in search_blob:
                continue
        out.append(rr)
    return out


def load_ui_state() -> UIState:
    if not UI_STATE_FILE.exists():
        return UIState()
    try:
        raw = json.loads(UI_STATE_FILE.read_text(encoding="utf-8"))
        return UIState(
            selected_tab=int(raw.get("selected_tab", 0)),
            summary_mode=bool(raw.get("summary_mode", True)),
            history_filter_text=str(raw.get("history_filter_text", "")),
            moving_only=bool(raw.get("moving_only", False)),
        )
    except Exception:
        return UIState()


def save_ui_state(state: UIState) -> None:
    UI_STATE_FILE.write_text(
        json.dumps(state.__dict__, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def parse_int_filter(text: str) -> Optional[int]:
    return _safe_int(text)


def _load_migration_meta() -> dict:
    if not MIGRATION_META_FILE.exists():
        return {}
    try:
        return json.loads(MIGRATION_META_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def migrate_legacy_history_once() -> MigrationResult:
    meta = _load_migration_meta()
    if meta.get("done") is True:
        return MigrationResult(
            already_migrated=True,
            attempted=int(meta.get("attempted", 0)),
            imported=int(meta.get("imported", 0)),
            skipped=int(meta.get("skipped", 0)),
            failed=int(meta.get("failed", 0)),
            message="이미 구형 데이터 병합이 완료되었습니다.",
        )

    result = MigrationResult()
    existing = load_history()
    existing_keys = {_record_key(x) for x in existing}

    legacy_rows = []
    if LEGACY_HISTORY_FILE.exists():
        try:
            loaded = json.loads(LEGACY_HISTORY_FILE.read_text(encoding="utf-8"))
            if isinstance(loaded, list):
                legacy_rows = loaded
        except Exception:
            result.failed += 1

    for row in legacy_rows:
        result.attempted += 1
        try:
            rr = RunResult.from_legacy_dict(row)
            key = _record_key(rr)
            if key in existing_keys:
                result.skipped += 1
                continue
            append_history(rr)
            existing_keys.add(key)
            result.imported += 1
        except Exception:
            result.failed += 1

    if not legacy_rows:
        result.message = "구형 history.json이 없어 병합할 데이터가 없습니다."
    else:
        result.message = (
            f"구형 데이터 병합 완료: 총 {result.attempted}건, "
            f"추가 {result.imported}건, 건너뜀 {result.skipped}건, 실패 {result.failed}건"
        )

    meta_payload = {
        "done": True,
        "migrated_at": datetime.now().isoformat(timespec="seconds"),
        "attempted": result.attempted,
        "imported": result.imported,
        "skipped": result.skipped,
        "failed": result.failed,
        "message": result.message,
    }
    MIGRATION_META_FILE.write_text(
        json.dumps(meta_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result


configure_runtime()

