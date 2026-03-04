from __future__ import annotations

import io
import sys
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional
from zoneinfo import ZoneInfo

import iching_text
import wenwang as yarrow_app
from models import RunResult


_DATASET: Dict[int, Dict[str, Any]] | None = None


def load_dataset() -> Dict[int, Dict[str, Any]]:
    global _DATASET
    if _DATASET is None:
        _DATASET = iching_text.load_wilhelm_dataset()
    return _DATASET


def run_cast_once(
    delay_sec: float = 0.0,
    show_banners: bool = False,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> RunResult:
    dt_kst = datetime.now(ZoneInfo("Asia/Seoul"))
    ts_iso = dt_kst.isoformat(timespec="seconds")

    if progress_callback is not None:
        try:
            progress_callback("시초 생성 시작 (1효→6효)")
        except Exception:
            pass

    lines, traces_by_line = yarrow_app.make_six_lines_yarrow(
        delay_sec=delay_sec,
        show_banners=show_banners,
        progress_callback=progress_callback,
    )

    if progress_callback is not None:
        try:
            progress_callback("시초 6효 생성 완료. 문왕괘/주역 텍스트 정리 중...")
        except Exception:
            pass
    base = yarrow_app.compute_hex_info(lines)
    changed_lines, changing_idx = yarrow_app.compute_changed(lines)
    changed_hex = yarrow_app.compute_hex_info(changed_lines)

    pillars = yarrow_app.four_pillars(dt_kst, use_zi_day_boundary=True)
    ww = yarrow_app.build_wenwang_report(lines, base, pillars)
    ww_rows = ww["rows"]

    buf = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf
    try:
        now_kst_str = dt_kst.strftime("%Y.%m.%d %H:%M:%S")
        print(f"실행 시간 (KST): {now_kst_str}")
        print("-" * 60)

        print("Yarrow result :", base.printed)
        print(f"BASE (본괘)   : #{base.no} {base.name_zh}  ({base.upper[2]} over {base.lower[2]})")
        print("Changing lines:", changing_idx if changing_idx else "None")
        print("Changed result:", changed_hex.printed)
        print(f"CHANGED(변괘) : #{changed_hex.no} {changed_hex.name_zh}  ({changed_hex.upper[2]} over {changed_hex.lower[2]})")

        print("\n--- Yarrow traces per line (bottom -> top) ---")
        for i, (v, traces) in enumerate(zip(lines, traces_by_line), start=1):
            afters = [t["after"] for t in traces]
            steps = " | ".join(yarrow_app._fmt_change(t) for t in traces)
            print(f"Line {i}: afters={' '.join(map(str, afters))}  -> line={v}  steps={steps}")

        yarrow_app.print_wenwang(ww)
    finally:
        sys.stdout = old_stdout

    return RunResult(
        ts_iso=ts_iso,
        lines=lines,
        changed_lines=changed_lines,
        changing_idx=changing_idx,
        base_no=base.no,
        base_name=base.name_zh,
        base_upper_symbol=base.upper[2],
        base_lower_symbol=base.lower[2],
        changed_no=changed_hex.no,
        changed_name=changed_hex.name_zh,
        changed_upper_symbol=changed_hex.upper[2],
        changed_lower_symbol=changed_hex.lower[2],
        output_text=buf.getvalue(),
        ww_rows=ww_rows,
    )


def build_iching_block(rr: RunResult, include_comments: bool) -> str:
    dataset = load_dataset()
    if not dataset:
        return (
            "⚠ 주역 텍스트 데이터가 없습니다.\n"
            "1) python3 app/download_iching_data.py 실행\n"
            "2) app/data/iching_wilhelm_translation.json(.js) 생성 확인\n"
        )

    base_text = iching_text.format_hex_text(
        dataset,
        rr.base_no,
        changing_idx=rr.changing_idx,
        include_comments=include_comments,
        show_lines_if_changing=True,
    )
    changed_text = iching_text.format_hex_text(
        dataset,
        rr.changed_no,
        changing_idx=None,
        include_comments=include_comments,
        show_lines_if_changing=False,
    )

    return (
        "====================\n"
        "본괘 (괘사 + 대상 + 동효)\n"
        "====================\n\n"
        f"{base_text}\n\n"
        "====================\n"
        "변괘 (괘사 + 대상)\n"
        "====================\n\n"
        f"{changed_text}"
    )


def _extract_line(text: str, prefix: str) -> str:
    for line in text.splitlines():
        if line.startswith(prefix):
            return line
    return ""


def _extract_primary_time(text: str) -> str:
    line = _extract_line(text, "실행 시간 (KST):")
    if line:
        return line
    line = _extract_line(text, "Timestamp     :")
    if line:
        return line
    return ""


def build_summary_log(rr: RunResult) -> str:
    lines = []
    tline = _extract_primary_time(rr.output_text)
    if tline:
        lines.append(tline)
    else:
        lines.append(f"실행 시간 (KST): {rr.ts_iso.replace('T', ' ')}")

    lines.append("-" * 60)
    lines.append(f"BASE (본괘)   : #{rr.base_no} {rr.base_name} ({rr.base_upper_symbol} over {rr.base_lower_symbol})")
    lines.append(f"CHANGED(변괘) : #{rr.changed_no} {rr.changed_name} ({rr.changed_upper_symbol} over {rr.changed_lower_symbol})")
    lines.append(f"Changing lines: {rr.changing_idx if rr.changing_idx else 'None'}")

    for prefix in ("Four Pillars    :", "XunKong (旬空)  :", "Palace (宫)     :"):
        row = _extract_line(rr.output_text, prefix)
        if row:
            lines.append(row)

    moving = [str(r.get("line")) for r in rr.ww_rows if r.get("moving")]
    if moving:
        lines.append(f"동효(표 기준): {', '.join(moving)}효")
    else:
        lines.append("동효(표 기준): 없음")
    return "\n".join(lines).strip() + "\n"


def format_log_for_display(rr: RunResult, summary_mode: bool) -> str:
    if summary_mode:
        return build_summary_log(rr)
    return rr.output_text


def extract_pillars_line(output_text: str) -> str:
    line = _extract_line(output_text, "Four Pillars    :")
    return line.replace("Four Pillars    :", "").strip() if line else ""
