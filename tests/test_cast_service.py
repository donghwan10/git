from __future__ import annotations

import cast_service
from models import RunResult


def _rr(output_text: str, changing_idx: list[int], ww_rows: list[dict]) -> RunResult:
    return RunResult(
        ts_iso="2026-02-09T12:30:00+09:00",
        lines=[7, 8, 8, 7, 8, 8],
        changed_lines=[7, 8, 8, 7, 8, 8],
        changing_idx=changing_idx,
        base_no=7,
        base_name="師",
        base_upper_symbol="☷",
        base_lower_symbol="☵",
        changed_no=7,
        changed_name="師",
        changed_upper_symbol="☷",
        changed_lower_symbol="☵",
        output_text=output_text,
        ww_rows=ww_rows,
    )


def test_build_summary_log_no_moving_lines() -> None:
    rr = _rr(
        output_text=(
            "실행 시간 (KST): 2026.02.09 12:30:00\n"
            "Four Pillars    : 丙午年  庚寅月  甲子日  甲子時\n"
            "XunKong (旬空)  : 戌亥\n"
            "Palace (宫)     : kan宫(水)  |  世=3  應=6\n"
        ),
        changing_idx=[],
        ww_rows=[],
    )
    text = cast_service.build_summary_log(rr)
    assert "실행 시간 (KST): 2026.02.09 12:30:00" in text
    assert "동효(표 기준): 없음" in text
    assert "Four Pillars    :" in text


def test_build_summary_log_with_moving_lines() -> None:
    rr = _rr(
        output_text="Timestamp     : 2026.02.09 12:30\n",
        changing_idx=[2, 6],
        ww_rows=[{"line": 2, "moving": True}, {"line": 6, "moving": True}],
    )
    text = cast_service.build_summary_log(rr)
    assert "Timestamp     : 2026.02.09 12:30" in text
    assert "Changing lines: [2, 6]" in text
    assert "동효(표 기준): 2, 6효" in text

