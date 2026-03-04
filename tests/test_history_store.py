from __future__ import annotations

import json

import history_store
from models import RunResult


def _make_rr(ts: str, base_no: int, changed_no: int, changing_idx: list[int]) -> RunResult:
    return RunResult(
        ts_iso=ts,
        lines=[7, 8, 8, 7, 8, 9],
        changed_lines=[7, 8, 8, 7, 8, 8],
        changing_idx=changing_idx,
        base_no=base_no,
        base_name="本",
        base_upper_symbol="☰",
        base_lower_symbol="☱",
        changed_no=changed_no,
        changed_name="變",
        changed_upper_symbol="☷",
        changed_lower_symbol="☳",
        output_text="dummy",
        ww_rows=[],
    )


def test_append_and_load_roundtrip(tmp_path) -> None:
    history_store.configure_runtime(tmp_path)
    rr1 = _make_rr("2026-01-01T12:00:00+09:00", 1, 2, [])
    rr2 = _make_rr("2026-01-02T12:00:00+09:00", 3, 4, [2])
    history_store.append_history(rr1)
    history_store.append_history(rr2)

    loaded = history_store.load_history()
    assert len(loaded) == 2
    assert loaded[0].base_no == 1
    assert loaded[1].changing_idx == [2]


def test_query_history_combined_filters(tmp_path) -> None:
    history_store.configure_runtime(tmp_path)
    rows = [
        _make_rr("2026-01-01T12:00:00+09:00", 10, 24, [2, 4]),
        _make_rr("2026-01-02T12:00:00+09:00", 7, 7, []),
        _make_rr("2026-01-03T12:00:00+09:00", 10, 7, [1]),
    ]

    out = history_store.query_history(rows, text_query="2026-01-03", moving_only=True, base_no=10, changed_no=7)
    assert len(out) == 1
    assert out[0].ts_iso.startswith("2026-01-03")

    out2 = history_store.query_history(rows, text_query="10", moving_only=False, base_no=10, changed_no=None)
    assert len(out2) == 2


def test_migrate_legacy_history_once_and_no_duplicate(tmp_path) -> None:
    history_store.configure_runtime(tmp_path)

    legacy_rows = [
        {
            "ts": "2026-01-21 18:40:10",
            "output": (
                "BASE (본괘)   : #10 履  (☰ over ☱)\n"
                "Changing lines: [2, 4]\n"
                "CHANGED(변괘) : #24 復  (☷ over ☳)\n"
            ),
            "base_no": 10,
            "changed_no": 24,
            "changing_idx": [2, 4],
        },
        {
            "ts": "2026-01-22 10:00:00",
            "output": (
                "BASE (본괘)   : #7 師  (☷ over ☵)\n"
                "Changing lines: None\n"
                "CHANGED(변괘) : #7 師  (☷ over ☵)\n"
            ),
            "base_no": 7,
            "changed_no": 7,
            "changing_idx": [],
        },
    ]
    history_store.LEGACY_HISTORY_FILE.write_text(json.dumps(legacy_rows, ensure_ascii=False), encoding="utf-8")

    first = history_store.migrate_legacy_history_once()
    assert first.already_migrated is False
    assert first.imported == 2

    second = history_store.migrate_legacy_history_once()
    assert second.already_migrated is True

    loaded = history_store.load_history()
    assert len(loaded) == 2

