from __future__ import annotations

from models import RunResult


def test_from_legacy_dict_success() -> None:
    row = {
        "ts": "2026-01-21 18:40:10",
        "base_no": 10,
        "changed_no": 24,
        "changing_idx": [2, 4, 5, 6],
        "output": (
            "Timestamp     : 2026.01.21 18:40\n"
            "BASE (본괘)   : #10 履  (☰ over ☱)\n"
            "Changing lines: [2, 4, 5, 6]\n"
            "CHANGED(변괘) : #24 復  (☷ over ☳)\n"
        ),
    }

    rr = RunResult.from_legacy_dict(row)
    assert rr.ts_iso.startswith("2026-01-21T18:40:10")
    assert rr.base_no == 10
    assert rr.changed_no == 24
    assert rr.base_name == "履"
    assert rr.changed_name == "復"
    assert rr.base_upper_symbol == "☰"
    assert rr.changed_lower_symbol == "☳"
    assert rr.changing_idx == [2, 4, 5, 6]


def test_from_legacy_dict_missing_and_invalid_fields() -> None:
    row = {
        "ts": "not-a-time",
        "base_no": "x",
        "changed_no": None,
        "changing_idx": "not-a-list",
        "output": (
            "BASE (본괘)   : #7 師  (☷ over ☵)\n"
            "Changing lines: None\n"
            "CHANGED(변괘) : #7 師  (☷ over ☵)\n"
        ),
    }

    rr = RunResult.from_legacy_dict(row)
    assert rr.ts_iso == "not-a-time"
    assert rr.base_no == 7
    assert rr.changed_no == 7
    assert rr.changing_idx == []
    assert rr.base_name == "師"
    assert rr.changed_name == "師"

