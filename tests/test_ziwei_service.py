from __future__ import annotations

import ziwei_service


def test_compute_ziwei_payload_shape() -> None:
    payload = ziwei_service.compute_ziwei_payload(
        ziwei_service.ZiweiInput(
            birth_date="1996-05-04",
            birth_time="08:10",
            gender="male",
            calendar_type="solar",
            lunar_leap_month=False,
        )
    )

    palaces = payload["palaces"]
    assert len(palaces) == 12
    assert sum(1 for p in palaces if p["is_ming"]) == 1
    assert sum(1 for p in palaces if p["is_shen"]) == 1
    assert set(payload["four_transforms"].keys()) == {"natal", "decadal", "yearly"}
    assert "normalized" in payload
    assert "chartMeta" in payload
    assert "indexes" in payload
    assert "relations" in payload
    assert "transformations" in payload
    assert "horoscope" in payload
    assert "dynamicPalaces" in payload
    assert "debug" in payload

    for key in ("calendarType", "year", "month", "day", "hour", "minute", "gender", "isLeapMonth", "timezone"):
        assert key in payload["input"]
    for key in ("solarDate", "lunarDate", "fourPillars", "timeLabel", "timeRange", "timeIndex", "dayShiftedByTimeRule"):
        assert key in payload["normalized"]


def test_build_chart_view_model_filters_and_modes() -> None:
    payload = ziwei_service.compute_ziwei_payload(
        ziwei_service.ZiweiInput(
            birth_date="1992-11-12",
            birth_time="23:40",
            gender="female",
            calendar_type="solar",
            lunar_leap_month=False,
        )
    )

    vm_triad = ziwei_service.build_chart_view_model(
        payload=payload,
        view_mode="triad",
        scope="natal",
        star_filter="major",
        show_hanja=False,
        selected_palace_index=None,
    )
    vm_transform = ziwei_service.build_chart_view_model(
        payload=payload,
        view_mode="four_transform",
        scope="yearly",
        star_filter="all",
        show_hanja=True,
        selected_palace_index=3,
    )

    assert len(vm_triad["cards"]) == 12
    assert vm_triad["summary"]["scope"] == "natal"
    assert vm_transform["summary"]["scope"] == "yearly"
    assert vm_transform["selected_detail"]["index"] == 3

    triad_preview_len = len(vm_triad["cards"][0]["stars_preview"])
    transform_preview_len = len(vm_transform["cards"][0]["stars_preview"])
    assert triad_preview_len <= transform_preview_len


def test_palace_detail_schema() -> None:
    payload = ziwei_service.compute_ziwei_payload(
        ziwei_service.ZiweiInput(
            birth_date="1988-07-21",
            birth_time="14:25",
            gender="male",
            calendar_type="solar",
            lunar_leap_month=False,
        )
    )
    p0 = payload["palaces"][0]
    expected_keys = {
        "palaceId",
        "labelKo",
        "labelZh",
        "index",
        "heavenlyStem",
        "earthlyBranch",
        "isLifePalace",
        "isBodyPalace",
        "isOriginalPalace",
        "isEmptyPalace",
        "oppositePalaceId",
        "threeHarmonyPalaceIds",
        "fourCardinalPalaceIds",
        "stage",
        "ages",
        "changsheng12",
        "boshi12",
        "jiangqian12",
        "suiqian12",
        "stars",
    }
    assert expected_keys.issubset(set(p0.keys()))

    stars = p0["stars"]
    assert {"majorStars", "minorStars", "adjectiveStars"}.issubset(set(stars.keys()))
    if stars["majorStars"]:
        s0 = stars["majorStars"][0]
        for key in (
            "starId",
            "labelKo",
            "labelZh",
            "type",
            "scope",
            "brightness",
            "isTransformed",
            "transformType",
            "sourceStem",
            "sourceScope",
        ):
            assert key in s0


def test_rule_based_determinism_and_transform_table() -> None:
    inp = ziwei_service.ZiweiInput(
        birth_date="2026-03-03",
        birth_time="11:30",
        gender="male",
        calendar_type="solar",
        lunar_leap_month=False,
    )
    p1 = ziwei_service.compute_ziwei_payload(inp)
    p2 = ziwei_service.compute_ziwei_payload(inp)

    assert p1["debug"]["checksum"] == p2["debug"]["checksum"]

    year_stem = p1["normalized"]["fourPillars"]["year"]["stem"]
    expected = ziwei_service.FOUR_TRANSFORM_RULES.get(year_stem, ziwei_service.FOUR_TRANSFORM_RULES["甲"])
    natal_by_star = p1["transformations"]["byScope"]["natal"]["byStar"]
    actual = {row["transformType"]: row["starId"] for row in natal_by_star}
    assert actual == expected

    for palace in p1["palaces"]:
        for key in ("majorStars", "minorStars", "adjectiveStars"):
            for row in palace["stars"][key]:
                assert row["brightness"] in ziwei_service.BRIGHTNESS_ORDER


def test_reference_lunar_layout_matches_expected_core_positions() -> None:
    payload = ziwei_service.compute_ziwei_payload(
        ziwei_service.ZiweiInput(
            birth_date="2026-01-15",
            birth_time="11:00",
            gender="male",
            calendar_type="lunar",
            lunar_leap_month=False,
        )
    )

    assert payload["chartMeta"]["fiveElementsClass"] == "화6국"
    assert payload["profile"]["ming_palace_index"] == 8
    assert payload["profile"]["shen_palace_index"] == 8

    expected_major = {
        0: ("관록궁", ["天機"]),
        1: ("노복궁", ["紫微", "破軍"]),
        2: ("천이궁", []),
        3: ("질액궁", ["天府"]),
        4: ("재백궁", ["太陰"]),
        5: ("자녀궁", ["廉貞", "貪狼"]),
        6: ("부처궁", ["巨門"]),
        7: ("형제궁", ["天相"]),
        8: ("명궁", ["天同", "天梁"]),
        9: ("부모궁", ["武曲", "七殺"]),
        10: ("복덕궁", ["太陽"]),
        11: ("전택궁", []),
    }

    for idx, palace in enumerate(payload["palaces"]):
        major_zh = [row["labelZh"] for row in palace["stars"]["majorStars"]]
        exp_name, exp_stars = expected_major[idx]
        assert palace["labelKo"] == exp_name
        assert major_zh == exp_stars


def test_require_exact_calendar_mode() -> None:
    inp = ziwei_service.ZiweiInput(
        birth_date="2026-03-03",
        birth_time="11:00",
        gender="male",
        calendar_type="solar",
        lunar_leap_month=False,
        require_exact_calendar=True,
    )
    if ziwei_service.KoreanLunarCalendar is None:
        try:
            ziwei_service.compute_ziwei_payload(inp)
        except RuntimeError:
            pass
        else:
            raise AssertionError("require_exact_calendar=True 에서 RuntimeError가 발생해야 합니다.")
    else:
        payload = ziwei_service.compute_ziwei_payload(inp)
        assert payload["normalized"]["lunarDate"]["month"] >= 1
