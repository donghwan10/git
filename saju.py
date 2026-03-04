"""
saju_gui.py
- 출생지: 서울(Asia/Seoul) 고정
- 출생시간: 선택 입력(미입력 가능)
- 출생일 달력: 양력/음력 선택(음력 입력 시 양력으로 변환 계산)
- 성별: 남/여 입력(대운 순행/역행 계산에 사용)
- 운세 조회일: 선택 입력(일운/시운 계산용)
- 자시(子時) 경계 정책: GUI 옵션(전통 23시 교체 vs 자정 00시 교체)
- 출력 상세도: GUI 옵션(pillars_only / summary / report / full)
- 추가 계산: 오행/십신/지장간/12운성(옵션)
"""

from __future__ import annotations

from datetime import datetime, date, time, timedelta, timezone
from zoneinfo import ZoneInfo
from functools import lru_cache
from typing import Optional, Dict, List, Tuple, Literal, Any
import math
import json
import tkinter as tk
from tkinter import ttk, messagebox
try:
    from korean_lunar_calendar import KoreanLunarCalendar
except Exception:
    KoreanLunarCalendar = None  # type: ignore[misc, assignment]

# =============================================================================
# 0) Constants / Locale
# =============================================================================

SEOUL_TZ = ZoneInfo("Asia/Seoul")
UTC = timezone.utc
UNIX_EPOCH_UTC = datetime(1970, 1, 1, tzinfo=UTC)

ZiPolicy = Literal["traditional_23", "midnight_00"]
DetailLevel = Literal["pillars_only", "summary", "full", "report"]
Gender = Literal["male", "female"]
CalendarType = Literal["solar", "lunar"]
SchoolPreset = Literal["modern_japyung_common"]

DEFAULT_SCHOOL: SchoolPreset = "modern_japyung_common"
SCHOOL_INFO: Dict[SchoolPreset, Dict[str, Any]] = {
    "modern_japyung_common": {
        "id": "modern_japyung_common",
        "name_ko": "현대 자평명리(한국 실무형)",
        "rules": [
            "연주 경계: 입춘(立春) 절입 시각",
            "월주 경계: 12절(절입) 기준",
            "일주 기준: 그레고리력 날짜(JDN) 기준",
            "시주 기준: 2시간 단위 시지",
            "자시 경계: 기본 전통 야자시(23시 교체)",
            "지장간: 정해 만세력 실무 표(초기/중기/본기)",
        ],
    }
}

STEMS_HANJA = ["甲", "乙", "丙", "丁", "戊", "己", "庚", "辛", "壬", "癸"]
STEMS_KOR = ["갑", "을", "병", "정", "무", "기", "경", "신", "임", "계"]

BRANCHES_HANJA = ["子", "丑", "寅", "卯", "辰", "巳", "午", "未", "申", "酉", "戌", "亥"]
BRANCHES_KOR = ["자", "축", "인", "묘", "진", "사", "오", "미", "신", "유", "술", "해"]

ELEMENTS_HANJA = ["木", "火", "土", "金", "水"]
ELEMENTS_KOR = ["목", "화", "토", "금", "수"]

TEN_GODS = {
    "비견": "比肩",
    "겁재": "劫財",
    "식신": "食神",
    "상관": "傷官",
    "편재": "偏財",
    "정재": "正財",
    "편관": "偏官",
    "정관": "正官",
    "편인": "偏印",
    "정인": "正印",
}

TWELVE_STAGES = ["长生", "沐浴", "冠带", "临官", "帝旺", "衰", "病", "死", "墓", "绝", "胎", "养"]
TWELVE_STAGES_KOR = ["장생", "목욕", "관대", "건록", "제왕", "쇠", "병", "사", "묘", "절", "태", "양"]

TWELVE_SALS = ["년살", "월살", "망신살", "장성살", "반안살", "역마살", "육해살", "화개살", "겁살", "재살", "천살", "지살"]

SOLAR_TERM_KOR = {
    "小寒": "소한",
    "立春": "입춘",
    "驚蟄": "경칩",
    "清明": "청명",
    "立夏": "입하",
    "芒種": "망종",
    "小暑": "소서",
    "立秋": "입추",
    "白露": "백로",
    "寒露": "한로",
    "立冬": "입동",
    "大雪": "대설",
}

MONTH_BRANCH_START_TERM = {
    0: "大雪",  # 子
    1: "小寒",  # 丑
    2: "立春",  # 寅
    3: "驚蟄",  # 卯
    4: "清明",  # 辰
    5: "立夏",  # 巳
    6: "芒種",  # 午
    7: "小暑",  # 未
    8: "立秋",  # 申
    9: "白露",  # 酉
    10: "寒露", # 戌
    11: "立冬", # 亥
}

# 12운성: 각 천간의 “长生 시작 지지”
CHANGSHENG_START_BRANCH: Dict[int, int] = {
    0: 11,  # 甲: 亥
    1: 6,   # 乙: 午
    2: 2,   # 丙: 寅
    3: 9,   # 丁: 酉
    4: 2,   # 戊: 寅
    5: 9,   # 己: 酉
    6: 5,   # 庚: 巳
    7: 0,   # 辛: 子
    8: 8,   # 壬: 申
    9: 3,   # 癸: 卯
}

# 지장간(정해 만세력 표 기반: 초기/중기/본기)
HIDDEN_STEMS_JEONGHAE: Dict[int, List[Tuple[int, str]]] = {
    0: [(8, "초기"), (9, "본기")],                 # 子: 壬, 癸
    1: [(9, "초기"), (7, "중기"), (5, "본기")],    # 丑: 癸, 辛, 己
    2: [(4, "초기"), (2, "중기"), (0, "본기")],    # 寅: 戊, 丙, 甲
    3: [(0, "초기"), (1, "본기")],                 # 卯: 甲, 乙
    4: [(1, "초기"), (9, "중기"), (4, "본기")],    # 辰: 乙, 癸, 戊
    5: [(4, "초기"), (6, "중기"), (2, "본기")],    # 巳: 戊, 庚, 丙
    6: [(2, "초기"), (5, "중기"), (3, "본기")],    # 午: 丙, 己, 丁
    7: [(3, "초기"), (1, "중기"), (5, "본기")],    # 未: 丁, 乙, 己
    8: [(4, "초기"), (8, "중기"), (6, "본기")],    # 申: 戊, 壬, 庚
    9: [(6, "초기"), (7, "본기")],                 # 酉: 庚, 辛
    10: [(7, "초기"), (3, "중기"), (4, "본기")],   # 戌: 辛, 丁, 戊
    11: [(4, "초기"), (0, "중기"), (8, "본기")],   # 亥: 戊, 甲, 壬
}

# =============================================================================
# 1) Basic helpers
# =============================================================================

def is_yang_stem(stem_i: int) -> bool:
    return (stem_i % 2) == 0  # 甲丙戊庚壬

def stem_element_index(stem_i: int) -> int:
    return stem_i // 2  # 甲乙木(0), 丙丁火(1), 戊己土(2), 庚辛金(3), 壬癸水(4)

def branch_element_index(branch_i: int) -> int:
    # 지지 자체의 대표 오행(지장간까지 합치면 더 정교)
    mapping = [4, 2, 0, 0, 2, 1, 1, 2, 3, 3, 2, 4]
    return mapping[branch_i]

def fmt_yinyang(is_yang: bool) -> str:
    return "陽" if is_yang else "陰"

def stem_meta(stem_i: int) -> Dict[str, Any]:
    e = stem_element_index(stem_i)
    return {
        "stem_index": stem_i,
        "hanja": STEMS_HANJA[stem_i],
        "kor": STEMS_KOR[stem_i],
        "yin_yang": fmt_yinyang(is_yang_stem(stem_i)),
        "element": {"index": e, "hanja": ELEMENTS_HANJA[e], "kor": ELEMENTS_KOR[e]},
    }

def pillar_str(stem_i: int, branch_i: int) -> str:
    return f"{STEMS_HANJA[stem_i]}{BRANCHES_HANJA[branch_i]}({STEMS_KOR[stem_i]}{BRANCHES_KOR[branch_i]})"

# =============================================================================
# 2) Julian day number + day pillar
# =============================================================================

def jdn_from_gregorian(d: date) -> int:
    y, m, day = d.year, d.month, d.day
    a = (14 - m) // 12
    y2 = y + 4800 - a
    m2 = m + 12 * a - 3
    jdn = day + (153 * m2 + 2) // 5 + 365 * y2 + y2 // 4 - y2 // 100 + y2 // 400 - 32045
    return jdn

def day_stem_branch_from_jdn_noon(jdn_noon: int) -> Tuple[int, int]:
    T = 1 + ((jdn_noon - 1) % 10)   # 1..10
    B = 1 + ((jdn_noon + 1) % 12)   # 1..12
    return (T - 1, B - 1)

def lunar_to_solar_date_kr(lunar_date_str: str, is_leap_month: bool = False) -> date:
    if KoreanLunarCalendar is None:
        raise RuntimeError("음력 변환 기능을 사용하려면 `korean_lunar_calendar` 패키지가 필요합니다.")

    y, m, d = map(int, lunar_date_str.split("-"))
    cal = KoreanLunarCalendar()
    ok = cal.setLunarDate(y, m, d, is_leap_month)
    if not ok:
        leap_txt = " (윤달)" if is_leap_month else ""
        raise ValueError(f"유효하지 않은 음력 날짜입니다: {lunar_date_str}{leap_txt}")

    sy, sm, sd = map(int, cal.SolarIsoFormat().split("-"))
    return date(sy, sm, sd)

def normalize_birth_date_input(
    birth_date_str: str,
    calendar_type: CalendarType,
    lunar_leap_month: bool,
) -> Tuple[date, Optional[str]]:
    if calendar_type == "solar":
        y, m, d = map(int, birth_date_str.split("-"))
        return date(y, m, d), None

    solar_date = lunar_to_solar_date_kr(birth_date_str, is_leap_month=lunar_leap_month)
    leap_txt = " (윤달)" if lunar_leap_month else ""
    note = f"입력 출생일(음력{leap_txt}) {birth_date_str}을 양력 {solar_date.isoformat()}로 변환해 계산했습니다."
    return solar_date, note

# =============================================================================
# 3) Solar terms (approx) for year/month boundaries
# =============================================================================

def normalize_deg(x: float) -> float:
    x = x % 360.0
    return x if x >= 0 else x + 360.0

def diff_deg(a: float, b: float) -> float:
    d = (a - b + 180.0) % 360.0 - 180.0
    return d

def dt_to_jd(dt_utc: datetime) -> float:
    if dt_utc.tzinfo is None:
        raise ValueError("dt_utc must be timezone-aware (UTC).")
    ts = (dt_utc.astimezone(UTC) - UNIX_EPOCH_UTC).total_seconds()
    return ts / 86400.0 + 2440587.5

def jd_to_dt_utc(jd: float) -> datetime:
    ts = (jd - 2440587.5) * 86400.0
    # Windows에서 1970년 이전 `fromtimestamp()`가 OSError(Errno 22)를 내는 경우가 있어
    # 에폭 + timedelta 방식으로 변환해 플랫폼 차이를 피한다.
    return UNIX_EPOCH_UTC + timedelta(seconds=ts)

def sun_apparent_longitude_deg(jd: float) -> float:
    # Low-order approximation (MVP)
    T = (jd - 2451545.0) / 36525.0
    L0 = 280.46646 + 36000.76983 * T + 0.0003032 * T * T
    L0 = normalize_deg(L0)
    M = 357.52911 + 35999.05029 * T - 0.0001537 * T * T
    M = math.radians(normalize_deg(M))
    C = (1.914602 - 0.004817 * T - 0.000014 * T * T) * math.sin(M) \
        + (0.019993 - 0.000101 * T) * math.sin(2 * M) \
        + 0.000289 * math.sin(3 * M)
    true_long = L0 + C
    Omega = math.radians(normalize_deg(125.04 - 1934.136 * T))
    lam = true_long - 0.00569 - 0.00478 * math.sin(Omega)
    return normalize_deg(lam)

def find_solar_term_time_utc(
    guess_utc: datetime,
    target_longitude_deg: float,
    search_days: int = 15,
    step_hours: int = 6,
    bisect_iters: int = 60,
) -> datetime:
    if guess_utc.tzinfo is None:
        raise ValueError("guess_utc must be timezone-aware (UTC).")

    target = normalize_deg(target_longitude_deg)

    def f(jd: float) -> float:
        return diff_deg(sun_apparent_longitude_deg(jd), target)

    start = guess_utc - timedelta(days=search_days)
    end = guess_utc + timedelta(days=search_days)

    prev_dt = start
    prev_jd = dt_to_jd(prev_dt)
    prev_f = f(prev_jd)

    found = None
    dt = start + timedelta(hours=step_hours)
    while dt <= end:
        jd = dt_to_jd(dt)
        cur_f = f(jd)
        if prev_f == 0.0:
            return prev_dt
        if cur_f == 0.0:
            return dt
        if (prev_f < 0 <= cur_f) or (prev_f > 0 >= cur_f):
            found = (prev_jd, jd, prev_f, cur_f)
            break
        prev_dt, prev_jd, prev_f = dt, jd, cur_f
        dt += timedelta(hours=step_hours)

    if not found:
        raise RuntimeError("Failed to bracket solar term root (try wider search window).")

    lo_jd, hi_jd, lo_f, hi_f = found
    for _ in range(bisect_iters):
        mid_jd = (lo_jd + hi_jd) / 2.0
        mid_f = f(mid_jd)
        if mid_f == 0.0:
            lo_jd = hi_jd = mid_jd
            break
        if (lo_f < 0 <= mid_f) or (lo_f > 0 >= mid_f):
            hi_jd, hi_f = mid_jd, mid_f
        else:
            lo_jd, lo_f = mid_jd, mid_f

    return jd_to_dt_utc((lo_jd + hi_jd) / 2.0)

SOLAR_TERMS_12 = [
    ("小寒", 285.0, 1, 5),
    ("立春", 315.0, 2, 4),
    ("驚蟄", 345.0, 3, 6),
    ("清明", 15.0, 4, 5),
    ("立夏", 45.0, 5, 5),
    ("芒種", 75.0, 6, 6),
    ("小暑", 105.0, 7, 7),
    ("立秋", 135.0, 8, 7),
    ("白露", 165.0, 9, 7),
    ("寒露", 195.0, 10, 8),
    ("立冬", 225.0, 11, 7),
    ("大雪", 255.0, 12, 7),
]

@lru_cache(maxsize=256)
def solar_terms_12_kst(year: int) -> Dict[str, datetime]:
    out: Dict[str, datetime] = {}
    for name, target_deg, mm, dd in SOLAR_TERMS_12:
        guess = datetime(year, mm, dd, 12, 0, tzinfo=UTC)
        dt_utc = find_solar_term_time_utc(guess, target_deg)
        out[name] = dt_utc.astimezone(SEOUL_TZ)
    return out

def get_term_kst(year: int, name: str) -> datetime:
    return solar_terms_12_kst(year)[name]

# =============================================================================
# 4) Pillars: year/month/day/hour
# =============================================================================

def year_pillar(dt_local: datetime) -> Tuple[int, int, int]:
    y = dt_local.year
    lichun = get_term_kst(y, "立春")
    cyc_year = y if dt_local >= lichun else y - 1
    stem_i = (cyc_year - 4) % 10
    branch_i = (cyc_year - 4) % 12
    return stem_i, branch_i, cyc_year

def month_branch_index(dt_local: datetime) -> int:
    y = dt_local.year
    terms_prev = solar_terms_12_kst(y - 1)
    terms_cur = solar_terms_12_kst(y)

    boundaries: List[Tuple[datetime, int]] = [
        (terms_prev["大雪"], 0),  # 子
        (terms_cur["小寒"], 1),   # 丑
        (terms_cur["立春"], 2),   # 寅
        (terms_cur["驚蟄"], 3),   # 卯
        (terms_cur["清明"], 4),   # 辰
        (terms_cur["立夏"], 5),   # 巳
        (terms_cur["芒種"], 6),   # 午
        (terms_cur["小暑"], 7),   # 未
        (terms_cur["立秋"], 8),   # 申
        (terms_cur["白露"], 9),   # 酉
        (terms_cur["寒露"], 10),  # 戌
        (terms_cur["立冬"], 11),  # 亥
        (terms_cur["大雪"], 0),   # 子 (late year)
    ]
    boundaries.sort(key=lambda x: x[0])

    mb = boundaries[0][1]
    for t, b in boundaries:
        if dt_local >= t:
            mb = b
        else:
            break
    return mb

def month_pillar(dt_local: datetime) -> Tuple[int, int]:
    y_stem, _, _ = year_pillar(dt_local)
    m_branch = month_branch_index(dt_local)
    # 寅월(2) 월간
    yin_stem = (y_stem * 2 + 2) % 10
    delta = (m_branch - 2) % 12
    m_stem = (yin_stem + delta) % 10
    return m_stem, m_branch

def hour_branch_index(dt_local: datetime) -> int:
    h = dt_local.hour
    if h == 23:
        return 0
    return (h + 1) // 2

def hour_pillar(day_stem: int, dt_local: datetime) -> Tuple[int, int]:
    hb = hour_branch_index(dt_local)
    zi_stem = (day_stem * 2) % 10
    hs = (zi_stem + hb) % 10
    return hs, hb

# =============================================================================
# 5) Ten-gods / Hidden stems / 12 stages / 5 elements
# =============================================================================

def generates(a_elem: int, b_elem: int) -> bool:
    return (a_elem + 1) % 5 == b_elem

def overcomes(a_elem: int, b_elem: int) -> bool:
    return {0: 2, 2: 4, 4: 1, 1: 3, 3: 0}[a_elem] == b_elem

def ten_god(day_stem_i: int, target_stem_i: int) -> Dict[str, str]:
    dm_e = stem_element_index(day_stem_i)
    tg_e = stem_element_index(target_stem_i)
    dm_yang = is_yang_stem(day_stem_i)
    tg_yang = is_yang_stem(target_stem_i)
    same_polarity = (dm_yang == tg_yang)

    if dm_e == tg_e:
        key = "비견" if same_polarity else "겁재"
    elif generates(dm_e, tg_e):
        key = "식신" if same_polarity else "상관"
    elif overcomes(dm_e, tg_e):
        key = "편재" if same_polarity else "정재"
    elif overcomes(tg_e, dm_e):
        key = "편관" if same_polarity else "정관"
    elif generates(tg_e, dm_e):
        key = "편인" if same_polarity else "정인"
    else:
        key = "비견"

    return {"kor": key, "hanja": TEN_GODS[key]}

def hidden_stems(branch_i: int) -> List[Dict[str, Any]]:
    items = []
    for stem_i, role in HIDDEN_STEMS_JEONGHAE[branch_i]:
        items.append({"role": role, **stem_meta(stem_i)})
    return items

def twelve_stage(day_stem_i: int, branch_i: int) -> Dict[str, Any]:
    cs = CHANGSHENG_START_BRANCH[day_stem_i]
    direction = 1 if is_yang_stem(day_stem_i) else -1
    steps = (branch_i - cs) % 12 if direction == 1 else (cs - branch_i) % 12
    return {"name": TWELVE_STAGES[steps], "index": steps}

def element_count_visible(pillars: Dict[str, Tuple[int, int]]) -> List[float]:
    counts = [0.0] * 5
    for _, (s, b) in pillars.items():
        counts[stem_element_index(s)] += 1.0
        counts[branch_element_index(b)] += 1.0
    return counts

def element_count_with_hidden_weighted(pillars: Dict[str, Tuple[int, int]]) -> List[float]:
    counts = element_count_visible(pillars)

    for _, (_, b) in pillars.items():
        hs = HIDDEN_STEMS_JEONGHAE[b]
        if len(hs) == 3:
            weights = {"초기": 0.3, "중기": 0.1, "본기": 0.6}
        elif len(hs) == 2:
            weights = {"초기": 0.3, "본기": 0.7}
        else:
            weights = {"본기": 1.0}
        for stem_i, role in hs:
            counts[stem_element_index(stem_i)] += weights.get(role, 0.0)

    return counts

def counts_to_obj(counts: List[float], mode: str) -> Dict[str, Any]:
    return {
        "mode": mode,
        "counts": [
            {"element": {"hanja": ELEMENTS_HANJA[i], "kor": ELEMENTS_KOR[i]}, "value": counts[i]}
            for i in range(5)
        ],
    }

def school_meta(school: SchoolPreset, zi_policy: ZiPolicy) -> Dict[str, Any]:
    base = SCHOOL_INFO[school]
    return {
        "id": base["id"],
        "name_ko": base["name_ko"],
        "rules": base["rules"],
        "zi_policy_applied": zi_policy,
    }

def ganji_hanja(stem_i: int, branch_i: int) -> str:
    return f"{STEMS_HANJA[stem_i]}{BRANCHES_HANJA[branch_i]}"

def branch_main_stem_index(branch_i: int) -> int:
    for stem_i, role in HIDDEN_STEMS_JEONGHAE[branch_i]:
        if role == "본기":
            return stem_i
    return HIDDEN_STEMS_JEONGHAE[branch_i][-1][0]

def twelve_stage_kor(day_stem_i: int, branch_i: int) -> str:
    stage = twelve_stage(day_stem_i, branch_i)
    return TWELVE_STAGES_KOR[stage["index"]]

def sexagenary_shift(stem_i: int, branch_i: int, steps: int) -> Tuple[int, int]:
    return (stem_i + steps) % 10, (branch_i + steps) % 12

def twelve_sal_from_year_branch(base_year_branch_i: int, target_branch_i: int) -> str:
    offset = (target_branch_i - base_year_branch_i) % 12
    return TWELVE_SALS[offset]

def daewoon_direction(year_stem_i: int, gender: Gender) -> int:
    # 양남/음녀 순행, 음남/양녀 역행
    is_yang_year = is_yang_stem(year_stem_i)
    if gender == "male":
        return 1 if is_yang_year else -1
    return -1 if is_yang_year else 1

def daewoon_direction_label(direction: int) -> str:
    return "순행" if direction > 0 else "역행"

def _all_terms_around(dt_local: datetime) -> List[Tuple[datetime, str]]:
    years = [dt_local.year - 1, dt_local.year, dt_local.year + 1]
    out: List[Tuple[datetime, str]] = []
    for yy in years:
        terms = solar_terms_12_kst(yy)
        for name, _, _, _ in SOLAR_TERMS_12:
            out.append((terms[name], name))
    out.sort(key=lambda x: x[0])
    return out

def daewoon_start_age(
    birth_dt_local: datetime,
    direction: int,
) -> Tuple[int, datetime, str, float]:
    terms = _all_terms_around(birth_dt_local)

    target: Optional[Tuple[datetime, str]] = None
    if direction > 0:
        for t in terms:
            if t[0] > birth_dt_local:
                target = t
                break
    else:
        for t in reversed(terms):
            if t[0] <= birth_dt_local:
                target = t
                break

    if target is None:
        raise RuntimeError("Failed to find adjacent solar term for Daewoon start age.")

    delta_days = abs((target[0] - birth_dt_local).total_seconds()) / 86400.0
    # 한국 실무에서 자주 쓰는 환산: 3일 = 1년
    start_age = max(1, math.ceil(delta_days / 3.0))
    return start_age, target[0], target[1], delta_days

def format_float_cell(x: float) -> str:
    if abs(x - round(x)) < 1e-9:
        return str(int(round(x)))
    return f"{x:.2f}"

def current_age_korean_style(birth_d: date, today_d: date) -> int:
    age = today_d.year - birth_d.year
    if (today_d.month, today_d.day) < (birth_d.month, birth_d.day):
        age -= 1
    return age

def _pair_key(a: int, b: int) -> Tuple[int, int]:
    return (a, b) if a <= b else (b, a)

GAN_COMBINE = {_pair_key(0, 5), _pair_key(1, 6), _pair_key(2, 7), _pair_key(3, 8), _pair_key(4, 9)}
GAN_CLASH = {_pair_key(0, 6), _pair_key(1, 7), _pair_key(2, 8), _pair_key(3, 9)}

BRANCH_LIUHAP = {_pair_key(0, 1), _pair_key(2, 11), _pair_key(3, 10), _pair_key(4, 9), _pair_key(5, 8), _pair_key(6, 7)}
BRANCH_CLASH = {_pair_key(0, 6), _pair_key(1, 7), _pair_key(2, 8), _pair_key(3, 9), _pair_key(4, 10), _pair_key(5, 11)}
BRANCH_BREAK = {_pair_key(0, 9), _pair_key(1, 4), _pair_key(2, 11), _pair_key(3, 6), _pair_key(5, 8), _pair_key(7, 10)}
BRANCH_HARM = {_pair_key(0, 7), _pair_key(1, 6), _pair_key(2, 5), _pair_key(3, 4), _pair_key(8, 11), _pair_key(9, 10)}
BRANCH_PUNISH_PAIRS = {
    _pair_key(2, 5), _pair_key(5, 8), _pair_key(2, 8),  # 寅巳申
    _pair_key(1, 7), _pair_key(7, 10), _pair_key(1, 10), # 丑未戌
    _pair_key(0, 3),                                      # 子卯
}
BRANCH_SELF_PUNISH = {4, 6, 9, 11}  # 辰午酉亥

def branch_is_punish(a: int, b: int) -> bool:
    if a == b and a in BRANCH_SELF_PUNISH:
        return True
    return _pair_key(a, b) in BRANCH_PUNISH_PAIRS

def compute_heavenly_relations(pillars_idx: Dict[str, Tuple[int, int]]) -> List[Dict[str, str]]:
    names = [k for k in ["year", "month", "day", "hour"] if k in pillars_idx]
    out: List[Dict[str, str]] = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            k1, k2 = names[i], names[j]
            s1 = pillars_idx[k1][0]
            s2 = pillars_idx[k2][0]
            pair = _pair_key(s1, s2)
            pair_name = f"{STEMS_HANJA[s1]}{STEMS_HANJA[s2]}"
            pillar_pair = f"{k1}-{k2}"
            if pair in GAN_CLASH:
                out.append({
                    "type": "충",
                    "pair": pair_name,
                    "pillars": pillar_pair,
                    "desc": f"{pair_name}충",
                })
            if pair in GAN_COMBINE:
                out.append({
                    "type": "합",
                    "pair": pair_name,
                    "pillars": pillar_pair,
                    "desc": f"{pair_name}합",
                })
    return out

def compute_earthly_relations(pillars_idx: Dict[str, Tuple[int, int]]) -> List[Dict[str, str]]:
    names = [k for k in ["year", "month", "day", "hour"] if k in pillars_idx]
    out: List[Dict[str, str]] = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            k1, k2 = names[i], names[j]
            b1 = pillars_idx[k1][1]
            b2 = pillars_idx[k2][1]
            pair = _pair_key(b1, b2)
            pair_name = f"{BRANCHES_HANJA[b1]}{BRANCHES_HANJA[b2]}"
            pillar_pair = f"{k1}-{k2}"

            if pair in BRANCH_LIUHAP:
                out.append({"type": "육합", "pair": pair_name, "pillars": pillar_pair})
            if pair in BRANCH_CLASH:
                out.append({"type": "충", "pair": pair_name, "pillars": pillar_pair})
            if branch_is_punish(b1, b2):
                out.append({"type": "형", "pair": pair_name, "pillars": pillar_pair})
            if pair in BRANCH_BREAK:
                out.append({"type": "파", "pair": pair_name, "pillars": pillar_pair})
            if pair in BRANCH_HARM:
                out.append({"type": "해", "pair": pair_name, "pillars": pillar_pair})
    return out

def build_daewoon_rows(
    birth_date_obj: date,
    birth_dt_local: datetime,
    gender: Gender,
    year_stem_i: int,
    year_branch_i: int,
    month_stem_i: int,
    month_branch_i: int,
    day_stem_i: int,
    count: int = 10,
) -> Dict[str, Any]:
    direction = daewoon_direction(year_stem_i, gender)
    start_age, ref_term_dt, ref_term_name, delta_days = daewoon_start_age(birth_dt_local, direction)
    direction_label = daewoon_direction_label(direction)
    today_local = datetime.now(SEOUL_TZ).date()
    current_age = current_age_korean_style(birth_date_obj, today_local)
    rows = []

    for i in range(count):
        step = direction * (i + 1)
        s_i, b_i = sexagenary_shift(month_stem_i, month_branch_i, step)
        age_start = start_age + i * 10
        rows.append({
            "age_start": age_start,
            "age_end": age_start + 9,
            "stem_index": s_i,
            "branch_index": b_i,
            "ganji": ganji_hanja(s_i, b_i),
            "start_year": birth_date_obj.year + age_start,
            "direction": direction_label,
            "current": "●" if age_start <= current_age <= age_start + 9 else "",
            "sal": twelve_sal_from_year_branch(year_branch_i, b_i),
            "stage": twelve_stage_kor(day_stem_i, b_i),
        })

    return {
        "direction": direction_label,
        "start_age": start_age,
        "reference_term_hanja": ref_term_name,
        "reference_term_kor": SOLAR_TERM_KOR.get(ref_term_name, ref_term_name),
        "reference_term_at": ref_term_dt.isoformat(),
        "delta_days": delta_days,
        "rows": rows,
    }

def build_sewoon_rows(
    base_year_branch_i: int,
    day_stem_i: int,
    year_to: Optional[int] = None,
    years: int = 10,
) -> List[Dict[str, Any]]:
    to_year = year_to if year_to is not None else datetime.now(SEOUL_TZ).year
    from_year = to_year - years + 1
    out: List[Dict[str, Any]] = []
    for yy in range(from_year, to_year + 1):
        s_i = (yy - 4) % 10
        b_i = (yy - 4) % 12
        out.append({
            "year": yy,
            "stem_index": s_i,
            "branch_index": b_i,
            "ganji": ganji_hanja(s_i, b_i),
            "sal": twelve_sal_from_year_branch(base_year_branch_i, b_i),
            "stage": twelve_stage_kor(day_stem_i, b_i),
        })
    return out

def build_monthly_rows(
    target_year: int,
    base_year_branch_i: int,
    day_stem_i: int,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for mm in range(12, 0, -1):
        dt_mid = datetime(target_year, mm, 15, 12, 0, tzinfo=SEOUL_TZ)
        m_s, m_b = month_pillar(dt_mid)
        main_stem = branch_main_stem_index(m_b)
        term_hanja = MONTH_BRANCH_START_TERM[m_b]
        out.append({
            "month": mm,
            "month_name": f"{mm}월",
            "stem_index": m_s,
            "branch_index": m_b,
            "ganji": ganji_hanja(m_s, m_b),
            "sal": twelve_sal_from_year_branch(base_year_branch_i, m_b),
            "stage": twelve_stage_kor(day_stem_i, m_b),
            "stem_ten_god": ten_god(day_stem_i, m_s)["kor"],
            "branch_ten_god": ten_god(day_stem_i, main_stem)["kor"],
            "term": SOLAR_TERM_KOR.get(term_hanja, term_hanja),
        })
    return out

def build_daily_hourly_fortune(
    target_date_obj: date,
    day_master_stem_i: int,
    natal_year_branch_i: int,
    zi_policy: ZiPolicy,
) -> Dict[str, Any]:
    # 일운(조회일의 일주)
    day_jdn = jdn_from_gregorian(target_date_obj)
    day_s, day_b = day_stem_branch_from_jdn_noon(day_jdn)
    day_main_branch_stem = branch_main_stem_index(day_b)

    daily = {
        "date": target_date_obj.isoformat(),
        "day_ganji": ganji_hanja(day_s, day_b),
        "day_stem_ten_god": ten_god(day_master_stem_i, day_s)["kor"],
        "day_branch_ten_god": ten_god(day_master_stem_i, day_main_branch_stem)["kor"],
        "stage": twelve_stage_kor(day_master_stem_i, day_b),
        "sal": twelve_sal_from_year_branch(natal_year_branch_i, day_b),
    }

    # 조회일의 전체 12시운
    # 자시(23~00:59)부터 해시(21~22:59)까지 표준 시각 구간
    hour_windows: List[Tuple[str, int]] = [
        ("23:00-00:59", 23), ("01:00-02:59", 1), ("03:00-04:59", 3), ("05:00-06:59", 5),
        ("07:00-08:59", 7), ("09:00-10:59", 9), ("11:00-12:59", 11), ("13:00-14:59", 13),
        ("15:00-16:59", 15), ("17:00-18:59", 17), ("19:00-20:59", 19), ("21:00-22:59", 21),
    ]

    hourly_rows = []
    for label, hh in hour_windows:
        dt_local = datetime(target_date_obj.year, target_date_obj.month, target_date_obj.day, hh, 30, tzinfo=SEOUL_TZ)

        eff_day = target_date_obj
        if hh == 23 and zi_policy == "traditional_23":
            eff_day = target_date_obj + timedelta(days=1)

        eff_jdn = jdn_from_gregorian(eff_day)
        eff_day_s, eff_day_b = day_stem_branch_from_jdn_noon(eff_jdn)
        h_s, h_b = hour_pillar(eff_day_s, dt_local)
        h_main_branch_stem = branch_main_stem_index(h_b)

        hourly_rows.append({
            "time_range": label,
            "hour_branch": BRANCHES_HANJA[h_b],
            "hour_ganji": ganji_hanja(h_s, h_b),
            "stem_ten_god": ten_god(day_master_stem_i, h_s)["kor"],
            "branch_ten_god": ten_god(day_master_stem_i, h_main_branch_stem)["kor"],
            "stage": twelve_stage_kor(day_master_stem_i, h_b),
            "sal": twelve_sal_from_year_branch(natal_year_branch_i, h_b),
            "effective_day": eff_day.isoformat(),
            "effective_day_ganji": ganji_hanja(eff_day_s, eff_day_b),
        })

    return {
        "daily": daily,
        "hourly": hourly_rows,
        "zi_policy_applied": zi_policy,
    }

def build_extended_detail(
    birth_date_obj: date,
    birth_dt_for_terms: datetime,
    gender: Gender,
    pillars_idx: Dict[str, Tuple[int, int]],
    day_stem_i: int,
    fortune_date_obj: Optional[date],
    zi_policy: ZiPolicy,
) -> Dict[str, Any]:
    year_s, year_b = pillars_idx["year"]
    month_s, month_b = pillars_idx["month"]

    pillar_values = {
        k: {
            "stem_index": s,
            "branch_index": b,
            "ganji_hanja": ganji_hanja(s, b),
            "ganji_full": pillar_str(s, b),
        }
        for k, (s, b) in pillars_idx.items()
    }

    analysis = {}
    for key in ["hour", "day", "month", "year"]:
        if key not in pillars_idx:
            continue
        s_i, b_i = pillars_idx[key]
        hidden = hidden_stems(b_i)
        analysis[key] = {
            "stem_ten_god": "비견" if key == "day" else ten_god(day_stem_i, s_i)["kor"],
            "main_branch_ten_god": ten_god(day_stem_i, branch_main_stem_index(b_i))["kor"],
            "hidden_ten_gods": [ten_god(day_stem_i, x["stem_index"])["kor"] for x in hidden],
            "stage_kor": twelve_stage_kor(day_stem_i, b_i),
        }

    branch_sals = {
        key: twelve_sal_from_year_branch(year_b, pillars_idx[key][1])
        for key in ["year", "month", "day", "hour"]
        if key in pillars_idx
    }

    daewoon = build_daewoon_rows(
        birth_date_obj=birth_date_obj,
        birth_dt_local=birth_dt_for_terms,
        gender=gender,
        year_stem_i=year_s,
        year_branch_i=year_b,
        month_stem_i=month_s,
        month_branch_i=month_b,
        day_stem_i=day_stem_i,
    )

    sewoon = build_sewoon_rows(
        base_year_branch_i=year_b,
        day_stem_i=day_stem_i,
    )

    monthly = build_monthly_rows(
        target_year=datetime.now(SEOUL_TZ).year,
        base_year_branch_i=year_b,
        day_stem_i=day_stem_i,
    )

    daily_hourly = None
    if fortune_date_obj is not None:
        daily_hourly = build_daily_hourly_fortune(
            target_date_obj=fortune_date_obj,
            day_master_stem_i=day_stem_i,
            natal_year_branch_i=year_b,
            zi_policy=zi_policy,
        )

    return {
        "pillars": pillar_values,
        "analysis": analysis,
        "daewoon": daewoon,
        "sewoon": sewoon,
        "monthly": monthly,
        "daily_hourly": daily_hourly,
        "heavenly_relations": compute_heavenly_relations(pillars_idx),
        "earthly_relations": compute_earthly_relations(pillars_idx),
        "branch_sals": branch_sals,
    }

def markdown_table(headers: List[str], rows: List[List[str]]) -> str:
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join([":--:" for _ in headers]) + " |")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)

# =============================================================================
# 6) Single-result API (no variants)
# =============================================================================

def _prune_detail(detail: Dict[str, Any], level: DetailLevel) -> Dict[str, Any]:
    if level == "pillars_only":
        return {}
    if level in ("full", "report"):
        return detail

    # summary
    out: Dict[str, Any] = {}
    out["day_master"] = detail.get("day_master")
    out["ten_gods"] = detail.get("ten_gods")
    stages = detail.get("twelve_stages", {})
    out["twelve_stages"] = {k: v.get("name") for k, v in stages.items() if isinstance(v, dict)}
    fe = detail.get("five_elements", {})
    out["five_elements"] = {
        "visible": fe.get("visible"),
        "visible_plus_hidden": fe.get("visible_plus_hidden"),
    }
    return out

def compute_saju_seoul_single(
    birth_date: str,
    birth_time: Optional[str],
    zi_policy: ZiPolicy,
    include_hidden: bool,
    detail_level: DetailLevel,
    gender: Gender = "male",
    school: SchoolPreset = DEFAULT_SCHOOL,
    fortune_date: Optional[str] = None,
    calendar_type: CalendarType = "solar",
    lunar_leap_month: bool = False,
) -> Dict[str, Any]:
    notes: List[str] = []
    assumed_time_for_terms: Optional[str] = None
    base_date, conversion_note = normalize_birth_date_input(
        birth_date_str=birth_date,
        calendar_type=calendar_type,
        lunar_leap_month=lunar_leap_month,
    )
    if conversion_note:
        notes.append(conversion_note)
    y, m, d = base_date.year, base_date.month, base_date.day

    fortune_date_obj: Optional[date] = None
    if fortune_date:
        fy, fm, fd = map(int, fortune_date.split("-"))
        fortune_date_obj = date(fy, fm, fd)

    # ---- Time unknown: use 12:00 KST for term boundary decisions (single result)
    if not birth_time:
        assumed_time_for_terms = "12:00"
        dt_local = datetime(y, m, d, 12, 0, tzinfo=SEOUL_TZ)

        # (Optional) warn if term boundary within the day
        dt_start = datetime.combine(base_date, time(0, 0), tzinfo=SEOUL_TZ)
        dt_end = datetime.combine(base_date, time(23, 59), tzinfo=SEOUL_TZ)
        if year_pillar(dt_start) != year_pillar(dt_end) or month_pillar(dt_start) != month_pillar(dt_end):
            notes.append("경고: 절기 경계(절입) 근처일 수 있어, 실제 출생시각에 따라 연주/월주가 달라질 가능성이 있습니다.")

        # Year / Month
        y_s, y_b, _ = year_pillar(dt_local)
        mo_s, mo_b = month_pillar(dt_local)

        # Day (no zi-policy shift when time unknown)
        jdn = jdn_from_gregorian(base_date)
        d_s, d_b = day_stem_branch_from_jdn_noon(jdn)

        pillars_idx: Dict[str, Tuple[int, int]] = {
            "year": (y_s, y_b),
            "month": (mo_s, mo_b),
            "day": (d_s, d_b),
        }

        dm = stem_meta(d_s)
        ten_gods_map = {
            "year_stem": ten_god(d_s, y_s),
            "month_stem": ten_god(d_s, mo_s),
            "day_stem": {"kor": "일간", "hanja": "日干"},
        }
        stages = {f"{k}_branch": twelve_stage(d_s, b) for k, (_, b) in pillars_idx.items()}

        detail: Dict[str, Any] = {
            "day_master": dm,
            "ten_gods": ten_gods_map,
            "twelve_stages": stages,
            "hidden_stems": {},
            "five_elements": {
                "visible": counts_to_obj(element_count_visible(pillars_idx), "visible_6_chars"),
                "visible_plus_hidden": None,
            },
        }

        if include_hidden:
            detail["hidden_stems"] = {
                f"{k}_branch_hidden_stems": [
                    {**hs, "ten_god": ten_god(d_s, hs["stem_index"])}
                    for hs in hidden_stems(b)
                ]
                for k, (_, b) in pillars_idx.items()
            }
            # 6글자 케이스는 보통 hidden 가중합까지는 과한 경우가 많아 기본 None 유지(원하면 켜면 됨)
            detail["five_elements"]["visible_plus_hidden"] = counts_to_obj(
                element_count_with_hidden_weighted(pillars_idx),
                "visible_6_chars_plus_hidden_weighted"
            )

        detail["extended"] = build_extended_detail(
            birth_date_obj=base_date,
            birth_dt_for_terms=dt_local,
            gender=gender,
            pillars_idx=pillars_idx,
            day_stem_i=d_s,
            fortune_date_obj=fortune_date_obj,
            zi_policy=zi_policy,
        )

        notes.insert(0, "출생시간 미입력: 시주(時柱)는 계산하지 않습니다.")
        notes.insert(0, "출생시간 미입력: 절기 경계 판단을 위해 서울 기준 12:00(정오)를 대표시각으로 가정했습니다.")

        return {
            "input": {
                "birth_date": birth_date,
                "birth_date_solar": base_date.isoformat(),
                "calendar_type": calendar_type,
                "lunar_leap_month": lunar_leap_month if calendar_type == "lunar" else False,
                "birth_time": None,
                "gender": gender,
                "fortune_date": fortune_date_obj.isoformat() if fortune_date_obj else None,
                "assumed_time_for_terms": assumed_time_for_terms,
                "timezone": "Asia/Seoul",
                "location": "Seoul, KR (fixed)",
                "school": school_meta(school, zi_policy),
                "zi_policy": zi_policy,
                "include_hidden": include_hidden,
                "detail_level": detail_level,
            },
            "pillars": {k: pillar_str(*v) for k, v in pillars_idx.items()},
            "detail": _prune_detail(detail, detail_level),
            "notes": notes,
        }

    # ---- Time known
    hh, mm = map(int, birth_time.split(":"))
    dt_local = datetime(y, m, d, hh, mm, tzinfo=SEOUL_TZ)

    # Year / Month fixed by moment
    y_s, y_b, _ = year_pillar(dt_local)
    mo_s, mo_b = month_pillar(dt_local)

    # Day effective date by Zi policy (only impacts 23:00~23:59)
    eff_day = dt_local.date()
    if dt_local.hour == 23 and zi_policy == "traditional_23":
        eff_day = eff_day + timedelta(days=1)
        notes.append("자시 정책(전통): 23:00~23:59 출생은 일주를 다음날로 계산했습니다.")

    jdn = jdn_from_gregorian(eff_day)
    d_s, d_b = day_stem_branch_from_jdn_noon(jdn)

    # Hour
    h_s, h_b = hour_pillar(d_s, dt_local)

    pillars_idx = {
        "year": (y_s, y_b),
        "month": (mo_s, mo_b),
        "day": (d_s, d_b),
        "hour": (h_s, h_b),
    }

    dm = stem_meta(d_s)

    detail = {
        "day_master": dm,
        "ten_gods": {
            "year_stem": ten_god(d_s, y_s),
            "month_stem": ten_god(d_s, mo_s),
            "hour_stem": ten_god(d_s, h_s),
            "day_stem": {"kor": "일간", "hanja": "日干"},
        },
        "twelve_stages": {f"{k}_branch": twelve_stage(d_s, b) for k, (_, b) in pillars_idx.items()},
        "hidden_stems": {},
        "five_elements": {
            "visible": counts_to_obj(element_count_visible(pillars_idx), "visible_8_chars"),
            "visible_plus_hidden": None,
        },
    }

    if include_hidden:
        detail["hidden_stems"] = {
            f"{k}_branch_hidden_stems": [
                {**hs, "ten_god": ten_god(d_s, hs["stem_index"])}
                for hs in hidden_stems(b)
            ]
            for k, (_, b) in pillars_idx.items()
        }
        detail["five_elements"]["visible_plus_hidden"] = counts_to_obj(
            element_count_with_hidden_weighted(pillars_idx),
            "visible_8_chars_plus_hidden_weighted"
        )

    detail["extended"] = build_extended_detail(
        birth_date_obj=base_date,
        birth_dt_for_terms=dt_local,
        gender=gender,
        pillars_idx=pillars_idx,
        day_stem_i=d_s,
        fortune_date_obj=fortune_date_obj,
        zi_policy=zi_policy,
    )

    return {
        "input": {
            "birth_date": birth_date,
            "birth_date_solar": base_date.isoformat(),
            "calendar_type": calendar_type,
            "lunar_leap_month": lunar_leap_month if calendar_type == "lunar" else False,
            "birth_time": birth_time,
            "gender": gender,
            "fortune_date": fortune_date_obj.isoformat() if fortune_date_obj else None,
            "timezone": "Asia/Seoul",
            "location": "Seoul, KR (fixed)",
            "school": school_meta(school, zi_policy),
            "zi_policy": zi_policy,
            "include_hidden": include_hidden,
            "detail_level": detail_level,
        },
        "pillars": {k: pillar_str(*v) for k, v in pillars_idx.items()},
        "detail": _prune_detail(detail, detail_level),
        "notes": notes,
    }

# =============================================================================
# 7) Pretty formatter for GUI
# =============================================================================

def format_report_markdown(result: Dict[str, Any]) -> str:
    detail = result.get("detail", {})
    extended = detail.get("extended", {})
    pillars = extended.get("pillars", {})
    analysis = extended.get("analysis", {})
    daewoon = extended.get("daewoon", {})
    sewoon = extended.get("sewoon", [])
    monthly = extended.get("monthly", [])
    daily_hourly = extended.get("daily_hourly")
    heavenly_rel = extended.get("heavenly_relations", [])
    earthly_rel = extended.get("earthly_relations", [])
    branch_sals = extended.get("branch_sals", {})
    notes = result.get("notes", [])

    def p_hanja(key: str) -> str:
        return pillars.get(key, {}).get("ganji_hanja", "-")

    def a_cell(key: str, field: str) -> str:
        value = analysis.get(key, {}).get(field)
        if value is None:
            return "-"
        if isinstance(value, list):
            return ",".join(value) if value else "-"
        return str(value)

    lines = []
    lines.append("## 사주 4주")
    lines.append("")
    lines.append(markdown_table(
        ["시주", "일주", "월주", "연주"],
        [[p_hanja("hour"), p_hanja("day"), p_hanja("month"), p_hanja("year")]],
    ))
    lines.append("")

    visible_counts = detail.get("five_elements", {}).get("visible", {}).get("counts", [])
    element_cells = ["-"] * 5
    if len(visible_counts) == 5:
        element_cells = [format_float_cell(float(x.get("value", 0.0))) for x in visible_counts]

    lines.append("## 오행 분포")
    lines.append("")
    lines.append(markdown_table(
        ["목", "화", "토", "금", "수"],
        [element_cells],
    ))
    lines.append("")

    lines.append("## 심층 분석: 십성 & 12운성")
    lines.append("")
    lines.append(markdown_table(
        ["구분", "시주", "일주", "월주", "연주"],
        [
            ["십성", a_cell("hour", "stem_ten_god"), a_cell("day", "stem_ten_god"), a_cell("month", "stem_ten_god"), a_cell("year", "stem_ten_god")],
            ["정기지지십성", a_cell("hour", "main_branch_ten_god"), a_cell("day", "main_branch_ten_god"), a_cell("month", "main_branch_ten_god"), a_cell("year", "main_branch_ten_god")],
            ["지지십성", a_cell("hour", "hidden_ten_gods"), a_cell("day", "hidden_ten_gods"), a_cell("month", "hidden_ten_gods"), a_cell("year", "hidden_ten_gods")],
            ["12운성", a_cell("hour", "stage_kor"), a_cell("day", "stage_kor"), a_cell("month", "stage_kor"), a_cell("year", "stage_kor")],
        ],
    ))
    lines.append("")

    lines.append("## 대운")
    lines.append("")
    daewoon_rows = daewoon.get("rows", [])
    daewoon_table_rows: List[List[str]] = []
    for row in reversed(daewoon_rows):
        daewoon_table_rows.append([
            f"{row['age_start']}~{row['age_end']}",
            row["ganji"],
            STEMS_HANJA[row["stem_index"]],
            BRANCHES_HANJA[row["branch_index"]],
            str(row["start_year"]),
            row["direction"],
            row["current"],
            row["sal"],
            row["stage"],
        ])
    if not daewoon_table_rows:
        daewoon_table_rows = [["-", "-", "-", "-", "-", "-", "-", "-", "-"]]
    lines.append(markdown_table(
        ["나이범위", "간지", "천간", "지지", "시작연도", "방향", "현재", "신살", "12운성"],
        daewoon_table_rows,
    ))
    lines.append("")

    lines.append("## 세운")
    lines.append("")
    sewoon_rows: List[List[str]] = []
    for row in sewoon:
        sewoon_rows.append([
            str(row["year"]),
            row["ganji"],
            STEMS_HANJA[row["stem_index"]],
            BRANCHES_HANJA[row["branch_index"]],
            row["sal"],
            row["stage"],
        ])
    if not sewoon_rows:
        sewoon_rows = [["-", "-", "-", "-", "-", "-"]]
    lines.append(markdown_table(
        ["연도", "간지", "천간", "지지", "신살", "12운성"],
        sewoon_rows,
    ))
    lines.append("")

    lines.append("## 월운")
    lines.append("")
    monthly_rows: List[List[str]] = []
    for row in monthly:
        monthly_rows.append([
            str(row["month"]),
            row["month_name"],
            row["ganji"],
            STEMS_HANJA[row["stem_index"]],
            BRANCHES_HANJA[row["branch_index"]],
            row["sal"],
            row["stage"],
            row["stem_ten_god"],
            row["branch_ten_god"],
            row["term"],
        ])
    if not monthly_rows:
        monthly_rows = [["-", "-", "-", "-", "-", "-", "-", "-", "-", "-"]]
    lines.append(markdown_table(
        ["월", "월명", "간지", "천간", "지지", "신살", "12운성", "천간십성", "지지십성", "절기"],
        monthly_rows,
    ))
    lines.append("")

    if daily_hourly:
        daily = daily_hourly.get("daily", {})
        hourly = daily_hourly.get("hourly", [])

        lines.append(f"## 일운 (조회일: {daily.get('date', '-')})")
        lines.append("")
        lines.append(markdown_table(
            ["날짜", "일주", "천간십성", "지지십성", "12운성", "신살"],
            [[
                daily.get("date", "-"),
                daily.get("day_ganji", "-"),
                daily.get("day_stem_ten_god", "-"),
                daily.get("day_branch_ten_god", "-"),
                daily.get("stage", "-"),
                daily.get("sal", "-"),
            ]],
        ))
        lines.append("")

        lines.append(f"## 시운 (조회일 전체 12시, 자시정책: {daily_hourly.get('zi_policy_applied', '-')})")
        lines.append("")
        hourly_rows: List[List[str]] = []
        for row in hourly:
            hourly_rows.append([
                row.get("time_range", "-"),
                row.get("hour_branch", "-"),
                row.get("hour_ganji", "-"),
                row.get("stem_ten_god", "-"),
                row.get("branch_ten_god", "-"),
                row.get("stage", "-"),
                row.get("sal", "-"),
                row.get("effective_day", "-"),
            ])
        if not hourly_rows:
            hourly_rows = [["-", "-", "-", "-", "-", "-", "-", "-"]]
        lines.append(markdown_table(
            ["시각대", "시지", "시간간지", "천간십성", "지지십성", "12운성", "신살", "적용일"],
            hourly_rows,
        ))
        lines.append("")

    lines.append("## 천간 특수관계")
    lines.append("")
    heavenly_rows: List[List[str]] = []
    for rel in heavenly_rel:
        heavenly_rows.append([rel["type"], rel["pair"], rel["pillars"], rel["desc"]])
    if not heavenly_rows:
        heavenly_rows = [["-", "-", "-", "-"]]
    lines.append(markdown_table(
        ["유형", "관계쌍", "기둥", "설명"],
        heavenly_rows,
    ))
    lines.append("")

    lines.append("## 지지 형·충·합·파·해")
    lines.append("")
    earthly_rows: List[List[str]] = []
    for rel in earthly_rel:
        earthly_rows.append([rel["type"], rel["pair"], rel["pillars"]])
    if not earthly_rows:
        earthly_rows = [["-", "-", "-"]]
    lines.append(markdown_table(
        ["유형", "관계쌍", "기둥"],
        earthly_rows,
    ))
    lines.append("")

    lines.append("### 지지 신살 (branchSals)")
    lines.append("")
    lines.append(markdown_table(
        ["연주", "월주", "일주", "시주"],
        [[
            branch_sals.get("year", "-"),
            branch_sals.get("month", "-"),
            branch_sals.get("day", "-"),
            branch_sals.get("hour", "-"),
        ]],
    ))
    lines.append("")

    if notes:
        lines.append("### 메모")
        lines.append("")
        for n in notes:
            lines.append(f"- {n}")
        lines.append("")

    return "\n".join(lines).strip()

def format_result(result: Dict[str, Any], detail_level: DetailLevel) -> str:
    pillars = result.get("pillars", {})
    notes = result.get("notes", [])
    detail = result.get("detail", {})

    lines = []
    lines.append("=== 사주 결과(서울 고정) ===")
    lines.append("")
    lines.append("[사주(柱)]")
    for k in ["year", "month", "day", "hour"]:
        if k in pillars:
            lines.append(f" - {k:>5}: {pillars[k]}")
    lines.append("")

    if notes:
        lines.append("[메모]")
        for n in notes:
            lines.append(f" - {n}")
        lines.append("")

    if detail_level == "pillars_only":
        return "\n".join(lines)

    if detail_level == "report":
        return format_report_markdown(result)

    if detail_level == "summary":
        lines.append("[요약 상세]")
        dm = detail.get("day_master", {})
        if dm:
            lines.append(f" - 일간(日干): {dm.get('hanja')}({dm.get('kor')}) / {dm.get('yin_yang')} / {dm.get('element', {}).get('kor')}")
        tg = detail.get("ten_gods", {})
        if tg:
            lines.append(" - 십신(天干 기준):")
            for key in ["year_stem", "month_stem", "hour_stem"]:
                if key in tg:
                    lines.append(f"    • {key}: {tg[key]['kor']}({tg[key]['hanja']})")
        stages = detail.get("twelve_stages", {})
        if stages:
            lines.append(" - 12운성(지지 기준):")
            for k in ["year_branch", "month_branch", "day_branch", "hour_branch"]:
                kk = f"{k}"
                if kk in stages:
                    lines.append(f"    • {kk}: {stages[kk]}")
        fe = detail.get("five_elements", {})
        if fe:
            lines.append(" - 오행 분포:")
            for part in ["visible", "visible_plus_hidden"]:
                obj = fe.get(part)
                if obj and obj.get("counts"):
                    compact = ", ".join([f"{c['element']['kor']}={c['value']:.2f}" for c in obj["counts"]])
                    lines.append(f"    • {part}: {compact}")
        return "\n".join(lines)

    # full
    lines.append("[FULL JSON]")
    lines.append(json.dumps(result, ensure_ascii=False, indent=2))
    return "\n".join(lines)

# =============================================================================
# 8) GUI (Tkinter)
# =============================================================================

class SajuPanel(ttk.Frame):
    def __init__(self, parent: tk.Widget):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        frm = ttk.Frame(self, padding=12)
        frm.pack(fill="both", expand=True)

        # ---- Input section
        input_box = ttk.LabelFrame(frm, text="입력", padding=10)
        input_box.pack(fill="x")

        ttk.Label(input_box, text="출생일(YYYY-MM-DD) *").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.var_date = tk.StringVar(value="1996-05-04")
        ttk.Entry(input_box, textvariable=self.var_date, width=18).grid(row=0, column=1, sticky="w", padx=5, pady=5)

        ttk.Label(input_box, text="출생시간(HH:MM) (선택)").grid(row=0, column=2, sticky="w", padx=5, pady=5)
        self.var_time = tk.StringVar(value="")
        ttk.Entry(input_box, textvariable=self.var_time, width=10).grid(row=0, column=3, sticky="w", padx=5, pady=5)

        ttk.Label(input_box, text="달력").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.var_calendar = tk.StringVar(value="solar")
        ttk.Radiobutton(input_box, text="양력", variable=self.var_calendar, value="solar").grid(row=1, column=1, sticky="w", padx=5, pady=5)
        ttk.Radiobutton(input_box, text="음력", variable=self.var_calendar, value="lunar").grid(row=1, column=2, sticky="w", padx=5, pady=5)
        self.var_lunar_leap = tk.BooleanVar(value=False)
        ttk.Checkbutton(input_box, text="윤달", variable=self.var_lunar_leap).grid(row=1, column=3, sticky="w", padx=5, pady=5)

        ttk.Label(input_box, text="성별").grid(row=2, column=0, sticky="w", padx=5, pady=5)
        self.var_gender = tk.StringVar(value="male")
        ttk.Radiobutton(input_box, text="남", variable=self.var_gender, value="male").grid(row=2, column=1, sticky="w", padx=5, pady=5)
        ttk.Radiobutton(input_box, text="여", variable=self.var_gender, value="female").grid(row=2, column=2, sticky="w", padx=5, pady=5)
        ttk.Label(input_box, text="운세조회일(YYYY-MM-DD) (선택)").grid(row=3, column=0, sticky="w", padx=5, pady=5)
        self.var_fortune_date = tk.StringVar(value="")
        self.ent_fortune_date = ttk.Entry(input_box, textvariable=self.var_fortune_date, width=18)
        self.ent_fortune_date.grid(row=3, column=1, sticky="w", padx=5, pady=5)
        self.var_fortune_today = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            input_box,
            text="오늘",
            variable=self.var_fortune_today,
            command=self.on_toggle_fortune_today,
        ).grid(row=3, column=2, sticky="w", padx=5, pady=5)

        # Zi policy
        policy_box = ttk.LabelFrame(frm, text="자시(子時) 경계 정책", padding=10)
        policy_box.pack(fill="x", pady=(10, 0))

        self.var_policy = tk.StringVar(value="traditional_23")
        ttk.Radiobutton(
            policy_box,
            text="전통(23시 교체): 23:00~23:59 출생은 일주를 다음날로",
            variable=self.var_policy,
            value="traditional_23"
        ).pack(anchor="w", pady=2)
        ttk.Radiobutton(
            policy_box,
            text="자정(00시 교체): 달력처럼 00:00 기준(23시는 당일 유지)",
            variable=self.var_policy,
            value="midnight_00"
        ).pack(anchor="w", pady=2)
        ttk.Label(
            policy_box,
            text="유파 기준: 현대 자평명리(입춘/절입 기준, 기본은 전통 23시 교체)",
            foreground="#555"
        ).pack(anchor="w", pady=(4, 0))

        # Options
        opt_box = ttk.LabelFrame(frm, text="옵션", padding=10)
        opt_box.pack(fill="x", pady=(10, 0))

        self.var_hidden = tk.BooleanVar(value=True)
        ttk.Checkbutton(opt_box, text="지장간(支藏干) 포함", variable=self.var_hidden).grid(row=0, column=0, sticky="w", padx=5, pady=5)

        ttk.Label(opt_box, text="출력 상세도").grid(row=0, column=1, sticky="e", padx=5, pady=5)
        self.var_detail = tk.StringVar(value="summary")
        detail_cb = ttk.Combobox(opt_box, textvariable=self.var_detail, state="readonly", width=14)
        detail_cb["values"] = ("pillars_only", "summary", "report", "full")
        detail_cb.grid(row=0, column=2, sticky="w", padx=5, pady=5)

        # Compute button
        btn_box = ttk.Frame(frm)
        btn_box.pack(fill="x", pady=(10, 0))
        ttk.Button(btn_box, text="계산", command=self.on_compute).pack(side="left")
        ttk.Button(btn_box, text="결과 지우기", command=self.on_clear).pack(side="left", padx=8)

        # Output
        out_box = ttk.LabelFrame(frm, text="출력", padding=10)
        out_box.pack(fill="both", expand=True, pady=(10, 0))

        self.txt = tk.Text(out_box, wrap="word")
        yscroll = ttk.Scrollbar(out_box, orient="vertical", command=self.txt.yview)
        self.txt.configure(yscrollcommand=yscroll.set)

        self.txt.pack(side="left", fill="both", expand=True)
        yscroll.pack(side="right", fill="y")

        # Footer
        foot = ttk.Label(
            frm,
            text="※ 출생시간 미입력 시: 연/월 절기 경계 판단을 위해 서울 기준 12:00(정오)를 대표 시각으로 가정합니다.",
            foreground="#555"
        )
        foot.pack(anchor="w", pady=(8, 0))

    def on_toggle_fortune_today(self):
        if bool(self.var_fortune_today.get()):
            self.var_fortune_date.set(datetime.now(SEOUL_TZ).date().isoformat())
            self.ent_fortune_date.state(["disabled"])
        else:
            self.ent_fortune_date.state(["!disabled"])

    def on_clear(self):
        self.txt.delete("1.0", "end")

    def on_compute(self):
        birth_date = self.var_date.get().strip()
        birth_time = self.var_time.get().strip()
        birth_time = birth_time if birth_time else None
        if bool(self.var_fortune_today.get()):
            self.var_fortune_date.set(datetime.now(SEOUL_TZ).date().isoformat())
        fortune_date = self.var_fortune_date.get().strip()
        fortune_date = fortune_date if fortune_date else None

        calendar_type: CalendarType = "solar" if self.var_calendar.get() == "solar" else "lunar"
        lunar_leap_month = bool(self.var_lunar_leap.get())
        zi_policy: ZiPolicy = "traditional_23" if self.var_policy.get() == "traditional_23" else "midnight_00"
        gender: Gender = "male" if self.var_gender.get() == "male" else "female"
        include_hidden = bool(self.var_hidden.get())
        detail_level: DetailLevel = self.var_detail.get()  # type: ignore

        # Basic validation (date/time)
        try:
            datetime.strptime(birth_date, "%Y-%m-%d")
        except ValueError:
            messagebox.showerror("입력 오류", "출생일 형식이 올바르지 않습니다. 예) 1996-05-04")
            return

        if birth_time:
            try:
                datetime.strptime(birth_time, "%H:%M")
            except ValueError:
                messagebox.showerror("입력 오류", "출생시간 형식이 올바르지 않습니다. 예) 08:10 (또는 비워두기)")
                return
        if calendar_type == "lunar" and KoreanLunarCalendar is None:
            messagebox.showerror("입력 오류", "음력 변환 라이브러리(`korean_lunar_calendar`)가 설치되어 있지 않습니다.")
            return
        if fortune_date:
            try:
                datetime.strptime(fortune_date, "%Y-%m-%d")
            except ValueError:
                messagebox.showerror("입력 오류", "운세조회일 형식이 올바르지 않습니다. 예) 2026-02-11 (또는 비워두기)")
                return

        # Compute
        try:
            result = compute_saju_seoul_single(
                birth_date=birth_date,
                birth_time=birth_time,
                zi_policy=zi_policy,
                include_hidden=include_hidden,
                detail_level=detail_level,
                gender=gender,
                fortune_date=fortune_date,
                school=DEFAULT_SCHOOL,
                calendar_type=calendar_type,
                lunar_leap_month=lunar_leap_month,
            )
            text = format_result(result, detail_level)
            self.txt.delete("1.0", "end")
            self.txt.insert("1.0", text)
        except Exception as e:
            messagebox.showerror("계산 실패", f"오류가 발생했습니다:\n{e}")


class SajuApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("사주 계산기 (서울 고정)")
        self.geometry("980x720")
        panel = SajuPanel(self)
        panel.pack(fill="both", expand=True)

# =============================================================================
# 9) Entrypoint
# =============================================================================

if __name__ == "__main__":
    app = SajuApp()
    app.mainloop()
