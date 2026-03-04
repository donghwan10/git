from __future__ import annotations

import math
import secrets
import time
from dataclasses import dataclass
from typing import Callable, List, Dict, Tuple, Any, Optional
from collections import Counter
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from functools import lru_cache

# =========================================================
# 0) 공통: 변괘/팔괘/64괘 계산
# =========================================================
def flip_changing(raw: int) -> int:
    if raw == 6:
        return 7
    if raw == 9:
        return 8
    return raw


def compute_changed(lines6: List[int]) -> Tuple[List[int], List[int]]:
    """
    Returns (changed_lines6, changing_indices)
      - changing_indices are 1..6 (bottom=1, top=6)
    """
    idx = [i + 1 for i, v in enumerate(lines6) if v in (6, 9)]  # 1..6
    changed = [flip_changing(v) for v in lines6]
    return changed, idx


def is_yang(raw: int) -> bool:
    return raw in (7, 9)


def trigram_code(lines3: List[int]) -> int:
    """
    lines3: bottom -> top (length=3)
    code: b0 + 2*b1 + 4*b2, where b=1 for yang(7/9), 0 for yin(6/8)
    """
    b0 = 1 if is_yang(lines3[0]) else 0
    b1 = 1 if is_yang(lines3[1]) else 0
    b2 = 1 if is_yang(lines3[2]) else 0
    return b0 + 2 * b1 + 4 * b2


TRIGRAM_BY_CODE: Dict[int, Tuple[str, str, str]] = {
    7: ("qian", "乾", "☰"),
    3: ("dui",  "兌", "☱"),
    5: ("li",   "離", "☲"),
    1: ("zhen", "震", "☳"),
    6: ("xun",  "巽", "☴"),
    2: ("kan",  "坎", "☵"),
    4: ("gen",  "艮", "☶"),
    0: ("kun",  "坤", "☷"),
}

KINGWEN_MATRIX = [
    [ 1, 43, 14, 34,  9,  5, 26, 11],
    [10, 58, 38, 54, 61, 60, 41, 19],
    [13, 49, 30, 55, 37, 63, 22, 36],
    [25, 17, 21, 51, 42,  3, 27, 24],
    [44, 28, 50, 32, 57, 48, 18, 46],
    [ 6, 47, 64, 40, 59, 29,  4,  7],
    [33, 31, 56, 62, 53, 39, 52, 15],
    [12, 45, 35, 16, 20,  8, 23,  2],
]

TRIGRAM_INDEX: Dict[str, int] = {
    "qian": 0, "dui": 1, "li": 2, "zhen": 3,
    "xun": 4, "kan": 5, "gen": 6, "kun": 7,
}

HEX_NAME_ZH = {
     1:"乾",  2:"坤",  3:"屯",  4:"蒙",  5:"需",  6:"訟",  7:"師",  8:"比",
     9:"小畜",10:"履",11:"泰",12:"否",13:"同人",14:"大有",15:"謙",16:"豫",
    17:"隨",18:"蠱",19:"臨",20:"觀",21:"噬嗑",22:"賁",23:"剝",24:"復",
    25:"无妄",26:"大畜",27:"頤",28:"大過",29:"坎",30:"離",31:"咸",32:"恆",
    33:"遯",34:"大壯",35:"晉",36:"明夷",37:"家人",38:"睽",39:"蹇",40:"解",
    41:"損",42:"益",43:"夬",44:"姤",45:"萃",46:"升",47:"困",48:"井",
    49:"革",50:"鼎",51:"震",52:"艮",53:"漸",54:"歸妹",55:"豐",56:"旅",
    57:"巽",58:"兌",59:"渙",60:"節",61:"中孚",62:"小過",63:"既濟",64:"未濟",
}


def kingwen_no(upper_key: str, lower_key: str) -> int:
    r = TRIGRAM_INDEX[lower_key]
    c = TRIGRAM_INDEX[upper_key]
    return KINGWEN_MATRIX[r][c]


def triplet_str(lines3: List[int]) -> str:
    return "".join(str(x) for x in lines3)


@dataclass(frozen=True)
class HexInfo:
    printed: str
    lines6: List[int]
    upper: Tuple[str, str, str]  # (key, zh, symbol)
    lower: Tuple[str, str, str]
    no: int
    name_zh: str


def compute_hex_info(lines6: List[int]) -> HexInfo:
    if len(lines6) != 6 or any(v not in (6, 7, 8, 9) for v in lines6):
        raise ValueError("lines6 must be length 6, each in {6,7,8,9} (bottom->top).")

    lower3 = lines6[0:3]
    upper3 = lines6[3:6]

    lower_code = trigram_code(lower3)
    upper_code = trigram_code(upper3)

    lower = TRIGRAM_BY_CODE[lower_code]
    upper = TRIGRAM_BY_CODE[upper_code]

    no = kingwen_no(upper[0], lower[0])
    printed = f"<{triplet_str(lower3)} {triplet_str(upper3)}>"
    return HexInfo(
        printed=printed,
        lines6=lines6,
        upper=upper,
        lower=lower,
        no=no,
        name_zh=HEX_NAME_ZH[no],
    )


# =========================================================
# 1) 시초점(대연법)
# =========================================================
DEFAULT_DELAY_SEC = 0.1


def suspense(
    step: str,
    delay_sec: float = DEFAULT_DELAY_SEC,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> None:
    msg = f"  · {step} ..."
    if progress_callback is not None:
        try:
            progress_callback(msg)
        except Exception:
            pass

    if delay_sec > 0:
        if progress_callback is None:
            print(msg)
        time.sleep(delay_sec)


def rem4_as_1to4(x: int) -> int:
    r = x % 4
    return 4 if r == 0 else r


def split_into_two_piles(n: int) -> Tuple[int, int]:
    while True:
        left = secrets.randbits(n).bit_count()  # 0..n, Binomial(n, 0.5)
        if 1 <= left <= n - 1:
            return left, n - left


def yarrow_change(
    n: int,
    delay_sec: float = DEFAULT_DELAY_SEC,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> Tuple[int, Dict[str, Any]]:
    suspense(f"[一變] 시작 (n={n})", delay_sec, progress_callback=progress_callback)

    left, right = split_into_two_piles(n)

    if right == 1:
        suspense(
            f"[一變] right=1 발생: left={left}, right={right} → 좌우 스왑",
            delay_sec,
            progress_callback=progress_callback,
        )
        left, right = right, left

    suspense(f"[一變] 분할: left={left}, right={right}", delay_sec, progress_callback=progress_callback)

    suspense("[一變] 오른쪽에서 1개 취함(人)", delay_sec, progress_callback=progress_callback)
    right_after_take = right - 1

    left_rem = rem4_as_1to4(left)
    right_rem = rem4_as_1to4(right_after_take)
    suspense(
        f"[一變] 4로 나눈 나머지: left_rem={left_rem}, right_rem={right_rem}",
        delay_sec,
        progress_callback=progress_callback,
    )

    removed = 1 + left_rem + right_rem
    n_after = n - removed
    suspense(
        f"[一變] 제거: 1 + {left_rem} + {right_rem} = {removed}개 → 남음 n={n_after}",
        delay_sec,
        progress_callback=progress_callback,
    )

    trace = {
        "start": n,
        "split_left": left,
        "split_right": right,
        "take_one_from_right": 1,
        "right_after_take": right_after_take,
        "left_rem": left_rem,
        "right_rem": right_rem,
        "removed_total": removed,
        "after": n_after,
    }
    return n_after, trace


def make_line_yarrow(
    delay_sec: float = DEFAULT_DELAY_SEC,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> Tuple[int, List[Dict[str, Any]]]:
    suspense("[효 생성] 태극 1개를 덜어내고 49로 시작", delay_sec, progress_callback=progress_callback)
    n = 49  # 50 - 1

    traces: List[Dict[str, Any]] = []
    for k in range(1, 4):
        suspense(f"[효 생성] {k}번째 변(一變) 진행", delay_sec, progress_callback=progress_callback)
        n, t = yarrow_change(n, delay_sec=delay_sec, progress_callback=progress_callback)
        traces.append(t)

    suspense(f"[효 생성] 3변 종료, 최종 잔량 n={n}", delay_sec, progress_callback=progress_callback)

    if n not in (24, 28, 32, 36):
        raise RuntimeError(f"Unexpected yarrow remaining count: {n}")

    line_value = n // 4
    suspense(f"[효 생성] n/4 = {line_value} → 효 확정", delay_sec, progress_callback=progress_callback)
    return line_value, traces


def _fmt_change(t: Dict[str, Any]) -> str:
    return (
        f"{t['start']}→split({t['split_left']},{t['split_right']})"
        f"→rem({t['left_rem']},{t['right_rem']})"
        f"→-{t['removed_total']}={t['after']}"
    )


def make_six_lines_yarrow(
    delay_sec: float = DEFAULT_DELAY_SEC,
    show_banners: bool = False,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> Tuple[List[int], List[List[Dict[str, Any]]]]:
    lines: List[int] = []
    traces_by_line: List[List[Dict[str, Any]]] = []

    for i in range(1, 7):
        if progress_callback is not None:
            try:
                progress_callback(f"===== {i}효(아래→위) 생성 시작 =====")
            except Exception:
                pass
        if show_banners and delay_sec > 0:
            if progress_callback is None:
                print(f"\n===== {i}효(아래→위) 생성 시작 =====")

        v, traces = make_line_yarrow(delay_sec=delay_sec, progress_callback=progress_callback)
        lines.append(v)
        traces_by_line.append(traces)

        if progress_callback is not None:
            try:
                progress_callback(f"===== {i}효 결과: {v} =====")
            except Exception:
                pass
        if show_banners and delay_sec > 0:
            if progress_callback is None:
                print(f"===== {i}효 결과: {v} =====")

    return lines, traces_by_line


# =========================================================
# 5) 文王卦(六爻) + 四柱(절기/立春 기준) 계산
# =========================================================
STEMS_CN = ["甲","乙","丙","丁","戊","己","庚","辛","壬","癸"]
BRANCHES_CN = ["子","丑","寅","卯","辰","巳","午","未","申","酉","戌","亥"]

BRANCH_TO_WX = {
    "子":"水","丑":"土","寅":"木","卯":"木","辰":"土","巳":"火",
    "午":"火","未":"土","申":"金","酉":"金","戌":"土","亥":"水",
}

TRI_TO_WX = {
    "qian":"金", "dui":"金",
    "zhen":"木", "xun":"木",
    "kan":"水",
    "li":"火",
    "gen":"土", "kun":"土",
}

NAJIA_BRANCH_SEQ = {
    "qian": ["子","寅","辰","午","申","戌"],
    "kun":  ["未","巳","卯","丑","亥","酉"],
    "zhen": ["子","寅","辰","午","申","戌"],
    "xun":  ["丑","亥","酉","未","巳","卯"],
    "kan":  ["寅","辰","午","申","戌","子"],
    "li":   ["卯","丑","亥","酉","未","巳"],
    "gen":  ["辰","午","申","戌","子","寅"],
    "dui":  ["巳","卯","丑","亥","酉","未"],
}

NAJIA_STEM_INNER_OUTER = {
    "qian": ("甲","壬"),
    "kun":  ("乙","癸"),
    "zhen": ("庚","庚"),
    "xun":  ("辛","辛"),
    "kan":  ("戊","戊"),
    "li":   ("己","己"),
    "gen":  ("丙","丙"),
    "dui":  ("丁","丁"),
}

SIX_SPIRITS_ORDER = ["青龍","朱雀","勾陳","螣蛇","白虎","玄武"]
SIX_SPIRITS_START = {
    0:"青龍", 1:"青龍",
    2:"朱雀", 3:"朱雀",
    4:"勾陳", 5:"勾陳",
    6:"白虎", 7:"白虎",
    8:"玄武", 9:"玄武",
}

def six_spirits_for_day_stem(day_stem_idx: int) -> list[str]:
    start = SIX_SPIRITS_START[day_stem_idx]
    si = SIX_SPIRITS_ORDER.index(start)
    return [SIX_SPIRITS_ORDER[(si + i) % 6] for i in range(6)]

XUNKONG_BY_XUN_INDEX = {
    0: ("戌","亥"),
    1: ("申","酉"),
    2: ("午","未"),
    3: ("辰","巳"),
    4: ("寅","卯"),
    5: ("子","丑"),
}

def jdn_from_ymd(y: int, m: int, d: int) -> int:
    a = (14 - m) // 12
    y2 = y + 4800 - a
    m2 = m + 12 * a - 3
    return d + (153 * m2 + 2) // 5 + 365 * y2 + y2 // 4 - y2 // 100 + y2 // 400 - 32045

def day_ganzhi_from_date(y: int, m: int, d: int) -> tuple[int,int,int]:
    JD_noon = jdn_from_ymd(y, m, d)
    day_stem_idx = (JD_noon - 1) % 10
    day_branch_idx = (JD_noon + 1) % 12
    sexagenary_idx = (JD_noon - 11) % 60
    return day_stem_idx, day_branch_idx, sexagenary_idx

def hour_branch_index(local_hour: int) -> int:
    if local_hour in (23, 0):
        return 0
    return (local_hour + 1) // 2

def hour_stem_index(day_stem_idx: int, hour_branch_idx: int) -> int:
    zi_stem = (day_stem_idx * 2) % 10
    return (zi_stem + hour_branch_idx) % 10

def jd_from_datetime(dt: datetime) -> float:
    if dt.tzinfo is None:
        raise ValueError("dt must be timezone-aware")
    dt_utc = dt.astimezone(timezone.utc)
    jdn = jdn_from_ymd(dt_utc.year, dt_utc.month, dt_utc.day)
    frac = (
        dt_utc.hour + dt_utc.minute / 60 + dt_utc.second / 3600 + dt_utc.microsecond / 3.6e9 - 12
    ) / 24
    return jdn + frac

def jd_to_datetime(jd: float, tz) -> datetime:
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    seconds = (jd - 2440587.5) * 86400.0
    return (epoch + timedelta(seconds=seconds)).astimezone(tz)

def sun_apparent_longitude(jd: float) -> float:
    T = (jd - 2451545.0) / 36525.0
    L0 = (280.46646 + 36000.76983 * T + 0.0003032 * T * T) % 360.0
    M = (357.52911 + 35999.05029 * T - 0.0001537 * T * T) % 360.0
    Mr = math.radians(M)
    C = (
        (1.914602 - 0.004817 * T - 0.000014 * T * T) * math.sin(Mr)
        + (0.019993 - 0.000101 * T) * math.sin(2 * Mr)
        + 0.000289 * math.sin(3 * Mr)
    )
    true_long = L0 + C
    Omega = 125.04 - 1934.136 * T
    lam = true_long - 0.00569 - 0.00478 * math.sin(math.radians(Omega))
    return lam % 360.0

def angle_diff(a: float, b: float) -> float:
    return ((a - b + 180) % 360) - 180

_TERM_APPROX_MD = {
    255: (12, 7),
    285: (1, 5),
    315: (2, 4),
    345: (3, 6),
    15:  (4, 5),
    45:  (5, 5),
    75:  (6, 6),
    105: (7, 7),
    135: (8, 7),
    165: (9, 7),
    195: (10, 8),
    225: (11, 7),
}

@lru_cache(maxsize=None)
def find_solar_term_jd(year: int, target_deg: int) -> float:
    m, d = _TERM_APPROX_MD[target_deg]
    guess = datetime(year, m, d, 12, tzinfo=timezone.utc)
    jd = jd_from_datetime(guess)

    for _ in range(20):
        diff = angle_diff(sun_apparent_longitude(jd), float(target_deg))
        if abs(diff) < 1e-6:
            break
        jd -= diff / 0.98564736

    span = 2.0
    lo, hi = jd - span, jd + span
    dlo = angle_diff(sun_apparent_longitude(lo), float(target_deg))
    dhi = angle_diff(sun_apparent_longitude(hi), float(target_deg))

    k = 0
    while dlo * dhi > 0 and k < 12:
        span *= 1.5
        lo, hi = jd - span, jd + span
        dlo = angle_diff(sun_apparent_longitude(lo), float(target_deg))
        dhi = angle_diff(sun_apparent_longitude(hi), float(target_deg))
        k += 1

    if dlo * dhi > 0:
        return jd

    for _ in range(70):
        mid = (lo + hi) / 2
        dm = angle_diff(sun_apparent_longitude(mid), float(target_deg))
        if abs(dm) < 1e-8:
            return mid
        if dlo * dm <= 0:
            hi = mid
            dhi = dm
        else:
            lo = mid
            dlo = dm
    return (lo + hi) / 2

MONTH_BOUNDARIES = [
    (255, 0),
    (285, 1),
    (315, 2),
    (345, 3),
    (15,  4),
    (45,  5),
    (75,  6),
    (105, 7),
    (135, 8),
    (165, 9),
    (195, 10),
    (225, 11),
]

def first_month_stem_idx(year_stem_idx: int) -> int:
    if year_stem_idx in (0, 5): return 2
    if year_stem_idx in (1, 6): return 4
    if year_stem_idx in (2, 7): return 6
    if year_stem_idx in (3, 8): return 8
    if year_stem_idx in (4, 9): return 0
    raise ValueError("invalid year_stem_idx")

def month_branch_from_dt(dt_kst: datetime) -> int:
    tz = ZoneInfo("Asia/Seoul")
    dt = dt_kst.astimezone(tz)

    years = {dt.year - 1, dt.year, dt.year + 1}
    boundaries = []
    for y in years:
        for deg, mb in MONTH_BOUNDARIES:
            jd = find_solar_term_jd(y, deg)
            boundaries.append((jd_to_datetime(jd, tz), mb))
    boundaries.sort(key=lambda x: x[0])

    last_mb = None
    for bd_dt, mb in boundaries:
        if bd_dt <= dt:
            last_mb = mb
        else:
            break
    if last_mb is None:
        return boundaries[0][1]
    return last_mb

def four_pillars(dt_kst: datetime, use_zi_day_boundary: bool = True) -> dict:
    tz = ZoneInfo("Asia/Seoul")
    dt = dt_kst.astimezone(tz)

    lichun_jd = find_solar_term_jd(dt.year, 315)
    lichun_dt = jd_to_datetime(lichun_jd, tz)

    bazi_year = dt.year if dt >= lichun_dt else dt.year - 1
    year_stem_idx = (bazi_year - 4) % 10
    year_branch_idx = (bazi_year - 4) % 12

    month_branch_idx = month_branch_from_dt(dt)
    first_m_stem = first_month_stem_idx(year_stem_idx)
    offset_from_yin = (month_branch_idx - 2) % 12
    month_stem_idx = (first_m_stem + offset_from_yin) % 10

    day_date = dt.date()
    if use_zi_day_boundary and dt.hour == 23:
        day_date = day_date + timedelta(days=1)

    day_stem_idx, day_branch_idx, sexagenary_idx = day_ganzhi_from_date(
        day_date.year, day_date.month, day_date.day
    )

    hb = hour_branch_index(dt.hour)
    hs = hour_stem_index(day_stem_idx, hb)

    xun_index = sexagenary_idx // 10
    xk = XUNKONG_BY_XUN_INDEX[xun_index]

    return {
        "dt_kst": dt,
        "lichun_dt_kst": lichun_dt,
        "year_stem_idx": year_stem_idx,
        "year_branch_idx": year_branch_idx,
        "month_stem_idx": month_stem_idx,
        "month_branch_idx": month_branch_idx,
        "day_stem_idx": day_stem_idx,
        "day_branch_idx": day_branch_idx,
        "hour_stem_idx": hs,
        "hour_branch_idx": hb,
        "sexagenary_idx": sexagenary_idx,
        "xunkong": xk,
        "year_gz": STEMS_CN[year_stem_idx] + BRANCHES_CN[year_branch_idx],
        "month_gz": STEMS_CN[month_stem_idx] + BRANCHES_CN[month_branch_idx],
        "day_gz": STEMS_CN[day_stem_idx] + BRANCHES_CN[day_branch_idx],
        "hour_gz": STEMS_CN[hs] + BRANCHES_CN[hb],
    }

WW_TRIGRAM_BITS = {
    "qian": (1,1,1), "dui": (1,1,0), "li": (1,0,1), "zhen": (1,0,0),
    "xun":  (0,1,1), "kan": (0,1,0), "gen": (0,0,1), "kun": (0,0,0),
}
WW_PALACE_MASKS = [
    [], [1], [1,2], [1,2,3], [1,2,3,4], [1,2,3,4,5], [1,2,3,5], [5]
]
WW_SHI_BY_GEN = [6, 1, 2, 3, 4, 5, 4, 3]

def _flip_bits(bits6: list[int], lines_to_flip: list[int]) -> list[int]:
    out = bits6[:]
    for ln in lines_to_flip:
        out[ln-1] ^= 1
    return out

def palace_and_shi_ying(bits6: list[int]) -> tuple[str,int,int,int]:
    for palace_key, tri_bits in WW_TRIGRAM_BITS.items():
        base = list(tri_bits) + list(tri_bits)
        for gen_idx, mask in enumerate(WW_PALACE_MASKS):
            if _flip_bits(base, mask) == bits6:
                shi = WW_SHI_BY_GEN[gen_idx]
                ying = ((shi + 2) % 6) + 1
                return palace_key, gen_idx, shi, ying
    raise RuntimeError("palace not found")

def six_rel(palace_wx: str, line_wx: str) -> str:
    gen = {"木":"火","火":"土","土":"金","金":"水","水":"木"}
    ke  = {"木":"土","土":"水","水":"火","火":"金","金":"木"}
    if line_wx == palace_wx:
        return "兄弟"
    if gen[palace_wx] == line_wx:
        return "子孫"
    if gen[line_wx] == palace_wx:
        return "父母"
    if ke[palace_wx] == line_wx:
        return "妻財"
    if ke[line_wx] == palace_wx:
        return "官鬼"
    return "?"

def build_wenwang_report(lines6: list[int], hex_info: "HexInfo", pillars: dict) -> dict:
    bits6 = [1 if is_yang(v) else 0 for v in lines6]

    palace_key, gen_idx, shi, ying = palace_and_shi_ying(bits6)
    palace_wx = TRI_TO_WX[palace_key]

    day_stem_idx = pillars["day_stem_idx"]
    spirits = six_spirits_for_day_stem(day_stem_idx)
    xunkong = pillars["xunkong"]

    lower_key = hex_info.lower[0]
    upper_key = hex_info.upper[0]

    inner_stem = NAJIA_STEM_INNER_OUTER[lower_key][0]
    outer_stem = NAJIA_STEM_INNER_OUTER[upper_key][1]
    lower_seq = NAJIA_BRANCH_SEQ[lower_key]
    upper_seq = NAJIA_BRANCH_SEQ[upper_key]

    rows = []
    for i in range(1, 7):
        raw = lines6[i-1]
        moving = raw in (6, 9)
        if i <= 3:
            stem = inner_stem
            br = lower_seq[i-1]
        else:
            stem = outer_stem
            br = upper_seq[i-1]
        wx = BRANCH_TO_WX[br]
        rel = six_rel(palace_wx, wx)

        rows.append({
            "line": i,
            "shi_ying": "世" if i == shi else ("應" if i == ying else ""),
            "raw": raw,
            "moving": moving,
            "ganzhi": f"{stem}{br}",
            "branch": br,
            "wx": wx,
            "six_rel": rel,
            "six_spirit": spirits[i-1],
            "is_xunkong": "空" if br in xunkong else "",
        })

    return {
        "palace_key": palace_key,
        "palace_wx": palace_wx,
        "gen_idx": gen_idx,
        "shi": shi,
        "ying": ying,
        "xunkong": xunkong,
        "pillars": pillars,
        "hex_no": hex_info.no,
        "hex_name": hex_info.name_zh,
        "upper": hex_info.upper,
        "lower": hex_info.lower,
        "rows": rows,
    }

def print_wenwang(report: dict) -> None:
    p = report["pillars"]
    print("\n======================")
    print("WENWANG (文王卦 / 六爻) AUTO REPORT")
    print("======================")
    print(f"Cast time (KST) : {p['dt_kst'].strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"LiChun (KST)    : {p['lichun_dt_kst'].strftime('%Y-%m-%d %H:%M:%S %Z')}  (year boundary)")
    print(f"Four Pillars    : {p['year_gz']}年  {p['month_gz']}月  {p['day_gz']}日  {p['hour_gz']}時")
    print(f"XunKong (旬空)  : {report['xunkong'][0]}{report['xunkong'][1]}")
    print(f"Palace (宫)     : {report['palace_key']}宫({report['palace_wx']})  |  世={report['shi']}  應={report['ying']}")
    print(f"Hexagram        : #{report['hex_no']} {report['hex_name']}  ({report['upper'][2]} over {report['lower'][2]})")

    header = ["효(상->하)","世/應","원효","동","納甲(干支)","지지오행","六親","六神","旬空"]
    print("\n" + " | ".join(header))
    print("-" * 92)
    for r in reversed(report["rows"]):
        print(
            f"{r['line']:>8} | {r['shi_ying']:^5} | {r['raw']:^4} | "
            f"{('동' if r['moving'] else ''):^2} | {r['ganzhi']:^8} | {r['wx']:^4} | "
            f"{r['six_rel']:^4} | {r['six_spirit']:^4} | {r['is_xunkong']:^2}"
        )

# =========================================================
# 6) 실행
# =========================================================
def run_once_yarrow(
    delay_sec: float = DEFAULT_DELAY_SEC,
    show_line_details: bool = True,
    show_banners: bool = False,
) -> None:
    dt_kst = datetime.now(ZoneInfo("Asia/Seoul"))
    now_kst_str = dt_kst.strftime("%Y.%m.%d %H:%M")

    lines, traces_by_line = make_six_lines_yarrow(
        delay_sec=delay_sec,
        show_banners=show_banners,
    )

    base = compute_hex_info(lines)
    changed, changing_idx = compute_changed(lines)
    changed_hex = compute_hex_info(changed)

    print(f"Timestamp     : {now_kst_str}")
    print("Yarrow result :", base.printed)
    print(f"BASE (본괘)   : #{base.no} {base.name_zh}  ({base.upper[2]} over {base.lower[2]})")
    print("Changing lines:", changing_idx if changing_idx else "None")
    print("Changed result:", changed_hex.printed)
    print(f"CHANGED(변괘) : #{changed_hex.no} {changed_hex.name_zh}  ({changed_hex.upper[2]} over {changed_hex.lower[2]})")

    if show_line_details:
        print("\n--- Yarrow traces per line (bottom -> top) ---")
        for i, (v, traces) in enumerate(zip(lines, traces_by_line), start=1):
            afters = [t["after"] for t in traces]
            steps = " | ".join(_fmt_change(t) for t in traces)
            print(f"Line {i}: afters={' '.join(map(str, afters))}  -> line={v}  steps={steps}")

    pillars = four_pillars(dt_kst, use_zi_day_boundary=True)
    ww = build_wenwang_report(lines, base, pillars)
    print_wenwang(ww)


if __name__ == "__main__":
    run_once_yarrow(delay_sec=0.1, show_line_details=True, show_banners=False)
