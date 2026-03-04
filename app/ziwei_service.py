from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

try:
    from korean_lunar_calendar import KoreanLunarCalendar
except Exception:
    KoreanLunarCalendar = None  # type: ignore[assignment]


ViewMode = Literal["triad", "four_transform"]
FortuneScope = Literal["natal", "decadal", "yearly"]
StarFilter = Literal["major", "major_minor", "all"]
Gender = Literal["male", "female"]
CalendarType = Literal["solar", "lunar"]

SEOUL_TZ = ZoneInfo("Asia/Seoul")

STEMS = ["甲", "乙", "丙", "丁", "戊", "己", "庚", "辛", "壬", "癸"]
BRANCHES = ["子", "丑", "寅", "卯", "辰", "巳", "午", "未", "申", "酉", "戌", "亥"]
ZODIAC_ANIMALS_KO = ["쥐", "소", "호랑이", "토끼", "용", "뱀", "말", "양", "원숭이", "닭", "개", "돼지"]
BRIGHTNESS_ORDER = ["廟", "旺", "得", "平", "陷"]

CHANGSHENG_12 = ["장생", "목욕", "관대", "건록", "제왕", "쇠", "병", "사", "묘", "절", "태", "양"]
BOSHI_12 = ["博士", "力士", "青龍", "小耗", "將軍", "奏書", "飛廉", "喜神", "病符", "大耗", "伏兵", "官符"]
JIANGQIAN_12 = ["將星", "攀鞍", "歲驛", "息神", "華蓋", "劫煞", "災煞", "天煞", "指背", "咸池", "月煞", "亡神"]
SUIQIAN_12 = ["歲建", "晦氣", "喪門", "貫索", "官符", "小耗", "歲破", "龍德", "白虎", "天德", "吊客", "病符"]

FOUR_TRANSFORM_META: List[Tuple[str, str, str]] = [
    ("lu", "록", "祿"),
    ("quan", "권", "權"),
    ("ke", "과", "科"),
    ("ji", "기", "忌"),
]

PALACE_SEQUENCE: List[Tuple[str, str, str]] = [
    # 명궁에서 역배할 때의 표준 순서(命, 兄, 夫, 子, 財, 疾, 遷, 奴, 官, 田, 福, 父)
    ("life", "명궁", "命宮"),
    ("siblings", "형제궁", "兄弟"),
    ("spouse", "부처궁", "夫妻"),
    ("children", "자녀궁", "子女"),
    ("wealth", "재백궁", "財帛"),
    ("health", "질액궁", "疾厄"),
    ("travel", "천이궁", "遷移"),
    ("servants", "노복궁", "僕役"),
    ("career", "관록궁", "官祿"),
    ("property", "전택궁", "田宅"),
    ("fortune", "복덕궁", "福德"),
    ("parents", "부모궁", "父母"),
]

GRID_RING_POSITIONS: List[Tuple[int, int]] = [
    (0, 0),
    (0, 1),
    (0, 2),
    (0, 3),
    (1, 3),
    (2, 3),
    (3, 3),
    (3, 2),
    (3, 1),
    (3, 0),
    (2, 0),
    (1, 0),
]


@dataclass(frozen=True)
class StarDef:
    star_id: str
    label_ko: str
    label_zh: str
    star_type: str
    bucket: str


STAR_DEFS: Dict[str, StarDef] = {
    # 14주성
    "ziwei": StarDef("ziwei", "자미", "紫微", "major", "majorStars"),
    "tianji": StarDef("tianji", "천기", "天機", "major", "majorStars"),
    "taiyang": StarDef("taiyang", "태양", "太陽", "major", "majorStars"),
    "wuqu": StarDef("wuqu", "무곡", "武曲", "major", "majorStars"),
    "tiantong": StarDef("tiantong", "천동", "天同", "major", "majorStars"),
    "lianzhen": StarDef("lianzhen", "염정", "廉貞", "major", "majorStars"),
    "tianfu": StarDef("tianfu", "천부", "天府", "major", "majorStars"),
    "taiyin": StarDef("taiyin", "태음", "太陰", "major", "majorStars"),
    "tanlang": StarDef("tanlang", "탐랑", "貪狼", "major", "majorStars"),
    "jumen": StarDef("jumen", "거문", "巨門", "major", "majorStars"),
    "tianxiang": StarDef("tianxiang", "천상", "天相", "major", "majorStars"),
    "tianliang": StarDef("tianliang", "천량", "天梁", "major", "majorStars"),
    "qisha": StarDef("qisha", "칠살", "七殺", "major", "majorStars"),
    "pojun": StarDef("pojun", "파군", "破軍", "major", "majorStars"),
    # 보성/잡성
    "wenchang": StarDef("wenchang", "문창", "文昌", "helper", "minorStars"),
    "wenqu": StarDef("wenqu", "문곡", "文曲", "helper", "minorStars"),
    "zuofu": StarDef("zuofu", "좌보", "左輔", "helper", "minorStars"),
    "youbi": StarDef("youbi", "우필", "右弼", "helper", "minorStars"),
    "tiankui": StarDef("tiankui", "천괴", "天魁", "helper", "minorStars"),
    "tianyue": StarDef("tianyue", "천월", "天鉞", "helper", "minorStars"),
    "lucun": StarDef("lucun", "녹존", "祿存", "minor", "minorStars"),
    "qingyang": StarDef("qingyang", "경양", "擎羊", "malefic", "minorStars"),
    "tuoluo": StarDef("tuoluo", "타라", "陀羅", "malefic", "minorStars"),
    "huoxing": StarDef("huoxing", "화성", "火星", "malefic", "minorStars"),
    "lingxing": StarDef("lingxing", "영성", "鈴星", "malefic", "minorStars"),
    "dikong": StarDef("dikong", "지공", "地空", "adjective", "adjectiveStars"),
    "dijie": StarDef("dijie", "지겁", "地劫", "adjective", "adjectiveStars"),
    "tianma": StarDef("tianma", "천마", "天馬", "tianma", "adjectiveStars"),
    "hongluan": StarDef("hongluan", "홍란", "紅鸞", "adjective", "adjectiveStars"),
    "tianxi": StarDef("tianxi", "천희", "天喜", "adjective", "adjectiveStars"),
    "guchen": StarDef("guchen", "고진", "孤辰", "adjective", "adjectiveStars"),
    "guasu": StarDef("guasu", "과숙", "寡宿", "adjective", "adjectiveStars"),
}

STAR_ORDER: Tuple[str, ...] = tuple(STAR_DEFS.keys())
STAR_BRIGHTNESS_ANCHOR: Dict[str, int] = {sid: (i * 3) % 12 for i, sid in enumerate(STAR_ORDER)}

# 甲乙丙丁戊己庚辛壬癸 사화
FOUR_TRANSFORM_RULES: Dict[str, Dict[str, str]] = {
    "甲": {"lu": "lianzhen", "quan": "pojun", "ke": "wuqu", "ji": "taiyang"},
    "乙": {"lu": "tianji", "quan": "tianliang", "ke": "ziwei", "ji": "taiyin"},
    "丙": {"lu": "tiantong", "quan": "tianji", "ke": "wenchang", "ji": "lianzhen"},
    "丁": {"lu": "taiyin", "quan": "tiantong", "ke": "tianji", "ji": "jumen"},
    "戊": {"lu": "tanlang", "quan": "taiyin", "ke": "youbi", "ji": "tianji"},
    "己": {"lu": "wuqu", "quan": "tanlang", "ke": "tianliang", "ji": "wenqu"},
    "庚": {"lu": "taiyang", "quan": "wuqu", "ke": "taiyin", "ji": "tiantong"},
    "辛": {"lu": "jumen", "quan": "taiyang", "ke": "wenqu", "ji": "wenchang"},
    "壬": {"lu": "tianliang", "quan": "ziwei", "ke": "zuofu", "ji": "wuqu"},
    "癸": {"lu": "pojun", "quan": "jumen", "ke": "taiyin", "ji": "tanlang"},
}

DAY_TO_START_INDEX_MAP: Dict[int, int] = {
    1: 2,
    2: 2,
    3: 3,
    4: 5,
    5: 6,
    6: 3,
    7: 4,
    8: 5,
    9: 6,
    10: 7,
    11: 8,
    12: 9,
    13: 10,
    14: 11,
    15: 2,
    16: 2,
    17: 3,
    18: 4,
    19: 5,
    20: 6,
    21: 7,
    22: 8,
    23: 9,
    24: 10,
    25: 11,
    26: 2,
    27: 3,
    28: 4,
    29: 5,
    30: 6,
    31: 7,
}

FIVE_ELEMENT_CLASS_BY_INDEX: List[Tuple[str, int]] = [
    ("수2국", 2),
    ("목3국", 3),
    ("금4국", 4),
    ("토5국", 5),
    ("화6국", 6),
]

NAYIN_TO_BUREAU: Dict[str, Tuple[str, int]] = {
    "水": ("수2국", 2),
    "木": ("목3국", 3),
    "金": ("금4국", 4),
    "土": ("토5국", 5),
    "火": ("화6국", 6),
}

NAYIN_ELEMENT_SEQ_30: List[str] = [
    "金",
    "火",
    "木",
    "土",
    "金",
    "火",
    "水",
    "土",
    "金",
    "木",
    "水",
    "土",
    "火",
    "木",
    "水",
    "金",
    "火",
    "木",
    "土",
    "金",
    "火",
    "水",
    "土",
    "金",
    "木",
    "水",
    "土",
    "火",
    "木",
    "水",
]

CHANGSHENG_START_BY_CLASS = {
    "수2국": 8,   # 申
    "목3국": 11,  # 亥
    "금4국": 5,   # 巳
    "토5국": 8,   # 申
    "화6국": 2,   # 寅
}


@dataclass(frozen=True)
class ZiweiInput:
    birth_date: str
    birth_time: Optional[str]
    gender: Gender
    calendar_type: CalendarType
    lunar_leap_month: bool
    tz_name: str = "Asia/Seoul"
    require_exact_calendar: bool = False


def _fix_index(value: int, length: int = 12) -> int:
    if length <= 0:
        raise ValueError("length must be positive")
    return (value + length) % length


def _parse_input_date(text: str) -> date:
    try:
        return date.fromisoformat(text)
    except ValueError as exc:
        raise ValueError("출생일 형식이 올바르지 않습니다. YYYY-MM-DD 형식으로 입력하세요.") from exc


def _parse_input_time(text: Optional[str]) -> Tuple[int, int]:
    if not text:
        return 12, 0
    raw = text.strip()
    try:
        hh, mm = raw.split(":", 1)
        hour = int(hh)
        minute = int(mm)
    except Exception as exc:
        raise ValueError("출생시간 형식이 올바르지 않습니다. HH:MM 형식으로 입력하세요.") from exc
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise ValueError("출생시간 범위가 올바르지 않습니다. 00:00~23:59")
    return hour, minute


def _convert_solar_to_lunar(solar_date: date) -> Optional[Dict[str, Any]]:
    if KoreanLunarCalendar is None:
        return None
    cal = KoreanLunarCalendar()
    ok = cal.setSolarDate(solar_date.year, solar_date.month, solar_date.day)
    if not ok:
        return None
    ly, lm, ld = map(int, cal.LunarIsoFormat().split("-"))
    return {
        "year": ly,
        "month": lm,
        "day": ld,
        "isLeapMonth": bool(getattr(cal, "isIntercalation", False)),
    }


def _convert_lunar_to_solar(lunar_year: int, lunar_month: int, lunar_day: int, is_leap: bool) -> Optional[date]:
    if KoreanLunarCalendar is None:
        return None
    cal = KoreanLunarCalendar()
    ok = cal.setLunarDate(lunar_year, lunar_month, lunar_day, is_leap)
    if not ok:
        return None
    sy, sm, sd = map(int, cal.SolarIsoFormat().split("-"))
    return date(sy, sm, sd)


def _resolve_solar_and_lunar(inp: ZiweiInput, input_date: date) -> Tuple[date, Dict[str, Any], List[str]]:
    warnings: List[str] = []

    if inp.calendar_type == "solar":
        solar_date = input_date
        lunar_info = _convert_solar_to_lunar(solar_date)
        if lunar_info is None:
            if inp.require_exact_calendar:
                raise RuntimeError(
                    "정확한 명반 계산을 위해 `korean_lunar_calendar`가 필요합니다. "
                    "환경에 패키지를 설치한 뒤 다시 시도하세요."
                )
            warnings.append("korean_lunar_calendar 미설치 또는 변환 실패로 음력값은 근사치(양력 동일값)로 표기됩니다.")
            lunar_info = {
                "year": solar_date.year,
                "month": solar_date.month,
                "day": solar_date.day,
                "isLeapMonth": False,
            }
        return solar_date, lunar_info, warnings

    solar_date = _convert_lunar_to_solar(input_date.year, input_date.month, input_date.day, inp.lunar_leap_month)
    if solar_date is None:
        if inp.require_exact_calendar:
            raise RuntimeError(
                "정확한 명반 계산을 위해 `korean_lunar_calendar`가 필요합니다. "
                "환경에 패키지를 설치한 뒤 다시 시도하세요."
            )
        warnings.append("음력→양력 변환 실패로 입력 날짜를 그대로 사용했습니다. 정밀도 저하 가능성이 있습니다.")
        solar_date = input_date

    lunar_info = {
        "year": input_date.year,
        "month": input_date.month,
        "day": input_date.day,
        "isLeapMonth": bool(inp.lunar_leap_month),
    }
    return solar_date, lunar_info, warnings


def _time_index(hour: int) -> int:
    # 子시 시작을 23:00 기준으로 0~11 인덱스
    return ((hour + 1) // 2) % 12


def _time_range_from_index(idx: int) -> str:
    if idx == 0:
        return "23:00-00:59"
    start = (idx * 2 - 1) % 24
    end = (start + 1) % 24
    return f"{start:02d}:00-{end:02d}:59"


def _sexagenary_for_datetime(dt_local: datetime) -> Dict[str, Dict[str, Any]]:
    t_idx = _time_index(dt_local.hour)

    # 간명식(절기 미적용): 자미두수 UI용 구조 데이터에 사용
    y_stem = (dt_local.year - 4) % 10
    y_branch = (dt_local.year - 4) % 12

    m_branch = (dt_local.month + 1) % 12
    m_stem = (y_stem * 2 + dt_local.month) % 10

    day_no = dt_local.date().toordinal()
    d_stem = (day_no + 4) % 10
    d_branch = (day_no + 2) % 12

    h_branch = t_idx
    h_stem = (d_stem * 2 + t_idx) % 10

    def pillar(stem_i: int, branch_i: int) -> Dict[str, Any]:
        return {
            "stemIndex": stem_i,
            "branchIndex": branch_i,
            "stem": STEMS[stem_i],
            "branch": BRANCHES[branch_i],
            "ganzhi": f"{STEMS[stem_i]}{BRANCHES[branch_i]}",
        }

    return {
        "year": pillar(y_stem, y_branch),
        "month": pillar(m_stem, m_branch),
        "day": pillar(d_stem, d_branch),
        "hour": pillar(h_stem, h_branch),
    }


def _sexagenary_year_indices(year: int) -> Tuple[int, int]:
    return (year - 4) % 10, (year - 4) % 12


def _western_sign(month: int, day: int) -> str:
    signs = [
        ((1, 20), "Capricorn"),
        ((2, 19), "Aquarius"),
        ((3, 21), "Pisces"),
        ((4, 20), "Aries"),
        ((5, 21), "Taurus"),
        ((6, 22), "Gemini"),
        ((7, 23), "Cancer"),
        ((8, 23), "Leo"),
        ((9, 24), "Virgo"),
        ((10, 24), "Libra"),
        ((11, 23), "Scorpio"),
        ((12, 22), "Sagittarius"),
        ((12, 32), "Capricorn"),
    ]
    for (m_limit, d_limit), sign in signs:
        if month < m_limit or (month == m_limit and day < d_limit):
            return sign
    return "Capricorn"


def _rotated_palaces(ming_index: int) -> List[Tuple[str, str, str]]:
    # branch index(idx: 子=0..亥=11)에서,
    # 명궁에서 역배한 궁순서를 매핑한다.
    # idx==ming_index => 命宮, idx==ming_index-1 => 兄弟, ...
    out: List[Tuple[str, str, str]] = []
    for idx in range(12):
        seq_idx = (ming_index - idx) % 12
        out.append(PALACE_SEQUENCE[seq_idx])
    return out


def _fix_lunar_month_index(month: int, is_leap_month: bool) -> int:
    if month < 1:
        return 1
    if month > 12:
        month = ((month - 1) % 12) + 1
    if is_leap_month:
        # 윤달은 다음 달로 넘겨 계산(iztro fixLeapMonth 옵션과 동일한 방향)
        return 1 if month == 12 else month + 1
    return month


def _is_yang_stem(stem_index: int) -> bool:
    return stem_index % 2 == 0


def _is_forward_daeun(gender: Gender, year_stem_index: int) -> bool:
    if gender == "male":
        return _is_yang_stem(year_stem_index)
    return not _is_yang_stem(year_stem_index)


def _palace_heavenly_stem_index(year_stem_index: int, branch_index: int) -> int:
    # 五虎遁: 甲己丙寅, 乙庚戊寅, 丙辛庚寅, 丁壬壬寅, 戊癸甲寅
    if year_stem_index in (0, 5):
        start_stem = 2
    elif year_stem_index in (1, 6):
        start_stem = 4
    elif year_stem_index in (2, 7):
        start_stem = 6
    elif year_stem_index in (3, 8):
        start_stem = 8
    else:
        start_stem = 0
    offset = _fix_index(branch_index - 2)
    return _fix_index(start_stem + offset, 10)


def _compute_five_elements_class(ming_stem_index: int, ming_branch_index: int) -> Tuple[str, int]:
    # 명궁의 간지(甲子~癸亥) 납음 오행으로 국수를 결정한다.
    jiazi_index: Optional[int] = None
    for i in range(60):
        if (i % 10) == ming_stem_index and (i % 12) == ming_branch_index:
            jiazi_index = i
            break

    if jiazi_index is None:
        # 비정상 입력 보호용 fallback
        stem_group = ming_stem_index // 2
        branch_group = ming_branch_index % 3
        return FIVE_ELEMENT_CLASS_BY_INDEX[(stem_group + branch_group) % 5]

    nayin_element = NAYIN_ELEMENT_SEQ_30[jiazi_index // 2]
    return NAYIN_TO_BUREAU[nayin_element]


def _calc_start_position(day: int, bureau_num: int) -> int:
    q = int(math.ceil(day / float(bureau_num)))
    remainder = q * bureau_num - day
    if remainder % 2 == 0:
        return q + remainder
    return q - remainder


def _get_start_index(lunar_day: int, bureau_num: int) -> int:
    fixed_day = max(1, min(31, int(lunar_day)))
    base = DAY_TO_START_INDEX_MAP[fixed_day]
    n = _fix_index(base - 2)
    # startPoint는 오행국 수치(2~6)를 사용한다.
    distance = max(2, min(6, int(bureau_num)))
    raw = _calc_start_position(fixed_day, distance) - 1 + n
    return _fix_index(raw)


def _yin_to_branch_index(idx: int) -> int:
    # 일부 전통식/라이브러리 내부 좌표는 寅=0으로 계산된다.
    # 본 프로젝트의 궁 인덱스는 子=0이므로 +2 보정이 필요하다.
    return _fix_index(idx + 2)


def _major_star_positions(lunar_day: int, bureau_num: int) -> Dict[str, int]:
    # 14주성 기본 배치는 寅궁을 0으로 두는 좌표에서 계산 후,
    # 최종적으로 子=0 좌표로 변환한다.
    ziwei_yin = _get_start_index(lunar_day, bureau_num)
    tianfu_yin = 0 if ziwei_yin == 0 else _fix_index(12 - ziwei_yin)

    yin_positions = {
        "ziwei": ziwei_yin,
        "tianji": _fix_index(ziwei_yin - 1),
        "taiyang": _fix_index(ziwei_yin - 3),
        "wuqu": _fix_index(ziwei_yin - 4),
        "tiantong": _fix_index(ziwei_yin - 5),
        "lianzhen": _fix_index(ziwei_yin - 8),
        "tianfu": tianfu_yin,
        "taiyin": _fix_index(tianfu_yin + 1),
        "tanlang": _fix_index(tianfu_yin + 2),
        "jumen": _fix_index(tianfu_yin + 3),
        "tianxiang": _fix_index(tianfu_yin + 4),
        "tianliang": _fix_index(tianfu_yin + 5),
        "qisha": _fix_index(tianfu_yin + 6),
        "pojun": _fix_index(tianfu_yin + 10),
    }
    return {sid: _yin_to_branch_index(pos) for sid, pos in yin_positions.items()}


def _lu_yang_tuo_ma_indices(year_stem_index: int) -> Dict[str, int]:
    lu_base = [2, 3, 5, 6, 5, 6, 8, 9, 11, 0][year_stem_index]
    return {
        "lucun": lu_base,
        "qingyang": _fix_index(lu_base + 1),
        "tuoluo": _fix_index(lu_base - 1),
        "tianma": _fix_index(lu_base + 6),
    }


def _kui_yue_indices(year_stem_index: int) -> Dict[str, int]:
    # 甲戊庚: 丑未, 乙己: 子申, 丙丁: 亥酉, 辛: 寅午, 壬癸: 卯巳
    kui = [1, 0, 11, 11, 1, 0, 1, 2, 3, 3][year_stem_index]
    yue = [7, 8, 9, 9, 7, 8, 7, 6, 5, 5][year_stem_index]
    return {"tiankui": kui, "tianyue": yue}


def _zuo_you_indices(lunar_month: int) -> Dict[str, int]:
    return {
        "zuofu": _fix_index(4 + lunar_month - 1),
        "youbi": _fix_index(10 - lunar_month + 1),
    }


def _chang_qu_indices(time_index: int) -> Dict[str, int]:
    return {
        "wenchang": _fix_index(10 - time_index),
        "wenqu": _fix_index(4 + time_index),
    }


def _kong_jie_indices(time_index: int) -> Dict[str, int]:
    return {
        "dikong": _fix_index(11 - time_index),
        "dijie": _fix_index(11 + time_index),
    }


def _huo_ling_indices(year_branch_index: int, time_index: int) -> Dict[str, int]:
    group = (year_branch_index + 12 - 2) % 4
    if group == 0:  # 寅午戌
        huo = _fix_index(1 + time_index)
        ling = _fix_index(3 + time_index)
    elif group == 1:  # 申子辰
        huo = _fix_index(2 + time_index)
        ling = _fix_index(10 + time_index)
    elif group == 2:  # 巳酉丑
        huo = _fix_index(3 + time_index)
        ling = _fix_index(10 + time_index)
    else:  # 亥卯未
        huo = _fix_index(9 + time_index)
        ling = _fix_index(10 + time_index)
    return {"huoxing": huo, "lingxing": ling}


def _hongluan_tianxi_indices(year_branch_index: int) -> Dict[str, int]:
    # 紅鸞은 연지 기준 역행, 天喜는 맞은편 궁
    hongluan = [3, 2, 1, 0, 11, 10, 9, 8, 7, 6, 5, 4][year_branch_index]
    return {"hongluan": hongluan, "tianxi": _fix_index(hongluan + 6)}


def _guchen_guasu_indices(year_branch_index: int) -> Dict[str, int]:
    if year_branch_index in (2, 3, 4):  # 寅卯辰
        return {"guchen": 1, "guasu": 7}  # 丑, 未
    if year_branch_index in (5, 6, 7):  # 巳午未
        return {"guchen": 4, "guasu": 10}  # 辰, 戌
    if year_branch_index in (8, 9, 10):  # 申酉戌
        return {"guchen": 7, "guasu": 1}  # 未, 丑
    return {"guchen": 10, "guasu": 4}  # 亥子丑 -> 戌, 辰


def _minor_star_positions(
    year_stem_index: int,
    year_branch_index: int,
    lunar_month: int,
    time_index: int,
) -> Dict[str, int]:
    out: Dict[str, int] = {}
    out.update(_lu_yang_tuo_ma_indices(year_stem_index))
    out.update(_kui_yue_indices(year_stem_index))
    out.update(_zuo_you_indices(lunar_month))
    out.update(_chang_qu_indices(time_index))
    out.update(_kong_jie_indices(time_index))
    out.update(_huo_ling_indices(year_branch_index, time_index))
    out.update(_hongluan_tianxi_indices(year_branch_index))
    out.update(_guchen_guasu_indices(year_branch_index))
    return out


def _brightness_for_star(star_id: str, palace_index: int) -> str:
    anchor = STAR_BRIGHTNESS_ANCHOR.get(star_id, 0)
    distance = _fix_index(palace_index - anchor)
    if distance in (0, 1):
        return BRIGHTNESS_ORDER[0]
    if distance in (2, 11, 10):
        return BRIGHTNESS_ORDER[1]
    if distance in (3, 9, 4):
        return BRIGHTNESS_ORDER[2]
    if distance in (5, 8):
        return BRIGHTNESS_ORDER[3]
    return BRIGHTNESS_ORDER[4]


def _star_object(star: StarDef, palace_index: int, scope: str = "origin") -> Dict[str, Any]:
    return {
        "starId": star.star_id,
        "labelKo": star.label_ko,
        "labelZh": star.label_zh,
        "type": star.star_type,
        "scope": scope,
        "brightness": _brightness_for_star(star.star_id, palace_index),
        "isTransformed": False,
        "transformType": None,
        "sourceStem": None,
        "sourceScope": "origin",
    }


def _find_star_in_palace(palace: Dict[str, Any], star_id: str) -> Optional[Dict[str, Any]]:
    stars = palace.get("stars", {})
    for key in ("majorStars", "minorStars", "adjectiveStars", "miscStars", "shenshaStars"):
        for row in stars.get(key, []):
            if row.get("starId") == star_id:
                return row
    return None


def _apply_star_positions(palaces: List[Dict[str, Any]], positions: Dict[str, int]) -> None:
    for palace in palaces:
        palace["stars"] = {
            "majorStars": [],
            "minorStars": [],
            "adjectiveStars": [],
            "miscStars": [],
            "shenshaStars": [],
            "major": [],
            "minor": [],
            "misc": [],
        }
        palace["brightness"] = {}

    for star_id in STAR_ORDER:
        if star_id not in positions:
            continue
        pos = _fix_index(int(positions[star_id]))
        palace = palaces[pos]
        star_def = STAR_DEFS[star_id]
        row = _star_object(star_def, pos, scope="origin")
        palace["stars"][star_def.bucket].append(row)

    for palace in palaces:
        stars = palace["stars"]
        for bucket in ("majorStars", "minorStars", "adjectiveStars", "miscStars", "shenshaStars"):
            stars[bucket].sort(key=lambda x: STAR_ORDER.index(x["starId"]) if x["starId"] in STAR_ORDER else 999)

        stars["major"] = [row["labelZh"] for row in stars["majorStars"]]
        stars["minor"] = [row["labelZh"] for row in stars["minorStars"]]
        stars["misc"] = [row["labelZh"] for row in stars["adjectiveStars"] + stars["miscStars"] + stars["shenshaStars"]]

        brightness_map: Dict[str, str] = {}
        for row in stars["majorStars"] + stars["minorStars"] + stars["adjectiveStars"] + stars["miscStars"] + stars["shenshaStars"]:
            brightness_map[row["labelZh"]] = row["brightness"]
        palace["brightness"] = brightness_map
        palace["isEmptyPalace"] = len(stars["majorStars"]) == 0


def _build_scope_transformations(
    scope_name: str,
    source_stem: str,
    palaces: List[Dict[str, Any]],
) -> Dict[str, Any]:
    rule = FOUR_TRANSFORM_RULES.get(source_stem)
    if rule is None:
        rule = FOUR_TRANSFORM_RULES["甲"]

    palace_by_star: Dict[str, Dict[str, Any]] = {}
    for p in palaces:
        for key in ("majorStars", "minorStars", "adjectiveStars"):
            for row in p.get("stars", {}).get(key, []):
                palace_by_star[row["starId"]] = p

    by_star: List[Dict[str, Any]] = []
    by_palace_map: Dict[str, Dict[str, Any]] = {}
    flying_edges: List[Dict[str, Any]] = []
    generated_map: Dict[str, List[Dict[str, Any]]] = {}

    natal_summary: Dict[str, Any] = {"heavenlyStem": source_stem, "lu": None, "quan": None, "ke": None, "ji": None}

    for tf_type, tf_ko, tf_zh in FOUR_TRANSFORM_META:
        star_id = rule[tf_type]
        palace = palace_by_star.get(star_id)
        if palace is None:
            continue

        row = {
            "starId": star_id,
            "labelKo": STAR_DEFS[star_id].label_ko,
            "labelZh": STAR_DEFS[star_id].label_zh,
            "palaceId": palace["palaceId"],
            "palaceNameKo": palace["labelKo"],
            "transformType": tf_type,
            "transformTypeKo": tf_ko,
            "transformTypeZh": tf_zh,
            "sourceScope": scope_name,
            "sourceStem": source_stem,
            "targetPalaceId": palace["palaceId"],
            "targetPalaceNameKo": palace["labelKo"],
        }
        by_star.append(row)
        natal_summary[tf_type] = star_id

        by_palace_map.setdefault(
            palace["palaceId"],
            {
                "palaceId": palace["palaceId"],
                "luStars": [],
                "quanStars": [],
                "keStars": [],
                "jiStars": [],
            },
        )
        by_palace_map[palace["palaceId"]][f"{tf_type}Stars"].append(star_id)

        flying_edges.append(
            {
                "fromPalaceId": palace["palaceId"],
                "toPalaceId": palace["palaceId"],
                "transformType": tf_type,
                "sourceScope": scope_name,
            }
        )
        generated_map.setdefault(palace["palaceId"], []).append(
            {"toPalaceId": palace["palaceId"], "transformType": tf_type}
        )

    generated_transformation_palaces = [
        {"palaceId": pid, "targets": targets} for pid, targets in generated_map.items()
    ]

    return {
        "scope": scope_name,
        "natal": natal_summary,
        "byPalace": list(by_palace_map.values()),
        "byStar": by_star,
        "flyingEdges": flying_edges,
        "generatedTransformationPalaces": generated_transformation_palaces,
    }


def _annotate_natal_transformations(palaces: List[Dict[str, Any]], natal_by_star: List[Dict[str, Any]]) -> None:
    palace_map = {p["palaceId"]: p for p in palaces}
    for row in natal_by_star:
        palace = palace_map.get(row["palaceId"])
        if palace is None:
            continue
        target = _find_star_in_palace(palace, row["starId"])
        if target is None:
            continue
        target["isTransformed"] = True
        target["transformType"] = row["transformType"]
        target["sourceStem"] = row["sourceStem"]
        target["sourceScope"] = row["sourceScope"]


def _build_indexes(palaces: List[Dict[str, Any]], natal_transforms: Dict[str, Any]) -> Dict[str, Any]:
    star_to_palace: Dict[str, List[str]] = {}
    palace_to_star_ids: Dict[str, List[str]] = {}
    major_star_to_palace: Dict[str, List[str]] = {}
    palace_stem_map: Dict[str, str] = {}
    palace_branch_map: Dict[str, str] = {}
    empty_palace_ids: List[str] = []

    for p in palaces:
        pid = p["palaceId"]
        palace_stem_map[pid] = p["heavenlyStem"]
        palace_branch_map[pid] = p["earthlyBranch"]
        if p.get("isEmptyPalace"):
            empty_palace_ids.append(pid)

        star_ids: List[str] = []
        stars = p.get("stars", {})
        for key in ("majorStars", "minorStars", "adjectiveStars", "miscStars", "shenshaStars"):
            for row in stars.get(key, []):
                sid = row["starId"]
                star_ids.append(sid)
                star_to_palace.setdefault(sid, []).append(pid)
                if key == "majorStars":
                    major_star_to_palace.setdefault(sid, []).append(pid)
        palace_to_star_ids[pid] = sorted(set(star_ids))

    transformed_star_map: Dict[str, List[Dict[str, Any]]] = {}
    for row in natal_transforms.get("byStar", []):
        transformed_star_map.setdefault(row["starId"], []).append(
            {
                "palaceId": row["palaceId"],
                "transformType": row["transformType"],
                "targetPalaceId": row["targetPalaceId"],
                "sourceScope": row["sourceScope"],
                "sourceStem": row["sourceStem"],
            }
        )

    ming = next((p for p in palaces if p.get("isLifePalace")), palaces[0])
    shen = next((p for p in palaces if p.get("isBodyPalace")), palaces[0])

    return {
        "starToPalaceMap": {k: sorted(set(v)) for k, v in star_to_palace.items()},
        "palaceToStarIdsMap": palace_to_star_ids,
        "majorStarToPalaceMap": {k: sorted(set(v)) for k, v in major_star_to_palace.items()},
        "transformedStarMap": transformed_star_map,
        "emptyPalaceIds": sorted(set(empty_palace_ids)),
        "lifeBodyAxis": {
            "lifePalaceId": ming["palaceId"],
            "bodyPalaceId": shen["palaceId"],
            "lifeOppositePalaceId": ming["oppositePalaceId"],
            "bodyOppositePalaceId": shen["oppositePalaceId"],
        },
        "palaceStemMap": palace_stem_map,
        "palaceBranchMap": palace_branch_map,
    }


def _build_relations(palaces: List[Dict[str, Any]], ming_palace_id: str) -> Dict[str, Any]:
    by_id = {p["palaceId"]: p for p in palaces}
    oppositions = [{"fromPalaceId": p["palaceId"], "toPalaceId": p["oppositePalaceId"]} for p in palaces]
    three_harmony_groups = [
        {"centerPalaceId": p["palaceId"], "relatedPalaceIds": list(p["threeHarmonyPalaceIds"])} for p in palaces
    ]
    four_cardinal_groups = [
        {"centerPalaceId": p["palaceId"], "relatedPalaceIds": list(p["fourCardinalPalaceIds"])} for p in palaces
    ]

    def trine_for(palace_id: str) -> List[str]:
        row = by_id[palace_id]
        return [palace_id] + list(row["threeHarmonyPalaceIds"])

    life_trine = trine_for(ming_palace_id)
    career_trine = trine_for("career") if "career" in by_id else life_trine
    wealth_trine = trine_for("wealth") if "wealth" in by_id else life_trine

    return {
        "oppositions": oppositions,
        "threeHarmonyGroups": three_harmony_groups,
        "fourCardinalGroups": four_cardinal_groups,
        "lifeTrine": life_trine,
        "careerTrine": career_trine,
        "wealthTrine": wealth_trine,
    }


def _build_horoscope_entry(
    scope: str,
    idx: int,
    stem_idx: int,
    branch_idx: int,
    palaces: List[Dict[str, Any]],
    transform_scope: Dict[str, Any],
    nominal_age: int,
    year_value: Optional[int] = None,
    month_value: Optional[int] = None,
    day_value: Optional[str] = None,
    time_label: Optional[str] = None,
) -> Dict[str, Any]:
    shift = idx % 12
    palace_shift_map = {p["palaceId"]: palaces[(p["index"] + shift) % 12]["palaceId"] for p in palaces}

    transformed = transform_scope.get("byStar", [])
    mutagen = [
        {
            "transformType": row["transformType"],
            "starId": row["starId"],
            "palaceId": row["palaceId"],
            "targetPalaceId": row["targetPalaceId"],
        }
        for row in transformed
    ]

    stars = [
        {
            "starId": row["starId"],
            "labelZh": row["labelZh"],
            "palaceId": row["palaceId"],
        }
        for row in transformed
    ]

    out = {
        "index": idx,
        "heavenlyStem": STEMS[_fix_index(stem_idx, 10)],
        "earthlyBranch": BRANCHES[_fix_index(branch_idx, 12)],
        "palaceNames": [palaces[(i + shift) % 12]["labelKo"] for i in range(12)],
        "palaceShiftMap": palace_shift_map,
        "mutagen": mutagen,
        "stars": stars,
        "age": {"index": idx + 1, "nominalAge": nominal_age},
        "scope": scope,
    }
    if year_value is not None:
        out["year"] = year_value
    if month_value is not None:
        out["month"] = month_value
    if day_value is not None:
        out["day"] = day_value
    if time_label is not None:
        out["timeLabel"] = time_label
    return out


def _build_horoscope(
    birth_dt: datetime,
    now_kst: datetime,
    palaces: List[Dict[str, Any]],
    by_scope_transforms: Dict[str, Dict[str, Any]],
    age_base: int,
    birth_pillars: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    decadal: List[Dict[str, Any]] = []
    month_stem = int(birth_pillars["month"]["stemIndex"])
    month_branch = int(birth_pillars["month"]["branchIndex"])
    for i in range(12):
        decadal.append(
            _build_horoscope_entry(
                "decadal",
                i,
                stem_idx=month_stem + i,
                branch_idx=month_branch + i,
                palaces=palaces,
                transform_scope=by_scope_transforms["decadal"],
                nominal_age=age_base + i * 10,
            )
        )

    yearly: List[Dict[str, Any]] = []
    start_year = now_kst.year - 5
    for i in range(12):
        year_value = start_year + i
        ys, yb = _sexagenary_year_indices(year_value)
        yearly.append(
            _build_horoscope_entry(
                "yearly",
                i,
                stem_idx=ys,
                branch_idx=yb,
                palaces=palaces,
                transform_scope=by_scope_transforms["yearly"],
                nominal_age=age_base + i,
                year_value=year_value,
            )
        )

    monthly: List[Dict[str, Any]] = []
    ys_now, _ = _sexagenary_year_indices(now_kst.year)
    for i in range(12):
        month_no = i + 1
        ms = (ys_now * 2 + month_no) % 10
        mb = (month_no + 1) % 12
        monthly.append(
            _build_horoscope_entry(
                "monthly",
                i,
                stem_idx=ms,
                branch_idx=mb,
                palaces=palaces,
                transform_scope=by_scope_transforms["yearly"],
                nominal_age=age_base + (i // 2),
                year_value=now_kst.year,
                month_value=month_no,
            )
        )

    daily: List[Dict[str, Any]] = []
    for i in range(7):
        target_day = (now_kst.date() - timedelta(days=3)) + timedelta(days=i)
        target_dt = datetime(target_day.year, target_day.month, target_day.day, 12, 0, tzinfo=SEOUL_TZ)
        dp = _sexagenary_for_datetime(target_dt)["day"]
        daily.append(
            _build_horoscope_entry(
                "daily",
                i,
                stem_idx=int(dp["stemIndex"]),
                branch_idx=int(dp["branchIndex"]),
                palaces=palaces,
                transform_scope=by_scope_transforms["yearly"],
                nominal_age=age_base,
                day_value=target_day.isoformat(),
            )
        )

    hourly: List[Dict[str, Any]] = []
    today_noon = datetime(now_kst.year, now_kst.month, now_kst.day, 12, 0, tzinfo=SEOUL_TZ)
    day_stem_idx = int(_sexagenary_for_datetime(today_noon)["day"]["stemIndex"])
    for i in range(12):
        hs = (day_stem_idx * 2 + i) % 10
        hb = i
        hourly.append(
            _build_horoscope_entry(
                "hourly",
                i,
                stem_idx=hs,
                branch_idx=hb,
                palaces=palaces,
                transform_scope=by_scope_transforms["yearly"],
                nominal_age=age_base,
                day_value=now_kst.date().isoformat(),
                time_label=f"{BRANCHES[i]}時",
            )
        )

    return {
        "decadal": decadal,
        "yearly": yearly,
        "monthly": monthly,
        "daily": daily,
        "hourly": hourly,
    }


def _build_dynamic_palace_snapshots(
    palaces: List[Dict[str, Any]],
    transform_scope: Dict[str, Any],
    offset: int,
) -> List[Dict[str, Any]]:
    by_palace = {row["palaceId"]: row for row in transform_scope.get("byPalace", [])}
    out: List[Dict[str, Any]] = []
    for p in palaces:
        shifted = palaces[(p["index"] + offset) % 12]
        majors = shifted.get("stars", {}).get("majorStars", [])[:2]
        trans = by_palace.get(p["palaceId"], {})
        dynamic_transformations = []
        for tf_type, _, _ in FOUR_TRANSFORM_META:
            for sid in trans.get(f"{tf_type}Stars", []):
                dynamic_transformations.append({"transformType": tf_type, "starId": sid})
        out.append(
            {
                "palaceId": p["palaceId"],
                "shiftedName": shifted["labelKo"],
                "dynamicStars": [row["starId"] for row in majors],
                "dynamicTransformations": dynamic_transformations,
                "hasSpecificTargetFlags": {
                    "hasLu": bool(trans.get("luStars")),
                    "hasQuan": bool(trans.get("quanStars")),
                    "hasKe": bool(trans.get("keStars")),
                    "hasJi": bool(trans.get("jiStars")),
                },
            }
        )
    return out


def _build_query_helpers(
    palaces: List[Dict[str, Any]],
    indexes: Dict[str, Any],
    relations: Dict[str, Any],
    transformations: Dict[str, Any],
) -> Dict[str, Any]:
    has_star_in_palace: Dict[str, Dict[str, bool]] = {}
    for p in palaces:
        pid = p["palaceId"]
        has_star_in_palace[pid] = {sid: True for sid in indexes["palaceToStarIdsMap"].get(pid, [])}

    by_palace = {row["palaceId"]: row for row in transformations.get("byPalace", [])}
    has_transformation_in_palace: Dict[str, Dict[str, bool]] = {}
    for p in palaces:
        pid = p["palaceId"]
        row = by_palace.get(pid, {})
        has_transformation_in_palace[pid] = {
            "lu": bool(row.get("luStars")),
            "quan": bool(row.get("quanStars")),
            "ke": bool(row.get("keStars")),
            "ji": bool(row.get("jiStars")),
        }

    opposite_map = {row["fromPalaceId"]: row["toPalaceId"] for row in relations.get("oppositions", [])}
    empty_map = {pid: True for pid in indexes.get("emptyPalaceIds", [])}

    flying_target_map: Dict[str, List[str]] = {}
    for row in transformations.get("flyingEdges", []):
        flying_target_map.setdefault(row["fromPalaceId"], []).append(row["toPalaceId"])

    return {
        "hasStarInPalace": has_star_in_palace,
        "hasTransformationInPalace": has_transformation_in_palace,
        "getPalaceOfStar": indexes.get("starToPalaceMap", {}),
        "getOppositePalace": opposite_map,
        "isEmptyPalace": empty_map,
        "doesPalaceGenerateFlyingToTarget": flying_target_map,
    }


def _first_star_label(palace: Dict[str, Any]) -> str:
    stars = palace.get("stars", {})
    for key in ("majorStars", "minorStars", "adjectiveStars"):
        rows = stars.get(key, [])
        if rows:
            return rows[0].get("labelZh", "")
    return ""


def compute_ziwei_payload(inp: ZiweiInput) -> Dict[str, Any]:
    input_date = _parse_input_date(inp.birth_date)
    hour, minute = _parse_input_time(inp.birth_time)

    solar_date, lunar_info, warnings = _resolve_solar_and_lunar(inp, input_date)
    dt_local = datetime(solar_date.year, solar_date.month, solar_date.day, hour, minute, tzinfo=SEOUL_TZ)
    now_kst = datetime.now(SEOUL_TZ)

    t_idx = _time_index(dt_local.hour)
    pillars = _sexagenary_for_datetime(dt_local)

    input_raw = {
        "calendarType": inp.calendar_type,
        "year": input_date.year,
        "month": input_date.month,
        "day": input_date.day,
        "hour": hour,
        "minute": minute,
        "gender": inp.gender,
        "isLeapMonth": bool(inp.lunar_leap_month),
        "timezone": inp.tz_name,
    }

    normalized = {
        "solarDate": solar_date.isoformat(),
        "lunarDate": {
            "year": int(lunar_info["year"]),
            "month": int(lunar_info["month"]),
            "day": int(lunar_info["day"]),
            "isLeapMonth": bool(lunar_info.get("isLeapMonth", False)),
        },
        "fourPillars": pillars,
        "timeLabel": f"{BRANCHES[t_idx]}時",
        "timeRange": _time_range_from_index(t_idx),
        "timeIndex": t_idx,
        "dayShiftedByTimeRule": bool(dt_local.hour == 23),
    }

    lunar_month_fixed = _fix_lunar_month_index(int(lunar_info["month"]), bool(lunar_info.get("isLeapMonth", False)))
    lunar_day = int(lunar_info["day"])
    lunar_year = int(lunar_info["year"])

    # 命宮/身宮: 寅궁(2)을 기준으로 음력 월(1월=0 오프셋)과 시지로 계산
    month_offset = lunar_month_fixed - 1
    ming_index = _fix_index(2 + month_offset - t_idx)
    shen_index = _fix_index(2 + month_offset + t_idx)

    rotated = _rotated_palaces(ming_index)
    # 본명반 핵심 배치는 음력 연간/연지 기준
    year_stem_idx, year_branch_idx = _sexagenary_year_indices(lunar_year)

    ming_stem_idx = _palace_heavenly_stem_index(year_stem_idx, ming_index)
    five_elements_class, bureau_num = _compute_five_elements_class(ming_stem_idx, ming_index)

    major_positions = _major_star_positions(lunar_day=lunar_day, bureau_num=bureau_num)
    minor_positions = _minor_star_positions(
        year_stem_index=year_stem_idx,
        year_branch_index=year_branch_idx,
        lunar_month=lunar_month_fixed,
        time_index=t_idx,
    )
    all_positions: Dict[str, int] = {**major_positions, **minor_positions}

    forward_daeun = _is_forward_daeun(inp.gender, year_stem_idx)
    lu_index = all_positions.get("lucun", ming_index)

    palaces: List[Dict[str, Any]] = []
    for idx in range(12):
        palace_id, label_ko, label_zh = rotated[idx]

        stem_idx = _palace_heavenly_stem_index(year_stem_idx, idx)
        stem = STEMS[stem_idx]
        branch = BRANCHES[idx]

        order = (idx - ming_index) % 12 if forward_daeun else (ming_index - idx) % 12
        range_start = order * 10 + 1
        range_end = range_start + 9

        opposite_idx = (idx + 6) % 12
        triad_idx = [(idx + 4) % 12, (idx + 8) % 12]
        cardinal_idx = [(idx + 3) % 12, (idx + 6) % 12, (idx + 9) % 12]

        changsheng_start = CHANGSHENG_START_BY_CLASS.get(five_elements_class, 2)
        changsheng = CHANGSHENG_12[_fix_index(idx - changsheng_start)]

        palaces.append(
            {
                "palaceId": palace_id,
                "labelKo": label_ko,
                "labelZh": label_zh,
                "index": idx,
                "heavenlyStem": stem,
                "earthlyBranch": branch,
                "isLifePalace": idx == ming_index,
                "isBodyPalace": idx == shen_index,
                "isOriginalPalace": True,
                "isEmptyPalace": False,
                "oppositePalaceId": rotated[opposite_idx][0],
                "threeHarmonyPalaceIds": [rotated[i][0] for i in triad_idx],
                "fourCardinalPalaceIds": [rotated[i][0] for i in cardinal_idx],
                "stage": {
                    "rangeStart": range_start,
                    "rangeEnd": range_end,
                    "stageHeavenlyStem": STEMS[_fix_index(year_stem_idx + order, 10)],
                },
                "ages": [range_start + 12 * i for i in range(10)],
                "changsheng12": changsheng,
                "boshi12": BOSHI_12[_fix_index(idx - lu_index)],
                "jiangqian12": JIANGQIAN_12[_fix_index(idx - year_branch_idx)],
                "suiqian12": SUIQIAN_12[_fix_index(idx - year_branch_idx)],
                "stars": {},
                "brightness": {},
                # UI backward compatibility
                "key": palace_id,
                "name_ko": label_ko,
                "name_hanja": label_zh,
                "stem": stem,
                "branch": branch,
                "is_ming": idx == ming_index,
                "is_shen": idx == shen_index,
                "opposite_index": opposite_idx,
                "triad_indices": triad_idx,
                "grid_position": {"row": GRID_RING_POSITIONS[idx][0], "col": GRID_RING_POSITIONS[idx][1]},
            }
        )

    _apply_star_positions(palaces, all_positions)

    natal_scope = _build_scope_transformations("natal", STEMS[year_stem_idx], palaces)
    decadal_scope = _build_scope_transformations("decadal", pillars["month"]["stem"], palaces)
    current_year_stem_idx, _ = _sexagenary_year_indices(now_kst.year)
    yearly_scope = _build_scope_transformations("yearly", STEMS[current_year_stem_idx], palaces)
    by_scope = {"natal": natal_scope, "decadal": decadal_scope, "yearly": yearly_scope}

    _annotate_natal_transformations(palaces, natal_scope["byStar"])

    indexes = _build_indexes(palaces, natal_scope)
    relations = _build_relations(palaces, rotated[ming_index][0])

    transformations = {
        "natal": natal_scope["natal"],
        "byPalace": natal_scope["byPalace"],
        "byStar": natal_scope["byStar"],
        "flyingEdges": natal_scope["flyingEdges"],
        "generatedTransformationPalaces": natal_scope["generatedTransformationPalaces"],
        "byScope": by_scope,
    }

    compatibility_four_transforms = {
        scope: [
            {
                "type": row["transformTypeKo"],
                "type_hanja": row["transformTypeZh"],
                "star": row["labelZh"],
                "palace_index": next((p["index"] for p in palaces if p["palaceId"] == row["palaceId"]), 0),
                "palace_name_ko": row["palaceNameKo"],
                "source": {"natal": "생년", "decadal": "대운", "yearly": "세운"}[scope],
            }
            for row in by_scope[scope]["byStar"]
        ]
        for scope in ("natal", "decadal", "yearly")
    }

    age_base = now_kst.year - solar_date.year + 1
    horoscope = _build_horoscope(
        birth_dt=dt_local,
        now_kst=now_kst,
        palaces=palaces,
        by_scope_transforms=by_scope,
        age_base=age_base,
        birth_pillars=pillars,
    )

    dynamic_palaces = {
        "decadalPalaces": _build_dynamic_palace_snapshots(palaces, decadal_scope, 2),
        "yearlyPalaces": _build_dynamic_palace_snapshots(palaces, yearly_scope, 1),
        "monthlyPalaces": _build_dynamic_palace_snapshots(palaces, yearly_scope, 3),
        "dailyPalaces": _build_dynamic_palace_snapshots(palaces, yearly_scope, 5),
        "hourlyPalaces": _build_dynamic_palace_snapshots(palaces, yearly_scope, 7),
    }

    chart_meta = {
        "zodiac": ZODIAC_ANIMALS_KO[year_branch_idx],
        "westernSign": _western_sign(solar_date.month, solar_date.day),
        "earthlyBranchOfSoulPalace": palaces[ming_index]["earthlyBranch"],
        "earthlyBranchOfBodyPalace": palaces[shen_index]["earthlyBranch"],
        "soulMaster": _first_star_label(palaces[ming_index]),
        "bodyMaster": _first_star_label(palaces[shen_index]),
        "fiveElementsClass": five_elements_class,
        "nominalAgeSystem": "korean_nominal_age",
        "isTrueSolarTimeApplied": False,
        "mingPalaceIndex": ming_index,
        "shenPalaceIndex": shen_index,
    }

    query_helpers = _build_query_helpers(palaces, indexes, relations, transformations)

    debug = {
        "rawInputEcho": input_raw,
        "normalizedInputEcho": normalized,
        "calculationPath": [
            "solar/lunar normalize",
            "time index(12 shichen)",
            "ming/shen palace (寅起正月, 命逆時/身順時)",
            "12-palace naming (命→兄→夫→...→父, reverse assignment)",
            "palace stem (五虎遁)",
            "five-elements bureau from ming gan-zhi (六十納音)",
            "ziwei & tianfu systems (14 major stars)",
            "year/month/time helper stars",
            "four transformations (祿權科忌)",
            "indexes/relations/horoscope build",
        ],
        "version": "ziwei-rule-engine-1.0.0",
        "warnings": warnings,
        "checksum": "",
        "referenceExamplesMatched": False,
    }

    meta = {
        "generated_at": now_kst.isoformat(timespec="seconds"),
        "tz": inp.tz_name,
        "engine": "local-rule-based",
        "calcVersion": "1.0.0",
        "schoolMode": "triad",
        "language": "ko-KR",
    }

    profile = {
        "wuxing_ju": chart_meta["fiveElementsClass"],
        "ming_palace_index": ming_index,
        "shen_palace_index": shen_index,
        "ming_master": chart_meta["soulMaster"],
        "shen_master": chart_meta["bodyMaster"],
        "fortune": {
            "natal": {"label": "본명반"},
            "decadal": {"label": "대운"},
            "yearly": {"label": "세운", "year": now_kst.year},
        },
    }

    payload: Dict[str, Any] = {
        "meta": meta,
        "input": input_raw,
        "normalized": normalized,
        "chartMeta": chart_meta,
        "palaces": palaces,
        "indexes": indexes,
        "relations": relations,
        "transformations": transformations,
        "horoscope": horoscope,
        "dynamicPalaces": dynamic_palaces,
        "queryHelpers": query_helpers,
        "debug": debug,
        # backward compatibility for existing UI paths
        "profile": profile,
        "four_transforms": compatibility_four_transforms,
    }

    checksum_source = (
        str(payload["input"])
        + str(payload["normalized"]["fourPillars"])
        + str(payload["chartMeta"])
        + str(payload["indexes"]["lifeBodyAxis"])
    )
    payload["debug"]["checksum"] = hashlib.sha256(checksum_source.encode("utf-8")).hexdigest()[:20]
    return payload


def _star_names_in_group(stars: Dict[str, Any], keys: Sequence[str]) -> List[str]:
    names: List[str] = []
    for key in keys:
        rows = stars.get(key, [])
        if not isinstance(rows, list):
            continue
        for row in rows:
            if isinstance(row, dict):
                name = row.get("labelZh") or row.get("labelKo") or row.get("starId")
                if name:
                    names.append(str(name))
            elif isinstance(row, str):
                names.append(row)
    return names


def _visible_stars(stars: Dict[str, Any], star_filter: StarFilter) -> List[str]:
    major = _star_names_in_group(stars, ("major", "majorStars"))
    minor = _star_names_in_group(stars, ("minor", "minorStars"))
    misc = _star_names_in_group(stars, ("misc", "adjectiveStars", "miscStars", "shenshaStars"))

    if star_filter == "major":
        return major
    if star_filter == "major_minor":
        return major + minor
    return major + minor + misc


def _transforms_by_palace(payload: Dict[str, Any], scope: FortuneScope) -> Dict[int, List[Dict[str, Any]]]:
    out: Dict[int, List[Dict[str, Any]]] = {i: [] for i in range(12)}

    legacy_rows = payload.get("four_transforms", {}).get(scope, [])
    if isinstance(legacy_rows, list) and legacy_rows:
        for row in legacy_rows:
            idx = int(row.get("palace_index", -1))
            if idx in out:
                out[idx].append(row)
        return out

    by_scope = payload.get("transformations", {}).get("byScope", {}).get(scope, {})
    rows = by_scope.get("byStar", [])
    palace_id_to_index = {p["palaceId"]: int(p["index"]) for p in payload.get("palaces", [])}
    for row in rows:
        idx = palace_id_to_index.get(row.get("palaceId"), -1)
        if idx in out:
            out[idx].append(
                {
                    "type": row.get("transformTypeKo", ""),
                    "type_hanja": row.get("transformTypeZh", ""),
                    "star": row.get("labelZh", ""),
                    "palace_index": idx,
                    "palace_name_ko": row.get("palaceNameKo", ""),
                    "source": {"natal": "생년", "decadal": "대운", "yearly": "세운"}.get(scope, ""),
                }
            )
    return out


def build_chart_view_model(
    payload: Dict[str, Any],
    view_mode: ViewMode,
    scope: FortuneScope,
    star_filter: StarFilter,
    show_hanja: bool,
    selected_palace_index: Optional[int],
) -> Dict[str, Any]:
    palaces = payload.get("palaces", [])
    if len(palaces) != 12:
        raise ValueError("유효하지 않은 자미두수 payload: palaces 길이는 12여야 합니다.")

    profile = payload.get("profile", {})
    ming_index = int(profile.get("ming_palace_index", payload.get("chartMeta", {}).get("mingPalaceIndex", 0)))
    shen_index = int(profile.get("shen_palace_index", payload.get("chartMeta", {}).get("shenPalaceIndex", 0)))
    selected = ming_index if selected_palace_index is None else selected_palace_index
    selected = max(0, min(11, int(selected)))

    transforms_map = _transforms_by_palace(payload, scope)
    cards: List[Dict[str, Any]] = []
    for p in palaces:
        idx = int(p["index"])
        stars_all = _visible_stars(p.get("stars", {}), star_filter)
        preview = stars_all[:4]
        extra_count = max(0, len(stars_all) - len(preview))
        transforms = transforms_map.get(idx, [])
        cards.append(
            {
                "index": idx,
                "grid_position": p["grid_position"],
                "name": p["labelZh"] if show_hanja else p["labelKo"],
                "name_ko": p["labelKo"],
                "name_hanja": p["labelZh"],
                "stem_branch": f"{p['heavenlyStem']}{p['earthlyBranch']}",
                "is_ming": bool(p.get("isLifePalace")),
                "is_shen": bool(p.get("isBodyPalace")),
                "is_selected": idx == selected,
                "stars_preview": preview,
                "stars_extra_count": extra_count,
                "transforms": transforms,
                "has_transform": bool(transforms),
                "raw": p,
            }
        )

    selected_row = palaces[selected]
    triad_ids = selected_row.get("threeHarmonyPalaceIds", [])
    palace_by_id = {p["palaceId"]: p for p in palaces}
    triad_names = [palace_by_id[x]["labelKo"] for x in triad_ids if x in palace_by_id]
    opposite_id = selected_row.get("oppositePalaceId")

    detail = {
        "index": selected,
        "name_ko": selected_row["labelKo"],
        "name_hanja": selected_row["labelZh"],
        "stem_branch": f"{selected_row['heavenlyStem']}{selected_row['earthlyBranch']}",
        "is_ming": bool(selected_row.get("isLifePalace")),
        "is_shen": bool(selected_row.get("isBodyPalace")),
        "stars_major": _star_names_in_group(selected_row.get("stars", {}), ("major", "majorStars")),
        "stars_minor": _star_names_in_group(selected_row.get("stars", {}), ("minor", "minorStars")),
        "stars_misc": _star_names_in_group(
            selected_row.get("stars", {}), ("misc", "adjectiveStars", "miscStars", "shenshaStars")
        ),
        "brightness": selected_row.get("brightness", {}),
        "triad_indices": [palace_by_id[x]["index"] for x in triad_ids if x in palace_by_id],
        "triad_names": triad_names,
        "opposite_index": palace_by_id[opposite_id]["index"] if opposite_id in palace_by_id else (selected + 6) % 12,
        "opposite_name": palace_by_id[opposite_id]["labelKo"] if opposite_id in palace_by_id else "",
        "transforms": transforms_map.get(selected, []),
        "scope": scope,
        "view_mode": view_mode,
    }

    chart_meta = payload.get("chartMeta", {})
    summary = {
        "view_mode": view_mode,
        "scope": scope,
        "wuxing_ju": chart_meta.get("fiveElementsClass", profile.get("wuxing_ju", "")),
        "ming_index": ming_index,
        "shen_index": shen_index,
        "ming_name": palaces[ming_index]["labelKo"],
        "shen_name": palaces[shen_index]["labelKo"],
        "ming_master": chart_meta.get("soulMaster", profile.get("ming_master", "")),
        "shen_master": chart_meta.get("bodyMaster", profile.get("shen_master", "")),
        "fortune_label": profile.get("fortune", {}).get(scope, {}).get("label", ""),
    }

    return {"summary": summary, "cards": cards, "selected_detail": detail}
