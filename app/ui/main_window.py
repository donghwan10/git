from __future__ import annotations

from datetime import datetime
from pathlib import Path
import queue
import re
import sys
import threading
from typing import Any, Dict, List, Optional

import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText

import cast_service
import history_store
from models import RunResult, UIState

try:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from saju import SajuPanel
    SAJU_IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:
    SajuPanel = None  # type: ignore[assignment]
    SAJU_IMPORT_ERROR = exc

try:
    from ui.ziwei_panel import ZiweiPanel
    ZIWEI_IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:
    ZiweiPanel = None  # type: ignore[assignment]
    ZIWEI_IMPORT_ERROR = exc


def is_yang(raw: int) -> bool:
    return raw in (7, 9)


_YARROW_TRACE_LINE_RE = re.compile(
    r"^Line\s+(?P<line_no>\d+):\s+afters=(?P<afters>[0-9 ]+)\s+->\s+line=(?P<line_value>\d+)\s+steps=(?P<steps>.+)$",
    re.MULTILINE,
)
_YARROW_STEP_RE = re.compile(
    r"^(?P<start>\d+)→split\((?P<split_left>\d+),(?P<split_right>\d+)\)"
    r"→rem\((?P<left_rem>\d+),(?P<right_rem>\d+)\)"
    r"→-(?P<removed_total>\d+)=(?P<after>\d+)$"
)


def parse_yarrow_trace_rows(output_text: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for match in _YARROW_TRACE_LINE_RE.finditer(output_text or ""):
        try:
            line_no = int(match.group("line_no"))
            line_value = int(match.group("line_value"))
        except ValueError:
            continue

        afters = [int(tok) for tok in match.group("afters").split() if tok.isdigit()]
        step_chunks = [chunk.strip() for chunk in match.group("steps").split("|")]
        steps: List[Dict[str, Any]] = []
        for idx, chunk in enumerate(step_chunks, start=1):
            sm = _YARROW_STEP_RE.match(chunk)
            if not sm:
                steps.append({"change_no": idx, "raw": chunk, "parse_error": True})
                continue
            step = {k: int(v) for k, v in sm.groupdict().items()}
            step["change_no"] = idx
            steps.append(step)

        rows.append(
            {
                "line_no": line_no,
                "line_value": line_value,
                "afters": afters,
                "steps": steps,
                "raw_line": match.group(0),
            }
        )
    return rows


def build_yarrow_trace_display_text(trace_rows: List[Dict[str, Any]]) -> str:
    if not trace_rows:
        return "시초 추적 정보가 없습니다.\n(구형 데이터이거나 로그에 Yarrow traces 구간이 없음)\n"

    lines: List[str] = ["[시초 진행 요약] 아래(1효) → 위(6효)", "-" * 64]
    for row in sorted(trace_rows, key=lambda x: x["line_no"]):
        afters = " / ".join(str(x) for x in row.get("afters", []))
        lines.append(f"{row['line_no']}효 확정값={row['line_value']} | 3변 후 잔량={afters}")
        for step in row.get("steps", []):
            if step.get("parse_error"):
                lines.append(f"  - {step.get('change_no', '?')}변: {step.get('raw', '')}")
                continue
            lines.append(
                "  - {change_no}변: {start}개 → 분할({split_left},{split_right}) → "
                "나머지({left_rem},{right_rem}) → 제거 {removed_total}개 → 남음 {after}개".format(**step)
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def draw_hexagram(
    canvas: tk.Canvas,
    lines6: List[int],
    changing_idx: Optional[List[int]],
    x: int,
    y: int,
    title: str,
    *,
    line_w: int = 220,
    line_h: int = 10,
    gap_y: int = 18,
    break_gap: int = 46,
) -> None:
    canvas.create_text(x, y - 24, text=title, anchor="w", font=("Malgun Gothic", 10, "bold"))

    for top_i in range(6):
        line_no = 6 - top_i
        raw = lines6[line_no - 1]
        yy = y + top_i * gap_y

        if is_yang(raw):
            canvas.create_line(x, yy, x + line_w, yy, width=line_h, capstyle=tk.PROJECTING)
        else:
            mid = x + line_w // 2
            canvas.create_line(x, yy, mid - break_gap // 2, yy, width=line_h, capstyle=tk.PROJECTING)
            canvas.create_line(mid + break_gap // 2, yy, x + line_w, yy, width=line_h, capstyle=tk.PROJECTING)

        if changing_idx and line_no in changing_idx:
            canvas.create_text(x + line_w + 10, yy, text="動", anchor="w", font=("Malgun Gothic", 10, "bold"))


def render_two_hexagrams(canvas: tk.Canvas, rr: RunResult) -> None:
    canvas.delete("all")
    if len(rr.lines) != 6 or len(rr.changed_lines) != 6:
        canvas.create_text(
            20,
            20,
            anchor="nw",
            text="구형 데이터는 괘 라인 정보가 없어 그래픽을 표시할 수 없습니다.",
            font=("Malgun Gothic", 10),
            fill="#555555",
        )
        return

    left_x = 20
    top_y = 60
    right_x = 360
    base_title = f"본괘  #{rr.base_no} {rr.base_name} ({rr.base_upper_symbol} over {rr.base_lower_symbol})"
    chg_title = f"변괘  #{rr.changed_no} {rr.changed_name} ({rr.changed_upper_symbol} over {rr.changed_lower_symbol})"
    draw_hexagram(canvas, rr.lines, rr.changing_idx, left_x, top_y, base_title)
    draw_hexagram(canvas, rr.changed_lines, [], right_x, top_y, chg_title)


class MainWindow:
    def __init__(self) -> None:
        history_store.configure_runtime()
        self.migration = history_store.migrate_legacy_history_once()
        self.ui_state = history_store.load_ui_state()
        self.history: List[RunResult] = history_store.load_history()
        self.filtered_indices: List[int] = []

        self.root = tk.Tk()
        self.root.title("주역 시초점 대시보드 (文王卦 + 주역 텍스트)")
        self.root.geometry("1500x930")

        self.status_var = tk.StringVar(value=f"대기 중 | 저장: {history_store.HISTORY_FILE}")
        self.log_error_var = tk.StringVar(value="")
        self.summary_card_var = tk.StringVar(value="아직 실행 결과가 없습니다.")

        self.search_var = tk.StringVar(value=self.ui_state.history_filter_text)
        self.moving_only_var = tk.BooleanVar(value=self.ui_state.moving_only)
        self.base_filter_var = tk.StringVar(value="")
        self.changed_filter_var = tk.StringVar(value="")

        self.delay_var = tk.DoubleVar(value=0.0)
        self.banner_var = tk.BooleanVar(value=False)
        self.comments_var = tk.BooleanVar(value=False)
        self.summary_var = tk.BooleanVar(value=self.ui_state.summary_mode)
        self.current_view = ""
        self._cast_running = False
        self._cast_queue: Optional[queue.Queue] = None
        self._cast_thread: Optional[threading.Thread] = None

        self._build_layout()
        self._bind_events()
        self._refresh_history_list(select_last=True)
        self._restore_initial_view()

        if self.migration.message:
            self.status_var.set(f"{self.migration.message} | 저장: {history_store.HISTORY_FILE}")

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_layout(self) -> None:
        status_lbl = tk.Label(self.root, textvariable=self.status_var, anchor="w")
        status_lbl.pack(fill="x", padx=10, pady=(8, 6))

        nav = tk.Frame(self.root)
        nav.pack(fill="x", padx=10, pady=(0, 6))
        self.btn_show_yukhyo = tk.Button(nav, text="육효점", width=12, command=lambda: self._show_view("yukhyo"))
        self.btn_show_yukhyo.pack(side="left", padx=(0, 6))
        self.btn_show_saju = tk.Button(nav, text="사주", width=12, command=lambda: self._show_view("saju"))
        self.btn_show_saju.pack(side="left", padx=(0, 6))
        self.btn_show_ziwei = tk.Button(nav, text="자미두수", width=12, command=lambda: self._show_view("ziwei"))
        self.btn_show_ziwei.pack(side="left")

        self.view_host = tk.Frame(self.root)
        self.view_host.pack(fill="both", expand=True)

        self.yukhyo_view = tk.Frame(self.view_host)
        self.saju_view = tk.Frame(self.view_host)
        self.ziwei_view = tk.Frame(self.view_host)

        self._build_yukhyo_view(self.yukhyo_view)
        self._build_saju_view(self.saju_view)
        self._build_ziwei_view(self.ziwei_view)
        self._show_view("yukhyo")

    def _build_yukhyo_view(self, parent: tk.Widget) -> None:
        main = tk.Frame(parent)
        main.pack(fill="both", expand=True)

        left = tk.LabelFrame(main, text="실행 히스토리", padx=8, pady=8)
        left.pack(side="left", fill="y", padx=(10, 6), pady=10)

        right = tk.Frame(main)
        right.pack(side="right", fill="both", expand=True, padx=(6, 10), pady=10)

        self._build_history_panel(left)
        self._build_right_panel(right)

    def _build_saju_view(self, parent: tk.Widget) -> None:
        if SajuPanel is None:
            msg = tk.Label(
                parent,
                text=(
                    "사주 UI를 로드하지 못했습니다.\n"
                    "원인: " + repr(SAJU_IMPORT_ERROR)
                ),
                justify="left",
                anchor="nw",
                fg="red",
            )
            msg.pack(fill="both", expand=True, padx=12, pady=12)
            return

        panel = SajuPanel(parent)
        panel.pack(fill="both", expand=True)

    def _build_ziwei_view(self, parent: tk.Widget) -> None:
        if ZiweiPanel is None:
            msg = tk.Label(
                parent,
                text=(
                    "자미두수 UI를 로드하지 못했습니다.\n"
                    "원인: " + repr(ZIWEI_IMPORT_ERROR)
                ),
                justify="left",
                anchor="nw",
                fg="red",
            )
            msg.pack(fill="both", expand=True, padx=12, pady=12)
            return

        panel = ZiweiPanel(parent)
        panel.pack(fill="both", expand=True)

    def _show_view(self, view: str) -> None:
        if view == self.current_view:
            return

        self.yukhyo_view.pack_forget()
        self.saju_view.pack_forget()
        self.ziwei_view.pack_forget()

        if view == "saju":
            self.saju_view.pack(fill="both", expand=True)
            self.btn_show_saju.config(relief=tk.SUNKEN)
            self.btn_show_yukhyo.config(relief=tk.RAISED)
            self.btn_show_ziwei.config(relief=tk.RAISED)
        elif view == "ziwei":
            self.ziwei_view.pack(fill="both", expand=True)
            self.btn_show_ziwei.config(relief=tk.SUNKEN)
            self.btn_show_yukhyo.config(relief=tk.RAISED)
            self.btn_show_saju.config(relief=tk.RAISED)
        else:
            self.yukhyo_view.pack(fill="both", expand=True)
            self.btn_show_yukhyo.config(relief=tk.SUNKEN)
            self.btn_show_saju.config(relief=tk.RAISED)
            self.btn_show_ziwei.config(relief=tk.RAISED)
            view = "yukhyo"

        self.current_view = view

    def _build_history_panel(self, parent: tk.Widget) -> None:
        tk.Label(parent, text="검색").pack(anchor="w")
        self.search_entry = tk.Entry(parent, textvariable=self.search_var, width=34)
        self.search_entry.pack(fill="x", pady=(2, 6))

        fline = tk.Frame(parent)
        fline.pack(fill="x", pady=(0, 6))
        tk.Checkbutton(fline, text="동효 있음만", variable=self.moving_only_var, command=self._on_filter_changed).pack(
            side="left"
        )

        num_filter = tk.Frame(parent)
        num_filter.pack(fill="x", pady=(0, 6))
        tk.Label(num_filter, text="본괘#").pack(side="left")
        self.base_filter_entry = tk.Entry(num_filter, textvariable=self.base_filter_var, width=7)
        self.base_filter_entry.pack(side="left", padx=(4, 10))
        tk.Label(num_filter, text="변괘#").pack(side="left")
        self.changed_filter_entry = tk.Entry(num_filter, textvariable=self.changed_filter_var, width=7)
        self.changed_filter_entry.pack(side="left", padx=(4, 0))

        self.hist_list = tk.Listbox(parent, height=32, exportselection=False, width=48)
        self.hist_list.pack(fill="y", expand=False)

        self.hist_meta = tk.Label(parent, text="", justify="left", anchor="w", wraplength=360)
        self.hist_meta.pack(fill="x", pady=(8, 0))

    def _build_right_panel(self, parent: tk.Widget) -> None:
        controls = tk.Frame(parent)
        controls.pack(fill="x", pady=(0, 8))

        self.run_btn = tk.Button(controls, text="점치기 실행(저장)", width=18, command=self.run_new_cast)
        self.run_btn.pack(side="left", padx=4)
        tk.Button(controls, text="전체 복사", width=12, command=self.copy_all).pack(side="left", padx=4)
        tk.Button(controls, text="선택 복사", width=12, command=self.copy_selected).pack(side="left", padx=4)
        tk.Button(controls, text="TXT 저장", width=12, command=self.save_txt).pack(side="left", padx=4)
        tk.Button(controls, text="히스토리 삭제", width=12, command=self.clear_history_ui).pack(side="left", padx=4)

        opts = tk.Frame(controls)
        opts.pack(side="right")
        tk.Label(opts, text="delay").pack(side="left", padx=(0, 4))
        tk.Entry(opts, textvariable=self.delay_var, width=6).pack(side="left")
        tk.Checkbutton(opts, text="배너", variable=self.banner_var).pack(side="left", padx=6)
        tk.Checkbutton(opts, text="주석 포함", variable=self.comments_var).pack(side="left", padx=6)
        tk.Checkbutton(opts, text="요약 모드", variable=self.summary_var, command=self._on_summary_toggle).pack(
            side="left", padx=6
        )

        if not cast_service.load_dataset():
            tk.Label(opts, text="⚠ 주역 텍스트 데이터 없음", fg="red").pack(side="left", padx=8)

        middle = tk.Frame(parent)
        middle.pack(fill="x", pady=(0, 8))

        self.canvas = tk.Canvas(middle, height=220)
        self.canvas.pack(side="left", fill="x", expand=True)

        summary_box = tk.LabelFrame(middle, text="핵심 요약", padx=8, pady=8)
        summary_box.pack(side="right", fill="y", padx=(8, 0))
        tk.Label(summary_box, textvariable=self.summary_card_var, justify="left", anchor="nw", width=42).pack(
            fill="both", expand=True
        )

        self.notebook = ttk.Notebook(parent)
        self.notebook.pack(fill="both", expand=True)

        self.tab_table = tk.Frame(self.notebook)
        self.tab_text = tk.Frame(self.notebook)
        self.tab_iching = tk.Frame(self.notebook)
        self.tab_yarrow_trace = tk.Frame(self.notebook)
        self.notebook.add(self.tab_table, text="文王卦 표")
        self.notebook.add(self.tab_text, text="텍스트 로그")
        self.notebook.add(self.tab_iching, text="주역 텍스트")
        self.notebook.add(self.tab_yarrow_trace, text="시초 과정")

        self._build_wenwang_table(self.tab_table)
        self._build_log_tab(self.tab_text)
        self._build_iching_tab(self.tab_iching)
        self._build_yarrow_trace_tab(self.tab_yarrow_trace)

    def _build_wenwang_table(self, parent: tk.Widget) -> None:
        columns = ("line", "shi_ying", "raw", "moving", "ganzhi", "wx", "six_rel", "six_spirit", "xunkong")
        self.tree = ttk.Treeview(parent, columns=columns, show="headings", height=16)
        self.tree.heading("line", text="효(상->하)")
        self.tree.heading("shi_ying", text="世/應")
        self.tree.heading("raw", text="원효")
        self.tree.heading("moving", text="동")
        self.tree.heading("ganzhi", text="納甲(干支)")
        self.tree.heading("wx", text="지지오행")
        self.tree.heading("six_rel", text="六親")
        self.tree.heading("six_spirit", text="六神")
        self.tree.heading("xunkong", text="旬空")

        self.tree.column("line", width=80, anchor="center")
        self.tree.column("shi_ying", width=70, anchor="center")
        self.tree.column("raw", width=60, anchor="center")
        self.tree.column("moving", width=50, anchor="center")
        self.tree.column("ganzhi", width=110, anchor="center")
        self.tree.column("wx", width=90, anchor="center")
        self.tree.column("six_rel", width=70, anchor="center")
        self.tree.column("six_spirit", width=70, anchor="center")
        self.tree.column("xunkong", width=60, anchor="center")
        self.tree.pack(fill="both", expand=True, padx=8, pady=8)

        scroll_y = ttk.Scrollbar(parent, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll_y.set)
        scroll_y.place(relx=1.0, rely=0.0, relheight=1.0, anchor="ne")

        self.tree.tag_configure("moving", font=("Malgun Gothic", 10, "bold"))
        self.tree.tag_configure("shi", font=("Malgun Gothic", 10, "bold"))
        self.tree.tag_configure("xunkong", font=("Malgun Gothic", 10, "bold"))

    def _build_log_tab(self, parent: tk.Widget) -> None:
        tk.Label(parent, textvariable=self.log_error_var, anchor="w", fg="red").pack(fill="x", padx=8, pady=(8, 0))
        self.output = ScrolledText(parent, width=150, height=33, undo=True)
        self.output.pack(fill="both", expand=True, padx=8, pady=8)

    def _build_iching_tab(self, parent: tk.Widget) -> None:
        self.iching_output = ScrolledText(parent, width=150, height=35, undo=True)
        self.iching_output.pack(fill="both", expand=True, padx=8, pady=8)

    def _build_yarrow_trace_tab(self, parent: tk.Widget) -> None:
        header = tk.Frame(parent)
        header.pack(fill="x", padx=8, pady=(8, 4))

        self.yarrow_trace_meta_var = tk.StringVar(value="시초 추적 정보가 없습니다.")
        tk.Label(header, textvariable=self.yarrow_trace_meta_var, anchor="w", justify="left").pack(fill="x")

        table_wrap = tk.Frame(parent)
        table_wrap.pack(fill="x", padx=8, pady=(0, 6))

        columns = (
            "line_no",
            "change_no",
            "start",
            "split_left",
            "split_right",
            "left_rem",
            "right_rem",
            "removed_total",
            "after",
            "line_value",
        )
        self.yarrow_trace_tree = ttk.Treeview(table_wrap, columns=columns, show="headings", height=12)
        self.yarrow_trace_tree.heading("line_no", text="효(아래→위)")
        self.yarrow_trace_tree.heading("change_no", text="변")
        self.yarrow_trace_tree.heading("start", text="시작")
        self.yarrow_trace_tree.heading("split_left", text="좌")
        self.yarrow_trace_tree.heading("split_right", text="우")
        self.yarrow_trace_tree.heading("left_rem", text="좌余4")
        self.yarrow_trace_tree.heading("right_rem", text="우余4")
        self.yarrow_trace_tree.heading("removed_total", text="제거")
        self.yarrow_trace_tree.heading("after", text="남음")
        self.yarrow_trace_tree.heading("line_value", text="확정효")

        self.yarrow_trace_tree.column("line_no", width=95, anchor="center")
        self.yarrow_trace_tree.column("change_no", width=55, anchor="center")
        self.yarrow_trace_tree.column("start", width=70, anchor="center")
        self.yarrow_trace_tree.column("split_left", width=70, anchor="center")
        self.yarrow_trace_tree.column("split_right", width=70, anchor="center")
        self.yarrow_trace_tree.column("left_rem", width=70, anchor="center")
        self.yarrow_trace_tree.column("right_rem", width=70, anchor="center")
        self.yarrow_trace_tree.column("removed_total", width=70, anchor="center")
        self.yarrow_trace_tree.column("after", width=70, anchor="center")
        self.yarrow_trace_tree.column("line_value", width=80, anchor="center")
        self.yarrow_trace_tree.pack(side="left", fill="x", expand=True)

        yscroll = ttk.Scrollbar(table_wrap, orient="vertical", command=self.yarrow_trace_tree.yview)
        self.yarrow_trace_tree.configure(yscrollcommand=yscroll.set)
        yscroll.pack(side="right", fill="y")

        self.yarrow_trace_tree.tag_configure("line_end", font=("Malgun Gothic", 9, "bold"))
        self.yarrow_trace_tree.tag_configure("moving_line", background="#fff5d6")

        self.yarrow_trace_output = ScrolledText(parent, width=150, height=16, undo=True)
        self.yarrow_trace_output.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self.yarrow_trace_output.insert(
            "end",
            "점치기 실행 후 또는 히스토리 선택 시 시초 진행 과정이 표시됩니다.\n",
        )

    def _bind_events(self) -> None:
        self.hist_list.bind("<<ListboxSelect>>", self.load_selected_history)
        self.search_entry.bind("<KeyRelease>", self._on_filter_changed)
        self.base_filter_entry.bind("<KeyRelease>", self._on_filter_changed)
        self.changed_filter_entry.bind("<KeyRelease>", self._on_filter_changed)

        self.root.bind_all("<Control-Return>", self._shortcut_run_cast)
        self.root.bind_all("<Control-s>", self._shortcut_save_txt)
        self.root.bind_all("<Control-Shift-c>", self._shortcut_copy_all)
        self.root.bind_all("<Control-Shift-C>", self._shortcut_copy_all)
        self.root.bind_all("<Control-f>", self._shortcut_focus_search)
        self.root.bind_all("<Alt-Up>", self._shortcut_prev_history)
        self.root.bind_all("<Alt-Down>", self._shortcut_next_history)

    def _restore_initial_view(self) -> None:
        if self.history:
            preferred = self._safe_tab_index(self.ui_state.selected_tab)
            self.notebook.select(preferred)
            return

        self.output.insert(
            "end",
            "👉 [점치기 실행(저장)]을 누르면 결과가 생성되고 히스토리에 저장됩니다.\n"
            "👉 좌측 히스토리에서 검색/필터 후 선택하면 결과가 복원됩니다.\n"
            f"👉 저장 파일: {history_store.HISTORY_FILE}\n",
        )

    def _safe_tab_index(self, idx: int) -> int:
        try:
            n = self.notebook.index("end")
            if 0 <= idx < n:
                return idx
        except Exception:
            pass
        return 0

    def _build_history_line(self, rr: RunResult) -> str:
        ts = rr.ts_iso.replace("T", " ") if rr.ts_iso else "(시간없음)"
        base = f"#{rr.base_no} {rr.base_name or '?'}"
        changed = f"#{rr.changed_no} {rr.changed_name or '?'}"
        moving = f"동효:{len(rr.changing_idx)}" if rr.changing_idx else "동효:0"
        return f"{ts} | {base} → {changed} | {moving}"

    def _selected_history_index(self) -> Optional[int]:
        sel = self.hist_list.curselection()
        if not sel:
            return None
        pos = sel[0]
        if pos < 0 or pos >= len(self.filtered_indices):
            return None
        return self.filtered_indices[pos]

    def _refresh_history_list(
        self,
        *,
        select_last: bool = False,
        preferred_history_index: Optional[int] = None,
    ) -> None:
        query = self.search_var.get().strip()
        base_no = history_store.parse_int_filter(self.base_filter_var.get())
        changed_no = history_store.parse_int_filter(self.changed_filter_var.get())

        filtered = history_store.query_history(
            self.history,
            text_query=query,
            moving_only=bool(self.moving_only_var.get()),
            base_no=base_no,
            changed_no=changed_no,
        )
        allowed_ids = {id(x) for x in filtered}
        self.filtered_indices = [idx for idx, rr in enumerate(self.history) if id(rr) in allowed_ids]

        self.hist_list.delete(0, "end")
        for idx in self.filtered_indices:
            self.hist_list.insert("end", self._build_history_line(self.history[idx]))

        if not self.filtered_indices:
            self.hist_meta.config(
                text=(
                    f"필터 결과 없음\n"
                    f"총 기록: {len(self.history)}건\n"
                    f"저장파일:\n{history_store.HISTORY_FILE}"
                )
            )
            self.canvas.delete("all")
            self._fill_wenwang_table([])
            self._clear_yarrow_trace_view("선택된 기록이 없습니다.")
            self.summary_card_var.set("선택된 기록이 없습니다.")
            return

        target_pos = None
        if preferred_history_index is not None and preferred_history_index in self.filtered_indices:
            target_pos = self.filtered_indices.index(preferred_history_index)
        elif select_last:
            target_pos = len(self.filtered_indices) - 1

        if target_pos is not None:
            self.hist_list.selection_clear(0, "end")
            self.hist_list.selection_set(target_pos)
            self.hist_list.activate(target_pos)
            self.load_selected_history()

    def _fill_wenwang_table(self, ww_rows: List[Dict[str, Any]]) -> None:
        for item in self.tree.get_children():
            self.tree.delete(item)
        for r in sorted(ww_rows, key=lambda x: x["line"], reverse=True):
            tags = []
            if r.get("moving"):
                tags.append("moving")
            if r.get("shi_ying") == "世":
                tags.append("shi")
            if r.get("is_xunkong") == "空":
                tags.append("xunkong")
            self.tree.insert(
                "",
                "end",
                values=(
                    r.get("line", ""),
                    r.get("shi_ying", ""),
                    r.get("raw", ""),
                    "동" if r.get("moving") else "",
                    r.get("ganzhi", ""),
                    r.get("wx", ""),
                    r.get("six_rel", ""),
                    r.get("six_spirit", ""),
                    r.get("is_xunkong", ""),
                ),
                tags=tuple(tags),
            )

    def _clear_yarrow_trace_view(self, message: str = "시초 추적 정보가 없습니다.") -> None:
        for item in self.yarrow_trace_tree.get_children():
            self.yarrow_trace_tree.delete(item)
        self.yarrow_trace_meta_var.set(message)
        self.yarrow_trace_output.delete("1.0", "end")
        self.yarrow_trace_output.insert("end", f"{message}\n")

    def _fill_yarrow_trace_view(self, rr: RunResult) -> None:
        trace_rows = parse_yarrow_trace_rows(rr.output_text)

        for item in self.yarrow_trace_tree.get_children():
            self.yarrow_trace_tree.delete(item)

        if not trace_rows:
            self.yarrow_trace_meta_var.set("시초 추적 정보가 없습니다. (구형 기록 또는 로그 형식 불일치)")
            self.yarrow_trace_output.delete("1.0", "end")
            self.yarrow_trace_output.insert("end", build_yarrow_trace_display_text(trace_rows))
            self.yarrow_trace_output.see("1.0")
            return

        moving_set = set(rr.changing_idx or [])
        total_steps = 0
        sorted_rows = sorted(trace_rows, key=lambda x: x["line_no"])
        for row in sorted_rows:
            line_no = int(row["line_no"])
            line_value = int(row["line_value"])
            for step in row.get("steps", []):
                total_steps += 1
                tags: List[str] = []
                if line_no in moving_set:
                    tags.append("moving_line")
                if step.get("change_no") == 3:
                    tags.append("line_end")

                if step.get("parse_error"):
                    self.yarrow_trace_tree.insert(
                        "",
                        "end",
                        values=(
                            line_no,
                            step.get("change_no", ""),
                            "",
                            "",
                            "",
                            "",
                            "",
                            "",
                            "",
                            line_value if step.get("change_no") == 3 else "",
                        ),
                        tags=tuple(tags),
                    )
                    continue

                self.yarrow_trace_tree.insert(
                    "",
                    "end",
                    values=(
                        line_no,
                        step.get("change_no", ""),
                        step.get("start", ""),
                        step.get("split_left", ""),
                        step.get("split_right", ""),
                        step.get("left_rem", ""),
                        step.get("right_rem", ""),
                        step.get("removed_total", ""),
                        step.get("after", ""),
                        line_value if step.get("change_no") == 3 else "",
                    ),
                    tags=tuple(tags),
                )

        self.yarrow_trace_meta_var.set(
            f"시초 과정 {len(sorted_rows)}효 / 총 {total_steps}변 | 동효: {rr.changing_idx if rr.changing_idx else 'None'}"
        )

        self.yarrow_trace_output.delete("1.0", "end")
        self.yarrow_trace_output.insert("end", build_yarrow_trace_display_text(sorted_rows))
        self.yarrow_trace_output.see("1.0")

    def _build_summary_card(self, rr: RunResult) -> str:
        pillars = cast_service.extract_pillars_line(rr.output_text)
        lines = [
            f"시간: {rr.ts_iso.replace('T', ' ') if rr.ts_iso else '(알 수 없음)'}",
            f"본괘: #{rr.base_no} {rr.base_name} ({rr.base_upper_symbol} over {rr.base_lower_symbol})",
            f"변괘: #{rr.changed_no} {rr.changed_name} ({rr.changed_upper_symbol} over {rr.changed_lower_symbol})",
            f"동효: {rr.changing_idx if rr.changing_idx else 'None'}",
            f"사주: {pillars if pillars else '(로그에서 추출 불가)'}",
        ]
        return "\n".join(lines)

    def _set_output(self, rr: RunResult) -> None:
        render_two_hexagrams(self.canvas, rr)
        self._fill_wenwang_table(rr.ww_rows)
        self._fill_yarrow_trace_view(rr)

        log_text = cast_service.format_log_for_display(rr, summary_mode=bool(self.summary_var.get()))
        self.output.delete("1.0", "end")
        self.output.insert("end", log_text)
        self.output.see("1.0")

        self.iching_output.delete("1.0", "end")
        self.iching_output.insert("end", cast_service.build_iching_block(rr, include_comments=bool(self.comments_var.get())))
        self.iching_output.see("1.0")

        self.summary_card_var.set(self._build_summary_card(rr))
        self.hist_meta.config(
            text=(
                f"Timestamp: {rr.ts_iso}\n"
                f"본괘: #{rr.base_no} {rr.base_name}\n"
                f"변괘: #{rr.changed_no} {rr.changed_name}\n"
                f"동효: {rr.changing_idx if rr.changing_idx else 'None'}\n\n"
                f"저장파일:\n{history_store.HISTORY_FILE}"
            )
        )

    def load_selected_history(self, _event: Any = None) -> None:
        idx = self._selected_history_index()
        if idx is None:
            return
        rr = self.history[idx]
        self.log_error_var.set("")
        self._set_output(rr)
        self.status_var.set(f"히스토리 로드됨 ✅ ({rr.ts_iso or 'legacy'})")

    def _active_text_widget(self) -> ScrolledText:
        idx = self.notebook.index(self.notebook.select())
        if idx == 1:
            return self.output
        if idx == 2:
            return self.iching_output
        if idx == 3:
            return self.yarrow_trace_output
        return self.output

    def copy_all(self) -> None:
        w = self._active_text_widget()
        data = w.get("1.0", "end-1c")
        if not data.strip():
            self.status_var.set("복사할 내용이 없음")
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(data)
        self.root.update()
        self.status_var.set("전체 복사됨 ✅")

    def copy_selected(self) -> None:
        w = self._active_text_widget()
        try:
            data = w.get("sel.first", "sel.last")
        except tk.TclError:
            self.status_var.set("선택 영역 없음 (드래그 후 클릭)")
            return
        if not data.strip():
            self.status_var.set("선택 영역이 비어있음")
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(data)
        self.root.update()
        self.status_var.set("선택 복사됨 ✅")

    def save_txt(self) -> None:
        idx = self._selected_history_index()
        if idx is None:
            self.status_var.set("저장할 히스토리를 선택하세요")
            return
        rr = self.history[idx]
        safe_ts = (rr.ts_iso or datetime.now().isoformat(timespec="seconds")).replace(":", "-")
        out_path = history_store.EXPORT_DIR / f"wenwang_{safe_ts}.txt"

        content = rr.output_text
        content += "\n\n" + "=" * 60 + "\n"
        content += "I CHING TEXT (괘사/대상/효사)\n"
        content += "=" * 60 + "\n\n"
        content += cast_service.build_iching_block(rr, include_comments=bool(self.comments_var.get()))
        out_path.write_text(content, encoding="utf-8")
        self.status_var.set(f"TXT 저장됨 ✅ ({out_path})")

    def clear_history_ui(self) -> None:
        history_store.clear_history()
        self.history.clear()
        self.filtered_indices.clear()
        self.hist_list.delete(0, "end")
        self.output.delete("1.0", "end")
        self.iching_output.delete("1.0", "end")
        self.canvas.delete("all")
        self._fill_wenwang_table([])
        self._clear_yarrow_trace_view("기록이 삭제되었습니다.")
        self.summary_card_var.set("기록이 삭제되었습니다.")
        self.hist_meta.config(text="")
        self.status_var.set("히스토리 삭제됨 ✅")

    def _prepare_live_yarrow_trace_ui(self) -> None:
        self.notebook.select(3)
        for item in self.yarrow_trace_tree.get_children():
            self.yarrow_trace_tree.delete(item)
        self.yarrow_trace_meta_var.set("시초 과정 실시간 진행 중...")
        self.yarrow_trace_output.delete("1.0", "end")
        self.yarrow_trace_output.insert(
            "end",
            f"[실시간 시초 진행 시작] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
        )
        if float(self.delay_var.get()) <= 0:
            self.yarrow_trace_output.insert("end", "참고: delay=0 이면 진행 메시지가 매우 빠르게 지나갑니다.\n")
        self.yarrow_trace_output.see("end")

    def _append_live_yarrow_message(self, msg: str) -> None:
        text = (msg or "").rstrip("\n")
        if not text:
            return
        self.yarrow_trace_output.insert("end", text + "\n")
        self.yarrow_trace_output.see("end")
        self.root.update_idletasks()

    def _run_cast_worker(self, delay_sec: float, show_banners: bool) -> None:
        q = self._cast_queue
        if q is None:
            return

        def emit_progress(msg: str) -> None:
            q.put(("progress", msg))

        try:
            rr = cast_service.run_cast_once(
                delay_sec=delay_sec,
                show_banners=show_banners,
                progress_callback=emit_progress,
            )
            q.put(("done", rr))
        except Exception as exc:
            q.put(("error", f"{exc}"))

    def _finish_cast_success(self, rr: RunResult) -> None:
        self._cast_running = False
        self._cast_thread = None
        self._cast_queue = None
        self.run_btn.config(state="normal")

        self.history.append(rr)
        history_store.append_history(rr)
        self._refresh_history_list(select_last=True, preferred_history_index=len(self.history) - 1)
        self.status_var.set(f"완료 ✅ (저장됨: {history_store.HISTORY_FILE})")

    def _finish_cast_error(self, error_message: str) -> None:
        self._cast_running = False
        self._cast_thread = None
        self._cast_queue = None
        self.run_btn.config(state="normal")
        self.log_error_var.set(f"실행 오류: {error_message}")
        self.status_var.set(f"실행 오류: {error_message}")
        self.output.delete("1.0", "end")
        self.output.insert("end", f"[오류]\n{error_message}\n")
        self._append_live_yarrow_message(f"[오류] {error_message}")
        self.yarrow_trace_meta_var.set("시초 과정 실행 중 오류")

    def _poll_cast_queue(self) -> None:
        q = self._cast_queue
        if q is None:
            return

        while True:
            try:
                item = q.get_nowait()
            except queue.Empty:
                break

            kind = item[0]
            if kind == "progress":
                self._append_live_yarrow_message(str(item[1]))
                continue
            if kind == "done":
                self._append_live_yarrow_message("[완료] 결과 정리 및 저장 반영 중...")
                self._finish_cast_success(item[1])
                return
            if kind == "error":
                self._finish_cast_error(str(item[1]))
                return

        if self._cast_running:
            self.root.after(50, self._poll_cast_queue)

    def run_new_cast(self) -> None:
        if self._cast_running:
            self.status_var.set("이미 실행 중입니다. 완료를 기다리세요.")
            return

        try:
            delay_sec = float(self.delay_var.get())
        except Exception:
            self.log_error_var.set("실행 오류: delay 값은 숫자여야 합니다.")
            self.status_var.set("실행 오류: delay 값은 숫자여야 합니다.")
            return

        self._cast_running = True
        self.run_btn.config(state="disabled")
        self.status_var.set("실행 중... (시초 과정을 UI에 실시간 표시)")
        self.log_error_var.set("")
        self._prepare_live_yarrow_trace_ui()
        self.root.update_idletasks()

        self._cast_queue = queue.Queue()
        self._cast_thread = threading.Thread(
            target=self._run_cast_worker,
            args=(delay_sec, bool(self.banner_var.get())),
            daemon=True,
        )
        self._cast_thread.start()
        self.root.after(50, self._poll_cast_queue)

    def _on_filter_changed(self, _event: Any = None) -> None:
        current = self._selected_history_index()
        self._refresh_history_list(preferred_history_index=current)

    def _on_summary_toggle(self) -> None:
        idx = self._selected_history_index()
        if idx is not None:
            self._set_output(self.history[idx])

    def _move_history_selection(self, step: int) -> None:
        n = len(self.filtered_indices)
        if n == 0:
            return
        sel = self.hist_list.curselection()
        if not sel:
            pos = 0 if step > 0 else n - 1
        else:
            pos = max(0, min(n - 1, sel[0] + step))
        self.hist_list.selection_clear(0, "end")
        self.hist_list.selection_set(pos)
        self.hist_list.activate(pos)
        self.load_selected_history()

    def _shortcut_run_cast(self, _event: Any = None) -> str:
        if self.current_view != "yukhyo":
            return "break"
        self.run_new_cast()
        return "break"

    def _shortcut_save_txt(self, _event: Any = None) -> str:
        if self.current_view != "yukhyo":
            return "break"
        self.save_txt()
        return "break"

    def _shortcut_copy_all(self, _event: Any = None) -> str:
        if self.current_view != "yukhyo":
            return "break"
        self.copy_all()
        return "break"

    def _shortcut_focus_search(self, _event: Any = None) -> str:
        if self.current_view != "yukhyo":
            return "break"
        self.search_entry.focus_set()
        self.search_entry.selection_range(0, "end")
        return "break"

    def _shortcut_prev_history(self, _event: Any = None) -> str:
        if self.current_view != "yukhyo":
            return "break"
        self._move_history_selection(-1)
        return "break"

    def _shortcut_next_history(self, _event: Any = None) -> str:
        if self.current_view != "yukhyo":
            return "break"
        self._move_history_selection(1)
        return "break"

    def _on_close(self) -> None:
        state = UIState(
            selected_tab=self.notebook.index(self.notebook.select()),
            summary_mode=bool(self.summary_var.get()),
            history_filter_text=self.search_var.get().strip(),
            moving_only=bool(self.moving_only_var.get()),
        )
        history_store.save_ui_state(state)
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def run_app() -> None:
    MainWindow().run()
