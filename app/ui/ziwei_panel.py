from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

import tkinter as tk
from tkinter import messagebox, ttk
from tkinter.scrolledtext import ScrolledText

import ziwei_service


COLORS = {
    "app_bg": "#0f2032",
    "chart_bg": "#12253a",
    "card_bg": "#f6f1e8",
    "card_fg": "#1f2430",
    "card_ming": "#fff1c6",
    "card_shen": "#dce9ff",
    "card_selected": "#f5db95",
    "card_transform": "#fbe2e2",
    "hub_bg": "#17304b",
    "hub_fg": "#f8f4e9",
    "badge_fg": "#ffffff",
    "badge_lok": "#2f8f4e",
    "badge_gwon": "#3958a7",
    "badge_gwa": "#9d6f00",
    "badge_gi": "#b33939",
}

SCOPE_LABELS = {"natal": "본명반", "decadal": "대운", "yearly": "세운"}
VIEW_LABELS = {"triad": "삼합파", "four_transform": "사화파"}
STAR_FILTER_LABELS = {"major": "주성만", "major_minor": "주성+보성", "all": "전체"}


class ZiweiPanel(ttk.Frame):
    def __init__(self, parent: tk.Widget) -> None:
        super().__init__(parent)
        self.payload: Optional[Dict[str, Any]] = None
        self.selected_palace_index: Optional[int] = None
        self._compact_layout = False

        self.var_birth_date = tk.StringVar(value="1996-05-04")
        self.var_birth_time = tk.StringVar(value="08:10")
        self.var_calendar = tk.StringVar(value="solar")
        self.var_lunar_leap = tk.BooleanVar(value=False)
        self.var_gender = tk.StringVar(value="male")
        self.var_view_mode = tk.StringVar(value="triad")
        self.var_scope = tk.StringVar(value="natal")
        self.var_star_filter = tk.StringVar(value="major_minor")
        self.var_show_hanja = tk.BooleanVar(value=False)
        self.var_status = tk.StringVar(value="출생 정보를 입력하고 [명반 생성]을 눌러주세요.")
        self.var_summary = tk.StringVar(value="자미두수 명반 요약이 여기에 표시됩니다.")
        self.var_detail_title = tk.StringVar(value="궁 상세")

        self.palace_buttons: Dict[int, tk.Button] = {}
        self._build_ui()
        self._render_empty_state()

    def _build_ui(self) -> None:
        self.configure(style="Ziwei.TFrame")
        root = ttk.Frame(self, padding=12)
        root.pack(fill="both", expand=True)

        self._build_input_area(root)
        self._build_view_controls(root)
        self._build_summary_bar(root)
        self._build_content_area(root)

        status = ttk.Label(root, textvariable=self.var_status, anchor="w")
        status.pack(fill="x", pady=(8, 0))

    def _build_input_area(self, parent: ttk.Frame) -> None:
        box = ttk.LabelFrame(parent, text="자미두수 입력", padding=10)
        box.pack(fill="x")

        ttk.Label(box, text="출생일(YYYY-MM-DD) *").grid(row=0, column=0, sticky="w", padx=4, pady=4)
        ttk.Entry(box, textvariable=self.var_birth_date, width=14).grid(row=0, column=1, sticky="w", padx=4, pady=4)

        ttk.Label(box, text="출생시간(HH:MM)").grid(row=0, column=2, sticky="w", padx=4, pady=4)
        ttk.Entry(box, textvariable=self.var_birth_time, width=10).grid(row=0, column=3, sticky="w", padx=4, pady=4)

        ttk.Label(box, text="달력").grid(row=1, column=0, sticky="w", padx=4, pady=4)
        ttk.Radiobutton(box, text="양력", variable=self.var_calendar, value="solar").grid(
            row=1, column=1, sticky="w", padx=4, pady=4
        )
        ttk.Radiobutton(box, text="음력", variable=self.var_calendar, value="lunar").grid(
            row=1, column=2, sticky="w", padx=4, pady=4
        )
        ttk.Checkbutton(box, text="윤달", variable=self.var_lunar_leap).grid(row=1, column=3, sticky="w", padx=4, pady=4)

        ttk.Label(box, text="성별").grid(row=2, column=0, sticky="w", padx=4, pady=4)
        ttk.Radiobutton(box, text="남", variable=self.var_gender, value="male").grid(row=2, column=1, sticky="w", padx=4, pady=4)
        ttk.Radiobutton(box, text="여", variable=self.var_gender, value="female").grid(row=2, column=2, sticky="w", padx=4, pady=4)

        ttk.Button(box, text="명반 생성", command=self.on_compute).grid(row=2, column=3, sticky="e", padx=4, pady=4)
        ttk.Button(box, text="초기화", command=self.on_reset).grid(row=2, column=4, sticky="e", padx=4, pady=4)

    def _build_view_controls(self, parent: ttk.Frame) -> None:
        controls = ttk.LabelFrame(parent, text="보기 옵션", padding=10)
        controls.pack(fill="x", pady=(8, 0))

        ttk.Label(controls, text="뷰").grid(row=0, column=0, sticky="w", padx=4, pady=4)
        ttk.Radiobutton(
            controls, text="삼합파", variable=self.var_view_mode, value="triad", command=self._on_display_option_changed
        ).grid(row=0, column=1, sticky="w", padx=4, pady=4)
        ttk.Radiobutton(
            controls,
            text="사화파",
            variable=self.var_view_mode,
            value="four_transform",
            command=self._on_display_option_changed,
        ).grid(row=0, column=2, sticky="w", padx=4, pady=4)

        ttk.Label(controls, text="시점").grid(row=0, column=3, sticky="w", padx=10, pady=4)
        ttk.Radiobutton(
            controls, text="본명반", variable=self.var_scope, value="natal", command=self._on_display_option_changed
        ).grid(row=0, column=4, sticky="w", padx=4, pady=4)
        ttk.Radiobutton(
            controls, text="대운", variable=self.var_scope, value="decadal", command=self._on_display_option_changed
        ).grid(row=0, column=5, sticky="w", padx=4, pady=4)
        ttk.Radiobutton(
            controls, text="세운", variable=self.var_scope, value="yearly", command=self._on_display_option_changed
        ).grid(row=0, column=6, sticky="w", padx=4, pady=4)

        ttk.Label(controls, text="별 표시").grid(row=1, column=0, sticky="w", padx=4, pady=4)
        filter_cb = ttk.Combobox(
            controls,
            textvariable=self.var_star_filter,
            state="readonly",
            width=12,
            values=["major", "major_minor", "all"],
        )
        filter_cb.grid(row=1, column=1, sticky="w", padx=4, pady=4)
        filter_cb.bind("<<ComboboxSelected>>", self._on_display_option_changed)
        ttk.Checkbutton(
            controls, text="한자 우선 표시", variable=self.var_show_hanja, command=self._on_display_option_changed
        ).grid(row=1, column=2, sticky="w", padx=4, pady=4)
        ttk.Label(
            controls,
            text="(사화파는 배지+리스트 중심으로 표시, 복잡한 선 시각화는 1차 범위에서 제외)",
            foreground="#555555",
        ).grid(row=1, column=3, columnspan=4, sticky="w", padx=8, pady=4)
        ttk.Button(controls, text="상세 복사", command=self.on_copy_detail).grid(row=1, column=7, sticky="e", padx=4, pady=4)
        ttk.Button(controls, text="LLM용 전체 복사", command=self.on_copy_llm_prompt).grid(
            row=1, column=8, sticky="e", padx=4, pady=4
        )

    def _build_summary_bar(self, parent: ttk.Frame) -> None:
        summary = ttk.LabelFrame(parent, text="요약", padding=8)
        summary.pack(fill="x", pady=(8, 0))
        ttk.Label(summary, textvariable=self.var_summary, justify="left", anchor="w").pack(fill="x")

    def _build_content_area(self, parent: ttk.Frame) -> None:
        self.content = ttk.Frame(parent)
        self.content.pack(fill="both", expand=True, pady=(8, 0))
        self.content.grid_columnconfigure(0, weight=3)
        self.content.grid_columnconfigure(1, weight=2)
        self.content.grid_rowconfigure(0, weight=1)

        self.chart_wrap = tk.Frame(self.content, bg=COLORS["chart_bg"], padx=12, pady=12)
        self.detail_wrap = ttk.LabelFrame(self.content, text=self.var_detail_title.get(), padding=8)

        self.chart_wrap.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        self.detail_wrap.grid(row=0, column=1, sticky="nsew")

        self._build_chart_grid(self.chart_wrap)
        self._build_detail_panel(self.detail_wrap)
        self.content.bind("<Configure>", self._on_resize)

    def _build_chart_grid(self, parent: tk.Frame) -> None:
        self.chart_grid = tk.Frame(parent, bg=COLORS["chart_bg"])
        self.chart_grid.pack(fill="both", expand=True)

        for r in range(4):
            self.chart_grid.grid_rowconfigure(r, weight=1, uniform="row")
            self.chart_grid.grid_columnconfigure(r, weight=1, uniform="col")

        for idx, (row, col) in enumerate(ziwei_service.GRID_RING_POSITIONS):
            btn = tk.Button(
                self.chart_grid,
                text="",
                justify="left",
                anchor="nw",
                padx=8,
                pady=6,
                relief="ridge",
                bd=1,
                wraplength=180,
                font=("Malgun Gothic", 9),
                cursor="hand2",
                takefocus=True,
                command=lambda i=idx: self.on_select_palace(i),
            )
            btn.grid(row=row, column=col, sticky="nsew", padx=4, pady=4)
            self.palace_buttons[idx] = btn

        self.center_hub = tk.Label(
            self.chart_grid,
            text="명반 허브",
            bg=COLORS["hub_bg"],
            fg=COLORS["hub_fg"],
            font=("Malgun Gothic", 10, "bold"),
            justify="left",
            anchor="center",
            padx=10,
            pady=8,
        )
        self.center_hub.grid(row=1, column=1, rowspan=2, columnspan=2, sticky="nsew", padx=6, pady=6)

    def _build_detail_panel(self, parent: ttk.LabelFrame) -> None:
        self.detail_text = ScrolledText(parent, wrap="word", height=28)
        self.detail_text.pack(fill="both", expand=True)
        self.detail_text.configure(state="disabled")

    def _on_resize(self, _event: tk.Event) -> None:
        width = self.content.winfo_width()
        compact = width < 1180
        if compact == self._compact_layout:
            return
        self._compact_layout = compact

        self.chart_wrap.grid_forget()
        self.detail_wrap.grid_forget()
        if compact:
            self.content.grid_rowconfigure(0, weight=3)
            self.content.grid_rowconfigure(1, weight=2)
            self.chart_wrap.grid(row=0, column=0, columnspan=2, sticky="nsew", padx=(0, 0), pady=(0, 8))
            self.detail_wrap.grid(row=1, column=0, columnspan=2, sticky="nsew")
        else:
            self.content.grid_rowconfigure(0, weight=1)
            self.content.grid_rowconfigure(1, weight=0)
            self.chart_wrap.grid(row=0, column=0, sticky="nsew", padx=(0, 8), pady=(0, 0))
            self.detail_wrap.grid(row=0, column=1, sticky="nsew")

    def _on_display_option_changed(self, _event: Any = None) -> None:
        if self.payload is None:
            return
        self._render_payload()

    def on_reset(self) -> None:
        self.payload = None
        self.selected_palace_index = None
        self.var_view_mode.set("triad")
        self.var_scope.set("natal")
        self.var_star_filter.set("major_minor")
        self.var_show_hanja.set(False)
        self.var_status.set("초기화되었습니다. 출생 정보를 입력하고 [명반 생성]을 눌러주세요.")
        self.var_summary.set("자미두수 명반 요약이 여기에 표시됩니다.")
        self._render_empty_state()

    def on_compute(self) -> None:
        birth_date = self.var_birth_date.get().strip()
        birth_time = self.var_birth_time.get().strip() or None
        gender = "male" if self.var_gender.get() == "male" else "female"
        calendar_type = "solar" if self.var_calendar.get() == "solar" else "lunar"

        self.var_status.set("명반 계산 중...")
        self.update_idletasks()

        try:
            inp = ziwei_service.ZiweiInput(
                birth_date=birth_date,
                birth_time=birth_time,
                gender=gender,  # type: ignore[arg-type]
                calendar_type=calendar_type,  # type: ignore[arg-type]
                lunar_leap_month=bool(self.var_lunar_leap.get()),
                require_exact_calendar=True,
            )
            payload = ziwei_service.compute_ziwei_payload(inp)
        except Exception as exc:
            messagebox.showerror("명반 생성 실패", f"오류가 발생했습니다:\n{exc}")
            self.var_status.set(f"오류: {exc}")
            self._render_error_state(str(exc))
            return

        self.payload = payload
        self.selected_palace_index = payload.get("profile", {}).get("ming_palace_index", 0)
        self.var_status.set("명반 생성 완료")
        self._render_payload()

    def on_select_palace(self, palace_index: int) -> None:
        self.selected_palace_index = palace_index
        self._render_payload()

    def _render_empty_state(self) -> None:
        for idx, btn in self.palace_buttons.items():
            btn.configure(
                text=f"{idx + 1}궁\n정보 없음",
                bg=COLORS["card_bg"],
                fg=COLORS["card_fg"],
                state="disabled",
                highlightthickness=0,
            )
        self.center_hub.configure(text="명반 생성 전")
        self._set_detail_text("아직 명반 데이터가 없습니다.\n\n출생 정보를 입력하고 [명반 생성]을 실행하세요.")

    def _render_error_state(self, error_message: str) -> None:
        for btn in self.palace_buttons.values():
            btn.configure(
                text="오류",
                bg="#f9e2e2",
                fg="#7a1f1f",
                state="disabled",
                highlightthickness=0,
            )
        self.center_hub.configure(text="명반 로드 실패")
        self._set_detail_text(f"명반을 불러오지 못했습니다.\n\n오류 내용:\n{error_message}\n\n입력을 확인하고 다시 시도하세요.")

    def _render_payload(self) -> None:
        if self.payload is None:
            self._render_empty_state()
            return

        try:
            vm = ziwei_service.build_chart_view_model(
                payload=self.payload,
                view_mode=self.var_view_mode.get(),  # type: ignore[arg-type]
                scope=self.var_scope.get(),  # type: ignore[arg-type]
                star_filter=self.var_star_filter.get(),  # type: ignore[arg-type]
                show_hanja=bool(self.var_show_hanja.get()),
                selected_palace_index=self.selected_palace_index,
            )
        except Exception as exc:
            self.var_status.set(f"렌더링 오류: {exc}")
            self._render_error_state(str(exc))
            return

        summary = vm["summary"]
        self.var_summary.set(
            "뷰: {view} | 시점: {scope} | 오행국: {wuxing}\n"
            "명궁: {ming_name} | 신궁: {shen_name} | 명주: {ming_master} | 신주: {shen_master}".format(
                view=VIEW_LABELS.get(summary["view_mode"], summary["view_mode"]),
                scope=SCOPE_LABELS.get(summary["scope"], summary["scope"]),
                wuxing=summary.get("wuxing_ju", "-"),
                ming_name=summary.get("ming_name", "-"),
                shen_name=summary.get("shen_name", "-"),
                ming_master=summary.get("ming_master", "-"),
                shen_master=summary.get("shen_master", "-"),
            )
        )

        cards = vm["cards"]
        for card in cards:
            idx = card["index"]
            btn = self.palace_buttons[idx]
            text_lines = [card["name"]]

            markers: List[str] = []
            if card["is_ming"]:
                markers.append("명궁")
            if card["is_shen"]:
                markers.append("신궁")
            if markers:
                text_lines[-1] = f"{text_lines[-1]} ({'/'.join(markers)})"

            text_lines.append(card["stem_branch"])
            if card["stars_preview"]:
                stars_line = "· " + ", ".join(card["stars_preview"])
                if card["stars_extra_count"] > 0:
                    stars_line += f"  +{card['stars_extra_count']}"
                text_lines.append(stars_line)
            else:
                text_lines.append("· 별 정보 없음")

            if self.var_view_mode.get() == "four_transform" and card["transforms"]:
                badges = "".join([f"[{row['type']}]" for row in card["transforms"]])
                text_lines.append(badges)

            card_bg = COLORS["card_bg"]
            if card["is_ming"]:
                card_bg = COLORS["card_ming"]
            if card["is_shen"]:
                card_bg = COLORS["card_shen"]
            if self.var_view_mode.get() == "four_transform" and card["has_transform"]:
                card_bg = COLORS["card_transform"]
            if card["is_selected"]:
                card_bg = COLORS["card_selected"]

            btn.configure(
                text="\n".join(text_lines),
                bg=card_bg,
                fg=COLORS["card_fg"],
                activebackground=card_bg,
                activeforeground=COLORS["card_fg"],
                state="normal",
                highlightthickness=2 if card["is_selected"] else 1,
                highlightbackground="#876c2f" if card["is_selected"] else "#8e8e8e",
            )

        selected = vm["selected_detail"]
        self.var_detail_title.set(
            f"궁 상세 - {selected['name_hanja'] if self.var_show_hanja.get() else selected['name_ko']} ({selected['stem_branch']})"
        )
        self.detail_wrap.configure(text=self.var_detail_title.get())

        center = (
            f"{VIEW_LABELS.get(summary['view_mode'], summary['view_mode'])} / "
            f"{SCOPE_LABELS.get(summary['scope'], summary['scope'])}\n"
            f"오행국: {summary.get('wuxing_ju', '-')}\n"
            f"명궁: {summary.get('ming_name', '-')}\n"
            f"신궁: {summary.get('shen_name', '-')}"
        )
        self.center_hub.configure(text=center)

        self._set_detail_text(self._format_detail_text(selected))
        status_text = (
            f"표시 모드: {VIEW_LABELS.get(self.var_view_mode.get(), self.var_view_mode.get())} | "
            f"별 필터: {STAR_FILTER_LABELS.get(self.var_star_filter.get(), self.var_star_filter.get())}"
        )
        warnings = self.payload.get("debug", {}).get("warnings", []) if isinstance(self.payload, dict) else []
        if warnings:
            status_text = f"{status_text} | 주의: {warnings[0]}"
        self.var_status.set(status_text)

    def _format_detail_text(self, selected: Dict[str, Any]) -> str:
        lines: List[str] = []
        lines.append(f"[기본] {selected['name_ko']} ({selected['name_hanja']})")
        lines.append(f"- 천간/지지: {selected['stem_branch']}")
        lines.append(f"- 명궁 여부: {'예' if selected['is_ming'] else '아니오'}")
        lines.append(f"- 신궁 여부: {'예' if selected['is_shen'] else '아니오'}")
        lines.append("")

        lines.append("[별 목록]")
        lines.append(f"- 주성: {', '.join(selected['stars_major']) if selected['stars_major'] else '-'}")
        lines.append(f"- 보성: {', '.join(selected['stars_minor']) if selected['stars_minor'] else '-'}")
        lines.append(f"- 기타: {', '.join(selected['stars_misc']) if selected['stars_misc'] else '-'}")
        lines.append("")

        lines.append("[별 밝기]")
        all_stars = selected["stars_major"] + selected["stars_minor"] + selected["stars_misc"]
        if all_stars:
            for star in all_stars:
                level = selected["brightness"].get(star, "-")
                lines.append(f"- {star}: {level}")
        else:
            lines.append("- 없음")
        lines.append("")

        lines.append("[구조 정보]")
        triad_names = ", ".join(selected.get("triad_names", [])) if selected.get("triad_names") else "-"
        lines.append(f"- 삼합 관련 궁: {triad_names}")
        lines.append(f"- 대궁: {selected.get('opposite_name', '-')}")
        lines.append("")

        lines.append(f"[사화 정보 - {SCOPE_LABELS.get(selected['scope'], selected['scope'])}]")
        transforms = selected.get("transforms", [])
        if transforms:
            for row in transforms:
                lines.append(f"- {row['type']}({row['type_hanja']}): {row['star']} [{row['source']}]")
        else:
            lines.append("- 해당 없음")

        return "\n".join(lines)

    def _copy_to_clipboard(self, text: str, success_message: str) -> None:
        if not text.strip():
            self.var_status.set("복사할 내용이 없습니다.")
            return
        self.clipboard_clear()
        self.clipboard_append(text)
        self.update_idletasks()
        self.var_status.set(success_message)

    def on_copy_detail(self) -> None:
        data = self.detail_text.get("1.0", "end-1c")
        self._copy_to_clipboard(data, "궁 상세 텍스트를 클립보드에 복사했습니다.")

    def _build_llm_prompt_text(self) -> str:
        if self.payload is None:
            return ""

        vm = ziwei_service.build_chart_view_model(
            payload=self.payload,
            view_mode=self.var_view_mode.get(),  # type: ignore[arg-type]
            scope=self.var_scope.get(),  # type: ignore[arg-type]
            star_filter=self.var_star_filter.get(),  # type: ignore[arg-type]
            show_hanja=bool(self.var_show_hanja.get()),
            selected_palace_index=self.selected_palace_index,
        )
        summary = vm["summary"]
        selected = vm["selected_detail"]
        palaces = self.payload.get("palaces", [])
        scope = self.var_scope.get()
        transforms_by_scope = self.payload.get("four_transforms", {}).get(scope, [])

        lines: List[str] = []
        lines.append("[요청]")
        lines.append("아래 자미두수 명반 데이터를 바탕으로 핵심 해석을 해줘.")
        lines.append("- 1) 전체 요약")
        lines.append("- 2) 강점/약점")
        lines.append("- 3) 직업/재물/관계 포인트")
        lines.append("- 4) 현재 시점(" + SCOPE_LABELS.get(scope, scope) + ")에서 주의점")
        lines.append("")

        lines.append("[기본 정보]")
        lines.append(f"- 출생일: {self.var_birth_date.get().strip()}")
        lines.append(f"- 출생시간: {self.var_birth_time.get().strip() or '(미입력)'}")
        lines.append(f"- 성별: {'남' if self.var_gender.get() == 'male' else '여'}")
        lines.append(f"- 달력: {'양력' if self.var_calendar.get() == 'solar' else '음력'}")
        lines.append(f"- 윤달: {'예' if self.var_lunar_leap.get() else '아니오'}")
        lines.append(f"- 타임존: Asia/Seoul")
        lines.append("")

        lines.append("[현재 보기]")
        lines.append(f"- 뷰: {VIEW_LABELS.get(summary['view_mode'], summary['view_mode'])}")
        lines.append(f"- 시점: {SCOPE_LABELS.get(summary['scope'], summary['scope'])}")
        lines.append(f"- 별 필터: {STAR_FILTER_LABELS.get(self.var_star_filter.get(), self.var_star_filter.get())}")
        lines.append(f"- 오행국: {summary.get('wuxing_ju', '-')}")
        lines.append(f"- 명궁: {summary.get('ming_name', '-')}")
        lines.append(f"- 신궁: {summary.get('shen_name', '-')}")
        lines.append(f"- 명주: {summary.get('ming_master', '-')}")
        lines.append(f"- 신주: {summary.get('shen_master', '-')}")
        lines.append("")

        lines.append("[12궁 데이터]")
        for p in palaces:
            marks: List[str] = []
            if p.get("is_ming"):
                marks.append("명궁")
            if p.get("is_shen"):
                marks.append("신궁")
            mark_text = f" ({'/'.join(marks)})" if marks else ""
            lines.append(f"- {p.get('name_ko')}[{p.get('name_hanja')}]{mark_text} | {p.get('stem')}{p.get('branch')}")
            lines.append("  · 주성: " + (", ".join(p.get("stars", {}).get("major", [])) or "-"))
            lines.append("  · 보성: " + (", ".join(p.get("stars", {}).get("minor", [])) or "-"))
            lines.append("  · 기타: " + (", ".join(p.get("stars", {}).get("misc", [])) or "-"))

        lines.append("")
        lines.append("[사화 정보]")
        if transforms_by_scope:
            for row in transforms_by_scope:
                lines.append(
                    f"- {row.get('type')}({row.get('type_hanja')}): {row.get('star')} -> {row.get('palace_name_ko')} [{row.get('source')}]"
                )
        else:
            lines.append("- 해당 없음")

        lines.append("")
        lines.append("[현재 선택 궁 상세]")
        lines.append(self._format_detail_text(selected))

        lines.append("")
        lines.append("[출력 형식 요청]")
        lines.append("- 핵심만 간결하게, 근거가 되는 궁/별/사화를 함께 적어줘.")
        lines.append(f"- 생성시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        return "\n".join(lines).strip() + "\n"

    def on_copy_llm_prompt(self) -> None:
        data = self._build_llm_prompt_text()
        self._copy_to_clipboard(data, "LLM 해석용 전체 프롬프트를 클립보드에 복사했습니다.")

    def _set_detail_text(self, text: str) -> None:
        self.detail_text.configure(state="normal")
        self.detail_text.delete("1.0", "end")
        self.detail_text.insert("1.0", text)
        self.detail_text.see("1.0")
        self.detail_text.configure(state="disabled")
