"""Desktop window that drives the score aggregation.

    python main.py

The window holds three parts: the table of committed rounds, a preview of the
overlay that goes to OBS, and the log.

KS, rank and penalty are edited in place in the table. Editing a rank swaps it
with the slot that held it.

The overlay is published by a small HTTP server; only the standings travel, so
rows can animate when the order changes.
"""

from __future__ import annotations

PROTOCOL = 2  # module interface version; mixing different values is unsafe
FILE_SET = "2026-09-04-k"  # release this file belongs to

from recognize import check_shadowed_modules

check_shadowed_modules()  # stop when a local file shadows a standard module

from settings import (
    CONFIG_PATH,
    DEFAULT_INTERVAL,
    OVERLAY_PORT,
    FIRST_FRAME_TIMEOUT,
    LOG_HEIGHT,
    OVERLAY_HEIGHT,
    OVERLAY_ROW_HEIGHT as ROW_HEIGHT,
    OVERLAY_WIDTH,
    PREVIEW_SCALE,
    TIMELINE_ON,
    WINDOW_HEIGHT,
    WINDOW_WIDTH,
)

PREVIEW_HEIGHT = 28 + 8 * ROW_HEIGHT

import queue
import threading
import time
import tkinter as tk
from pathlib import Path
import webbrowser
from tkinter import messagebox, ttk

from capture import WindowSource, list_windows
from config import Config
from digits import load_reader
from model import MatchState
from rules import PLACEMENT_POINTS, TEAM_COUNT, ScanError, points_of
from history import Timeline, repair_frame
from scan import fill_missing, read_slots
from server import OverlayServer


class Reader(threading.Thread):
    """Read the window on a timer and push the results onto a queue.

    Kept off the UI thread so a slow read never freezes the window.
    """

    def __init__(self, cfg: Config, reader, source, out: "queue.Queue") -> None:
        super().__init__(daemon=True)
        self.cfg = cfg
        self.reader = reader
        self.source = source
        self.out = out
        self.interval = DEFAULT_INTERVAL
        self._stop = threading.Event()
        self._paused = threading.Event()

    def pause(self) -> None:
        self._paused.set()

    def resume(self) -> None:
        self._paused.clear()

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        while not self._stop.is_set():
            if self._paused.is_set():
                time.sleep(0.1)
                continue
            frame = self.source.grab()
            if frame is None:
                self.out.put(("문제", getattr(self.source, "last_reason", "그림 없음")))
            else:
                self.out.put(("읽음", read_slots(frame, self.cfg, self.reader)))
            time.sleep(max(0.1, self.interval))


def pick_window(root: tk.Misc, process: str | None, title: str | None):
    """Show the open windows and let the user pick one.

    Returns:
        The chosen process, title and handle, or None when cancelled.
    """
    windows = list_windows()
    if not windows:
        messagebox.showerror("창", "열려 있는 창을 찾을 수 없습니다")
        return None

    chosen: dict = {}
    dialog = tk.Toplevel(root)
    dialog.title("창 선택")
    dialog.geometry("620x420")
    ttk.Label(dialog, text="읽을 창을 선택한 뒤 확인을 클릭").pack(anchor="w", padx=8, pady=6)

    table = ttk.Treeview(dialog, columns=("process", "title"), show="headings")
    table.heading("process", text="프로세스")
    table.heading("title", text="창 제목")
    table.column("process", width=180)
    table.column("title", width=400)
    table.pack(fill="both", expand=True, padx=8)

    for i, w in enumerate(windows):
        table.insert("", "end", iid=str(i), values=(w.process or "", w.title))
        if (process and w.process == process) or (title and title in w.title):
            table.selection_set(str(i))

    def confirm() -> None:
        picked = table.selection()
        if picked:
            w = windows[int(picked[0])]
            chosen["process"], chosen["title"] = w.process, w.title
            chosen["hwnd"] = w.hwnd
        dialog.destroy()

    button_row = ttk.Frame(dialog)
    button_row.pack(fill="x", pady=6)
    ttk.Button(button_row, text="confirm", command=confirm).pack(side="right", padx=8)
    ttk.Button(button_row, text="취소", command=dialog.destroy).pack(side="right")
    table.bind("<Double-1>", lambda e: confirm())

    dialog.transient(root)
    dialog.grab_set()
    root.wait_window(dialog)
    return chosen or None


class App:
    def __init__(self, config_path: str = CONFIG_PATH) -> None:
        if not Path(config_path).exists():
            raise SystemExit(
                f"{config_path} 파일이 없습니다. 실행 파일과 같은 폴더에 넣어 주세요. "
                "이 판은 전체 화면 비율에 맞춰 지정된 영역을 사용합니다."
            )
        self.cfg = Config.load(config_path)
        self.config_path = config_path
        self.reader, _, _ = load_reader(self.cfg)
        self.match = MatchState(list(self.cfg.team_names))
        self.queue: "queue.Queue" = queue.Queue()
        self.worker: Reader | None = None
        self.interval = DEFAULT_INTERVAL
        self.round_open = True
        self.target = {"process": None, "title": None, "hwnd": None}
        self.timeline = Timeline(enabled=TIMELINE_ON)
        self.claimed_round: int | None = None

        self.server = OverlayServer(port=OVERLAY_PORT)
        try:
            self.server.start()
        except OSError as e:
            raise SystemExit(f"오버레이 포트를 열지 못했습니다. {e}")

        self.root = tk.Tk()
        self.root.title("이터널 리턴 점수 집계")
        self._build()
        self.fit_window()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        wanted = OVERLAY_PORT
        if self.server.port != wanted:
            self.say(f"{wanted}번 포트가 사용 중이어서 {self.server.port}번으로 열었습니다")
        self.root.after(200, self._pump)

    # ---- layout ------------------------------------------------------
    def _build(self) -> None:
        top = ttk.Frame(self.root, padding=8)
        top.pack(fill="x")
        self.top_bar = top

        ttk.Label(top, text="주기(초)").pack(side="left")
        self.interval_var = tk.StringVar(value=str(self.interval))
        ttk.Entry(top, textvariable=self.interval_var, width=6).pack(side="left", padx=(4, 4))
        ttk.Button(top, text="적용", command=self.apply_interval).pack(side="left")

        self.timeline_var = tk.BooleanVar(value=TIMELINE_ON)
        ttk.Checkbutton(
            top, text="시계열 보정 (beta)", variable=self.timeline_var,
            command=self.apply_timeline,
        ).pack(side="left", padx=12)

        self.start_btn = ttk.Button(top, text="집계 시작", command=self.toggle)
        self.start_btn.pack(side="left", padx=4)
        ttk.Button(top, text="대상 창 바꾸기", command=self.change_target).pack(side="left")
        ttk.Button(top, text="다음 라운드 시작", command=self.next_round).pack(side="left", padx=4)
        self.round_no_var = tk.StringVar(value="1")
        ttk.Label(top, text="라운드").pack(side="left")
        ttk.Entry(top, textvariable=self.round_no_var, width=4).pack(side="left", padx=(2, 8))
        ttk.Button(top, text="라운드 수동 확정", command=self.finish_round).pack(side="left")
        ttk.Button(top, text="현재 라운드 폐기", command=self.discard_round).pack(
            side="left", padx=4
        )

        self.state_label = ttk.Label(top, text="멈춤")
        self.state_label.pack(side="right")
        self.log_open = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            top, text="로그 보기", variable=self.log_open, command=self.toggle_log
        ).pack(side="right", padx=8)

        body = ttk.Panedwindow(self.root, orient="horizontal")
        body.pack(fill="both", expand=True)

        left = ttk.Frame(body, padding=6)
        body.add(left, weight=3)
        head = ttk.Frame(left)
        head.pack(fill="x")
        self.left_head = head
        ttk.Label(head, text="점수기록표").pack(side="left")
        self.sort_var = tk.StringVar(value="round")
        ttk.Radiobutton(
            head, text="라운드순", value="round", variable=self.sort_var,
            command=self.refresh,
        ).pack(side="left", padx=(12, 2))
        ttk.Radiobutton(
            head, text="팀번호순", value="team", variable=self.sort_var,
            command=self.refresh,
        ).pack(side="left")
        ttk.Button(head, text="라운드 추가", command=self.add_round_dialog).pack(side="right")
        ttk.Button(head, text="오프셋 넣기", command=self.offset_dialog).pack(side="right", padx=4)
        ttk.Button(head, text="새 경기 시작", command=self.new_match).pack(side="right")

        self.left_help = ttk.Label(
            left,
            justify="left",
            text=(
                "라운드/팀번호/팀명 더블클릭시 - 해당 행 점수 초기화\n"
                "점수 칸 더블클릭시 - 점수 편집 가능"
            ),
        )
        self.left_help.pack(anchor="w", pady=(2, 4))

        holder = ttk.Frame(left)
        holder.pack(fill="both", expand=True)
        self.grid_canvas = tk.Canvas(holder, highlightthickness=0)
        self.grid_bar = ttk.Scrollbar(
            holder, orient="vertical", command=self.grid_canvas.yview
        )
        self.grid_frame = ttk.Frame(self.grid_canvas)
        self.grid_frame.bind("<Configure>", lambda e: self.fit_scroll())
        self.grid_canvas.bind("<Configure>", lambda e: self.fit_scroll())
        self.grid_canvas.create_window((0, 0), window=self.grid_frame, anchor="nw")
        self.grid_canvas.configure(yscrollcommand=self.grid_bar.set)
        self.grid_canvas.pack(side="left", fill="both", expand=True)
        # the scrollbar appears only when the table overflows
        self.grid_canvas.bind_all("<MouseWheel>", self.on_wheel)

        right = ttk.Frame(body, padding=6)
        self.right_pane = right
        body.add(right, weight=2)

        ttk.Label(right, text="OBS 오버레이").pack(anchor="w")
        self.url_var = tk.StringVar(value=self.server.url)
        ttk.Entry(right, textvariable=self.url_var, state="readonly").pack(fill="x")
        ttk.Label(
            right, text="OBS 브라우저 소스에 이 주소를 넣고 크기를 207×242로 설정"
        ).pack(anchor="w", fill="x", pady=(2, 4))
        ttk.Button(right, text="브라우저로 열기", command=self.open_overlay).pack(fill="x")

        self.preview = self._make_preview(right)

        ttk.Separator(right).pack(fill="x", pady=6)
        ttk.Label(right, text="팀 이름 (1번부터 8번까지)").pack(anchor="w")
        self.name_vars: list[tk.StringVar] = []
        for i in range(TEAM_COUNT):
            line = ttk.Frame(right)
            line.pack(fill="x", pady=1)
            ttk.Label(line, text=f"{i + 1}", width=3).pack(side="left")
            var = tk.StringVar(value=self.match.team_names[i])
            ttk.Entry(line, textvariable=var).pack(side="left", fill="x", expand=True)
            self.name_vars.append(var)
        ttk.Button(right, text="이름 적용", command=self.apply_names).pack(fill="x", pady=4)

        # the log is collapsed by default and toggled from the top bar
        self.log = tk.Text(self.root, height=8)


    def fit_window(self) -> None:
        """Open the window at a fixed size.

        Measuring the contents would always add the height of the bottom bar, and
        the layout does not reflow, so measured values are not used.
        """
        width = min(WINDOW_WIDTH, self.root.winfo_screenwidth() - 40)
        height = min(WINDOW_HEIGHT, self.root.winfo_screenheight() - 80)
        self.root.geometry(f"{width}x{height}")
        self.root.minsize(900, 500)


    def toggle_log(self) -> None:
        """Show or hide the log, adjusting the window height to match."""
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        if self.log_open.get():
            self.log.pack(fill="both", expand=True, padx=6, pady=(0, 6))
            self.root.geometry(f"{width}x{height + LOG_HEIGHT}")
        else:
            self.log.pack_forget()
            self.root.geometry(f"{width}x{max(400, height - LOG_HEIGHT)}")

    def _make_preview(self, parent):
        """Draw the overlay preview inside the window.

        An embedded browser laid the table out badly, so the same geometry is
        drawn directly.
        """
        holder = ttk.Frame(parent)
        holder.pack(anchor="w", pady=4)
        canvas = tk.Canvas(
            holder,
            width=OVERLAY_WIDTH * PREVIEW_SCALE,
            height=PREVIEW_HEIGHT * PREVIEW_SCALE,
            background="#0b1016",
            highlightthickness=1,
            highlightbackground="#3a3f47",
        )
        canvas.pack()
        return canvas

    def draw_preview(self, standings) -> None:
        """Draw the standings with the same geometry as the overlay."""
        canvas = self.preview
        if canvas is None:
            return
        canvas.delete("all")
        k = PREVIEW_SCALE

        canvas.create_text(
            OVERLAY_WIDTH * k / 2, 13 * k, text=self.server.title,
            fill="#ffffff", font=("Arial", int(11 * k), "bold"),
        )
        canvas.create_line(
            0, 26 * k, OVERLAY_WIDTH * k, 26 * k, fill="#3a3f47",
        )

        out_slots = self.out_slots()
        for i, team in enumerate(standings):
            top = (28 + i * ROW_HEIGHT) * k
            bottom = top + ROW_HEIGHT * k
            gray = team.slot in out_slots
            color = "#8b95a2" if gray else "#ffffff"

            if i == 0:
                canvas.create_rectangle(
                    0, top, OVERLAY_WIDTH * k, bottom,
                    fill="#3b3a1c", outline="",
                )
            canvas.create_text(
                12 * k, (top + bottom) / 2, text=f"#{i + 1}",
                fill=color, font=("Arial", int(8 * k), "bold"),
            )
            canvas.create_text(
                (OVERLAY_WIDTH / 2) * k, (top + bottom) / 2, text=team.name,
                fill=color, font=("Arial", int(8 * k)),
            )
            canvas.create_text(
                (OVERLAY_WIDTH - 21) * k, (top + bottom) / 2,
                text=f"{team.total:g}", fill=color,
                font=("Arial", int(8 * k), "bold"),
            )
            if gray:
                canvas.create_line(
                    2 * k, bottom - 2, (OVERLAY_WIDTH - 2) * k, top + 2,
                    fill="#c9ccd1",
                )
            canvas.create_line(
                0, bottom, OVERLAY_WIDTH * k, bottom, fill="#2a2f36",
            )
        canvas.create_line(24 * k, 26 * k, 24 * k, PREVIEW_HEIGHT * k, fill="#2a2f36")
        canvas.create_line(
            (OVERLAY_WIDTH - 42) * k, 26 * k,
            (OVERLAY_WIDTH - 42) * k, PREVIEW_HEIGHT * k, fill="#2a2f36",
        )

    def out_slots(self) -> set[int]:
        """Return the slots whose rank is already fixed in the current round."""
        current = self.match.current.committed
        if current is None:
            return set()
        return {i for i, rank in enumerate(current.rank) if rank is not None}

    # ---- helpers -----------------------------------------------------
    def say(self, text: str) -> None:
        self.log.insert("end", f"[{time.strftime('%H:%M:%S')}] {text}\n")
        self.log.see("end")

    def open_overlay(self) -> None:
        webbrowser.open(self.server.url)

    def apply_interval(self) -> None:
        try:
            value = max(0.1, float(self.interval_var.get()))
        except ValueError:
            messagebox.showerror("주기", "숫자로 입력해 주세요. 0.5와 같이 소수점도 사용할 수 있습니다.")
            return
        self.interval = value
        self.interval_var.set(str(value))
        if self.worker is not None:
            self.worker.interval = value
        self.say(f"주기를 {value}초로 변경")

    def apply_timeline(self) -> None:
        on = self.timeline_var.get()
        self.timeline.enabled = on
        self.say("시계열 보정 켜짐" if on else "시계열 보정 꺼짐")

    # ---- target window -----------------------------------------------
    def change_target(self) -> None:
        chosen = pick_window(self.root, self.target["process"], self.target["title"])
        if chosen:
            self.target = chosen
            self.say(f"대상 창을 '{chosen['title']}'로 변경")

    def toggle(self) -> None:
        if self.worker is None:
            self.start()
        else:
            self.stop()

    def start(self) -> None:
        if not self.target["process"] and not self.target["title"]:
            self.change_target()
            if not self.target["process"] and not self.target["title"]:
                return
        if self.claimed_round is None and not self.claim_round():
            return
        try:
            source = WindowSource(
                self.target["process"], self.target["title"], self.target.get("hwnd")
            )
            source.first_frame(timeout=FIRST_FRAME_TIMEOUT)
        except (SystemExit, RuntimeError) as e:
            messagebox.showerror("창", str(e))
            return
        self.worker = Reader(self.cfg, self.reader, source, self.queue)
        self.worker.start()
        self.start_btn.config(text="집계 멈춤")
        self.state_label.config(text="읽는 중")
        self.say(f"'{source.title}' 창을 {self.interval}초 주기로 읽는 중")

    def stop(self) -> None:
        if self.worker is not None:
            self.worker.stop()
            self.worker = None
        self.start_btn.config(text="집계 시작")
        self.state_label.config(text="멈춤")
        self.say("집계 중지")

    def close(self) -> None:
        self.stop()
        self.server.stop()
        self.root.destroy()

    # ---- round_index -------------------------------------------------------
    def free_round_no(self, default: int) -> int | None:
        """Return a free round number, or None when the user cancels.

        A number already used, or the one claimed by the running aggregation, is
        rejected on the spot rather than after the values were typed.
        """
        from tkinter import simpledialog

        number = default
        while True:
            taken = set(self.match.round_numbers)
            if self.claimed_round is not None:
                taken.add(self.claimed_round)
            if number is not None and number not in taken:
                return number
            answer = simpledialog.askinteger(
                "라운드 번호",
                f"{number}라운드는 이미 존재합니다. 사용할 번호를 입력해 주세요.",
                parent=self.root,
                initialvalue=(max(taken) + 1) if taken else 1,
                minvalue=1,
            )
            if answer is None:
                return None
            number = answer

    def claim_round(self) -> bool:
        """Claim the round number for this run.

        Once claimed, editing the entry no longer changes the round in progress.
        """
        text = self.round_no_var.get().strip()
        number = self.free_round_no(int(text) if text.isdigit() else 1)
        if number is None:
            return False
        self.claimed_round = number
        self.match.next_round_no = number
        self.round_no_var.set(str(number))
        self.refresh(force=True)
        return True

    def next_round(self) -> None:
        """Start the next round, claiming its number once."""
        if not self.claim_round():
            return
        self.round_open = True
        self.timeline.reset()
        if self.worker is not None:
            self.worker.resume()
        self.state_label.config(text="읽는 중")
        self.say(f"{self.match.next_round_no}라운드 시작")

    def finish_round(self) -> None:
        """Commit the current round, asking for the order when teams are still alive."""
        if self.claimed_round is None and not self.claim_round():
            return
        self.match.next_round_no = self.claimed_round
        committed = self.match.current.committed
        order = None
        if committed is not None and not committed.finished:
            order = self.ask_order(list(committed.alive))
            if order is None:
                return
        try:
            result = self.match.finish_round(order=order)
        except ScanError as e:
            messagebox.showerror("round_index", str(e))
            return
        self.round_open = False
        if self.worker is not None:
            self.worker.pause()
        self.state_label.config(text="라운드 종료. 다음 라운드 시작을 클릭")
        self.say(f"{self.match.round_numbers[-1]}라운드 확정: 등수 {result.rank}")
        self.claimed_round = None
        self.round_no_var.set(str(self.match.next_round_no))
        self.refresh()

    def ask_order(self, alive: list[int]) -> list[int] | None:
        """Ask the operator to rank the living slots.

        The last wipe is sometimes missing from the broadcast, so the round still
        has to be committable.
        """
        order: list[int] = []
        dialog = tk.Toplevel(self.root)
        dialog.title("남은 팀의 등수")
        ttk.Label(
            dialog,
            text="생존한 팀이 남아 있습니다. 높은 등수부터 차례로 클릭해 주세요.",
            padding=8,
        ).pack(anchor="w")

        chosen = ttk.Label(dialog, text="", padding=(8, 0))
        chosen.pack(anchor="w")
        buttons: dict[int, ttk.Button] = {}

        def press(slot: int) -> None:
            order.append(slot)
            buttons[slot].state(["disabled"])
            chosen.config(
                text=" → ".join(
                    f"{i + 1}top {self.match.team_names[s]}"
                    for i, s in enumerate(order)
                )
            )
            if len(order) == len(alive):
                dialog.destroy()

        frame = ttk.Frame(dialog, padding=8)
        frame.pack(fill="x")
        for slot in alive:
            button = ttk.Button(
                frame, text=f"{slot + 1}번 {self.match.team_names[slot]}",
                command=lambda s=slot: press(s),
            )
            button.pack(fill="x", pady=2)
            buttons[slot] = button

        ttk.Button(dialog, text="취소", command=dialog.destroy).pack(pady=6)
        dialog.transient(self.root)
        dialog.grab_set()
        self.root.wait_window(dialog)
        return order if len(order) == len(alive) else None

    def discard_round(self) -> None:
        """Discard the round in progress and start reading it again."""
        if not messagebox.askyesno(
            "현재 라운드 폐기", "현재 라운드의 판독 값을 삭제합니다. 확정된 라운드는 유지됩니다. 계속할까요?"
        ):
            return
        self.match.discard_round()
        self.claimed_round = None
        self.timeline.reset()
        self.round_open = True
        self.round_no_var.set(str(self.match.next_round_no))
        self.say("현재 라운드 폐기")
        self.refresh()

    def apply_names(self) -> None:
        names = [v.get().strip() or f"팀 {i + 1}" for i, v in enumerate(self.name_vars)]
        self.match.team_names = names
        self.cfg.team_names = names
        self.cfg.save(self.config_path)
        self.say("팀 이름 변경")
        self.refresh()

    def fit_scroll(self) -> None:
        """Show the scrollbar while the table is taller than the view."""
        box = self.grid_canvas.bbox("all")
        if box is None:
            return
        self.grid_canvas.configure(scrollregion=box)
        need = box[3] - box[1] > self.grid_canvas.winfo_height()
        if need and not self.grid_bar.winfo_ismapped():
            self.grid_bar.pack(side="right", fill="y")
        elif not need and self.grid_bar.winfo_ismapped():
            self.grid_bar.pack_forget()

    def on_wheel(self, event) -> None:
        box = self.grid_canvas.bbox("all")
        if box is None or box[3] - box[1] <= self.grid_canvas.winfo_height():
            return
        self.grid_canvas.yview_scroll(-1 if event.delta > 0 else 1, "units")

    # ---- table -------------------------------------------------------

    COLUMNS = (
        ("group", "라운드", 70), ("slot", "팀번호", 60), ("team", "팀", 150),
        ("ks", "KS", 70), ("rank", "등수", 60), ("place", "순위점수", 80),
        ("penalty", "감점", 70),
    )

    EMPTY_BG = "#d9d9d9"     # empty cell
    CURRENT_BG = "#fff6c2"   # round in progress
    LINE_BG = "#8a8f96"      # separator between groups
    DUP_BG = "#ffb3b3"       # duplicated rank inside one round

    def record_rows(self) -> list[dict]:
        """Build the rows to draw; the offset comes first, then the rounds."""
        rows: list[dict] = []

        if self.match.offset:
            for slot in sorted(self.match.offset):
                values = self.match.offset[slot]
                rows.append({
                    "group": "오프셋", "round_index": None, "order": -1, "slot": slot,
                    "ks": values.get("ks"), "rank": None,
                    "place": None if values.get("ts") is None else round(
                        values["ts"] - (values.get("ks") or 0.0), 1
                    ),
                    "penalty": None, "current": False,
                })

        for index, record in enumerate(self.match.rounds):
            if self.match.is_empty_round(index):
                continue
            penalties = self.match.penalties[index] if index < len(self.match.penalties) else {}
            duplicates = self.match.duplicate_ranks(index)
            for slot in range(TEAM_COUNT):
                rank = record.rank[slot]
                rows.append({
                    "group": f"{self.match.number_of(index)}라운드",
                    "round_index": index, "order": self.match.number_of(index),
                    "slot": slot,
                    "dup": rank is not None and rank in duplicates,
                    "ks": record.ks[slot], "rank": rank,
                    "place": None if rank is None else points_of(rank),
                    "penalty": penalties.get(slot), "current": False,
                })

        current = self.match.current.committed
        if current is not None:
            index = len(self.match.rounds)
            penalties = self.match.penalties[index] if index < len(self.match.penalties) else {}
            for slot in range(TEAM_COUNT):
                rank = current.rank[slot]
                rows.append({
                    "group": f"{self.claimed_round or self.match.next_round_no}라운드",
                    "round_index": index,
                    "order": self.claimed_round or self.match.next_round_no,
                    "slot": slot,
                    "ks": current.ks[slot], "rank": rank,
                    "place": current.points(slot),
                    "penalty": penalties.get(slot), "current": True,
                })
        return rows

    def table_signature(self) -> tuple:
        """Summarise the table contents; an unchanged summary skips the redraw."""
        rows = self.record_rows()
        return (
            self.sort_var.get(),
            tuple(self.match.team_names),
            tuple(
                (r["group"], r["slot"], r["ks"], r["rank"], r["place"], r["penalty"], r["current"])
                for r in rows
            ),
        )

    def draw_table(self, force: bool = False) -> None:
        """Redraw the table.

        Nothing is drawn when the values are unchanged or while a cell is being
        edited, so a periodic refresh never clears what is being typed.
        """
        signature = self.table_signature()
        if not force and signature == getattr(self, "_table_signature", None):
            return
        if getattr(self, "_editing", False):
            return
        self._table_signature = signature

        for child in self.grid_frame.winfo_children():
            child.destroy()

        for column, (_, label, width) in enumerate(self.COLUMNS):
            head = tk.Label(
                self.grid_frame, text=label, width=width // 8, relief="ridge",
                background="#eaecef",
            )
            head.grid(row=0, column=column, sticky="nsew")

        rows = self.record_rows()
        if self.sort_var.get() == "team":
            rows.sort(key=lambda r: (r["slot"], r["order"]))
            key = "slot"
        else:
            # ordered by round number, not by insertion; the offset stays first
            rows.sort(key=lambda r: (r["order"], r["slot"]))
            key = "group"

        line = 1
        previous = None
        for row in rows:
            if previous is not None and row[key] != previous:
                tk.Frame(self.grid_frame, height=2, background=self.LINE_BG).grid(
                    row=line, column=0, columnspan=len(self.COLUMNS), sticky="ew"
                )
                line += 1
            previous = row[key]

            for column, (name, _, width) in enumerate(self.COLUMNS):
                text, empty = self.cell_text(row, name)
                if name == "rank" and self.rank_clash(row):
                    background = self.DUP_BG
                elif empty:
                    background = self.EMPTY_BG
                else:
                    background = self.CURRENT_BG if row["current"] else "white"
                cell = tk.Label(
                    self.grid_frame, text=text, width=width // 8, relief="ridge",
                    background=background, anchor="center",
                )
                cell.grid(row=line, column=column, sticky="nsew")
                editable = ("ks", "rank", "penalty")
                if row["round_index"] is None:
                    editable = ("ks", "place", "penalty")  # the offset row takes placement points directly
                if name in editable:
                    cell.bind(
                        "<Double-1>",
                        lambda e, r=row, f=name, w=cell: self.edit_cell(r, f, w),
                    )
                elif name in ("group", "slot", "team"):
                    cell.bind("<Double-1>", lambda e, r=row: self.clear_row(r))
            line += 1

    def rank_clash(self, row: dict) -> bool:
        """Return True when the same rank occurs twice in that round."""
        index, rank = row["round_index"], row["rank"]
        if index is None or rank is None:
            return False
        if index >= len(self.match.rounds):
            return False
        return list(self.match.rounds[index].rank).count(rank) > 1

    def cell_text(self, row: dict, name: str) -> tuple[str, bool]:
        if name == "group":
            return row["group"], False
        if name == "slot":
            return str(row["slot"] + 1), False
        if name == "team":
            return self.match.team_names[row["slot"]], False
        value = row.get(name)
        if value is None:
            return "", True
        return f"{value:g}", False

    def edit_cell(self, row: dict, name: str, widget) -> None:
        """Edit one cell in place; an empty value clears it."""
        entry = tk.Entry(self.grid_frame)
        entry.insert(0, widget.cget("text"))
        entry.place(x=widget.winfo_x(), y=widget.winfo_y(),
                    width=widget.winfo_width(), height=widget.winfo_height())
        entry.focus_set()
        entry.select_range(0, "end")
        self._editing = True

        def release() -> None:
            guard = getattr(self, "_click_guard", None)
            if guard:
                self.root.unbind("<Button-1>", guard)
                self._click_guard = None
            self._editing = False

        def done(_=None) -> None:
            if not entry.winfo_exists():
                return
            text = entry.get().strip()
            entry.destroy()
            release()
            self.write_cell(row, name, text)

        def cancel(_=None) -> None:
            entry.destroy()
            release()

        entry.bind("<Return>", done)
        entry.bind("<FocusOut>", done)
        entry.bind("<Escape>", cancel)
        # Clicking a label does not move focus, so FocusOut never fires; a click
        # anywhere in the window ends the edit instead.
        self._click_guard = self.root.bind(
            "<Button-1>",
            lambda e: done() if e.widget is not entry else None,
            add="+",
        )

    def write_cell(self, row: dict, name: str, text: str) -> None:
        index, slot = row["round_index"], row["slot"]
        try:
            if text == "":
                self.match.clear_cell(index, slot, name)
            elif index is None:
                values = dict(self.match.offset.get(slot, {"ts": None, "ks": None}))
                ks = values.get("ks")
                place = None if values.get("ts") is None else values["ts"] - (ks or 0.0)
                if name == "ks":
                    ks = float(text)
                elif name == "place":
                    place = float(text)
                elif name == "rank":
                    place = float(points_of(int(text)))
                # keep the placement points and let TS follow the new KS
                ts = None if place is None else place + (ks or 0.0)
                self.match.set_offset(slot, ts, ks)
            elif name == "ks":
                self.match.edit_round(index, slot, ks=float(text))
            elif name == "rank":
                self.match.edit_round(index, slot, rank=int(text))
            elif name == "place":
                # a placement value is converted to the rank that awards it
                points = float(text)
                found = [r for r, v in PLACEMENT_POINTS.items() if abs(v - points) < 1e-6]
                if not found:
                    raise ScanError(f"{points}점에 해당하는 등수가 없습니다")
                self.match.edit_round(index, slot, rank=found[0])
            else:
                self.match.set_penalty(index, slot, float(text))
        except (ValueError, ScanError) as e:
            messagebox.showerror("값 수정", str(e))
            return
        self.refresh(force=True)

    def clear_row(self, row: dict) -> None:
        name = self.match.team_names[row["slot"]]
        if not messagebox.askyesno("행 지우기", f"{row['group']}의 {name} 행을 비울까요?"):
            return
        self.match.clear_row(row["round_index"], row["slot"])
        self.match.drop_empty_rounds()
        self.refresh(force=True)

    def new_match(self) -> None:
        if not messagebox.askyesno(
            "새 경기 시작", "모든 기록을 삭제합니다. 팀 이름은 유지됩니다. 계속할까요?"
        ):
            return
        self.match.new_match()
        self.grid_canvas.yview_moveto(0)
        self.round_no_var.set("1")
        self.timeline.reset()
        self.round_open = True
        self.say("새 경기 시작")
        self.refresh(force=True)
        self.grid_canvas.yview_moveto(0)

    def add_round_dialog(self) -> None:
        """Add one round by hand; a partial row is allowed."""
        text = self.round_no_var.get().strip()
        number = self.free_round_no(int(text) if text.isdigit() else 1)
        if number is None:
            return

        def apply(values) -> None:
            self.match.add_round(
                [v[0] for v in values],
                [None if v[1] is None else int(v[1]) for v in values],
                number=number,
            )
            if self.claimed_round is None:
                self.round_no_var.set(str(self.match.next_round_no))

        self._entry_dialog(
            f"{number}라운드 추가", ("KS", "등수"), apply,
            "입력하지 않은 칸은 빈 값으로 저장됩니다.",
        )

    def offset_dialog(self) -> None:
        """Add earlier results as a single offset row instead of separate rounds."""
        def apply(values) -> None:
            # the whole offset is overwritten; blank fields stay blank
            for slot, (ts, ks) in enumerate(values):
                self.match.set_offset(slot, ts, ks)

        self._entry_dialog(
            "오프셋 넣기", ("TS", "KS"), apply,
            "입력하지 않은 칸은 빈 값으로 저장됩니다.\n"
            "표에 오프셋이 이미 있으면 지금 값으로 덮어씁니다.",
        )

    def _entry_dialog(
        self, title: str, fields: tuple[str, str], apply, note: str = ""
    ) -> None:
        dialog = tk.Toplevel(self.root)
        dialog.title(title)
        ttk.Label(dialog, text=note, padding=8, justify="left").grid(
            row=0, column=0, columnspan=3, sticky="w"
        )
        ttk.Label(dialog, text="팀").grid(row=1, column=0)
        ttk.Label(dialog, text=fields[0]).grid(row=1, column=1)
        ttk.Label(dialog, text=fields[1]).grid(row=1, column=2)

        boxes = []
        for slot in range(TEAM_COUNT):
            ttk.Label(dialog, text=f"{slot + 1} {self.match.team_names[slot]}").grid(
                row=slot + 2, column=0, sticky="w", padx=6
            )
            first, second = tk.StringVar(), tk.StringVar()
            ttk.Entry(dialog, textvariable=first, width=8).grid(row=slot + 2, column=1)
            ttk.Entry(dialog, textvariable=second, width=8).grid(row=slot + 2, column=2)
            boxes.append((first, second))

        def submit() -> None:
            values = []
            for first, second in boxes:
                a = first.get().strip()
                b = second.get().strip()
                values.append((float(a) if a else None, float(b) if b else None))
            try:
                apply(values)
            except (ValueError, ScanError) as e:
                messagebox.showerror(title, str(e))
                return
            dialog.destroy()
            self.refresh()

        ttk.Button(dialog, text="넣기", command=submit).grid(
            row=TEAM_COUNT + 2, column=0, columnspan=3, pady=6
        )
        dialog.transient(self.root)
        dialog.grab_set()

    # ---- incoming readings -------------------------------------------
    def _pump(self) -> None:
        while not self.queue.empty():
            kind, value = self.queue.get()
            if kind == "문제":
                self.say(f"화면을 받지 못했습니다: {value}")
            else:
                self._handle(value)
        self.root.after(200, self._pump)

    def _handle(self, reading) -> None:
        if not self.round_open:
            return
        filled = fill_missing(reading, self.match.current.last_readings)
        if filled is None:
            if reading.missing:
                self.state_label.config(
                    text="못 읽은 팀번호: " + ",".join(str(i + 1) for i in reading.missing)
                )
            return

        repaired = repair_frame(filled, self.out_slots())
        for text in repaired.changed:
            self.say("보정: " + text)
        filled, notes = self.timeline.feed(filled, repaired.readings)
        for text in notes:
            self.say(text)

        result = self.match.current.observe(filled)
        for text in result.warnings:
            self.say(text)
        if not result.committed:
            return

        self.refresh()
        committed = self.match.current.committed
        if committed is not None and committed.finished:
            # a first place ends the round; wait for the operator to start the next
            self.say("1등이 확정되어 라운드를 확정하고 집계를 중지합니다")
            self.finish_round()

    # ---- publishing --------------------------------------------------
    def refresh(self, force: bool = False) -> None:
        self.draw_table(force)
        standings = self.match.standings()
        self.server.update(
            standings, round_no=len(self.match.rounds), out_slots=self.out_slots()
        )
        self.draw_preview(standings)

    def penalties_of(self, round_index: int) -> dict[int, float]:
        if round_index < len(self.match.penalties):
            return self.match.penalties[round_index]
        return {}

    def run(self) -> None:
        self.refresh()
        self.root.mainloop()


def main() -> None:
    from recognize import check_file_set

    check_file_set()
    App().run()


if __name__ == "__main__":
    main()
