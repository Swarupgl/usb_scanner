from __future__ import annotations

import json
import os
import time
from pathlib import Path
from tkinter import BOTH, END, LEFT, RIGHT, TOP, X, Y, Frame, Label, Text, Tk
from tkinter import ttk


def _default_repo_root() -> Path:
    here = Path(__file__).resolve()
    candidates = [here.parent, *here.parents]
    for candidate in candidates[:5]:
        if (
            (candidate / "hardware" / "pi" / "pi_usb_sanitizer.py").exists()
            or (candidate / "hardware" / "pi").is_dir()
            or (candidate / "requirements-pi.txt").exists()
        ):
            return candidate
    return here.parent


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _tail_lines(path: Path, max_lines: int = 30) -> list[str]:
    try:
        data = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return []
    lines = data.splitlines()
    return lines[-max_lines:]


THEME = {
    "bg": "#0b1020",
    "panel": "#111a33",
    "fg": "#e6eaf2",
    "muted": "#a8b0c2",
    "accent": "#6aa9ff",
    "good": "#2bd576",
    "warn": "#ffb020",
    "bad": "#ff4d5e",
}


def _safe_int(v, default: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        return default


def _format_state_badge(state: str, quarantined: int, errors: int) -> tuple[str, str]:
    """Return (badge_text, badge_color) based on state + counters."""
    state = (state or "idle").lower()

    if state in {"idle", "starting"}:
        return "READY • INSERT USB", THEME["accent"]
    if state == "scanning":
        return "SCANNING…", THEME["accent"]
    if state == "done":
        if errors > 0:
            return "DONE • CHECK ERRORS", THEME["warn"]
        if quarantined > 0:
            return "THREATS FOUND", THEME["bad"]
        return "CLEAN", THEME["good"]
    if state == "error":
        return "ERROR", THEME["bad"]

    return state.upper(), THEME["muted"]


def _render_log(text: Text, lines: list[str]) -> None:
    text.delete("1.0", END)
    for line in lines:
        tag = None
        if line.startswith("[quarantined"):
            tag = "bad"
        elif line.startswith("[error]"):
            tag = "warn"
        elif line.startswith("[clean]"):
            tag = "good"
        elif line.startswith("==="):
            tag = "muted"

        if tag:
            text.insert(END, line + "\n", tag)
        else:
            text.insert(END, line + "\n")
    text.see(END)


class KioskApp:
    def __init__(self, root: Tk, repo: Path):
        self.root = root
        self.repo = repo
        self.status_path = repo / "outputs" / "pi" / "logs" / "status.json"
        self.log_path = repo / "outputs" / "pi" / "logs" / "usb_sanitizer.log"

        root.title("USB Sanitizer")
        root.configure(background=THEME["bg"])

        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass

        # ttk theme overrides (keeps widgets consistent on Pi)
        style.configure("TFrame", background=THEME["bg"])
        style.configure("TLabel", background=THEME["bg"], foreground=THEME["fg"])
        style.configure(
            "Kiosk.TProgressbar",
            troughcolor=THEME["panel"],
            background=THEME["accent"],
            bordercolor=THEME["panel"],
            lightcolor=THEME["accent"],
            darkcolor=THEME["accent"],
        )

        self.frame = ttk.Frame(root, padding=16)
        self.frame.pack(fill=BOTH, expand=True)

        # Header row
        header = Frame(self.frame, bg=THEME["bg"])
        header.pack(side=TOP, fill=X)

        self.title = Label(
            header,
            text="USB Sanitizer",
            font=("Helvetica", 26, "bold"),
            bg=THEME["bg"],
            fg=THEME["fg"],
        )
        self.title.pack(side=LEFT, anchor="w")

        self.clock = Label(
            header,
            text="",
            font=("Helvetica", 12),
            bg=THEME["bg"],
            fg=THEME["muted"],
        )
        self.clock.pack(side=RIGHT, anchor="e")

        # Status badge
        self.badge = Label(
            self.frame,
            text="STARTING…",
            font=("Helvetica", 20, "bold"),
            bg=THEME["panel"],
            fg=THEME["fg"],
            padx=14,
            pady=10,
        )
        self.badge.pack(side=TOP, fill=X, pady=(14, 8))

        # Progress bar (indeterminate during scanning)
        self.pbar = ttk.Progressbar(self.frame, mode="indeterminate", style="Kiosk.TProgressbar")
        self.pbar.pack(side=TOP, fill=X)

        # Details + counters
        self.details_line = Label(
            self.frame,
            text="",
            font=("Helvetica", 12),
            bg=THEME["bg"],
            fg=THEME["muted"],
            justify="left",
            anchor="w",
        )
        self.details_line.pack(side=TOP, fill=X, pady=(8, 0))

        self.counts_line = Label(
            self.frame,
            text="",
            font=("Helvetica", 14, "bold"),
            bg=THEME["bg"],
            fg=THEME["fg"],
            anchor="w",
        )
        self.counts_line.pack(side=TOP, fill=X, pady=(6, 12))

        # Log viewer
        self.log_label = Label(
            self.frame,
            text="RECENT LOG",
            font=("Helvetica", 12, "bold"),
            bg=THEME["bg"],
            fg=THEME["muted"],
            anchor="w",
        )
        self.log_label.pack(side=TOP, fill=X)

        log_wrap = Frame(self.frame, bg=THEME["bg"])
        log_wrap.pack(side=TOP, fill=BOTH, expand=True, pady=(6, 0))

        self.log_text = Text(
            log_wrap,
            height=14,
            wrap="word",
            bg=THEME["panel"],
            fg=THEME["fg"],
            insertbackground=THEME["fg"],
            relief="flat",
            padx=10,
            pady=10,
            font=("Consolas", 11),
        )
        self.log_text.pack(side=LEFT, fill=BOTH, expand=True)

        scrollbar = ttk.Scrollbar(log_wrap, command=self.log_text.yview)
        scrollbar.pack(side=RIGHT, fill=Y)
        self.log_text.configure(yscrollcommand=scrollbar.set)

        # Log color tags
        self.log_text.tag_configure("good", foreground=THEME["good"])
        self.log_text.tag_configure("bad", foreground=THEME["bad"])
        self.log_text.tag_configure("warn", foreground=THEME["warn"])
        self.log_text.tag_configure("muted", foreground=THEME["muted"])

        self.root.bind("<Escape>", lambda _e: self.root.destroy())

        # Fullscreen by default for touchscreen
        self.root.attributes("-fullscreen", True)

        self._last_log_render = None
        self._last_state = None

    def tick(self) -> None:
        status = _read_json(self.status_path)
        state = status.get("state") or "idle"
        device = status.get("device")
        mountpoint = status.get("mountpoint")
        last_line = status.get("last_line")
        progress = status.get("progress") or {}

        message = status.get("message")

        now = time.strftime("%Y-%m-%d %H:%M:%S")
        self.clock.configure(text=now)

        detail_parts = []
        if device:
            detail_parts.append(f"device={device}")
        if mountpoint:
            detail_parts.append(f"mount={mountpoint}")
        if message:
            detail_parts.append(str(message)[:140])
        if last_line:
            detail_parts.append(f"last= {last_line[:110]}")
        self.details_line.configure(text=" | ".join(detail_parts))

        scanned = _safe_int(progress.get("scanned", 0) or 0)
        quarantined = _safe_int(progress.get("quarantined", 0) or 0)
        errors = _safe_int(progress.get("errors", 0) or 0)
        self.counts_line.configure(text=f"Scanned: {scanned}    Quarantined: {quarantined}    Errors: {errors}")

        badge_text, badge_color = _format_state_badge(state, quarantined=quarantined, errors=errors)
        self.badge.configure(text=badge_text, bg=badge_color)

        # Progress bar behavior
        state_norm = (state or "").lower()
        if state_norm == "scanning":
            if self._last_state != "scanning":
                try:
                    self.pbar.start(10)
                except Exception:
                    pass
        else:
            if self._last_state == "scanning":
                try:
                    self.pbar.stop()
                except Exception:
                    pass

        # Update log viewer (cheap tail)
        log_lines = _tail_lines(self.log_path, max_lines=60)
        log_tail = "\n".join(log_lines)
        if log_tail != self._last_log_render:
            _render_log(self.log_text, log_lines)
            self._last_log_render = log_tail

        self._last_state = state_norm

        self.root.after(750, self.tick)


def main() -> int:
    repo = _default_repo_root()
    # Allow override (useful if you copy scripts elsewhere)
    env_repo = os.environ.get("USB_SANITIZER_REPO")
    if env_repo:
        repo = Path(env_repo).expanduser().resolve()

    root = Tk()
    app = KioskApp(root, repo=repo)
    app.tick()
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
