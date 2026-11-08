"""A tiny framework for narrated explainer videos (matplotlib frames + macOS speech + ffmpeg).

A video is a list of scenes. Each scene has narration (a list of sentences) and a
``draw(fig, t)`` function, where ``t`` runs from 0 to 1 over the scene. Every sentence is
spoken separately with the macOS ``say`` command, so captions are timed exactly.

    python tools/videos/lstn.py OUT_DIR [--voice Samantha] [--rate 175]

Outputs ``OUT_DIR/<name>.mp4`` (1920x1080, H.264/AAC) and ``OUT_DIR/<name>.srt`` (captions).
"""

from __future__ import annotations

import argparse
import math
import subprocess
import tempfile
import textwrap
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

W, H, FPS = 1920, 1080, 24
FONT = "Avenir Next"
INK, MUTED, LIGHT = "#1f2328", "#6b6a65", "#e1e0d9"
BLUE, ORANGE, RED, GREEN, PURPLE, GOLD, GRAY = ("#2a78d6", "#eb6834", "#c4473a", "#3a9a5b", "#8e6bbf",
                                               "#d9a62b", "#898781")
REPO = "github.com/ruikelyu/indflex"

plt.rcParams.update({
    # DejaVu Sans supplies symbols that Avenir Next lacks (→, ∈)
    "font.family": [FONT, "DejaVu Sans"], "font.size": 22, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 26, "axes.titleweight": "medium", "axes.labelsize": 22, "legend.frameon": False,
    "legend.fontsize": 20, "axes.grid": True, "grid.color": LIGHT, "grid.linewidth": 1,
})


@dataclass
class Scene:
    narration: list  # sentences; a (caption, spoken) pair when the spoken form should differ
    draw: Callable[[plt.Figure, float], None]
    static: bool = False  # draw once and hold (faster for still scenes)
    pad: float = 0.6  # pause after each sentence (s)
    tail: float = 0.8  # extra hold at the end of the scene (s)


@dataclass
class Video:
    name: str
    title: str
    scenes: list[Scene] = field(default_factory=list)


# ------------------------------------------------------------------ drawing helpers
def plain_log_axes(ax, xticks=None, yticks=None):
    """Log axes labelled with plain numbers (1, 10, 100) instead of powers of ten."""
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

    fmt = FuncFormatter(lambda v, _: f"{v:g}")
    for axis, ticks in ((ax.xaxis, xticks), (ax.yaxis, yticks)):
        if ticks is not None:
            axis.set_major_locator(FixedLocator(ticks))
            axis.set_minor_locator(NullLocator())
            axis.set_major_formatter(fmt)


def ease(t: float) -> float:
    """Smooth 0→1 easing."""
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


def stage(t: float, start: float, end: float) -> float:
    """Progress of a sub-animation running from ``start`` to ``end`` (fractions of the scene)."""
    return ease((t - start) / max(end - start, 1e-9))


def heading(fig, title: str, subtitle: str = ""):
    fig.text(0.06, 0.92, title, fontsize=44, fontweight="demibold", color=INK, va="center")
    if subtitle:
        fig.text(0.06, 0.865, subtitle, fontsize=26, color=MUTED, va="center")


def footer(fig, text: str = REPO):
    fig.text(0.94, 0.965, text, fontsize=18, color=MUTED, ha="right", va="center")


def bullet_list(fig, items, x=0.08, y=0.72, dy=0.085, size=30, reveal: float = 1.0):
    """Bullets that appear one by one as ``reveal`` goes 0→1."""
    n = len(items)
    for i, item in enumerate(items):
        a = min(max(reveal * n - i, 0.0), 1.0)
        if a <= 0:
            continue
        fig.text(x, y - i * dy, "•", fontsize=size, color=BLUE, alpha=a, va="center")
        fig.text(x + 0.025, y - i * dy, item, fontsize=size, color=INK, alpha=a, va="center")


def title_card(paper: str, title: str, citation: str, tagline: str = "indflex · industrial demand flexibility"):
    size = min(58, int(58 * 46 / max(len(title), 1)))  # long titles shrink to stay on one line

    def draw(fig, t):
        a = stage(t, 0, 0.25)
        fig.text(0.5, 0.62, title, fontsize=size, fontweight="demibold", color=INK, ha="center", va="center", alpha=a)
        fig.text(0.5, 0.5, paper, fontsize=30, color=BLUE, ha="center", va="center", alpha=a)
        fig.text(0.5, 0.42, citation, fontsize=22, color=MUTED, ha="center", va="center", alpha=stage(t, 0.15, 0.4))
        fig.text(0.5, 0.27, tagline, fontsize=20, color=MUTED, ha="center", va="center", alpha=stage(t, 0.2, 0.5))
    return draw


def outro_card(lines: list[str]):
    def draw(fig, t):
        heading(fig, "Try it yourself")
        bullet_list(fig, lines, y=0.68, reveal=stage(t, 0, 0.6), size=30)
        fig.text(0.5, 0.18, REPO, fontsize=40, color=BLUE, ha="center", va="center", alpha=stage(t, 0.3, 0.7),
                 fontweight="medium")
    return draw


def draw_caption(fig, text: str):
    text = text.replace("$", r"\$")  # a pair of dollar signs would otherwise start matplotlib math mode
    lines = textwrap.wrap(text, 92)
    y = 0.035 + 0.04 * (len(lines) - 1)
    box_h = 0.05 + 0.04 * (len(lines) - 1)
    fig.patches.append(plt.Rectangle((0, 0), 1, box_h + 0.025, transform=fig.transFigure, color="#ffffff",
                                     alpha=0.92, zorder=50))
    for i, line in enumerate(lines):
        fig.text(0.5, y - 0.04 * i + 0.012, line, fontsize=24, color=INK, ha="center", va="center", zorder=51)


# ------------------------------------------------------------------ rendering
def speak(text: str, path: Path, voice: str, rate: int) -> float:
    subprocess.run(["say", "-v", voice, "-r", str(rate), "-o", str(path), text], check=True)
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                         check=True, capture_output=True, text=True)
    return float(out.stdout.strip())


def _srt_time(s: float) -> str:
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{int(h):02d}:{int(m):02d}:{int(sec):02d},{int(round((sec % 1) * 1000)):03d}"


def render(video: Video, out_dir: Path, voice: str = "Samantha", rate: int = 175, captions: bool = True) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=f"{video.name}_"))
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    srt, clock, parts = [], 0.0, []

    for si, scene in enumerate(video.scenes):
        # 1. narration: one audio clip per sentence, with pauses, joined into one track
        timeline, wavs = [], []
        t0 = 0.0
        for k, item in enumerate(scene.narration):
            sentence, spoken = item if isinstance(item, tuple) else (item, item)
            wav = work / f"s{si:02d}_{k:02d}.aiff"
            dur = speak(spoken, wav, voice, rate)
            timeline.append((t0, t0 + dur, sentence))
            wavs.append((wav, dur))
            t0 += dur + scene.pad
        total = t0 + scene.tail
        audio = work / f"s{si:02d}.wav"
        inputs, filters = [], []
        for k, (wav, dur) in enumerate(wavs):
            inputs += ["-i", str(wav)]
            filters.append(f"[{k}:a]aresample=44100,apad=pad_dur={scene.pad}[a{k}]")
        concat = "".join(f"[a{k}]" for k in range(len(wavs)))
        filters.append(f"{concat}concat=n={len(wavs)}:v=0:a=1,apad=whole_dur={total}[out]")
        subprocess.run(["ffmpeg", "-y", "-v", "error", *inputs, "-filter_complex", ";".join(filters), "-map", "[out]",
                        "-ac", "2", str(audio)], check=True)

        # 2. frames, piped to ffmpeg together with the audio
        n = math.ceil(total * FPS)
        part = work / f"s{si:02d}.mp4"
        enc = subprocess.Popen(
            ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{W}x{H}", "-r", str(FPS),
             "-i", "-", "-i", str(audio), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-preset", "medium",
             "-c:a", "aac", "-b:a", "160k", "-shortest", str(part)], stdin=subprocess.PIPE)
        cache = {}
        for f in range(n):
            sec = f / FPS
            t = f / max(n - 1, 1)
            sentence = next((s for a, b, s in timeline if a <= sec < b + scene.pad), timeline[-1][2])
            key = ("static", sentence) if scene.static else None
            if key and key in cache:
                enc.stdin.write(cache[key])
                continue
            fig.clf()
            fig.patches.clear()
            scene.draw(fig, 1.0 if scene.static else t)
            if captions:
                draw_caption(fig, sentence)
            fig.canvas.draw()
            frame = bytes(fig.canvas.buffer_rgba())
            if key:
                cache[key] = frame
            enc.stdin.write(frame)
        enc.stdin.close()
        enc.wait()
        parts.append(part)
        for a, b, s in timeline:
            srt.append((clock + a, clock + b, s))
        clock += n / FPS
        print(f"  scene {si + 1}/{len(video.scenes)}: {total:.1f} s", flush=True)

    # 3. join the scenes
    lst = work / "parts.txt"
    lst.write_text("".join(f"file '{p}'\n" for p in parts))
    mp4 = out_dir / f"{video.name}.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy",
                    str(mp4)], check=True)
    (out_dir / f"{video.name}.srt").write_text(
        "".join(f"{i + 1}\n{_srt_time(a)} --> {_srt_time(b)}\n{s}\n\n" for i, (a, b, s) in enumerate(srt)))
    (out_dir / f"{video.name}_script.md").write_text(
        f"# {video.title}\n\n" + "\n\n".join(" ".join(i[0] if isinstance(i, tuple) else i for i in sc.narration)
                                               for sc in video.scenes) + "\n")
    plt.close(fig)
    print(f"wrote {mp4} ({clock:.0f} s)")
    return mp4


def main(video_factory: Callable[[], Video]):
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--voice", default="Samantha")
    ap.add_argument("--rate", type=int, default=175)
    ap.add_argument("--no-captions", action="store_true")
    args = ap.parse_args()
    render(video_factory(), Path(args.out_dir), args.voice, args.rate, not args.no_captions)


__all__ = ["Scene", "Video", "render", "main", "plain_log_axes", "ease", "stage", "heading", "footer", "bullet_list", "title_card",
           "outro_card", "np", "plt", "INK", "MUTED", "LIGHT", "BLUE", "ORANGE", "RED", "GREEN", "PURPLE", "GOLD",
           "GRAY", "REPO"]
