#!/usr/bin/env python3
"""Build a Search-R1 training metric dashboard from the current A100 PPO log."""

from __future__ import annotations

import csv
import math
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont


REMOTE_HOST = "nuist-a100-via-windows"
REMOTE_LOG = (
    "/mnt/xu/xu_exp/Search-R1/logs/"
    "phase6-ppo-qwen2.5-3b-instruct-300step-nq256-offfalse-safe-"
    "20260628_ppo_grpo300_nq256_offfalse_054206.log"
)

OUT_DIR = Path(__file__).resolve().parent
CSV_PATH = OUT_DIR / "phase6_metric_dashboard.csv"
PNG_PATH = OUT_DIR / "fig_phase6_metric_dashboard.png"

STEP_RE = re.compile(r"\bstep:(\d+)\s*-\s*(.*)")
KV_RE = re.compile(
    r"([A-Za-z0-9_./-]+):"
    r"(-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|nan|inf|-inf)"
)
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


COLORS = {
    "blue": "#0072B2",
    "sky": "#56B4E9",
    "green": "#009E73",
    "orange": "#E69F00",
    "red": "#D55E00",
    "pink": "#CC79A7",
    "dark": "#243447",
    "gray": "#7A869A",
    "grid": "#D7DBE3",
    "panel": "#FFFFFF",
    "bg": "#F6F7F9",
}


@dataclass(frozen=True)
class SeriesSpec:
    key: str
    label: str
    color: str
    smooth: int = 7
    dashed: bool = False
    markers_only: bool = False


@dataclass(frozen=True)
class PanelSpec:
    title: str
    ylabel: str
    series: tuple[SeriesSpec, ...]
    y_bounds: tuple[float, float] | None = None
    note: str | None = None


def fetch_log() -> str:
    proc = subprocess.run(
        [
            "ssh",
            "-o",
            "ConnectTimeout=25",
            "-o",
            "ServerAliveInterval=10",
            "-o",
            "ServerAliveCountMax=2",
            REMOTE_HOST,
            "cat",
            REMOTE_LOG,
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise SystemExit(f"failed to fetch remote log: exit {proc.returncode}")
    return proc.stdout


def parse_metrics(text: str) -> dict[int, dict[str, float]]:
    by_step: dict[int, dict[str, float]] = {}
    for raw_line in text.splitlines():
        line = ANSI_RE.sub("", raw_line)
        match = STEP_RE.search(line)
        if not match:
            continue
        step = int(match.group(1))
        rest = match.group(2)
        step_metrics = by_step.setdefault(step, {})
        for key, value_text in KV_RE.findall(rest):
            try:
                value = float(value_text)
            except ValueError:
                continue
            if math.isfinite(value):
                step_metrics[key] = value
    return by_step


def write_csv(by_step: dict[int, dict[str, float]]) -> list[str]:
    all_keys = sorted({key for metrics in by_step.values() for key in metrics})
    with CSV_PATH.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", *all_keys])
        for step in sorted(by_step):
            row = [step]
            metrics = by_step[step]
            row.extend(metrics.get(key, "") for key in all_keys)
            writer.writerow(row)
    return all_keys


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Helvetica.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        if path and os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


FONT_TITLE = load_font(24, bold=True)
FONT_SUBTITLE = load_font(16)
FONT_PANEL = load_font(17, bold=True)
FONT_AXIS = load_font(13)
FONT_SMALL = load_font(12)
FONT_LEGEND = load_font(13)


def hex_to_rgb(color: str) -> tuple[int, int, int]:
    color = color.lstrip("#")
    return tuple(int(color[i : i + 2], 16) for i in (0, 2, 4))


def lighten(color: str, alpha: float = 0.18) -> tuple[int, int, int, int]:
    r, g, b = hex_to_rgb(color)
    return (r, g, b, int(255 * alpha))


def smooth_values(points: list[tuple[int, float]], window: int) -> list[tuple[int, float]]:
    if window <= 1 or len(points) <= 2:
        return points
    result: list[tuple[int, float]] = []
    half = window // 2
    for i, (step, _) in enumerate(points):
        lo = max(0, i - half)
        hi = min(len(points), i + half + 1)
        vals = [v for _, v in points[lo:hi]]
        result.append((step, sum(vals) / len(vals)))
    return result


def series_points(by_step: dict[int, dict[str, float]], key: str) -> list[tuple[int, float]]:
    return [
        (step, metrics[key])
        for step, metrics in sorted(by_step.items())
        if key in metrics and math.isfinite(metrics[key])
    ]


def nice_ticks(lo: float, hi: float, count: int = 5) -> list[float]:
    if not math.isfinite(lo) or not math.isfinite(hi):
        return [0.0, 1.0]
    if hi <= lo:
        pad = 1.0 if hi == 0 else abs(hi) * 0.1
        lo -= pad
        hi += pad
    span = hi - lo
    raw = span / max(1, count - 1)
    exp = math.floor(math.log10(raw)) if raw > 0 else 0
    base = 10**exp
    frac = raw / base
    if frac <= 1:
        step = base
    elif frac <= 2:
        step = 2 * base
    elif frac <= 5:
        step = 5 * base
    else:
        step = 10 * base
    start = math.floor(lo / step) * step
    ticks = []
    value = start
    while value <= hi + step * 0.5:
        if value >= lo - step * 0.5:
            ticks.append(value)
        value += step
    return ticks[: count + 2]


def format_tick(value: float) -> str:
    if abs(value) >= 100:
        return f"{value:.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


def draw_line(
    draw: ImageDraw.ImageDraw,
    points: list[tuple[int, float]],
    x_map,
    y_map,
    color: str,
    width: int = 3,
    dashed: bool = False,
) -> None:
    if len(points) < 2:
        return
    xy = [(x_map(step), y_map(value)) for step, value in points]
    if not dashed:
        draw.line(xy, fill=color, width=width, joint="curve")
        return
    dash = 8
    gap = 6
    for (x1, y1), (x2, y2) in zip(xy, xy[1:]):
        dx, dy = x2 - x1, y2 - y1
        dist = max(1.0, math.hypot(dx, dy))
        pos = 0.0
        while pos < dist:
            end = min(dist, pos + dash)
            sx = x1 + dx * pos / dist
            sy = y1 + dy * pos / dist
            ex = x1 + dx * end / dist
            ey = y1 + dy * end / dist
            draw.line([(sx, sy), (ex, ey)], fill=color, width=width)
            pos += dash + gap


def draw_panel(
    base: Image.Image,
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    panel: PanelSpec,
    by_step: dict[int, dict[str, float]],
    max_step: int,
) -> None:
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=12, fill=COLORS["panel"], outline="#DDE2EA", width=1)
    draw.text((x0 + 18, y0 + 14), panel.title, fill=COLORS["dark"], font=FONT_PANEL)
    if panel.note:
        draw.text((x0 + 18, y0 + 39), panel.note, fill=COLORS["gray"], font=FONT_SMALL)

    plot_left = x0 + 68
    plot_right = x1 - 24
    plot_top = y0 + 83
    plot_bottom = y1 - 45

    all_points: list[tuple[int, float]] = []
    for spec in panel.series:
        all_points.extend(series_points(by_step, spec.key))
    if not all_points:
        draw.text((plot_left, plot_top + 30), "metric not found in log", fill=COLORS["gray"], font=FONT_AXIS)
        return

    if panel.y_bounds:
        y_min, y_max = panel.y_bounds
    else:
        values = [value for _, value in all_points]
        y_min, y_max = min(values), max(values)
        span = y_max - y_min
        pad = max(span * 0.10, 0.02 if y_max <= 1.5 else 1.0)
        y_min -= pad
        y_max += pad
    if y_max <= y_min:
        y_max = y_min + 1

    def x_map(step: int) -> float:
        return plot_left + (plot_right - plot_left) * step / max(1, max_step)

    def y_map(value: float) -> float:
        clipped = min(y_max, max(y_min, value))
        return plot_bottom - (plot_bottom - plot_top) * (clipped - y_min) / (y_max - y_min)

    for tick in nice_ticks(y_min, y_max, count=5):
        y = y_map(tick)
        draw.line([(plot_left, y), (plot_right, y)], fill=COLORS["grid"], width=1)
        draw.text((x0 + 13, y - 7), format_tick(tick), fill=COLORS["gray"], font=FONT_SMALL)
    for step in range(0, max_step + 1, 50):
        x = x_map(step)
        draw.line([(x, plot_top), (x, plot_bottom)], fill="#EEF1F6", width=1)
        draw.text((x - 11, plot_bottom + 10), str(step), fill=COLORS["gray"], font=FONT_SMALL)
    for step in (100, 200, 300):
        if step <= max_step:
            x = x_map(step)
            draw.line([(x, plot_top), (x, plot_bottom)], fill="#C4CAD5", width=1)

    draw.line([(plot_left, plot_bottom), (plot_right, plot_bottom)], fill="#9BA5B5", width=1)
    draw.line([(plot_left, plot_top), (plot_left, plot_bottom)], fill="#9BA5B5", width=1)
    draw.text((x0 + 18, y0 + 58), panel.ylabel, fill=COLORS["gray"], font=FONT_SMALL)
    draw.text((plot_right - 28, plot_bottom + 26), "step", fill=COLORS["gray"], font=FONT_SMALL)

    legend_x = x1 - 260
    legend_y = y0 + 17
    for idx, spec in enumerate(panel.series):
        y = legend_y + idx * 18
        color = COLORS.get(spec.color, spec.color)
        draw.line([(legend_x, y + 7), (legend_x + 22, y + 7)], fill=color, width=3)
        if spec.markers_only:
            draw.ellipse((legend_x + 8, y + 3, legend_x + 16, y + 11), fill=color)
        draw.text((legend_x + 29, y), spec.label, fill=COLORS["dark"], font=FONT_LEGEND)

    overlay = Image.new("RGBA", base.size, (255, 255, 255, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    for spec in panel.series:
        points = series_points(by_step, spec.key)
        if not points:
            continue
        color = COLORS.get(spec.color, spec.color)
        if spec.markers_only:
            for step, value in points:
                x = x_map(step)
                y = y_map(value)
                overlay_draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill=hex_to_rgb(color) + (235,), outline=(255, 255, 255, 255), width=2)
            continue
        if len(points) > 2:
            raw_xy = [(x_map(step), y_map(value)) for step, value in points]
            overlay_draw.line(raw_xy, fill=lighten(color, 0.22), width=2)
        smoothed = smooth_values(points, spec.smooth)
        draw_line(overlay_draw, smoothed, x_map, y_map, color, width=4, dashed=spec.dashed)
    base.alpha_composite(overlay)


def latest_value(by_step: dict[int, dict[str, float]], key: str) -> tuple[int, float] | None:
    for step in sorted(by_step, reverse=True):
        if key in by_step[step]:
            return step, by_step[step][key]
    return None


def best_value(by_step: dict[int, dict[str, float]], key: str) -> tuple[int, float] | None:
    candidates = [(step, metrics[key]) for step, metrics in by_step.items() if key in metrics]
    return max(candidates, key=lambda item: item[1]) if candidates else None


def draw_dashboard(by_step: dict[int, dict[str, float]], all_keys: Iterable[str]) -> None:
    max_step = max(by_step) if by_step else 1
    image = Image.new("RGBA", (1900, 1880), COLORS["bg"])
    draw = ImageDraw.Draw(image)

    title = "Search-R1 Phase6 PPO Metric Dashboard"
    subtitle = (
        "Qwen2.5-3B-Instruct, 300-step PPO run, NQ-256 validation every 100 steps; "
        "raw lines are faint, 7-step moving averages are bold."
    )
    draw.text((58, 36), title, fill=COLORS["dark"], font=FONT_TITLE)
    draw.text((58, 72), subtitle, fill=COLORS["gray"], font=FONT_SUBTITLE)

    latest_step = max_step
    latest_score = latest_value(by_step, "critic/score/mean")
    latest_reward = latest_value(by_step, "critic/rewards/mean")
    latest_finish = latest_value(by_step, "env/finish_ratio")
    latest_valid = latest_value(by_step, "env/ratio_of_valid_action")
    best_val = best_value(by_step, "val/test_score/nq")
    summary_parts = [f"latest logged step: {latest_step}"]
    if best_val:
        summary_parts.append(f"best NQ EM: {best_val[1]:.3f} @ step {best_val[0]}")
    if latest_score:
        summary_parts.append(f"latest train score: {latest_score[1]:.3f}")
    if latest_reward:
        summary_parts.append(f"latest PPO reward: {latest_reward[1]:.3f}")
    if latest_finish:
        summary_parts.append(f"finish ratio: {latest_finish[1]:.3f}")
    if latest_valid:
        summary_parts.append(f"valid action ratio: {latest_valid[1]:.3f}")
    draw.text((58, 103), " | ".join(summary_parts), fill=COLORS["dark"], font=FONT_SUBTITLE)

    panels = [
        PanelSpec(
            "Correctness Signal vs PPO Reward",
            "score / reward",
            (
                SeriesSpec("critic/score/mean", "train EM score", "blue"),
                SeriesSpec("critic/rewards/mean", "post-KL PPO reward", "orange"),
                SeriesSpec("val/test_score/nq", "NQ validation EM", "green", markers_only=True),
            ),
            y_bounds=(0.0, 0.65),
            note="score is rule EM; reward is the value PPO optimizes after KL shaping",
        ),
        PanelSpec(
            "Validation NQ EM",
            "EM",
            (SeriesSpec("val/test_score/nq", "NQ-256 EM", "green", markers_only=True),),
            y_bounds=(0.0, 0.40),
            note="held-out NQ subset; checkpoint selection should follow this, not train reward alone",
        ),
        PanelSpec(
            "Response Length",
            "tokens",
            (
                SeriesSpec("response_length/mean", "mean response length", "pink"),
                SeriesSpec("response_length/max", "max response length", "gray", dashed=True),
            ),
            note="longer outputs cost more and often signal search/format loops after drift",
        ),
        PanelSpec(
            "Search Format and Completion",
            "ratio",
            (
                SeriesSpec("env/finish_ratio", "finished answers", "green"),
                SeriesSpec("env/ratio_of_valid_action", "valid actions", "red"),
            ),
            y_bounds=(0.0, 1.05),
            note="format health: can the rollout end cleanly and issue valid search actions?",
        ),
        PanelSpec(
            "Search Action Budget",
            "count",
            (
                SeriesSpec("env/number_of_actions/mean", "all actions", "blue"),
                SeriesSpec("env/number_of_valid_search", "valid searches", "orange"),
            ),
            y_bounds=(0.0, 4.2),
            note="actions near the turn limit with fewer valid searches suggest inefficient search",
        ),
        PanelSpec(
            "KL Drift",
            "KL",
            (
                SeriesSpec("critic/kl", "signed KL metric", "red"),
                SeriesSpec("critic/kl_coeff", "KL coefficient", "gray", dashed=True),
            ),
            note="negative signed KL can make post-KL reward look better than pure EM score",
        ),
        PanelSpec(
            "Policy Entropy and PPO Clipping",
            "value",
            (
                SeriesSpec("actor/entropy_loss", "entropy", "sky"),
                SeriesSpec("actor/pg_clipfrac", "clip fraction", "orange"),
            ),
            note="falling entropy means less exploration; high clip fraction means stronger PPO constraint hits",
        ),
        PanelSpec(
            "Value Model Fit",
            "value",
            (
                SeriesSpec("critic/vf_explained_var", "explained variance", "green"),
                SeriesSpec("critic/vpred_mean", "value prediction", "blue", dashed=True),
            ),
            y_bounds=(-0.1, 0.35),
            note="critic quality matters for stable advantages under sparse EM reward",
        ),
    ]

    left = 58
    top = 150
    panel_w = 870
    panel_h = 390
    gap_x = 44
    gap_y = 34
    for i, panel in enumerate(panels):
        row, col = divmod(i, 2)
        x0 = left + col * (panel_w + gap_x)
        y0 = top + row * (panel_h + gap_y)
        draw_panel(image, draw, (x0, y0, x0 + panel_w, y0 + panel_h), panel, by_step, max_step)

    footer = (
        f"source: {REMOTE_HOST}:{REMOTE_LOG} | CSV keeps {len(list(all_keys))} logged metrics"
    )
    draw.text((58, 1842), footer, fill=COLORS["gray"], font=FONT_SMALL)
    rgb = image.convert("RGB")
    rgb.save(PNG_PATH, quality=95)


def main() -> None:
    text = fetch_log()
    by_step = parse_metrics(text)
    if not by_step:
        raise SystemExit("no step metrics parsed")
    all_keys = write_csv(by_step)
    draw_dashboard(by_step, all_keys)
    print(f"parsed_steps={len(by_step)} max_step={max(by_step)} metrics={len(all_keys)}")
    for key in [
        "val/test_score/nq",
        "critic/score/mean",
        "critic/rewards/mean",
        "response_length/mean",
        "env/finish_ratio",
        "env/ratio_of_valid_action",
        "env/number_of_actions/mean",
        "critic/kl",
        "critic/vf_explained_var",
    ]:
        latest = latest_value(by_step, key)
        best = best_value(by_step, key)
        if latest:
            best_text = f" best={best[1]:.6g}@{best[0]}" if best else ""
            print(f"{key}: latest={latest[1]:.6g}@{latest[0]}{best_text}")
    print(f"csv={CSV_PATH}")
    print(f"png={PNG_PATH}")


if __name__ == "__main__":
    main()
