#!/usr/bin/env python3
"""Plot Search-R1 training reward comparison from A100 console logs."""

from __future__ import annotations

import csv
import re
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
SSH_HOST = "nuist-a100-via-windows"

RUNS = [
    {
        "label": "Previous phase5 PPO 500-step",
        "short": "previous_phase5",
        "path": "/mnt/xu/xu_exp/Search-R1/logs/phase5-v02-qwen2.5-3b-instruct-500step-3turn-b512-gpu04-nqfull-20260628_v02m3_gpu04_500_nqfull_004150.outer.log",
        "color": "#0072B2",
    },
    {
        "label": "Current phase6 PPO offload=false",
        "short": "current_phase6",
        "path": "/mnt/xu/xu_exp/Search-R1/logs/phase6-ppo-qwen2.5-3b-instruct-300step-nq256-offfalse-safe-20260628_ppo_grpo300_nq256_offfalse_054206.log",
        "color": "#D55E00",
    },
]

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
REWARD_RE = re.compile(r"step:(\d+).*?critic/rewards/mean:([-+0-9.]+)")


def fetch_log(path: str) -> str:
    result = subprocess.run(
        ["ssh", SSH_HOST, f"cat {path}"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout


def parse_rewards(text: str) -> list[tuple[int, float]]:
    rows: list[tuple[int, float]] = []
    seen: set[int] = set()
    for raw_line in text.splitlines():
        line = ANSI_RE.sub("", raw_line)
        match = REWARD_RE.search(line)
        if not match:
            continue
        step = int(match.group(1))
        reward = float(match.group(2))
        if step in seen:
            rows = [(s, r) for s, r in rows if s != step]
        seen.add(step)
        rows.append((step, reward))
    return sorted(rows)


def moving_average(values: np.ndarray, window: int = 10) -> np.ndarray:
    if len(values) < window:
        return values.copy()
    kernel = np.ones(window) / window
    valid = np.convolve(values, kernel, mode="valid")
    pad = np.full(window - 1, np.nan)
    return np.concatenate([pad, valid])


def write_csv(data: dict[str, list[tuple[int, float]]]) -> Path:
    out = ROOT / "training_reward_compare_phase5_phase6.csv"
    max_step = max((step for rows in data.values() for step, _ in rows), default=0)
    by_run = {
        name: {step: reward for step, reward in rows}
        for name, rows in data.items()
    }
    with out.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", *data.keys()])
        for step in range(1, max_step + 1):
            writer.writerow([step, *[by_run[name].get(step, "") for name in data]])
    return out


def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))


def blend(color: tuple[int, int, int], alpha: float, bg: tuple[int, int, int] = (255, 255, 255)) -> tuple[int, int, int]:
    return tuple(int(alpha * c + (1 - alpha) * b) for c, b in zip(color, bg))


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
        "/Library/Fonts/Arial.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_polyline(
    draw: ImageDraw.ImageDraw,
    points: list[tuple[float, float]],
    color: tuple[int, int, int],
    width: int,
) -> None:
    clean = [(int(round(x)), int(round(y))) for x, y in points if np.isfinite(x) and np.isfinite(y)]
    if len(clean) >= 2:
        draw.line(clean, fill=color, width=width, joint="curve")


def plot(data: dict[str, list[tuple[int, float]]]) -> Path:
    scale = 2
    width, height = 1500, 760
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    title_font = load_font(34, bold=True)
    label_font = load_font(26, bold=True)
    tick_font = load_font(21)
    small_font = load_font(20)
    legend_font = load_font(22)

    left, right, top, bottom = 150, 70, 95, 115
    plot_w = width - left - right
    plot_h = height - top - bottom

    max_seen = max((step for rows in data.values() for step, _ in rows), default=100)
    x_max = int(np.ceil((max_seen + 8) / 20) * 20)
    y_min, y_max = 0.15, 0.45

    def x_to_px(x: float) -> float:
        return left + (x / x_max) * plot_w

    def y_to_px(y: float) -> float:
        return top + (y_max - y) / (y_max - y_min) * plot_h

    # Grid and axes.
    grid_color = (224, 229, 235)
    axis_color = (45, 52, 60)
    for x in range(0, x_max + 1, 20):
        px = x_to_px(x)
        draw.line([(px, top), (px, top + plot_h)], fill=grid_color, width=1)
        draw.text((px - 12, top + plot_h + 16), str(x), fill=(70, 76, 84), font=tick_font)
    for y in np.arange(y_min, y_max + 0.001, 0.05):
        py = y_to_px(float(y))
        draw.line([(left, py), (left + plot_w, py)], fill=grid_color, width=1)
        draw.text((left - 70, py - 12), f"{y:.2f}", fill=(70, 76, 84), font=tick_font)
    draw.line([(left, top), (left, top + plot_h)], fill=axis_color, width=2)
    draw.line([(left, top + plot_h), (left + plot_w, top + plot_h)], fill=axis_color, width=2)

    # Curves.
    for run in RUNS:
        rows = data.get(run["short"], [])
        if not rows:
            continue
        steps = np.array([s for s, _ in rows], dtype=float)
        rewards = np.array([r for _, r in rows], dtype=float)
        color = hex_to_rgb(run["color"])
        raw_points = [(x_to_px(s), y_to_px(r)) for s, r in zip(steps, rewards)]
        smooth = moving_average(rewards, window=10)
        smooth_points = [(x_to_px(s), y_to_px(r)) for s, r in zip(steps, smooth)]

        draw_polyline(draw, raw_points, blend(color, 0.23), width=2)
        draw_polyline(draw, smooth_points, color, width=5)
        last_x, last_y = raw_points[-1]
        draw.ellipse((last_x - 7, last_y - 7, last_x + 7, last_y + 7), fill=color)
        draw.text(
            (min(last_x + 13, left + plot_w - 105), last_y - 15),
            f"{int(steps[-1])}: {rewards[-1]:.3f}",
            fill=color,
            font=small_font,
        )

    # Titles and labels.
    title = "Search-R1 PPO Train Reward Comparison"
    title_bbox = draw.textbbox((0, 0), title, font=title_font)
    draw.text(((width - (title_bbox[2] - title_bbox[0])) / 2, 30), title, fill=(25, 30, 35), font=title_font)
    xlabel = "Training Step"
    xlabel_bbox = draw.textbbox((0, 0), xlabel, font=label_font)
    draw.text((left + (plot_w - (xlabel_bbox[2] - xlabel_bbox[0])) / 2, height - 62), xlabel, fill=axis_color, font=label_font)
    ylabel = "Train Reward"
    label_img = Image.new("RGBA", (260, 48), (255, 255, 255, 0))
    label_draw = ImageDraw.Draw(label_img)
    label_draw.text((0, 0), ylabel, fill=axis_color, font=label_font)
    label_img = label_img.rotate(90, expand=True)
    img.paste(label_img, (35, top + plot_h // 2 - label_img.height // 2), label_img)

    # Legend.
    legend_x, legend_y = left + plot_w - 515, top + plot_h - 115
    draw.rounded_rectangle((legend_x - 18, legend_y - 18, legend_x + 490, legend_y + 94), radius=10, fill=(255, 255, 255), outline=(215, 220, 225), width=2)
    for i, run in enumerate(RUNS):
        y = legend_y + i * 45
        color = hex_to_rgb(run["color"])
        draw.line((legend_x, y + 12, legend_x + 55, y + 12), fill=color, width=5)
        draw.ellipse((legend_x + 23, y + 6, legend_x + 33, y + 16), fill=color)
        draw.text((legend_x + 70, y), f"{run['label']} 10-step avg", fill=(35, 40, 46), font=legend_font)

    note = "Raw curves are faint; bold curves are 10-step moving averages."
    draw.text((left, height - 28), note, fill=(95, 100, 108), font=small_font)

    if scale != 1:
        img = img.resize((width // scale, height // scale), Image.Resampling.LANCZOS)

    png = ROOT / "fig_training_reward_compare_phase5_phase6.png"
    img.save(png)
    return png


def main() -> None:
    data: dict[str, list[tuple[int, float]]] = {}
    for run in RUNS:
        text = fetch_log(run["path"])
        rows = parse_rewards(text)
        data[run["short"]] = rows
        if rows:
            first = rows[0]
            last = rows[-1]
            avg_last10 = float(np.mean([r for _, r in rows[-10:]]))
            print(
                f"{run['short']}: n={len(rows)} first={first[0]}:{first[1]:.3f} "
                f"last={last[0]}:{last[1]:.3f} last10_avg={avg_last10:.3f}"
            )
        else:
            print(f"{run['short']}: no reward rows found")

    csv_path = write_csv(data)
    png = plot(data)
    print(f"csv={csv_path}")
    print(f"png={png}")


if __name__ == "__main__":
    main()
