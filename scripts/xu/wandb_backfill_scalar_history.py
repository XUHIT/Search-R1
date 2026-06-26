#!/usr/bin/env python3
"""Backfill scalar W&B history from Search-R1 console logs."""

import argparse
import re
from pathlib import Path

import wandb


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
STEP_RE = re.compile(r"step:(\d+)\s*-\s*(.*)")


def parse_scalar_steps(log_path: Path, min_step: int):
    rows = []
    for raw_line in log_path.read_text(errors="replace").splitlines():
        line = ANSI_RE.sub("", raw_line)
        match = STEP_RE.search(line)
        if not match:
            continue

        step = int(match.group(1))
        if step < min_step:
            continue

        metrics = {}
        for item in match.group(2).split(" - "):
            if ":" not in item:
                continue
            key, value = item.rsplit(":", 1)
            try:
                metrics[key.strip()] = float(value)
            except ValueError:
                continue

        if metrics:
            rows.append((step, metrics))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--entity", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--min-step", type=int, default=1)
    args = parser.parse_args()

    rows = parse_scalar_steps(args.log, args.min_step)
    if not rows:
        raise SystemExit(f"no scalar step rows found in {args.log}")

    run = wandb.init(
        entity=args.entity,
        project=args.project,
        id=args.run_id,
        resume="allow",
    )
    for step, metrics in rows:
        wandb.log(metrics, step=step)
        print(f"backfilled step={step} metrics={len(metrics)}")
    run.finish(exit_code=0)


if __name__ == "__main__":
    main()
