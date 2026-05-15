"""
checkpoint.py — Resume-aware checkpoint system for the IJDAR pipeline.

After each (arch, aug, seed) run completes, the result is immediately
written to a checkpoint file. When the pipeline restarts, it reads that
file and skips all already-finished runs.

Checkpoint file: outputs/results/checkpoint.json
"""

import os
import json
import pandas as pd
from datetime import datetime

from config import RESULTS_DIR

CHECKPOINT_PATH = os.path.join(RESULTS_DIR, "checkpoint.json")


# ─── Read / Write ─────────────────────────────────────────────────────────────

def load_checkpoint() -> dict:
    """
    Load the checkpoint file.

    Returns
    -------
    dict with keys:
        'completed' : list of {arch, n_aug, seed, exact, cer, wer, ...}
        'started_at': ISO timestamp of first run
        'updated_at': ISO timestamp of last completed run
    """
    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        completed = len(data.get("completed", []))
        print(f"  [checkpoint] Resuming — {completed} run(s) already done.")
        return data
    return {"completed": [], "started_at": _now(), "updated_at": _now()}


def save_checkpoint(state: dict):
    """Write the checkpoint dict to disk immediately after each run."""
    state["updated_at"] = _now()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def mark_done(state: dict, record: dict):
    """
    Append a completed run record to the checkpoint and save.

    Parameters
    ----------
    state  : current checkpoint dict (mutated in place)
    record : dict with at minimum keys: arch, n_aug, seed
    """
    # Strip non-serialisable objects (model, numpy arrays, etc.)
    safe = {
        k: (v.item() if hasattr(v, "item") else v)
        for k, v in record.items()
        if k not in ("gt", "pred", "model", "history")
        and isinstance(v, (str, int, float, bool, type(None)))
    }
    state["completed"].append(safe)
    save_checkpoint(state)


def is_done(state: dict, arch: str, n_aug: int, seed: int) -> bool:
    """Return True if this (arch, n_aug, seed) combo is already in checkpoint."""
    for rec in state["completed"]:
        if (rec.get("arch") == arch
                and rec.get("n_aug") == n_aug
                and rec.get("seed") == seed):
            return True
    return False


# ─── Merge helpers ────────────────────────────────────────────────────────────

def checkpoint_to_records(state: dict) -> list:
    """Return the completed runs as a list of dicts (for train.py aggregation)."""
    return state.get("completed", [])


def print_remaining(state: dict, architectures: list,
                    aug_degrees: list, seeds: list):
    """Print a summary of what still needs to run."""
    total = len(architectures) * len(aug_degrees) * len(seeds)
    done  = len(state.get("completed", []))
    remaining = []
    for arch in architectures:
        for n_aug in aug_degrees:
            for seed in seeds:
                if not is_done(state, arch, n_aug, seed):
                    remaining.append((arch, n_aug, seed))

    print(f"\n  [checkpoint] Progress: {done}/{total} runs completed")
    print(f"  [checkpoint] Remaining: {len(remaining)} runs")
    if remaining:
        print("  Next up:")
        for arch, n_aug, seed in remaining[:5]:
            print(f"    arch={arch}  aug={n_aug}  seed={seed}")
        if len(remaining) > 5:
            print(f"    … and {len(remaining)-5} more")


def _now() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


# ─── Reset ────────────────────────────────────────────────────────────────────

def reset_checkpoint():
    """Delete the checkpoint file to start fresh."""
    if os.path.exists(CHECKPOINT_PATH):
        os.remove(CHECKPOINT_PATH)
        print("  [checkpoint] Reset — starting fresh.")
