"""
statistical_tests.py — Paired t-tests across model pairs and augmentation
conditions for IJDAR statistical rigor requirements.

Outputs
-------
outputs/results/statistical_tests.csv   — full pairwise table
outputs/results/aug_effect_tests.csv    — augmentation effect per architecture
"""

import os
import itertools
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

from config import RESULTS_DIR, ARCHITECTURES


def _significance_stars(p: float) -> str:
    """Return *** / ** / * / ns based on p-value."""
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


def run_pairwise_tests(raw_records: list,
                       target_aug: int | None = None) -> pd.DataFrame:
    """
    Perform paired t-tests across all architecture pairs.

    For each pair (A, B), the per-seed exact-match scores are collected and
    compared with ttest_rel.  If target_aug is None, uses the best aug level
    per architecture (highest mean exact).  Otherwise fixes the aug degree.

    Parameters
    ----------
    raw_records : list of dicts from train.run_multi_seed_training
    target_aug  : int or None

    Returns
    -------
    pd.DataFrame with columns:
        model_A, model_B, mean_A, mean_B, t_stat, p_value,
        significant, stars, effect_dir
    """
    # Build dict: (arch, n_aug) → list of exact scores over seeds
    score_map: dict = {}
    for rec in raw_records:
        key = (rec["arch"], rec["n_aug"])
        score_map.setdefault(key, []).append(rec["exact"])

    # Select best aug per arch (or fixed aug)
    best_scores: dict[str, list] = {}
    for arch in ARCHITECTURES:
        candidates = {
            n_aug: scores
            for (a, n_aug), scores in score_map.items()
            if a == arch and (target_aug is None or n_aug == target_aug)
        }
        if not candidates:
            continue
        best_aug = max(candidates, key=lambda k: np.mean(candidates[k]))
        best_scores[arch] = candidates[best_aug]

    rows = []
    for arch_a, arch_b in itertools.combinations(best_scores.keys(), 2):
        s_a = best_scores[arch_a]
        s_b = best_scores[arch_b]
        # Ensure equal length (pad with mean if needed — rare edge case)
        min_len = min(len(s_a), len(s_b))
        s_a, s_b = s_a[:min_len], s_b[:min_len]

        if len(s_a) < 2:
            # Cannot run t-test with < 2 paired observations
            continue

        stat, pval = ttest_rel(s_a, s_b)
        rows.append(dict(
            model_A     = arch_a,
            model_B     = arch_b,
            mean_A      = round(np.mean(s_a), 4),
            mean_B      = round(np.mean(s_b), 4),
            t_stat      = round(stat,  4),
            p_value     = round(pval,  6),
            significant = pval < 0.05,
            stars       = _significance_stars(pval),
            effect_dir  = "A > B" if np.mean(s_a) > np.mean(s_b) else "B > A",
        ))

    df = pd.DataFrame(rows)
    df.to_csv(
        os.path.join(RESULTS_DIR, "statistical_tests.csv"), index=False
    )
    print("\nPAIRWISE STATISTICAL TESTS:")
    print(df.to_string(index=False))
    return df


def run_augmentation_effect_tests(raw_records: list) -> pd.DataFrame:
    """
    For each architecture, test whether best-aug significantly outperforms
    no-aug (n_aug=0) using paired t-test across seeds.

    Returns
    -------
    pd.DataFrame with columns:
        arch, aug_0_mean, best_aug, best_aug_mean,
        t_stat, p_value, significant, stars
    """
    # Build dict: arch → { n_aug → [exact scores per seed] }
    arch_aug_scores: dict = {}
    for rec in raw_records:
        arch_aug_scores \
            .setdefault(rec["arch"], {}) \
            .setdefault(rec["n_aug"], []) \
            .append(rec["exact"])

    rows = []
    for arch, aug_dict in arch_aug_scores.items():
        if 0 not in aug_dict or len(aug_dict) < 2:
            continue
        no_aug_scores = aug_dict[0]
        # Find best non-zero aug
        best_aug = max(
            (k for k in aug_dict if k > 0),
            key=lambda k: np.mean(aug_dict[k]),
        )
        best_aug_scores = aug_dict[best_aug]

        min_len = min(len(no_aug_scores), len(best_aug_scores))
        s0 = no_aug_scores[:min_len]
        s1 = best_aug_scores[:min_len]

        if len(s0) < 2:
            continue

        stat, pval = ttest_rel(s0, s1)
        rows.append(dict(
            arch          = arch,
            aug_0_mean    = round(np.mean(s0), 4),
            best_aug      = best_aug,
            best_aug_mean = round(np.mean(s1), 4),
            t_stat        = round(stat, 4),
            p_value       = round(pval, 6),
            significant   = pval < 0.05,
            stars         = _significance_stars(pval),
        ))

    df = pd.DataFrame(rows)
    df.to_csv(
        os.path.join(RESULTS_DIR, "aug_effect_tests.csv"), index=False
    )
    print("\nAUGMENTATION EFFECT TESTS:")
    print(df.to_string(index=False))
    return df
