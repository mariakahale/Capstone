"""
alectra_age_model.py
====================
Assigns a wood-pole age to every pole on the feeder so that the ages follow
Alectra Utilities' 2023 wood pole age distribution, with poles on the same line
section sharing a common installation age (see age_assignment_explanation.txt).

    spans table (start_pole, end_pole)  ->  line sections  ->  one age per section
                                                               + spot replacements

Used by pipeline/asset_adapter.py when AGE_MODE = "alectra". It only needs the
pole ids and the spans table, so it does not depend on the polegen2 CSV layout.

Run on its own to check the result against the Alectra histogram:
    python pipeline/agedistribution_alectra/alectra_age_model.py
"""

import numpy as np
import pandas as pd

# =============================================================================
# SOURCE DATA -- Alectra Utilities DSP (EB-2025-0252), Exhibit 2A, Appendix B01,
# Figure 17 "2023 Wood Pole Age Distribution" (105,262 wood poles).
# The 5,986 poles of Unknown age are left out and the rest renormalized.
# The 76+ bin has no upper limit; it is capped at 85 years (assumption).
# =============================================================================
BIN_LO = np.array([0, 6, 11, 16, 21, 26, 31, 36, 41, 46, 51, 56, 61, 66, 71, 76])
BIN_HI = np.array([5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 85])
BIN_COUNTS = np.array([10603, 11447, 9116, 8871, 8452, 8463, 7708, 5882, 4415,
                       4228, 4756, 3549, 3220, 3726, 3986, 854])
BIN_P = BIN_COUNTS / BIN_COUNTS.sum()          # 99,276 poles of known age
BIN_LABELS = [f"{lo}-{hi}" if lo < 76 else "76+" for lo, hi in zip(BIN_LO, BIN_HI)]
MAX_AGE = int(BIN_HI[-1])

# =============================================================================
# DEFAULT SETTINGS (asset_adapter.py passes its own values)
# =============================================================================
RHO = 0.20        # fraction of poles individually replaced (age independent of section)
JITTER = 2        # +/- years of spread inside a section's cohort
SEED = 2026


# =============================================================================
# 1. LINE SECTIONS
# =============================================================================
def line_sections(pole_ids, spans):
    """Split the network into line sections = chains of poles between junctions.

    A junction is a pole with 3+ spans. Poles with 1 span (line ends) or 2 spans
    (in-line) stay in their chain. Each junction pole joins the largest section it
    touches. Returns a pd.Series: pole_id -> section id (int).
    """
    pole_ids = pd.Index(pole_ids)
    idx = pd.Series(np.arange(len(pole_ids)), index=pole_ids)
    a = idx[spans["start_pole"]].to_numpy()
    b = idx[spans["end_pole"]].to_numpy()
    degree = np.bincount(np.concatenate([a, b]), minlength=len(pole_ids))
    junction = degree >= 3

    # union-find over spans whose two ends are both non-junction poles
    parent = np.arange(len(pole_ids))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in zip(a, b):
        if not junction[i] and not junction[j]:
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[rj] = ri
    root = np.array([find(i) for i in range(len(pole_ids))])

    # attach each junction pole to the largest neighbouring (non-junction) section
    size = pd.Series(root[~junction]).value_counts()
    neighbours = {}
    for i, j in zip(a, b):
        neighbours.setdefault(i, []).append(j)
        neighbours.setdefault(j, []).append(i)
    for k in np.where(junction)[0]:
        cands = [root[n] for n in neighbours.get(k, []) if not junction[n]]
        if cands:
            root[k] = max(cands, key=lambda r: size.get(r, 0))
        # else: junction surrounded by junctions keeps its own section

    section = pd.Series(pd.factorize(root)[0], index=pole_ids, name="section_id")
    return section


# =============================================================================
# 2. AGE ASSIGNMENT
# =============================================================================
def _age_in_bin(rng, b):
    return rng.integers(BIN_LO[b], BIN_HI[b], endpoint=True)


def assign_alectra_ages(pole_ids, spans, rho=RHO, jitter=JITTER, seed=SEED,
                        years_since_2023=0):
    """Returns a DataFrame indexed by pole_id with columns:
        age_years   assigned age (whole years, 0-85 + years_since_2023)
        section_id  line section the pole belongs to
        spot        True if the pole was individually replaced (age drawn on its own)

    rho              fraction of poles whose age is independent of their section
                     (0 = pure cohorts, 1 = fully independent draws)
    jitter           cohort poles get section age +/- jitter years
    years_since_2023 shift every age forward if modelling a later year than the
                     2023 snapshot (default 0 = use the snapshot as-is)
    """
    rng = np.random.default_rng(seed)
    pole_ids = pd.Index(pole_ids)
    section = line_sections(pole_ids, spans)
    n = len(pole_ids)

    spot = rng.random(n) < rho
    age = np.zeros(n, dtype=int)

    # (a) spot replacements: independent draws from the Alectra histogram
    for k in np.where(spot)[0]:
        age[k] = _age_in_bin(rng, rng.choice(len(BIN_P), p=BIN_P))

    # (b) cohorts: one age per section; bin picked in proportion to the pole quota
    #     each bin still needs, so the feeder total tracks the Alectra histogram
    cohort = np.where(~spot)[0]
    members = pd.Series(cohort).groupby(section.to_numpy()[cohort]).apply(list)
    remaining = BIN_P * len(cohort)
    for sec in rng.permutation(members.index.to_numpy()):
        idx = members[sec]
        w = np.clip(remaining, 0, None)
        p = w / w.sum() if w.sum() > 0 else BIN_P
        b = rng.choice(len(BIN_P), p=p)
        remaining[b] -= len(idx)
        base = _age_in_bin(rng, b)
        for k in idx:
            age[k] = int(np.clip(base + rng.integers(-jitter, jitter, endpoint=True),
                                 0, MAX_AGE))

    return pd.DataFrame({"age_years": age + years_since_2023,
                         "section_id": section.to_numpy(),
                         "spot": spot}, index=pole_ids)


# =============================================================================
# 3. CHECK: assigned histogram vs Alectra
# =============================================================================
def compare_to_alectra(ages):
    ages = np.asarray(ages)
    b = np.searchsorted(BIN_HI, ages)          # bin index of each age
    b = np.clip(b, 0, len(BIN_HI) - 1)
    got = np.bincount(b, minlength=len(BIN_HI)) / len(ages)
    return pd.DataFrame({"bin": BIN_LABELS,
                         "alectra_%": (100 * BIN_P).round(1),
                         "feeder_%": (100 * got).round(1)})


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from pipeline.asset_adapter import load_polegen

    raw_poles, raw_spans = load_polegen()
    out = assign_alectra_ages(raw_poles["pole_id"], raw_spans)
    sizes = out.groupby("section_id").size()
    print(f"{len(out)} poles in {len(sizes)} line sections "
          f"(median {sizes.median():.0f} poles, max {sizes.max()}), "
          f"{out.spot.sum()} spot replacements ({100 * out.spot.mean():.1f}%)")
    print(f"mean age {out.age_years.mean():.1f} yr (Alectra: "
          f"{(BIN_P * (BIN_LO + BIN_HI) / 2).sum():.1f}), "
          f"{(out.age_years > 60).sum()} poles over 60 yr")
    print(compare_to_alectra(out.age_years).to_string(index=False))
