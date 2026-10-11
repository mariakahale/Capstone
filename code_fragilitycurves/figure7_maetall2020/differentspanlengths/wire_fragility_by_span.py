"""
wire_fragility_by_span.py
=========================
Span-dependent conductor (wire) fragility, rebuilt from the wire model in
Ma, Dai & Pang (2020), "Reliability Assessment of Electrical Grids Subjected to
Wind Hazards and Ice Accretion with Concurrent Wind", J. Struct. Eng. 146(7).

WHAT THIS REPLACES
------------------
Ma et al. publish ONE wire fragility: Fig. 7(d), a surface P_w(ice, wind) for a
single 91.44 m span. This module produces that same surface for any span, so the
pipeline can look up P_w by span bin instead of using the 91.44 m surface for
every wire. Everything downstream (Eq. 10 integration, Eq. 11 pole-wire element,
Eq. 13 line, Eq. 14 system) is unchanged; it just receives a span-specific P_w.

MODEL (mirrors Ma et al., "Structural Model of Wires")
------------------------------------------------------
* Wire = elastic catenary, clamped at level supports. Ma used OpenSees; here we
  use the closed-form elastic catenary (Irvine 1981), which is exact for a single
  span under uniform load and runs in microseconds.
* Failure = support tension > breaking strength (77,850 N), deterministic.
* Loads per metre:
    vertical  w_v = self-weight + ice tube weight             (Ma Eq. 9, per metre)
    lateral   w_h = 0.613 Kz Kzt v^2 G Cf (D + 2r)              (Ma Eq. 5, per metre)
  A cable has no bending stiffness, so it swings into the plane of the resultant
  load and the tension depends only on |w| = sqrt(w_v^2 + w_h^2). That gives a
  single critical load w_crit(L) per span, i.e. the failure boundary is a
  quarter-circle in (wind load, vertical load) space. Checked against Ma's Fig. 2:
  every digitized point lies at radius 117.2-117.4 N/m (std 0.08 N/m), so Ma's
  5th-order polynomial is just a curve fit to that circle.
* Randomness (Ma, from Vickery et al. 2000): Kz ~ N(1.024, 0.06 COV),
  G ~ N(0.801, 0.11 COV); Cf = 1.2 fixed for iced wires, ~N(1.0, 0.12 COV) for
  the wind-only case. Ice weight and wire strength are deterministic, as in Ma.
* Span scaling: WireFragility(stringing="slack") keeps Ma's slack ratio (91.50/91.44);
  stringing="tension" keeps Ma's bare-wire installation tension instead (more realistic).

CALIBRATION (read this)
-----------------------
Ma gives E = 3e7 MPa but not the conductor area, and that E is not physical for
ACSR (~75-80 GPa). So the axial stiffness EA is calibrated so the 91.44 m wire's
critical resultant load equals Ma's Fig. 2 radius, 117.3 N/m - i.e. it matches
their OpenSees output directly. EA is a material property, so the same EA is used
for every span.
Note: Fig. 2 implies ~53.6 mm of ice at failure with no wind, while Ma's text says
~57.5 mm. The paper is slightly inconsistent; Fig. 2 is used because it is the
model output their Monte Carlo actually used. calibrate_EA(..., r_fail_zero_wind=)
is kept if you want the text value instead.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

G_ACC = 9.81            # m/s^2
ICE_DENSITY = 917.0     # kg/m^3  (Ma prints 0.917 kg/m^3, a units typo)
BASE_SPAN = 91.44       # m, Ma et al.
SLACK_RATIO = 91.50 / 91.44
N_SAMPLES = 100_000     # Ma used 1e5 samples per point
_trapz = getattr(np, "trapezoid", None) or np.trapz


# ----------------------------------------------------------------------------
# Conductor
# ----------------------------------------------------------------------------
@dataclass
class Conductor:
    """ACSR 336.4 kcmil 30/7 'Oriole' (0.741 in, 526 lb/kft, 17,300 lb RBS), Short (2005) Table 2.3.
    Total metal area ~210 mm^2 (170.5 Al + 39.8 steel); only enters via EA, which is calibrated."""
    diameter: float = 0.01882          # m
    unit_mass: float = 0.7766          # kg/m
    breaking_strength: float = 77_850  # N
    EA: float | None = None            # N; None -> calibrate_EA() is called

    @property
    def radius(self) -> float:
        return self.diameter / 2

    @property
    def self_weight(self) -> float:    # N/m
        return self.unit_mass * G_ACC


# ----------------------------------------------------------------------------
# Elastic catenary mechanics
# ----------------------------------------------------------------------------
def horizontal_tension(w: float, span: float, L0: float, EA: float) -> float:
    """Solve Irvine's level-support elastic catenary for H [N].

    span = H*L0/EA + (2H/w) * asinh(w*L0 / (2H))
    """
    def f(H):
        return H * L0 / EA + (2 * H / w) * math.asinh(w * L0 / (2 * H)) - span
    return brentq(f, 1e-6, 1e10, xtol=1e-6, maxiter=500)


def support_tension(w: float, span: float, EA: float, slack_ratio: float = SLACK_RATIO) -> float:
    """Max (support) tension [N] for resultant uniform load w [N/m]."""
    L0 = span * slack_ratio
    H = horizontal_tension(w, span, L0, EA)
    return math.hypot(H, w * L0 / 2)


def critical_load(span: float, cond: Conductor, slack_ratio: float = SLACK_RATIO) -> float:
    """Resultant load per metre [N/m] at which support tension = breaking strength."""
    g = lambda w: support_tension(w, span, cond.EA, slack_ratio) - cond.breaking_strength
    return brentq(g, 1e-3, 1e6, xtol=1e-6)


# ----------------------------------------------------------------------------
# Loads (per metre of wire)
# ----------------------------------------------------------------------------
def vertical_load(r_ice, cond: Conductor):
    """Self-weight + ice tube weight [N/m]. r_ice in metres (array OK). Ma Eq. 9 / L."""
    rw = cond.radius
    return cond.self_weight + ICE_DENSITY * G_ACC * np.pi * ((rw + r_ice) ** 2 - rw ** 2)


def wind_load_coeff(r_ice, v, cond: Conductor):
    """Deterministic part of Ma Eq. 5 per metre: 0.613 v^2 (D + 2r). Multiply by Kz*G*Cf."""
    return 0.613 * v ** 2 * (cond.diameter + 2 * r_ice)


def sample_wind_factor(n: int, iced: bool, rng: np.random.Generator) -> np.ndarray:
    """Samples of Kz * Kzt * G * Cf (Ma's distributions). Sorted for fast survival lookup."""
    Kz = rng.normal(1.024, 0.06 * 1.024, n)
    G = rng.normal(0.801, 0.11 * 0.801, n)
    Cf = np.full(n, 1.2) if iced else rng.normal(1.0, 0.12, n)
    X = np.clip(Kz * G * Cf, 0, None)   # Kzt = 1.0
    return np.sort(X)


def _survival(X_sorted: np.ndarray, thresh: np.ndarray) -> np.ndarray:
    """Empirical P(X > thresh), vectorised."""
    n = X_sorted.size
    return (n - np.searchsorted(X_sorted, thresh, side="right")) / n


# ----------------------------------------------------------------------------
# Calibration
# ----------------------------------------------------------------------------
MA_FIG2_WCRIT = 117.3   # N/m, radius of Ma Fig. 2 failure circle (91.44 m span)


def calibrate_EA(cond: Conductor, w_crit_target: float = MA_FIG2_WCRIT,
                 r_fail_zero_wind: float | None = None,
                 span: float = BASE_SPAN) -> float:
    """Find EA so the `span` wire's critical resultant load = w_crit_target [N/m]
    (default: Ma Fig. 2). If r_fail_zero_wind [m] is given, match that instead
    (e.g. 0.0575 for Ma's text value)."""
    target = vertical_load(r_fail_zero_wind, cond) if r_fail_zero_wind else w_crit_target

    def g(logEA):
        c = Conductor(cond.diameter, cond.unit_mass, cond.breaking_strength, 10 ** logEA)
        return critical_load(span, c) - target

    return 10 ** brentq(g, 5, 12)


# ----------------------------------------------------------------------------
# Fragility
# ----------------------------------------------------------------------------
class WireFragility:
    """Span-dependent wire fragility. Build once, query many times."""

    def __init__(self, cond: Conductor | None = None, stringing: str = "slack",
                 n_samples: int = N_SAMPLES, seed: int = 0):
        """stringing = how a non-91.44 m span is assumed to be installed:
          "slack"   - same slack ratio as Ma (91.50/91.44). Sag grows with span.
          "tension" - same bare-wire installation tension as Ma's 91.44 m span
                      (closer to how utilities string lines; short spans are tighter).
        """
        self.cond = cond or Conductor()
        if self.cond.EA is None:
            self.cond.EA = calibrate_EA(self.cond)
        if stringing not in ("slack", "tension"):
            raise ValueError("stringing must be 'slack' or 'tension'")
        self.stringing = stringing
        self._T0 = support_tension(self.cond.self_weight, BASE_SPAN, self.cond.EA, SLACK_RATIO)
        rng = np.random.default_rng(seed)
        self._X_ice = sample_wind_factor(n_samples, iced=True, rng=rng)
        self._X_wind = sample_wind_factor(n_samples, iced=False, rng=rng)
        self._wcrit_cache: dict[float, float] = {}

    def w_crit(self, span: float) -> float:
        if span not in self._wcrit_cache:
            self._wcrit_cache[span] = critical_load(span, self.cond, self.slack_ratio(span))
        return self._wcrit_cache[span]

    def slack_ratio(self, span: float) -> float:
        """Unstretched wire length / span for this span under the stringing rule."""
        if self.stringing == "slack":
            return SLACK_RATIO
        ws, EA = self.cond.self_weight, self.cond.EA
        g = lambda s: support_tension(ws, span, EA, s) - self._T0
        return brentq(g, 0.98, 1.10, xtol=1e-12)

    def p_fail_ice(self, span: float, r_ice_mm, v_concurrent):
        """P(failure | radial ice [mm], concurrent wind [m/s]) - the Fig. 7(d) surface.
        r_ice_mm and v_concurrent broadcast against each other."""
        r = np.asarray(r_ice_mm, float) / 1000
        v = np.asarray(v_concurrent, float)
        r, v = np.broadcast_arrays(r, v)
        wc = self.w_crit(span)
        wv = vertical_load(r, self.cond)
        # need w_h > sqrt(wc^2 - wv^2);  w_h = coeff * X
        need = np.sqrt(np.clip(wc ** 2 - wv ** 2, 0, None))
        coeff = wind_load_coeff(r, v, self.cond)
        with np.errstate(divide="ignore", invalid="ignore"):
            thresh = np.where(coeff > 0, need / coeff, np.inf)
        p = _survival(self._X_ice, thresh)
        return np.where(wv >= wc, 1.0, p)

    def p_fail_wind(self, span: float, v):
        """P(failure | extreme wind [m/s], no ice). Ma found this negligible."""
        v = np.asarray(v, float)
        wc = self.w_crit(span)
        wv = self.cond.self_weight
        need = math.sqrt(max(wc ** 2 - wv ** 2, 0))
        coeff = wind_load_coeff(0.0, v, self.cond)
        with np.errstate(divide="ignore"):
            thresh = np.where(coeff > 0, need / coeff, np.inf)
        return _survival(self._X_wind, thresh)

    def lognormal_fit_ice(self, span: float, v_concurrent: float,
                          r_grid_mm=np.arange(0, 300.01, 0.05)):
        """Fit theta [mm], beta to the 1-D ice fragility at a fixed concurrent wind.
        theta = 50th percentile; beta = (ln r84 - ln r16)/2.
        beta ~ 0 means the curve is effectively a step (deterministic) - see notes.
        Returns (nan, nan) if the wire doesn't reach 84% failure within 300 mm of ice
        (i.e. it effectively never fails from ice)."""
        p = self.p_fail_ice(span, r_grid_mm, v_concurrent)
        if p[-1] < 0.84:
            return float("nan"), float("nan")
        def pct(q):
            idx = np.searchsorted(p, q)
            return r_grid_mm[min(idx, len(r_grid_mm) - 1)]
        r16, r50, r84 = pct(0.16), pct(0.50), pct(0.84)
        beta = (math.log(max(r84, 1e-6)) - math.log(max(r16, 1e-6))) / 2
        return r50, beta


# ----------------------------------------------------------------------------
# Span binning (round UP -> conservative)
# ----------------------------------------------------------------------------
def span_bin(length_m: float, bin_width: float = 5.0, min_bin: float = 10.0,
             max_bin: float = 100.0) -> float:
    """Round a span up to the next bin. Spans above max_bin get max_bin + flagged by caller."""
    b = math.ceil(length_m / bin_width) * bin_width
    return float(min(max(b, min_bin), max_bin))


def build_lookup(frag: WireFragility, bins=np.arange(10, 100.1, 5),
                 v_concurrent_list=(0.0, 9.0, 15.0, 20.0, 26.817)):
    """Table of (span_bin, v_concurrent) -> theta_mm, beta, w_crit."""
    import pandas as pd
    rows = []
    for L in bins:
        for v in v_concurrent_list:
            th, be = frag.lognormal_fit_ice(float(L), v)
            rows.append(dict(span_bin_m=float(L), v_concurrent_ms=v, theta_mm=th,
                             beta=be, w_crit_N_per_m=frag.w_crit(float(L))))
    return pd.DataFrame(rows)


def build_surface(frag: WireFragility, bins=np.arange(10, 100.1, 5),
                  r_mm=np.arange(0, 80.01, 0.5), v=np.arange(0, 40.01, 0.5)):
    """Long-format Fig. 7(d)-style surface for every span bin (Ma's grid: 0.5 mm x 0.5 m/s)."""
    import pandas as pd
    R, V = np.meshgrid(r_mm, v, indexing="ij")
    out = []
    for L in bins:
        P = frag.p_fail_ice(float(L), R, V)
        out.append(pd.DataFrame(dict(span_bin_m=float(L), ice_mm=R.ravel(),
                                     wind_ms=V.ravel(), p_fail=P.ravel())))
    return pd.concat(out, ignore_index=True)


# ----------------------------------------------------------------------------
# Validation against Ma et al. (2020)
# ----------------------------------------------------------------------------
def annual_pf_gpd(frag: WireFragility, span: float, v_conc: float,
                  alpha: float, k: float, u: float = 0.0) -> float:
    """Ma Eq. 10 with the GPD ice hazard, threshold u [mm].
    NOTE: Ma's printed Eq. 8 has the sign of k flipped relative to Eq. 7. With k < 0
    (Table 3), printed Eq. 8 gives a bounded tail and annual P_w ~1e-8. The pdf
    consistent with Eq. 7 (exceedance = [1 - k(r-u)/alpha]^(1/k)) is used here, and it
    reproduces Ma's Table 4 numbers."""
    r = np.linspace(u, u + 400, 40_001)
    base = 1 - k * (r - u) / alpha
    f = np.where(base > 0, (1 / alpha) * np.abs(base) ** (1 / k - 1), 0.0)
    F = frag.p_fail_ice(span, r, v_conc)
    return float(_trapz(F * f, r))


def validate(frag: WireFragility) -> None:
    L = BASE_SPAN
    print(f"Calibrated EA = {frag.cond.EA:,.3e} N   (w_crit @ 91.44 m = {frag.w_crit(L):.1f} N/m; Ma Fig. 2: 117.3)")
    th0, _ = frag.lognormal_fit_ice(L, 0.0)
    print(f"[1] Zero wind, 91.44 m: fails at {th0:.1f} mm      (Ma Fig. 2: ~53.6 mm; Ma text: ~57.5 mm)")
    r = np.arange(0, 80, 0.1)
    p = frag.p_fail_ice(L, r, 26.817)
    lo, hi = r[np.argmax(p > 0.01)], r[np.argmax(p > 0.99)]
    print(f"[2] 26.817 m/s, 91.44 m: 1%->99% between {lo:.1f}-{hi:.1f} mm   (Ma: ~41-45 mm)")
    gm = annual_pf_gpd(frag, L, 26.817, alpha=6.6357, k=-0.1037)
    se = annual_pf_gpd(frag, L, 9.0, alpha=0.9087, k=-0.3733)
    print(f"[3] Annual P_w Grand Marais: {gm:.2e}   (Ma: 5.9e-03)")
    print(f"    Annual P_w Seattle:      {se:.2e}   (Ma: 1.64e-04; Seattle threshold u unknown, u=0 used)")
    pw = frag.p_fail_wind(L, np.array([40, 60, 90]))
    print(f"[4] Wind-only P_w at 40/60/90 m/s: {pw}   (Ma: negligible)")


if __name__ == "__main__":
    import pandas as pd
    frag = WireFragility()
    validate(frag)

    frag_T = WireFragility(stringing="tension")
    print("\nSpan sensitivity (zero-wind ice thickness at failure / theta @ 26.817 m/s):")
    print("   span |  slack rule: w_crit  r_fail(0 wind) | tension rule: w_crit  r_fail(0 wind)")
    for L in (20, 30, 45, 60, 75, 91.44):
        a = frag.lognormal_fit_ice(L, 0.0)[0]
        b = frag_T.lognormal_fit_ice(L, 0.0)[0]
        print(f"  {L:5.1f} | {frag.w_crit(L):10.1f} N/m {a:8.1f} mm   |"
              f" {frag_T.w_crit(L):10.1f} N/m {b:8.1f} mm")

    table = build_lookup(frag)
    table.to_csv("wire_fragility_lookup.csv", index=False)
    build_lookup(frag_T).to_csv("wire_fragility_lookup_tension_rule.csv", index=False)
    print("\nLookup table (first rows):\n", table.head(10).to_string(index=False))
    build_surface(frag).to_csv("wire_fragility_surfaces.csv", index=False)
    print("\nSaved wire_fragility_lookup.csv and wire_fragility_surfaces.csv")
