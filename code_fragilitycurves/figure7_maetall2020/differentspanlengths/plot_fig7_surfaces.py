"""
plot_fig7_surfaces.py
=====================
3D plots of the Fig. 7 fragility surfaces (Ma, Dai & Pang 2020) for EVERY span bin
and age bin in the lookup files, so you can see what the pipeline is looking up.

    x = concurrent 3-s gust (m/s)    y = equivalent radial ice (mm)    z = P(fail)

Same axes as Ma et al. Fig. 7. Reads the precomputed surfaces only (numpy + matplotlib,
no scipy), so it shows exactly what pole_fig7_lookup.py / wire_fig7d_lookup.py return
(they interpolate these same grids).

    pole_fig7_surfaces_by_span.npz   p_fail[phase, span, age, ice, wind]
                                     3 phase counts x 30 spans x 13 ages
    wire_fig7d_surfaces_by_span.npz  p_slack / p_tension[span, ice, wind]   30 spans

HOW MANY SURFACES
    poles : 3 phase counts (1, 2, 3) x 30 tributary-span bins x 13 ages = 1170
    wires : 30 span bins x 2 stringing rules (slack, tension)          =   60
    The wire surface is per conductor, so it is the same for 1- and 3-phase lines.

1-PHASE vs 3-PHASE
    A 1-phase pole carries 1 phase wire, a 3-phase pole carries 3 (Ma et al. = 3;
    neutral neglected). The pole's iced-wire wind area and weight scale with that.
    In the viewer, the "Pole phases" buttons switch the pole surface, and the Result
    panel prints P(pole fails) for every phase count at the storm point.

RUN
    # 1) Interactive viewer -- just press Run in VS Code (no arguments needed).
    #    Sliders pick the pole age, pole tributary span and wire span bins; buttons
    #    pick the pole phase count and wire stringing. Type ice and wind in the boxes,
    #    press Enter or click "Plot point": the dot appears on both surfaces and
    #    P(fail) shows in the Result panel. Drag a plot to rotate it.
    python plot_fig7_surfaces.py
    python plot_fig7_surfaces.py --ice 14.7 --wind 12.3      # optional: pre-fill the boxes
    python plot_fig7_surfaces.py --phases 1                  # optional: start on 1-phase

    # 2) Save every combination as PNGs, one sheet per span bin and phase count
    #    (13 pole ages + wire slack + wire tension on each sheet)
    python plot_fig7_surfaces.py --save                       # 1- and 3-phase (default)
    python plot_fig7_surfaces.py --save --phases 1            # 1-phase only
    python plot_fig7_surfaces.py --save --phases 1 2 3
    python plot_fig7_surfaces.py --save --feeder-bins         # only the bins feeder 1878 uses
    python plot_fig7_surfaces.py --save --ice 14.7 --wind 12.3 --ice-max 150

Output of --save goes to ./fig7_surface_plots/ next to this script.
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
POLE_NPZ = HERE / "pole_fig7_surfaces_by_span.npz"
WIRE_NPZ = HERE / "wire_fig7d_surfaces_by_span.npz"
OUT_DIR = HERE / "fig7_surface_plots"

# Span bins feeder 1878 (suburban) actually hits, from pipeline/outputs/*_for_fragility_*.csv
# (pole tributary spans 0.5-79 m, wire spans 1-49.6 m, after rounding UP to a bin).
FEEDER_POLE_BINS = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 80]
FEEDER_WIRE_BINS = [10, 15, 20, 25, 30, 35, 40, 45, 50]

STRIDE = 4          # plot every 4th grid point (2 mm ice, 2 m/s wind) -- fast, still smooth


# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------
def load():
    for p in (POLE_NPZ, WIRE_NPZ):
        if not p.exists():
            raise FileNotFoundError(f"Missing {p.name}. Run build_pole_fig7_lookup.py / "
                                    f"build_wire_fig7d_lookup.py first.")
    pole, wire = np.load(POLE_NPZ), np.load(WIRE_NPZ)
    if "phases" not in pole or "p_fail_u16" not in pole:
        raise ValueError(f"{POLE_NPZ.name} is the old 3-phase-only file. "
                         f"Re-run build_pole_fig7_lookup.py.")
    if not (np.allclose(pole["span_bins_m"], wire["span_bins_m"])
            and np.allclose(pole["ice_mm"], wire["ice_mm"])
            and np.allclose(pole["wind_ms"], wire["wind_ms"])):
        raise ValueError("Pole and wire npz files use different grids -- rebuild one of them.")
    return dict(phases=pole["phases"], spans=pole["span_bins_m"], ages=pole["ages"],
                ice=pole["ice_mm"],
                wind=pole["wind_ms"],
                pole=(pole["p_fail_u16"] / np.float32(pole["p_fail_scale"])).astype(np.float32),
                wire={"slack": wire["p_slack"], "tension": wire["p_tension"]},
                v_toronto=float(pole["site_toronto"][3]))


def bilinear(surf, ice_grid, wind_grid, ice, wind):
    """Same linear interpolation the lookup files do (RegularGridInterpolator)."""
    i = np.clip(np.searchsorted(ice_grid, ice) - 1, 0, len(ice_grid) - 2)
    j = np.clip(np.searchsorted(wind_grid, wind) - 1, 0, len(wind_grid) - 2)
    ti = (ice - ice_grid[i]) / (ice_grid[i + 1] - ice_grid[i])
    tj = (wind - wind_grid[j]) / (wind_grid[j + 1] - wind_grid[j])
    p = ((1 - ti) * (1 - tj) * surf[i, j] + ti * (1 - tj) * surf[i + 1, j]
         + (1 - ti) * tj * surf[i, j + 1] + ti * tj * surf[i + 1, j + 1])
    return float(np.clip(p, 0, 1))


# -----------------------------------------------------------------------------
# One 3D panel
# -----------------------------------------------------------------------------
def draw_surface(ax, D, surf, title, ice_max, scenario=None, stride=STRIDE):
    """stride=1 for wires: their surface is a near-vertical cliff that looks jagged
    when sub-sampled."""
    ax.cla()
    keep = D["ice"] <= ice_max + 1e-9
    ice, wind = D["ice"][keep], D["wind"]
    Z = surf[keep]
    W, R = np.meshgrid(wind[::stride], ice[::stride])
    ax.plot_surface(W, R, Z[::stride, ::stride], cmap="viridis", vmin=0, vmax=1,
                    linewidth=0, antialiased=True, alpha=0.9)

    # Toronto concurrent gust (the default when CONCURRENT_WIND_MS = None)
    vt = D["v_toronto"]
    pt = [bilinear(surf, D["ice"], D["wind"], r, vt) for r in ice]
    ax.plot(np.full(ice.size, vt), ice, pt, color="red", lw=1.5,
            label=f"Toronto default {vt:.1f} m/s")

    label = title
    if scenario is not None:
        r, v = scenario
        p = bilinear(surf, D["ice"], D["wind"], r, v)
        ax.scatter([v], [r], [p], color="black", s=40, depthshade=False, zorder=10)
        ax.plot([v, v], [r, r], [0, p], color="black", lw=1, ls="--")
        label += f"\nP({r:g} mm, {v:g} m/s) = {p:.3f}"

    ax.set_title(label, fontsize=9)
    ax.set_xlabel("Concurrent gust (m/s)", fontsize=7)
    ax.set_ylabel("Radial ice (mm)", fontsize=7)
    ax.set_zlabel("P(fail)", fontsize=7)
    ax.set_xlim(0, wind[-1]); ax.set_ylim(0, ice_max); ax.set_zlim(0, 1)
    ax.tick_params(labelsize=6)
    ax.view_init(elev=25, azim=-130)     # similar angle to Ma Fig. 7


def span_label(L):
    return "91.44 (Ma)" if abs(L - 91.44) < 1e-6 else f"{L:g}"


def phase_index(D, n_phases):
    k = np.where(D["phases"] == n_phases)[0]
    if k.size == 0:
        raise ValueError(f"n_phases={n_phases} not in the pole file ({D['phases'].tolist()})")
    return int(k[0])


# -----------------------------------------------------------------------------
# Mode 1: interactive viewer (text boxes + button)
# -----------------------------------------------------------------------------
def bin_up(values, x):
    """Round UP to the next bin, exactly like the lookup files (span_bin / age_bin)."""
    k = int(np.searchsorted(values, x - 1e-9))
    if k >= len(values):
        raise ValueError(f"{x:g} is above the largest bin ({values[-1]:g})")
    return k


def interactive(D, ice_max, scenario, start_phases=3):
    from matplotlib.widgets import TextBox, Button, RadioButtons, Slider

    fig = plt.figure(figsize=(15, 8.5))
    try:
        fig.canvas.manager.set_window_title("Fig. 7 fragility surfaces (Ma et al. 2020)")
    except Exception:
        pass
    ax_p = fig.add_axes([0.00, 0.29, 0.50, 0.60], projection="3d")
    ax_w = fig.add_axes([0.50, 0.29, 0.50, 0.60], projection="3d")

    # ---- storm: text boxes + buttons (left) ----
    fig.text(0.02, 0.245, "Storm", fontsize=11, weight="bold")
    tb_ice = TextBox(fig.add_axes([0.13, 0.18, 0.07, 0.04]), "Ice (mm)  ",
                     initial="" if scenario is None else f"{scenario[0]:g}")
    tb_wind = TextBox(fig.add_axes([0.13, 0.12, 0.07, 0.04]), "Wind, 3-s gust (m/s)  ",
                      initial="" if scenario is None else f"{scenario[1]:g}")
    btn_clear = Button(fig.add_axes([0.02, 0.05, 0.08, 0.045]), "Clear point")
    btn = Button(fig.add_axes([0.12, 0.05, 0.08, 0.045]), "Plot point")

    # ---- bins: sliders (middle) ----
    fig.text(0.24, 0.245, "Bins", fontsize=11, weight="bold")
    n_s, n_a = len(D["spans"]), len(D["ages"])
    i40, j30 = int(np.searchsorted(D["spans"], 40)), int(np.searchsorted(D["ages"], 30))
    s_age = Slider(fig.add_axes([0.36, 0.19, 0.20, 0.03]), "Pole age",
                   0, n_a - 1, valinit=j30, valstep=1)
    s_trib = Slider(fig.add_axes([0.36, 0.13, 0.20, 0.03]), "Pole tributary span",
                    0, n_s - 1, valinit=i40, valstep=1)
    s_span = Slider(fig.add_axes([0.36, 0.07, 0.20, 0.03]), "Wire span length",
                    0, n_s - 1, valinit=i40, valstep=1)

    fig.text(0.635, 0.235, "Pole phases", fontsize=9)
    ph_labels = tuple(f"{ph}-phase" for ph in D["phases"])
    radio_ph = RadioButtons(fig.add_axes([0.635, 0.135, 0.07, 0.095]), ph_labels,
                            active=phase_index(D, start_phases))
    fig.text(0.635, 0.105, "Wire stringing", fontsize=9)
    radio = RadioButtons(fig.add_axes([0.635, 0.035, 0.07, 0.065]), ("slack", "tension"))

    # ---- results (right) ----
    fig.text(0.72, 0.245, "Result", fontsize=11, weight="bold")
    result = fig.text(0.72, 0.225, "", fontsize=10, va="top", family="monospace")
    fig.text(0.02, 0.012, "Click a box, type, then press Enter or click 'Plot point'. "
             "Drag the sliders to change bins. Drag a plot to rotate it. "
             "Red line = Toronto default concurrent gust.", fontsize=8, color="0.35")

    state = {"show_point": scenario is not None}

    def num(box, name):
        txt = box.text.strip()
        if txt == "":
            return None
        try:
            x = float(txt)
        except ValueError:
            raise ValueError(f"{name}: '{txt}' is not a number")
        if x < 0:
            raise ValueError(f"{name} must be >= 0")
        return x

    def update(_=None):
        j, i_p, i_w = int(s_age.val), int(s_trib.val), int(s_span.val)
        s_age.valtext.set_text(f"{D['ages'][j]} yr")
        s_trib.valtext.set_text(f"{span_label(D['spans'][i_p])} m")
        s_span.valtext.set_text(f"{span_label(D['spans'][i_w])} m")
        try:
            ice, wind = num(tb_ice, "Ice"), num(tb_wind, "Wind")
            if (ice is None) != (wind is None):
                raise ValueError("give both ice and wind (or leave both empty)")
            if ice is not None and (ice > D["ice"][-1] or wind > D["wind"][-1]):
                raise ValueError(f"grid only covers ice 0-{D['ice'][-1]:g} mm,\n"
                                 f"wind 0-{D['wind'][-1]:g} m/s")
        except ValueError as e:
            result.set_text(f"Error:\n{e}")
            result.set_color("crimson")
            fig.canvas.draw_idle()
            return

        sc = (ice, wind) if (ice is not None and state["show_point"]) else None
        top = ice_max if sc is None else min(max(ice_max, ice * 1.25), D["ice"][-1])
        mode = radio.value_selected
        k = ph_labels.index(radio_ph.value_selected)
        ph = int(D["phases"][k])
        pole_surf, wire_surf = D["pole"][k, i_p, j], D["wire"][mode][i_w]

        draw_surface(ax_p, D, pole_surf,
                     f"Pole ({ph}-phase): age {D['ages'][j]} yr, tributary span "
                     f"{span_label(D['spans'][i_p])} m", top, sc)
        draw_surface(ax_w, D, wire_surf,
                     f"Wire: span {span_label(D['spans'][i_w])} m ({mode})",
                     top, sc, stride=1)
        ax_w.legend(loc="upper left", fontsize=7)

        if sc is not None:
            pp = bilinear(pole_surf, D["ice"], D["wind"], ice, wind)
            pw = bilinear(wire_surf, D["ice"], D["wind"], ice, wind)
            others = [f"  {int(q)}-phase pole: "
                      f"{bilinear(D['pole'][kk, i_p, j], D['ice'], D['wind'], ice, wind):.4f}"
                      for kk, q in enumerate(D["phases"])]
            lines = [f"At {ice:g} mm ice, {wind:g} m/s gust:", "",
                     f"P(pole fails) = {pp:.4f}  ({ph}-ph)",
                     f"P(wire fails) = {pw:.4f}",
                     f"P(either)     = {1 - (1 - pp) * (1 - pw):.4f}",
                     "  (pole and wire independent)", "",
                     "Pole, all phase counts:"] + others
        else:
            lines = ["No storm point.", "Enter ice and wind,", "then press Enter."]
        result.set_text("\n".join(lines))
        result.set_color("black")
        fig.canvas.draw_idle()

    def plot_point(_=None):
        state["show_point"] = True
        update()

    def clear_point(_=None):
        state["show_point"] = False
        update()

    tb_ice.on_submit(plot_point)           # Enter in a box
    tb_wind.on_submit(plot_point)
    btn.on_clicked(plot_point)
    btn_clear.on_clicked(clear_point)
    for sl in (s_age, s_trib, s_span):
        sl.on_changed(update)
    radio.on_clicked(update)
    radio_ph.on_clicked(update)

    # widgets must stay referenced or matplotlib garbage-collects them (dead boxes)
    fig._widgets = (tb_ice, tb_wind, btn, btn_clear, s_age, s_trib, s_span, radio, radio_ph)
    update()
    plt.show()


# -----------------------------------------------------------------------------
# Mode 2: save every combination
# -----------------------------------------------------------------------------
def save_all(D, ice_max, scenario, feeder_only, phases=(1, 3)):
    OUT_DIR.mkdir(exist_ok=True)
    idx = range(len(D["spans"]))
    if feeder_only:
        want = sorted(set(FEEDER_POLE_BINS) | set(FEEDER_WIRE_BINS))
        idx = [i for i, L in enumerate(D["spans"]) if any(abs(L - w) < 1e-6 for w in want)]

    n_ages = len(D["ages"])
    n_sheets = 0
    for ph, i in [(ph, i) for ph in phases for i in idx]:
        k = phase_index(D, ph)
        L = D["spans"][i]
        fig = plt.figure(figsize=(18, 17))
        panels = [(D["pole"][k, i, j], f"Pole ({ph}-phase), {age} yr", STRIDE)
                  for j, age in enumerate(D["ages"])]
        panels += [(D["wire"]["slack"][i], "Wire (slack, default)", 1),
                   (D["wire"]["tension"][i], "Wire (tension)", 1)]
        ncol = 4
        nrow = int(np.ceil(len(panels) / ncol))
        for k, (surf, ttl, stride) in enumerate(panels):
            ax = fig.add_subplot(nrow, ncol, k + 1, projection="3d")
            draw_surface(ax, D, surf, ttl, ice_max, scenario, stride)
        sc = f"   |   scenario: {scenario[0]:g} mm ice, {scenario[1]:g} m/s" if scenario else ""
        fig.suptitle(f"Ma et al. (2020) Fig. 7 surfaces, {ph}-phase pole, span bin "
                     f"{span_label(L)} m (pole: tributary span, wire: span length){sc}\n"
                     f"red line = Toronto concurrent gust {D['v_toronto']:.1f} m/s",
                     fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        name = f"fig7_span{('91.44' if abs(L - 91.44) < 1e-6 else f'{L:g}')}m_{ph}ph.png"
        fig.savefig(OUT_DIR / name, dpi=110)
        plt.close(fig)
        n_sheets += 1
        print(f"saved {name}  ({n_ages} pole ages + 2 wire surfaces)")
    print(f"\n{n_sheets} sheets in {OUT_DIR}")


# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--save", action="store_true", help="save PNG sheets instead of the viewer")
    ap.add_argument("--feeder-bins", action="store_true",
                    help="with --save: only span bins feeder 1878 uses")
    ap.add_argument("--ice", type=float, help="scenario ice thickness (mm) to mark")
    ap.add_argument("--wind", type=float, help="scenario concurrent 3-s gust (m/s) to mark")
    ap.add_argument("--phases", type=int, nargs="+", choices=[1, 2, 3], default=None,
                    help="pole phase count(s). --save: sheets for each (default 1 3); "
                         "viewer: starting phase count (default 3)")
    ap.add_argument("--ice-max", type=float, default=80,
                    help="ice axis limit, mm (default 80 = Ma's range; grid goes to 150)")
    a = ap.parse_args()

    if (a.ice is None) != (a.wind is None):
        ap.error("give both --ice and --wind, or neither")
    scenario = (a.ice, a.wind) if a.ice is not None else None

    D = load()
    if a.save:
        matplotlib.use("Agg")
        save_all(D, a.ice_max, scenario, a.feeder_bins,
                 phases=tuple(a.phases) if a.phases else (1, 3))
    else:
        interactive(D, a.ice_max, scenario, start_phases=a.phases[0] if a.phases else 3)


if __name__ == "__main__":
    main()
