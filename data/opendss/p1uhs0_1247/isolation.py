"""Step 4: locate device-bounded isolation zones in feeder1878 and display them.
Place next to Lines.dss and Buscoords.dss, then run: python3 isolation_step4.py
"""
from pathlib import Path
import re
import networkx as nx
import plotly.graph_objects as go
import opendssdirect as dss

MODEL = Path(__file__).resolve().parent
FAILED = "Line.l(r:p1udt2390-p1udt3160)"
USE_FUSES = False  # Change to False to compare against switch-only isolation


def prop(line, key, default=None):
    m = re.search(r"(?<!\S)" + re.escape(key) + r"\s*=\s*([^\s!]+)", line, re.I)
    return m.group(1) if m else default


def nodes(bus_spec, number_of_phases):
    parts = bus_spec.lower().split(".")
    phase_ids = [int(p) for p in parts[1:] if p.isdigit()]
    if not phase_ids:
        phase_ids = list(range(1, number_of_phases + 1))
    return [(parts[0], p) for p in phase_ids]


# Read actual electrical connections, including intermediate x/xx buses.
elements = []
for raw in (MODEL / "Lines.dss").read_text().splitlines():
    m = re.match(r"\s*new\s+line\.(\S+)", raw, re.I)
    if not m or prop(raw, "enabled", "y").lower() == "n":
        continue
    name = "Line." + m.group(1)
    bus1, bus2 = prop(raw, "bus1"), prop(raw, "bus2")
    if not bus1 or not bus2:
        continue
    b1, b2 = bus1.split(".")[0].lower(), bus2.split(".")[0].lower()
    if not (b1.startswith("p1udt") and b2.startswith("p1udt")):
        continue
    if "lv" in b1 or "lv" in b2:
        continue
    nphases = int(prop(raw, "phases", "3"))
    ends1, ends2 = nodes(bus1, nphases), nodes(bus2, nphases)
    if len(ends1) != len(ends2):
        raise ValueError("Unmatched phases on " + name)
    kind = ("fuse" if name.lower().startswith("line.fuse(") else
            "switch" if prop(raw, "switch", "n").lower() == "y" else
            "conductor")
    elements.append({"name": name, "kind": kind, "edges": list(zip(ends1, ends2)),
                     "bus1": b1, "bus2": b2})

operable = {"switch", "fuse"} if USE_FUSES else {"switch"}

# Omit operable devices to find zones between isolation boundaries.
G = nx.Graph()
for e in elements:
    if e["kind"] not in operable:
        G.add_edges_from(e["edges"])

# Account for feeder1878's primary-to-primary voltage-regulating transformer.
for phase in (1, 2, 3):
    G.add_edge(("p1udt1878-p1udt2480x", phase), ("p1udt2480", phase))

target = next((e for e in elements if e["name"].lower() == FAILED.lower()), None)
if target is None or target["kind"] != "conductor":
    raise ValueError("Choose a valid physical primary conductor: " + FAILED)

zone_nodes = set()
for u, v in target["edges"]:
    group = nx.node_connected_component(G, u)
    if v not in group:
        raise ValueError("The target's terminals aren't connected in the fixed graph")
    zone_nodes.update(group)

zone_lines = [e for e in elements if e["kind"] == "conductor"
              and any(u in zone_nodes and v in zone_nodes for u, v in e["edges"])]

boundary = [e for e in elements if e["kind"] in operable
            and any((u in zone_nodes) != (v in zone_nodes) for u, v in e["edges"])]

source = {("p1udt1878-p1uhs0_1247x", p) for p in (1, 2, 3)}
print("\nFailed conductor:", target["name"])
print("Conductors in isolation zone:", len(zone_lines))
print("Selected isolation devices:")
for e in boundary:
    print("  ", e["kind"].upper(), e["name"])
print("Zone touches source:", bool(zone_nodes & source))
if zone_nodes & source:
    raise RuntimeError("A source-side isolation device is missing; do not assume this zone can be isolated.")

# Operate the selected devices and run the same load-flow test as test.py.
def delivered_kw():
    total = 0.0
    for name in dss.Loads.AllNames():
        dss.Circuit.SetActiveElement("Load." + name)
        total += sum(dss.CktElement.Powers()[::2])
    return total


dss.Text.Command("Clear")
dss.Text.Command(f'Compile "{MODEL / "Master.dss"}"')
if not dss.Solution.Converged():
    raise RuntimeError("Baseline power flow did not converge")
before_kw = delivered_kw()
for e in boundary:
    if dss.Circuit.SetActiveElement(e["name"]) <= 0:
        raise RuntimeError("OpenDSS device not found: " + e["name"])
    dss.CktElement.Open(1, 0)   # All phases of this actual device
    print("Opened:", e["name"], "=>", dss.CktElement.IsOpen(1, 0))

if dss.Circuit.SetActiveElement(target["name"]) <= 0:
    raise RuntimeError("OpenDSS conductor not found: " + target["name"])
dss.CktElement.Enabled(False)
dss.Solution.Solve()
for e in boundary:
    dss.Circuit.SetActiveElement(e["name"])
    print("Still open after solve:", e["name"], "=>", dss.CktElement.IsOpen(1, 0))
print("Power flow converged:", dss.Solution.Converged())
if dss.Solution.Converged():
    after_kw = delivered_kw()
    print("Baseline delivered kW:", round(before_kw, 2))
    print("Post-isolation delivered kW:", round(after_kw, 2))
    print("Change in delivered kW:", round(before_kw - after_kw, 2))
    print("Note: load-power change is not yet the finalized unserved-load metric.")

# Plot using actual bus coordinates (for display only, not for graph connectivity).
coords = {}
for row in (MODEL / "Buscoords.dss").read_text().splitlines():
    parts = row.split()
    if len(parts) >= 3:
        try:
            coords[parts[0].lower()] = (float(parts[1]), float(parts[2]))
        except ValueError:
            pass


def endpoints(e):
    # Original geographic names are in r:busA-busB, unlike the internal x/xx buses.
    m = re.search(r"\(r:([^)]+)\)", e["name"], re.I)
    if not m or "-" not in m.group(1):
        return None
    a, b = m.group(1).lower().split("-", 1)
    return (coords[a], coords[b]) if a in coords and b in coords else None


fig = go.Figure()
for group, label, color, width in [
    ([e for e in elements if e["kind"] == "conductor"], "Primary lines", "#9299A2", 2),
    (zone_lines, "Isolation zone", "#E64B4B", 5),
]:
    for e in group:
        pts = endpoints(e)
        if pts:
            fig.add_trace(go.Scatter(
                x=[pts[0][0], pts[1][0]], y=[pts[0][1], pts[1][1]],
                mode="lines", line=dict(color=color, width=width),
                hovertext=e["name"], hoverinfo="text", name=label,
                showlegend=False))

for kind, color, marker in [("switch", "orange", "square"), ("fuse", "purple", "diamond")]:
    xs, ys, names = [], [], []
    for e in elements:
        if e["kind"] != kind:
            continue
        # One physical endpoint is usually a named (non-intermediate) bus.
        location = next((coords[b] for b in (e["bus1"], e["bus2"]) if b in coords), None)
        if location is None:
            pts = endpoints(e)
            location = ((pts[0][0] + pts[1][0]) / 2,
                        (pts[0][1] + pts[1][1]) / 2) if pts else None
        if location:
            xs.append(location[0]); ys.append(location[1]); names.append(e["name"])
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="markers", text=names,
        hovertemplate="%{text}<extra></extra>",
        marker=dict(color=color, size=7, symbol=marker), name=kind.title() + "s"))

xs, ys, names = [], [], []
for e in boundary:
    location = next((coords[b] for b in (e["bus1"], e["bus2"]) if b in coords), None)
    if location is None:
        pts = endpoints(e)
        location = ((pts[0][0] + pts[1][0]) / 2,
                    (pts[0][1] + pts[1][1]) / 2) if pts else None
    if location:
        xs.append(location[0]); ys.append(location[1]); names.append(e["name"])
fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers", text=names,
    hovertemplate="Open: %{text}<extra></extra>",
    marker=dict(color="black", size=13, symbol="x"), name="Isolation boundaries"))

if "p1udt1878" in coords:
    sx, sy = coords["p1udt1878"]
    fig.add_trace(go.Scatter(x=[sx], y=[sy], mode="markers",
        marker=dict(color="green", size=15, symbol="star"), name="Source"))

fig.update_layout(title="Feeder1878: proposed isolation zone", width=1100, height=800,
                  xaxis_title="Longitude", yaxis_title="Latitude")
fig.update_yaxes(scaleanchor="x", scaleratio=1)
fig.write_html(str(MODEL / "isolation_step4.html"), auto_open=True)
print("Map saved to isolation_step4.html")
