"""
Divide Canada into one region per Texas7k substation.

Inputs (put next to this script):
  canada_boundary.geojson        Canada outline (e.g. Statistics Canada cartographic boundary)
  texas7k_substations.csv        columns: sub_id, lat, lon   (from the Texas7k .aux/.csv files)
  canada_population_points.csv   OPTIONAL columns: lat, lon, pop  (e.g. census dissemination areas)

Output:
  canada_regions.geojson         one polygon per substation, name in AREA_NAME
                                 (loads straight into the Toronto/Leaflet app)

pip install geopandas shapely scipy scikit-learn pandas numpy
"""
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.ops import voronoi_diagram
from shapely.geometry import MultiPoint
from sklearn.cluster import MiniBatchKMeans
import time

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

CANADA_FILE = "lpr_000b21g_e.gml"
SUBS_FILE = "texas7k_substations.csv"
POP_FILE = "canada_population_points.csv"
CANADA_CRS = 3347   # Statistics Canada Lambert
TEXAS_CRS = 3081    # Texas Centric Albers

subs = pd.read_csv(SUBS_FILE)
n = len(subs)
log(f"{n} substations loaded")

canada = gpd.read_file(CANADA_FILE)
if canada.crs is None:
    canada = canada.set_crs(3347)
canada = canada.to_crs(CANADA_CRS)
canada["geometry"] = canada.geometry.simplify(5000)   # 2 km tolerance, BEFORE merging
canada_poly = canada.geometry.union_all()
log("Canada boundary loaded")

# 1) Choose n region centres in Canada, weighted by population so regions are
#    small where people (load) are and huge where there is wilderness.
try:
    pop = pd.read_csv(POP_FILE)
    g = gpd.GeoDataFrame(pop, geometry=gpd.points_from_xy(pop.lon, pop.lat), crs=4326).to_crs(CANADA_CRS)
    xy = np.c_[g.geometry.x, g.geometry.y]
    w = g["pop"].to_numpy(dtype=float)
except FileNotFoundError:
    # fallback: uniform points inside Canada (gives equal-AREA regions)
    import shapely
    minx, miny, maxx, maxy = canada_poly.bounds
    rng = np.random.default_rng(0)
    pts = np.empty((0, 2))
    while len(pts) < 30000:
        cand = np.c_[rng.uniform(minx, maxx, 100000), rng.uniform(miny, maxy, 100000)]
        pts = np.vstack([pts, cand[shapely.contains_xy(canada_poly, cand[:, 0], cand[:, 1])]])
    xy, w = pts[:30000], np.ones(30000)

log("placing region centres (k-means)...")
centres = MiniBatchKMeans(n_clusters=n, n_init=1, random_state=0, batch_size=4096, max_iter=100).fit(xy, sample_weight=w).cluster_centers_

# 2) Match each Texas7k substation to one Canadian centre so the layout keeps
#    Texas7k's rough geometry (neighbours stay neighbours). Hungarian assignment
#    on coordinates normalised to the unit square.
def unit(a):
    return (a - a.min(0)) / (a.max(0) - a.min(0))

tx = gpd.GeoSeries(gpd.points_from_xy(subs.lon, subs.lat), crs=4326).to_crs(TEXAS_CRS)
T = unit(np.c_[tx.x, tx.y])
C = unit(centres)
log("matching to Texas7k substations...")
rows, cols = linear_sum_assignment(cdist(T, C))   # O(n^3): fine up to ~10k substations
centre_for_sub = centres[cols[np.argsort(rows)]]

# 3) Voronoi cells around the centres, clipped to Canada's outline.
log("building Voronoi cells...")
cells = voronoi_diagram(MultiPoint([tuple(c) for c in centre_for_sub]), envelope=canada_poly)
cells = gpd.GeoDataFrame(geometry=list(cells.geoms), crs=CANADA_CRS)

sites = gpd.GeoDataFrame(
    {"AREA_NAME": subs.sub_id.astype(str)},
    geometry=gpd.points_from_xy(centre_for_sub[:, 0], centre_for_sub[:, 1]),
    crs=CANADA_CRS,
)
regions = gpd.sjoin(cells, sites, predicate="contains").drop(columns="index_right")
log("clipping to Canada...")
regions["geometry"] = regions.geometry.intersection(canada_poly)
regions = regions[~regions.is_empty]

regions.to_crs(4326).to_file("canada_regions.geojson", driver="GeoJSON")
print(f"wrote {len(regions)} regions for {n} substations")