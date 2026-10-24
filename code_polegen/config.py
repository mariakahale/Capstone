from pathlib import Path

FEEDER_NAME = "1878"

# Resolved relative to this file so it works on any machine with the repo cloned
BASE_DIR = Path(__file__).resolve().parent
MASTER_PATH = str(BASE_DIR / "p1uhs0_1247--p1udt1878" / "Master.dss")

# Nominal line-to-line primary voltage
PRIMARY_KV_LL = 12.47

# Maximum assumed synthetic pole span
POLE_SPACING_M = 60.0