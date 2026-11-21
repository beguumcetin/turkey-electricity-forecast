from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
FIGURES_DIR = ROOT / "reports" / "figures"
MODELS_DIR = ROOT / "models"

TIMEZONE = "Europe/Istanbul"
START_DATE = "2019-01-01"

# Test period: the last 12 months are held out. Everything before is used for training.
TEST_MONTHS = 12

# Six most populous provinces: (latitude, longitude, weight). Weights are their
# approximate population shares among these six, used to build one national
# temperature series.
CITIES = {
    "istanbul": (41.01, 28.98, 0.46),
    "ankara": (39.93, 32.86, 0.17),
    "izmir": (38.42, 27.14, 0.13),
    "bursa": (40.19, 29.06, 0.09),
    "antalya": (36.90, 30.70, 0.08),
    "adana": (37.00, 35.32, 0.07),
}
