import pandas as pd

# ===============================
# Paths
# ===============================
BASE = "/media/gustavo/e5069d0b-8c3d-4850-8e10-319862c4c082/home/claudio/Downloads/dataset_TIST2015/"
CHECKINS_PATH = f"{BASE}dataset_TIST2015_Checkins.txt"
POIS_PATH = f"{BASE}dataset_TIST2015_POIs.txt"
CITIES_PATH = f"{BASE}dataset_TIST2015_Cities.txt"
OUTPUT_PATH = f"{BASE}dataset_tist2015_joined.csv"

# ===============================
# Load datasets
# ===============================
checkins = pd.read_csv(
    CHECKINS_PATH,
    sep="\t",
    header=None,
    names=["user_id", "venue_id", "utc_time", "timezone_offset"]
)

pois = pd.read_csv(
    POIS_PATH,
    sep="\t",
    header=None,
    names=["venue_id", "latitude", "longitude", "category", "country_code"]
)

cities = pd.read_csv(
    CITIES_PATH,
    sep="\t",
    header=None,
    names=["city", "city_lat", "city_lon", "country_code", "timezone"]
)

print(checkins.head())

print(pois.head())

print(cities)

# ===============================
# Timestamp processing
# ===============================
checkins["utc_time"] = pd.to_datetime(
    checkins["utc_time"],
    format="%a %b %d %H:%M:%S %z %Y",
    errors="coerce"
)

# Remove registros inválidos
checkins = checkins.dropna(subset=["utc_time"])

# ===============================
# Join: Checkins ⨝ POIs
# ===============================
df = checkins.merge(
    pois,
    on="venue_id",
    how="inner"
)

# ===============================
# Join: (Checkins + POIs) ⨝ Cities
# (via country_code)
# ===============================
df = df.merge(
    cities,
    on="country_code",
    how="left"
)

# ===============================
# Sorting (important for sequences)
# ===============================
df = df.sort_values(by=["user_id", "utc_time"])

# ===============================
# Cleanup / final columns
# ===============================
df = df[
    [
        "user_id",
        "venue_id",
        "category",
        "utc_time",
        "latitude",
        "longitude",
        "city",
        "timezone"
    ]
]

# ===============================
# Save
# ===============================
df.to_csv(OUTPUT_PATH, index=False)

print("✅ Dataset final salvo em:", OUTPUT_PATH)
print("Total de registros:", len(df))
print("Usuários únicos:", df["user_id"].nunique())
print("Categorias únicas:", df["category"].nunique())
