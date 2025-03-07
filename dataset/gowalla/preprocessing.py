import json
import pandas as pd

import geopandas as gpd

base = "/media/claudio/e5069d0b-8c3d-4850-8e10-319862c4c082/home/claudio/Documentos/gowalla/"

def p_spot(df):

    cat = df["spot_categories"].to_numpy()

    placeids = []
    urls = []
    subcategories = []
    for e in cat:
        e = e.replace("[", "").replace("]", "").replace("Dunkin' ", "Dunkin ").replace("Peet's ", "Peets ").replace("Doctor's", "Doctors").replace("Joe's", "Joes").replace("Lowe's", "Lowes").replace("McDonald's", "McDonalds").replace("Victoria's", "Victorias").replace("Jerry's", "Jerrys").replace("Women's", "Womens").replace("Seattle's", "Seattles").replace("Men's", "Mens").replace("Children's", "Childrens").replace("'", "\"")
        e = json.loads(e)
        urls.append(e["url"].replace("/categories/", ""))
        subcategories.append(e["name"])

    return pd.DataFrame({"placeid": df["placeid"].to_numpy(), "id_subcategory": urls, "subcategory": subcategories, "lng": df["lng"].to_numpy(), "lat": df["lat"].to_numpy()}).drop_duplicates()

def category_structure(df):

    spotcategories = df["spot_categories"].to_numpy()


    categories = []
    subcategories = []

    for e in spotcategories:

        # e = e.replace("[", "").replace("]", "").replace("Dunkin' ", "Dunkin ").replace("Peet's ", "Peets ").replace("Doctor's", "Doctors").replace("Joe's", "Joes").replace("Lowe's", "Lowes").replace("McDonald's", "McDonalds").replace("Victoria's", "Victorias").replace("Jerry's", "Jerrys").replace("Women's", "Womens").replace("Seattle's", "Seattles").replace("Men's", "Mens").replace("Children's", "Childrens").replace("'", "\"")
        # e = json.loads(e)
        spot_categories = e['spot_categories']
        for j in spot_categories:
            subcategory = j["url"].replace("/categories/", "")
            categories.append(e["name"])
            subcategories.append(subcategory)

    return pd.DataFrame({"category": categories, "id_subcategory": subcategories}).drop_duplicates()





df = pd.read_csv(base + "gowalla_checkins.csv")
df['datetime'] = pd.to_datetime(df['datetime'])
print(df)
spot = pd.read_csv(base + "gowalla_spots_subset1.csv")[["id", "spot_categories", "lng", "lat"]]
spot["placeid"] = spot["id"].to_numpy()
spot = spot[["placeid", "spot_categories", "lng", "lat"]]
spot = p_spot(spot)
df_json = pd.read_json(base + "gowalla_category_structure.json")
df_json = category_structure(df_json)

spot = spot.join(df_json.set_index("id_subcategory"), on="id_subcategory", how="inner")
df = df.join(spot.set_index("placeid"), on="placeid", how="inner")

print(spot)
print(df_json)
print(df)

df = gpd.GeoDataFrame(
    df, geometry=gpd.points_from_xy(df.lng, df.lat), crs="EPSG:4326"
)



us_filnname = "/media/claudio/e5069d0b-8c3d-4850-8e10-319862c4c082/home/claudio/Documentos/gowalla/us_states/cb_2018_us_state_500k.shp"
gdf = gpd.GeoDataFrame.from_file(us_filnname).query("STUSPS == 'TX'")

df = df.sjoin(gdf, how="inner", predicate="within")

print(df.columns)

df = df[['userid', 'placeid', 'datetime', 'id_subcategory', 'subcategory', 'category', 'STUSPS', 'NAME', 'lat', 'lng']]

print(df)

df['datetime'] = pd.to_datetime(df['datetime'], infer_datetime_format=True)
df['hour'] = df['datetime'].dt.hour + (df['datetime'].dt.dayofweek // 5) * 24

print(df)

df.to_csv(base + "gowalla_checkins_texas.csv", index=False)



