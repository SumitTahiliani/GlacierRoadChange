# lake_extractor.py
from pathlib import Path
from matplotlib import pyplot as plt
import pandas as pd
import geopandas as gpd
import shapely.geometry as sgeom
from skimage.morphology import remove_small_holes, remove_small_objects
from rasterio.features import shapes
import rioxarray as rxr
import xarray as xr
from mapminer.miners import Sentinel2Miner

# ----------------------------
# CONFIG — keep editable
THRESH = 0.05            # NDWI threshold
MIN_PIX = 20            # remove blobs < 50 pixels
BANDS   = ["B03", "B08"] # Green, NIR
OUT_DIR = Path("glacial_lakes_output")
OUT_DIR.mkdir(exist_ok=True)
# ----------------------------

def to_reflectance(arr):          # Sentinel‑2 DN -> reflectance
    return (arr * 0.0001).clip(0, 1)

def extract_lakes(aoi_geom, date_range, tag):
    """Return GeoDataFrame of lake polygons for given date window."""
    miner = Sentinel2Miner()
    ds_raw = miner.fetch(polygon=aoi_geom, daterange=date_range)
    ds      = ds_raw.isel(time=0)[BANDS]

    green   = to_reflectance(ds["B03"])
    nir     = to_reflectance(ds["B08"])
    ndwi    = (green - nir) / (green + nir)

    tmpl = ds["B03"]
    ndwi  = ndwi.rio.write_transform(tmpl.rio.transform(), inplace=False)
    ndwi  = ndwi.rio.write_crs(tmpl.rio.crs, inplace=False)

    mask = (ndwi > THRESH).compute().values
    mask = remove_small_objects(mask, MIN_PIX)
    mask = remove_small_holes(mask, area_threshold=MIN_PIX)

    geoms = shapes(mask.astype("uint8"), mask=mask, transform=ndwi.rio.transform())
    records = []

    for geom, val in geoms:
        if val == 1:
            poly = sgeom.shape(geom)
            if poly.is_valid and poly.area > 0:
                records.append({
                    "geometry": poly,
                    "ndwi_thr": THRESH,
                    "year": tag
                })
    if not records:                          # nothing extracted
        raise RuntimeError(
            f"No lake polygons detected for {tag}. "
            "Try lowering THRESH or MIN_PIXELS, or check the NDWI image."
        )
    lakes = gpd.GeoDataFrame(pd.DataFrame(records),
                             geometry="geometry",
                             crs=tmpl.rio.crs)

    # Save per‑year GeoJSON / Shapefile
    lakes.to_file(OUT_DIR / f"lakes_{tag}.geojson", driver="GeoJSON")
    lakes.to_file(OUT_DIR / f"lakes_{tag}.shp")
    ndwi.plot(cmap="Blues", vmin=-0.5, vmax=1.0)
    plt.show()
    return lakes
