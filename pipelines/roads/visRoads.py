import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx      # basemap provider

# 1. Load vectors and re‑project for web basemap
gdf = gpd.read_file("road_output/roads_vector.geojson")
gdf3857 = gdf.to_crs(epsg=3857)          # Web‑Mercator
print("Total segments:", len(gdf))
print("Total length (km):", gdf.to_crs(32644).length.sum()/1000)

# 2. Plot
ax = gdf3857.plot(figsize=(8, 8), linewidth=1.2, color="red")
ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery)
ax.set_title("Extracted road centre‑lines")
ax.axis("off")
plt.tight_layout()
plt.show()
