import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import contextily as ctx             # adds a basemap

combo = gpd.read_file("glacial_lakes_output/lakes_2018_2023_combo.geojson")

# Split into layers
g18   = combo.query("year == '2018'")
g23   = combo.query("year == '2023'")
gain  = combo.query("change == 'gain_2018‑23'")

# Project to Web‑Mercator for basemap
g18_web  = g18.to_crs(3857)
g23_web  = g23.to_crs(3857)
gain_web = gain.to_crs(3857)

ax = g18_web.plot(figsize=(8, 8), color="dodgerblue", edgecolor="black", alpha=0.4, linewidth=0.5)
g23_web.plot(ax=ax, color="limegreen", edgecolor="black", alpha=0.4, linewidth=0.5)
gain_web.plot(ax=ax, color="red", edgecolor="black", alpha=0.7)

ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery)  # satellite background
ax.set_title("Glacial lakes 2018 (blue) vs 2023 (green) + gains (red)")

# Custom legend
legend_elems = [
    Line2D([0], [0], marker='s', color='w', label='2018', markerfacecolor='dodgerblue', markersize=10, alpha=0.5),
    Line2D([0], [0], marker='s', color='w', label='2023', markerfacecolor='limegreen', markersize=10, alpha=0.5),
    Line2D([0], [0], marker='s', color='w', label='Gain', markerfacecolor='red', markersize=10, alpha=0.7),
]
ax.legend(handles=legend_elems, loc="lower right")
plt.tight_layout()
plt.show()
