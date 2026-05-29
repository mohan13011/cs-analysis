import awpy.data
import awpy.nav
import awpy.plot.nav
import json
import shapely.geometry as geom
import matplotlib.pyplot as plt

nav_path = awpy.data.NAVS_DIR / "de_dust2.json"
nav = awpy.nav.Nav.from_json(path=nav_path)

b_polygon = geom.Polygon([
    (-2300, 1600),
    (-1280, 1600),
    (-1280, 3200),
    (-2300, 3200),
])

b_tiles = []
for area_id, area in nav.areas.items():
    cx, cy = area.centroid.x, area.centroid.y
    if b_polygon.contains(geom.Point(cx, cy)):
        b_tiles.append(area_id)

# 手动去掉多余的瓦片
remove_tiles = [1612]
b_tiles = [t for t in b_tiles if t not in remove_tiles]

print(f"B点瓦片数: {len(b_tiles)}")
print(f"瓦片ID: {sorted(b_tiles)}")
# 保存B点瓦片ID
with open(r"C:\Users\yangmohan\OneDrive\桌面\cs-analysis\b_site_tiles.json", "w") as f:
    json.dump(sorted(b_tiles), f)

print("B点瓦片ID已保存到 b_site_tiles.json")
awpy.plot.nav.plot_map_tiles_selected(map_name="de_dust2", selected_tiles=b_tiles)
plt.show()