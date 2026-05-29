from awpy import Demo
import awpy.data
import awpy.nav
import polars as pl
import math
from shapely.geometry import Point, Polygon as ShapelyPolygon
from shapely.strtree import STRtree
import matplotlib.pyplot as plt

# === 1. 加载数据 ===
demo_path = r"C:\Users\yangmohan\OneDrive\桌面\demo样本\g161-n-20260329215254871005578_de_dust2.dem"
print("加载demo...")
dem = Demo(path=demo_path, verbose=False)
dem.parse(player_props=["health", "pitch", "yaw"])

print("加载导航网格...")
nav_path = awpy.data.NAVS_DIR / "de_dust2.json"
nav = awpy.nav.Nav.from_json(path=nav_path)

ticks = dem.ticks
rounds = dem.rounds
damages = dem.damages

B_site = {"x_min": -2300, "x_max": -1280, "y_min": 1600, "y_max": 3200}

# === 2. 构建回合活跃时段（freeze_start ~ 下一回合 freeze_start）===
print("构建回合索引...")
active_ticks = set()
tick_to_round = {}
round_list = list(rounds.iter_rows(named=True))
for i, r in enumerate(round_list):
    start_tick = r["start"]
    if i + 1 < len(round_list):
        end_tick = round_list[i + 1]["start"]
    else:
        end_tick = r["official_end"]
    for t in range(start_tick, end_tick + 1):
        active_ticks.add(t)
        tick_to_round[t] = r["round_num"]

# === 3. 构建伤害索引 ===
print("构建伤害索引...")
dmg_index = {}
for d in damages.iter_rows(named=True):
    t = d["tick"]
    victim = d["victim_name"]
    dmg = d["dmg_health"]
    if t not in dmg_index:
        dmg_index[t] = {}
    dmg_index[t][victim] = dmg_index[t].get(victim, 0) + dmg

# === 4. 空间索引 ===
print("构建空间索引...")
nav_polygons = {}
for area_id, area in nav.areas.items():
    corners = [(c.x, c.y) for c in area.corners]
    nav_polygons[area_id] = ShapelyPolygon(corners)

polygon_list = list(nav_polygons.values())
tree = STRtree(polygon_list)

def point_in_nav(x, y):
    candidates = tree.query(Point(float(x), float(y)))
    for poly in candidates:
        try:
            if poly.contains(Point(float(x), float(y))):
                return True
        except AttributeError:
            continue
    return False

# === 5. 视野判断 ===
def can_see(observer, target, max_distance=1500, fov_h=90, fov_v=70, steps=40):
    ox, oy, oz, oyaw, opitch = observer
    tx, ty, tz = target

    dx, dy, dz = tx - ox, ty - oy, tz - oz
    dist = math.sqrt(dx**2 + dy**2 + dz**2)

    if dist < 1:
        return True
    if dist > max_distance:
        return False

    yaw_rad = math.radians(oyaw)
    look_x = math.cos(yaw_rad)
    look_y = math.sin(yaw_rad)
    dot_h = max(-1, min(1, look_x * dx / dist + look_y * dy / dist))
    if math.degrees(math.acos(dot_h)) >= fov_h / 2:
        return False

    dist_2d = math.sqrt(dx**2 + dy**2)
    target_pitch = math.degrees(math.atan2(-dz, dist_2d))
    pitch_diff = abs(opitch - target_pitch)
    if pitch_diff > 180:
        pitch_diff = 360 - pitch_diff
    if pitch_diff >= fov_v / 2:
        return False

    for i in range(1, steps):
        px = ox + dx * i / steps
        py = oy + dy * i / steps
        if not point_in_nav(px, py):
            return False

    return True

# === 6. 区域判断 ===
def get_location(x, y):
    if B_site["x_min"] <= x <= B_site["x_max"] and B_site["y_min"] <= y <= B_site["y_max"]:
        return "B"
    return "Other"

# === 7. 全帧计算所有玩家的压力（不限阵营）===
print("计算所有玩家的压力...")

results = []
tick_list = ticks["tick"].unique().sort().to_list()
total = len(tick_list)

for idx, tick_id in enumerate(tick_list):
    if idx % 5000 == 0:
        print(f"  进度: {idx}/{total}")

    if tick_id not in active_ticks:
        continue

    frame = ticks.filter(ticks["tick"] == tick_id)
    round_num = tick_to_round.get(tick_id, -1)
    dmg_this_tick = dmg_index.get(tick_id, {})

    # 不限阵营，所有玩家都算
    for player in frame.iter_rows(named=True):

        is_dead = player["health"] <= 0

        if is_dead:
            results.append({
                "tick": tick_id,
                "round_num": round_num,
                "name": player["name"],
                "side": player["side"],
                "location": get_location(player["X"], player["Y"]),
                "visible_t": 0,
                "hp": 0,
                "nearby_teammate": 0,
                "dmg_taken": 0,
                "pressure": 0.0
            })
            continue

        # 视野内敌人（对方阵营）
        enemy_side = "t" if player["side"] == "ct" else "ct"
        visible_enemy = 0
        for enemy in frame.filter(pl.col("side") == enemy_side).iter_rows(named=True):
            if enemy["health"] <= 0:
                continue
            if can_see(
                (player["X"], player["Y"], player["Z"], player["yaw"], player["pitch"]),
                (enemy["X"], enemy["Y"], enemy["Z"])
            ):
                visible_enemy += 1

        # 附近队友
        nearby_teammate = 0
        for mate in frame.filter((pl.col("side") == player["side"]) & (pl.col("name") != player["name"])).iter_rows(named=True):
            if mate["health"] <= 0:
                continue
            dist = math.sqrt((mate["X"] - player["X"])**2 + (mate["Y"] - player["Y"])**2)
            if dist < 800:
                nearby_teammate += 1

        # 伤害因子
        dmg_taken = dmg_this_tick.get(player["name"], 0)

        f_visible = visible_enemy / 5
        f_hp = 1 - player["health"] / 100
        f_alone = 1 - min(nearby_teammate, 3) / 3
        f_dmg = min(dmg_taken / 50, 1.0)

        pressure = 0.35 * f_visible + 0.25 * f_hp + 0.15 * f_alone + 0.25 * f_dmg

        results.append({
            "tick": tick_id,
            "round_num": round_num,
            "name": player["name"],
            "side": player["side"],
            "location": get_location(player["X"], player["Y"]),
            "visible_enemy": visible_enemy,
            "hp": player["health"],
            "nearby_teammate": nearby_teammate,
            "dmg_taken": dmg_taken,
            "pressure": pressure
        })

result_df = pl.DataFrame(results)

# === 8. 输出 ===
print(f"\n=== 全局压力统计（全阵营）===")
print(f"总记录数: {result_df.height}")

print("\n--- 各玩家压力 ---")
player_stats = result_df.group_by("name").agg([
    pl.col("pressure").mean().alias("avg_pressure"),
    pl.col("pressure").max().alias("max_pressure"),
    pl.col("dmg_taken").sum().alias("total_dmg_taken"),
    pl.len().alias("frames")
]).sort("avg_pressure", descending=True)
print(player_stats.to_pandas().to_string())

# === 9. 画图 ===
player_name = "november02"
player_df = result_df.filter(pl.col("name") == player_name).sort("tick")

rounds_list = sorted(player_df["round_num"].unique().to_list())
n_rounds = len(rounds_list)
print(f"\n{player_name} 回合数: {n_rounds}")

fig, axes = plt.subplots(n_rounds, 1, figsize=(14, 1.5 * n_rounds), sharex=True)

if n_rounds == 1:
    axes = [axes]

for i, rn in enumerate(rounds_list):
    rd = player_df.filter(pl.col("round_num") == rn)
    x = (rd["tick"] - rd["tick"].min()) / 128
    
    axes[i].plot(x, rd["pressure"], linewidth=0.5, color="red", alpha=0.9)
    axes[i].fill_between(x, 0, rd["pressure"], alpha=0.15, color="red")
    
    axes[i].set_ylabel(f"R{rn}", fontsize=8, rotation=0, labelpad=20)
    axes[i].set_ylim(-0.02, 0.75)
    axes[i].grid(True, alpha=0.2)
    axes[i].tick_params(labelsize=7)

axes[-1].set_xlabel("Time within round (seconds)")
fig.suptitle(f"{player_name} Pressure by Round (All Sides)", fontsize=12, y=1.01)
plt.tight_layout()
plt.show()