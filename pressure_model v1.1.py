from awpy import Demo
import awpy.data
import awpy.nav
from awpy.stats import rating
import polars as pl
import math
import os
from shapely.geometry import Point, Polygon as ShapelyPolygon
from shapely.strtree import STRtree
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
from tqdm import tqdm

# === 1. 加载数据 ===
demo_path = r"C:\Users\yangmohan\OneDrive\桌面\demo样本\g161-n-20260329215254871005578_de_dust2.dem"
cache_path = r"C:\Users\yangmohan\OneDrive\桌面\cs-analysis\result_cache.parquet"

# 检查缓存
if os.path.exists(cache_path):
    print("从缓存加载结果...")
    result_df = pl.read_parquet(cache_path)
    print(f"总记录数: {result_df.height}")
    
    # 需要重新构建回合信息才能画图
    dem = Demo(path=demo_path, verbose=False)
    dem.parse()
    rounds = dem.rounds
    # 死亡索引
    death_ticks = {}
    for k in dem.kills.iter_rows(named=True):
        death_ticks[(k["victim_name"], k["round_num"])] = k["tick"]
    # 回合胜负
    round_winner = {}
    for r in rounds.iter_rows(named=True):
        round_winner[r["round_num"]] = r["winner"]
    # 下包/爆炸/拆包
    bomb_plant_ticks = {}
    bomb_explode_ticks = {}
    bomb_defuse_ticks = {}
    for r in rounds.iter_rows(named=True):
        rn = r["round_num"]
        if r["bomb_plant"] is not None:
            bomb_plant_ticks[rn] = int(r["bomb_plant"])
        if r.get("reason") == "bomb_exploded":
            bomb_explode_ticks[rn] = int(r["end"])
        if r.get("reason") == "bomb_defused":
            bomb_defuse_ticks[rn] = int(r["end"])

else:
    print("加载demo...")
    dem = Demo(path=demo_path, verbose=False)
    dem.parse(player_props=["health", "pitch", "yaw"])

    print("加载导航网格...")
    nav_path = awpy.data.NAVS_DIR / "de_dust2.json"
    nav = awpy.nav.Nav.from_json(path=nav_path)

    ticks = dem.ticks
    rounds = dem.rounds
    damages = dem.damages
    grenades = dem.grenades
    infernos = dem.infernos

    B_site = {"x_min": -2300, "x_max": -1280, "y_min": 1600, "y_max": 3200}

    # === 1.5. 个人能力计算 ===
    print("计算个人能力...")
    rating_df = rating(dem)

    rating_all = rating_df.filter(pl.col("side") == "all").select(["name", "rating"])
    r_min = rating_all["rating"].min()
    r_max = rating_all["rating"].max()
    if r_max > r_min:
        rating_all = rating_all.with_columns(
            ((pl.col("rating") - r_min) / (r_max - r_min)).alias("global_skill")
        )
    else:
        rating_all = rating_all.with_columns(pl.lit(0.5).alias("global_skill"))

    global_skill_map = {}
    for row in rating_all.iter_rows(named=True):
        global_skill_map[row["name"]] = row["global_skill"]

    side_skill_map = {}
    for side in ["ct", "t"]:
        side_ratings = rating_df.filter(pl.col("side") == side)
        if side_ratings.height > 0:
            r_min_s = side_ratings["rating"].min()
            r_max_s = side_ratings["rating"].max()
            if r_max_s > r_min_s:
                side_ratings = side_ratings.with_columns(
                    ((pl.col("rating") - r_min_s) / (r_max_s - r_min_s)).alias("side_skill")
                )
            else:
                side_ratings = side_ratings.with_columns(pl.lit(0.5).alias("side_skill"))
            for row in side_ratings.iter_rows(named=True):
                side_skill_map[(row["name"], side)] = row["side_skill"]

    player_side_rounds = {}
    for row in rating_df.filter(pl.col("side") != "all").iter_rows(named=True):
        player_side_rounds[(row["name"], row["side"])] = row["n_rounds"]

    def get_skill(name, side):
        gs = global_skill_map.get(name, 0.5)
        ss = side_skill_map.get((name, side), gs)
        side_rds = player_side_rounds.get((name, side), 0)
        if side_rds <= 2: w_g, w_s = 0.8, 0.2
        elif side_rds <= 5: w_g, w_s = 0.5, 0.5
        else: w_g, w_s = 0.2, 0.8
        return w_g * gs + w_s * ss

    print("玩家能力系数 (全局为基础，分边为修正):")
    for name in global_skill_map:
        for side in ["ct", "t"]:
            skill = get_skill(name, side)
            rds = player_side_rounds.get((name, side))
            if rds is None:
                print(f"  {name} ({side.upper()}, 无数据): {skill:.3f}")
            else:
                print(f"  {name} ({side.upper()}, {rds}回合): {skill:.3f}")

    # === 2. 回合索引 ===
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

    round_winner = {}
    for r in rounds.iter_rows(named=True):
        round_winner[r["round_num"]] = r["winner"]

    # === 2.5. 下包索引 ===
    print("构建下包索引...")
    bomb_planted_ticks = set()
    bomb_site_map = {}
    bomb_plant_ticks = {}
    bomb_explode_ticks = {}
    bomb_defuse_ticks = {}

    for r in rounds.iter_rows(named=True):
        rn = r["round_num"]
        if r["bomb_plant"] is not None:
            plant_tick = int(r["bomb_plant"])
            end_tick = r["end"]
            bomb_plant_ticks[rn] = plant_tick
            for t in range(plant_tick, int(end_tick) + 1):
                bomb_planted_ticks.add(t)
        if r.get("reason") == "bomb_exploded":
            bomb_explode_ticks[rn] = int(r["end"])
        if r.get("reason") == "bomb_defused":
            bomb_defuse_ticks[rn] = int(r["end"])

    # === 2.6. 死亡索引 ===
    death_ticks = {}
    for k in dem.kills.iter_rows(named=True):
        death_ticks[(k["victim_name"], k["round_num"])] = k["tick"]

    # === 3-9. 各种索引 + 视野判断（同前，省略注释）===
    print("构建伤害索引...")
    dmg_index = {}
    for d in damages.iter_rows(named=True):
        t = d["tick"]
        victim = d["victim_name"]
        dmg = d["dmg_health"]
        if t not in dmg_index:
            dmg_index[t] = {}
        dmg_index[t][victim] = dmg_index[t].get(victim, 0) + dmg

    print("构建闪光弹索引...")
    flash_proj = grenades.filter(pl.col("grenade_type") == "CFlashbangProjectile").sort("tick")
    flash_explosion_pos = {}
    for g in flash_proj.iter_rows(named=True):
        eid = g["entity_id"]
        if g["X"] is not None and g["Y"] is not None:
            flash_explosion_pos[eid] = (g["X"], g["Y"], g["Z"])

    flash_index = {}
    for g in grenades.filter(pl.col("grenade_type") == "CFlashbang").iter_rows(named=True):
        t = g["tick"]
        eid = g["entity_id"]
        if eid in flash_explosion_pos:
            x, y, z = flash_explosion_pos[eid]
            if math.isnan(x) or math.isnan(y): continue
            if t not in flash_index:
                flash_index[t] = []
            flash_index[t].append((x, y, z))

    print("构建燃烧弹索引...")
    inferno_index = {}
    for inf in infernos.iter_rows(named=True):
        x, y, z = inf["X"], inf["Y"], inf["Z"]
        for t in range(inf["start_tick"], inf["end_tick"] + 1):
            if t not in inferno_index:
                inferno_index[t] = []
            inferno_index[t].append((x, y, z))

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
                if poly.contains(Point(float(x), float(y))): return True
            except AttributeError: continue
        return False

    def can_see(observer, target, max_distance=1500, fov_h=90, fov_v=70, steps=40):
        ox, oy, oz, oyaw, opitch = observer
        tx, ty, tz = target
        dx, dy, dz = tx - ox, ty - oy, tz - oz
        dist = math.sqrt(dx**2 + dy**2 + dz**2)
        if dist < 1: return True
        if dist > max_distance: return False
        yaw_rad = math.radians(oyaw)
        look_x, look_y = math.cos(yaw_rad), math.sin(yaw_rad)
        dot_h = max(-1, min(1, look_x * dx/dist + look_y * dy/dist))
        if math.degrees(math.acos(dot_h)) >= fov_h / 2: return False
        dist_2d = math.sqrt(dx**2 + dy**2)
        target_pitch = math.degrees(math.atan2(-dz, dist_2d))
        pitch_diff = abs(opitch - target_pitch)
        if pitch_diff > 180: pitch_diff = 360 - pitch_diff
        if pitch_diff >= fov_v / 2: return False
        for i in range(1, steps):
            px = ox + dx * i / steps
            py = oy + dy * i / steps
            if not point_in_nav(px, py): return False
        return True

    def facing_flash(player, flash_pos, max_dist=800, fov=180):
        dx, dy = flash_pos[0] - player["X"], flash_pos[1] - player["Y"]
        dist_2d = math.sqrt(dx**2 + dy**2)
        if dist_2d < 1: return True
        dist_3d = math.sqrt(dx**2 + dy**2 + (flash_pos[2] - player["Z"])**2)
        if dist_3d > max_dist: return False
        yaw_rad = math.radians(player["yaw"])
        look_x, look_y = math.cos(yaw_rad), math.sin(yaw_rad)
        target_x, target_y = dx / dist_2d, dy / dist_2d
        dot = max(-1, min(1, look_x * target_x + look_y * target_y))
        angle = math.degrees(math.acos(dot))
        return angle < fov / 2

    def get_location(x, y):
        if B_site["x_min"] <= x <= B_site["x_max"] and B_site["y_min"] <= y <= B_site["y_max"]:
            return "B"
        return "Other"

    # === 10. 全帧计算 ===
    print("计算所有玩家的压力...")
    tick_set = set(ticks["tick"].unique().to_list())
    tick_set.update(grenades["tick"].unique().to_list())
    tick_list = sorted(tick_set)[::3]

    results = []

    for tick_id in tqdm(tick_list, desc="计算压力", ncols=80):
        if tick_id not in active_ticks: continue
        frame = ticks.filter(ticks["tick"] == tick_id)
        round_num = tick_to_round.get(tick_id, -1)
        dmg_this_tick = dmg_index.get(tick_id, {})
        flashes_this_tick = flash_index.get(tick_id, [])
        fires_this_tick = inferno_index.get(tick_id, [])
        bomb_planted = 1 if tick_id in bomb_planted_ticks else 0

        for player in frame.iter_rows(named=True):
            is_dead = player["health"] <= 0
            if is_dead:
                results.append({
                    "tick": tick_id, "round_num": round_num,
                    "name": player["name"], "side": player["side"],
                    "location": get_location(player["X"], player["Y"]),
                    "visible_enemy": 0, "hp": 0, "nearby_teammate": 0,
                    "dmg_taken": 0, "flashed": 0, "near_fire": 0,
                    "bomb_planted": bomb_planted,
                    "pressure": 0.0, "death_prob": 0.0
                })
                continue

            enemy_side = "t" if player["side"] == "ct" else "ct"
            visible_enemy = 0
            for enemy in frame.filter(pl.col("side") == enemy_side).iter_rows(named=True):
                if enemy["health"] <= 0: continue
                if can_see((player["X"], player["Y"], player["Z"], player["yaw"], player["pitch"]),
                           (enemy["X"], enemy["Y"], enemy["Z"])):
                    visible_enemy += 1

            nearby_teammate = 0
            for mate in frame.filter((pl.col("side") == player["side"]) & (pl.col("name") != player["name"])).iter_rows(named=True):
                if mate["health"] <= 0: continue
                if math.sqrt((mate["X"]-player["X"])**2 + (mate["Y"]-player["Y"])**2) < 800:
                    nearby_teammate += 1

            dmg_taken = dmg_this_tick.get(player["name"], 0)

            flashed = 0
            for fx, fy, fz in flashes_this_tick:
                if facing_flash(player, (fx, fy, fz)):
                    flashed = 1
                    break

            near_fire = 0
            for fx, fy, fz in fires_this_tick:
                if math.sqrt((player["X"]-fx)**2 + (player["Y"]-fy)**2 + (player["Z"]-fz)**2) < 300:
                    near_fire = 1
                    break

            f_visible = visible_enemy / 5
            f_hp = 1 - player["health"] / 100
            f_alone = 1 - min(nearby_teammate, 3) / 3
            f_dmg = min(dmg_taken / 50, 1.0)
            f_flash = flashed
            f_fire = near_fire
            f_bomb = bomb_planted if player["side"] == "ct" else -0.3 * bomb_planted

            # 压力值
            raw_p = 0.28*f_visible + 0.18*f_hp + 0.09*f_alone + 0.18*f_dmg + 0.12*f_flash + 0.05*f_fire + 0.10*f_bomb
            k, mid = 8, 0.25
            raw_p = 1 / (1 + math.exp(-k * (raw_p - mid)))
            skill = get_skill(player["name"], player["side"])
            pressure = raw_p * (1 - 0.25 * skill)

            # 死亡概率
            death_raw = (5.4518*f_visible + 2.0660*f_hp + 0.7983*f_alone 
                         + 4.6412*f_dmg - 0.2320*f_flash + 0.8431*f_fire + 0.5447*f_bomb)
            death_prob = 1 / (1 + math.exp(-(-1.6194 + death_raw)))

            results.append({
                "tick": tick_id, "round_num": round_num,
                "name": player["name"], "side": player["side"],
                "location": get_location(player["X"], player["Y"]),
                "visible_enemy": visible_enemy, "hp": player["health"],
                "nearby_teammate": nearby_teammate, "dmg_taken": dmg_taken,
                "flashed": flashed, "near_fire": near_fire,
                "bomb_planted": bomb_planted,
                "pressure": pressure, "death_prob": death_prob
            })

    result_df = pl.DataFrame(results)
    result_df.write_parquet(cache_path)
    print(f"结果已缓存到 {cache_path}")

# === 11. 输出 ===
print(f"\n=== 全局压力 + 死亡概率统计 ===")
print(f"总记录数: {result_df.height}")

print("\n--- 各玩家压力 ---")
player_stats = result_df.group_by("name").agg([
    pl.col("pressure").mean().alias("avg_pressure"),
    pl.col("pressure").max().alias("max_pressure"),
    pl.col("death_prob").mean().alias("avg_death_prob"),
    pl.col("death_prob").max().alias("max_death_prob"),
    pl.col("dmg_taken").sum().alias("total_dmg"),
    pl.col("flashed").sum().alias("total_flashed"),
    pl.col("near_fire").sum().alias("total_near_fire"),
    pl.col("bomb_planted").sum().alias("bomb_frames"),
    pl.len().alias("frames")
]).sort("avg_pressure", descending=True)
print(player_stats.to_pandas().to_string())

print("\n--- 压力峰值 Top 15 ---")
top15 = result_df.top_k(15, by="pressure").select([
    "tick", "round_num", "name", "side", "visible_enemy", "hp", "dmg_taken", "flashed", "pressure", "death_prob"
])
print(top15.to_pandas().to_string())

# === 12. 画图 ===
player_name = "november02"
player_df = result_df.filter(pl.col("name") == player_name).sort("tick")

rounds_list = sorted(player_df["round_num"].unique().to_list())
n_rounds = len(rounds_list)

fig, axes = plt.subplots(n_rounds, 1, figsize=(14, 2.0 * n_rounds), sharex=True)
if n_rounds == 1: axes = [axes]

freeze_end_ticks = {}
round_end_ticks = {}
round_end_reason = {}
official_end_ticks = {}

for r in rounds.iter_rows(named=True):
    rn = r["round_num"]
    freeze_end_ticks[rn] = r["freeze_end"]
    round_end_ticks[rn] = r["end"]
    round_end_reason[rn] = r.get("reason", "unknown")
    official_end_ticks[rn] = r["official_end"]

reason_labels = {
    "ct_killed": "CT全灭", "t_killed": "T全灭",
    "bomb_exploded": "炸弹爆炸", "bomb_defused": "拆包成功", "time": "时间到"
}

for i, rn in enumerate(rounds_list):
    rd = player_df.filter(pl.col("round_num") == rn)
    x = (rd["tick"] - rd["tick"].min()) / 128
    round_start_tick = rd["tick"].min()

    side_label = rd["side"].to_list()[0].upper() if rd.height > 0 else "?"
    winner = round_winner.get(rn, "?")
    player_side = rd["side"].to_list()[0] if rd.height > 0 else "?"
    win_label = "W" if player_side == winner else "L"

    axes[i].plot(x, rd["pressure"], linewidth=0.5, color="red", alpha=0.9)
    axes[i].fill_between(x, 0, rd["pressure"], alpha=0.10, color="red")
    axes[i].plot(x, rd["death_prob"], linewidth=0.5, color="black", alpha=0.7, linestyle="--")

    if rn in freeze_end_ticks:
        sec = (freeze_end_ticks[rn] - round_start_tick) / 128
        axes[i].axvline(x=sec, color="green", linewidth=0.7, linestyle="--", alpha=0.7)
        axes[i].text(sec + 0.3, 0.86, "START", fontsize=5, color="green")
    if rn in bomb_plant_ticks:
        sec = (bomb_plant_ticks[rn] - round_start_tick) / 128
        axes[i].axvline(x=sec, color="orange", linewidth=0.8, linestyle="--", alpha=0.8)
        axes[i].text(sec + 0.3, 0.76, "BOMB", fontsize=6, color="orange")
    if rn in bomb_defuse_ticks:
        sec = (bomb_defuse_ticks[rn] - round_start_tick) / 128
        axes[i].axvline(x=sec, color="blue", linewidth=0.8, linestyle="--", alpha=0.8)
        axes[i].text(sec + 0.3, 0.66, "DEFUSE", fontsize=5, color="blue")
    if rn in bomb_explode_ticks:
        sec = (bomb_explode_ticks[rn] - round_start_tick) / 128
        axes[i].axvline(x=sec, color="red", linewidth=0.8, linestyle=":", alpha=0.8)
        axes[i].text(sec + 0.3, 0.56, "BOOM", fontsize=6, color="red")
    if rn in round_end_ticks:
        sec = (round_end_ticks[rn] - round_start_tick) / 128
        reason = round_end_reason.get(rn, "?")
        label = reason_labels.get(reason, reason)
        axes[i].axvline(x=sec, color="grey", linewidth=0.8, linestyle="-", alpha=0.7)
        axes[i].text(sec + 0.3, 0.06, label, fontsize=5, color="grey", rotation=90)
    if rn in official_end_ticks:
        sec = (official_end_ticks[rn] - round_start_tick) / 128
        axes[i].axvline(x=sec, color="black", linewidth=0.5, linestyle=":", alpha=0.5)

    axes[i].set_ylabel(f"R{rn} {side_label} {win_label}", fontsize=8, rotation=0, labelpad=35)
    axes[i].set_ylim(-0.02, 1.02)
    axes[i].grid(True, alpha=0.2)
    axes[i].tick_params(labelsize=7)

    if (player_name, rn) in death_ticks:
        death_sec = (death_ticks[(player_name, rn)] - round_start_tick) / 128
        axes[i].axvline(x=death_sec, color="darkred", linewidth=1.0, linestyle="--", alpha=0.9)
        axes[i].text(death_sec + 0.3, 0.48, "DEAD", fontsize=6, color="darkred")

axes[-1].set_xlabel("Time within round (seconds)")
fig.suptitle(f"{player_name} Pressure (Red) + Death Probability (Black Dash)", fontsize=12, y=1.01)
plt.tight_layout()
plt.show()