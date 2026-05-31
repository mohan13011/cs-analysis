import os
import glob
import time
from awpy import Demo
import polars as pl
import math
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

start_time = time.time()

demo_dir = r"C:\Users\yangmohan\OneDrive\桌面\demo样本"
demo_files = glob.glob(os.path.join(demo_dir, "*.dem"))
print(f"找到 {len(demo_files)} 个 demo")

all_train_data = []

for demo_path in tqdm(demo_files, desc="处理demo"):
    dem = Demo(path=demo_path, verbose=False)
    dem.parse(player_props=["health", "pitch", "yaw"])

    ticks = dem.ticks
    rounds = dem.rounds
    damages = dem.damages
    grenades = dem.grenades
    infernos = dem.infernos

    # 死亡索引
    death_ticks = {}
    for k in dem.kills.iter_rows(named=True):
        death_ticks[(k["victim_name"], k["round_num"])] = k["tick"]

    # 伤害索引
    dmg_index = {}
    for d in damages.iter_rows(named=True):
        t = d["tick"]
        v = d["victim_name"]
        dmg = d["dmg_health"]
        if t not in dmg_index:
            dmg_index[t] = {}
        dmg_index[t][v] = dmg_index[t].get(v, 0) + dmg

    # 闪光弹爆炸位置
    flash_proj = grenades.filter(pl.col("grenade_type") == "CFlashbangProjectile").sort("tick")
    flash_explosion_pos = {}
    for g in flash_proj.iter_rows(named=True):
        if (g["X"] is not None and g["Y"] is not None 
            and not math.isnan(g["X"]) and not math.isnan(g["Y"])):
            flash_explosion_pos[g["entity_id"]] = (g["X"], g["Y"], g["Z"])

    flash_index = {}
    for g in grenades.filter(pl.col("grenade_type") == "CFlashbang").iter_rows(named=True):
        eid = g["entity_id"]
        if eid in flash_explosion_pos:
            x, y, z = flash_explosion_pos[eid]
            if not (math.isnan(x) or math.isnan(y)):
                t = g["tick"]
                if t not in flash_index:
                    flash_index[t] = []
                flash_index[t].append((x, y, z))

    # 燃烧弹索引
    inferno_index = {}
    for inf in infernos.iter_rows(named=True):
        if inf["start_tick"] is None or inf["end_tick"] is None:
            continue
        x, y, z = inf["X"], inf["Y"], inf["Z"]
        for t in range(int(inf["start_tick"]), int(inf["end_tick"]) + 1):
            if t not in inferno_index:
                inferno_index[t] = []
            inferno_index[t].append((x, y, z))

    # 下包索引
    bomb_planted_ticks = set()
    for r in rounds.iter_rows(named=True):
        if r["bomb_plant"] is not None:
            for t in range(int(r["bomb_plant"]), r["end"] + 1):
                bomb_planted_ticks.add(t)

    # 回合索引
    active_ticks = set()
    tick_to_round = {}
    round_list = list(rounds.iter_rows(named=True))
    for i, r in enumerate(round_list):
        s = r["start"]
        e = round_list[i+1]["start"] if i+1 < len(round_list) else r["official_end"]
        for t in range(s, e+1):
            active_ticks.add(t)
            tick_to_round[t] = r["round_num"]

    # 视野判断简化版
    def can_see_simple(ox, oy, oyaw, tx, ty, max_dist=1500, fov=90):
        dx, dy = tx - ox, ty - oy
        dist = math.sqrt(dx**2 + dy**2)
        if dist < 1: return True
        if dist > max_dist: return False
        yaw_rad = math.radians(oyaw)
        lx, ly = math.cos(yaw_rad), math.sin(yaw_rad)
        tx_u, ty_u = dx/dist, dy/dist
        dot = max(-1, min(1, lx*tx_u + ly*ty_u))
        return math.degrees(math.acos(dot)) < fov/2

    # 构建训练数据
    DEATH_WINDOW = 3 * 128
    tick_list = sorted(ticks["tick"].unique().to_list())[::5]

    for tick_id in tqdm(tick_list, desc=f"  帧", leave=False):
        if tick_id not in active_ticks:
            continue
        frame = ticks.filter(ticks["tick"] == tick_id)
        flashes = flash_index.get(tick_id, [])
        fires = inferno_index.get(tick_id, [])
        dmg_tick = dmg_index.get(tick_id, {})
        bomb_planted = 1 if tick_id in bomb_planted_ticks else 0
        rn = tick_to_round.get(tick_id, -1)

        for player in frame.iter_rows(named=True):
            if player["health"] is None or player["health"] <= 0:
                continue
            enemy_side = "t" if player["side"] == "ct" else "ct"

            vis = 0
            for enemy in frame.filter(pl.col("side") == enemy_side).iter_rows(named=True):
                if enemy["health"] is not None and enemy["health"] > 0 and can_see_simple(
                    player["X"], player["Y"], player["yaw"],
                    enemy["X"], enemy["Y"]
                ):
                    vis += 1

            nb = 0
            for mate in frame.filter((pl.col("side") == player["side"]) & (pl.col("name") != player["name"])).iter_rows(named=True):
                if mate["health"] is not None and mate["health"] > 0 and math.sqrt((mate["X"]-player["X"])**2 + (mate["Y"]-player["Y"])**2) < 800:
                    nb += 1

            fl = 0
            for fx, fy, fz in flashes:
                if math.sqrt((player["X"]-fx)**2 + (player["Y"]-fy)**2 + (player["Z"]-fz)**2) < 800:
                    dx, dy = fx-player["X"], fy-player["Y"]
                    d2d = math.sqrt(dx**2+dy**2)
                    if d2d > 0:
                        yr = math.radians(player["yaw"])
                        lx, ly = math.cos(yr), math.sin(yr)
                        dot = max(-1, min(1, lx*dx/d2d + ly*dy/d2d))
                        if math.degrees(math.acos(dot)) < 90:
                            fl = 1
                            break

            nf = 0
            for fx, fy, fz in fires:
                if math.sqrt((player["X"]-fx)**2 + (player["Y"]-fy)**2 + (player["Z"]-fz)**2) < 300:
                    nf = 1
                    break

            dt = death_ticks.get((player["name"], rn))
            died = 0
            if dt and 0 < dt - tick_id <= DEATH_WINDOW:
                died = 1

            all_train_data.append({
                "f_visible": vis / 5,
                "f_hp": 1 - player["health"]/100,
                "f_alone": 1 - min(nb, 3)/3,
                "f_dmg": min(dmg_tick.get(player["name"], 0)/50, 1.0),
                "f_flash": fl,
                "f_fire": nf,
                "f_bomb": bomb_planted if player["side"]=="ct" else -0.3*bomb_planted,
                "died_soon": died
            })

# 合并所有 demo 的数据
train_df = pl.DataFrame(all_train_data)
print(f"\n总训练数据: {train_df.height} 条, 正样本(死亡): {train_df['died_soon'].sum()} 条")

# 逻辑回归
feature_cols = ["f_visible", "f_hp", "f_alone", "f_dmg", "f_flash", "f_fire", "f_bomb"]
X = train_df.select(feature_cols).to_numpy()
y = train_df["died_soon"].to_numpy()

model = LogisticRegression(l1_ratio=1.0, solver='saga', C=0.1, class_weight='balanced')
model.fit(X, y)

print(f"\n=== 死亡概率权重（{len(demo_files)} 个 demo）===")
print(f"截距: {model.intercept_[0]:.4f}")
for name, coef in zip(feature_cols, model.coef_[0]):
    print(f"  {name}: {coef:.4f}")
print(f"训练准确率: {model.score(X, y):.3f}")

print(f"\n=== 对比：压力权重 vs 死亡权重 ===")
print(f"{'因子':<15} {'压力权重':>8} {'死亡权重':>8}")
pressure_weights = [0.28, 0.18, 0.09, 0.18, 0.12, 0.05, 0.10]
for name, pw, dw in zip(feature_cols, pressure_weights, model.coef_[0]):
    print(f"{name:<15} {pw:>8.2f} {dw:>8.4f}")

# === 数字版评估 ===
from sklearn.metrics import confusion_matrix, roc_curve, auc
import numpy as np

y_prob = model.predict_proba(X)[:, 1]
y_pred = (y_prob >= 0.3).astype(int)

cm = confusion_matrix(y, y_pred)
print(f"\n=== 混淆矩阵（阈值0.3）===")
print(f"真实安全, 预测安全: {cm[0][0]}")
print(f"真实安全, 预测死亡: {cm[0][1]}")
print(f"真实死亡, 预测安全: {cm[1][0]}")
print(f"真实死亡, 预测死亡: {cm[1][1]}")

tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

print(f"\n=== 评估指标 ===")
print(f"准确率 (Accuracy): {accuracy:.4f}")
print(f"精确率 (Precision): {precision:.4f}")
print(f"召回率 (Recall): {recall:.4f}")
print(f"特异度 (Specificity): {specificity:.4f}")
print(f"F1分数: {f1:.4f}")

fpr, tpr, thresholds = roc_curve(y, y_prob)
roc_auc = auc(fpr, tpr)
print(f"\n=== ROC ===")
print(f"AUC: {roc_auc:.4f}")

print(f"\n=== 预测概率分布 ===")
print(f"安全样本平均预测死亡概率: {np.mean(y_prob[y == 0]):.4f}")
print(f"死亡样本平均预测死亡概率: {np.mean(y_prob[y == 1]):.4f}")

print(f"\n=== 特征与死亡的相关性 ===")
for col in feature_cols:
    corr = np.corrcoef(train_df[col].to_numpy(), y)[0, 1]
    print(f"  {col}: {corr:.4f}")

print(f"\n总耗时: {time.time() - start_time:.1f} 秒")