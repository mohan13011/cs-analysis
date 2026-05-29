CS2 Pressure Analyzer

基于.dem文件的CS2玩家视角压力分析工具。

模拟真实玩家视野，计算每位玩家在比赛中每一帧的心理压力值。

功能

- **全阵营计算**：CT 和 T 的压力都可分析
- **真实视野模拟**：yaw/pitch 双角度 + 40 段射线检测墙体遮挡
- **导航网格加速**：STRtree 空间索引，7.7 万帧 < 2 分钟
- **四维压力因子**：
  - 视野内敌人数量（权重 0.35）
  - 残血程度（权重 0.25）
  - 孤立程度（权重 0.15）
  - 实时伤害量（权重 0.25）
- **B 点精确标定**：307 个导航瓦片，矩形坐标 X=-2300~-1280, Y=1600~3200
- **按回合/区域/玩家可视化**：15 回合独立子图，压力曲线一目了然

快速开始

bash
安装依赖
pip install -r requirements.txt

下载地图数据（首次使用）
awpy get maps
awpy get navs

修改 pressure_model.py 中的 demo_path 为你的 .dem 文件路径
python pressure_model.py
```

压力公式

```
pressure = 0.35 × (视野内敌人数 / 5)
         + 0.25 × (1 - 血量 / 100)
         + 0.15 × (1 - min(附近队友数, 3) / 3)
         + 0.25 × min(受到伤害 / 50, 1)
```

死亡后压力归零。

技术栈

Python 3.13 · awpy 2.0.2 · Polars · Shapely · Matplotlib

计划

- [ ] 道具干扰因子（闪光/烟雾/火）
- [ ] 下包状态因子
- [ ] A 点标定 + A/B 对比分析
- [ ] 对方威胁等级（基于历史伤害量）
- [ ] Streamlit 交互界面

License

MIT
```

---

复制到 `README.md`，push 上去就行。
