# A2 Wall Climbing MuJoCo Simulation

基于MuJoCo的宇树A2四足机器人集装箱壁面磁吸附爬行动力学仿真平台。

## 项目结构

```
a2_wall_climbing/
├── models/           # MuJoCo XML模型（机器人、环境、场景）
├── config/           # YAML配置文件
├── controllers/      # 运动控制与步态模块
├── simulation/       # 工况测试脚本
├── analysis/         # 数据分析与绘图
├── results/          # 输出（csv/figures/videos/reports）
└── assets/           # URDF原始资源（需自行放入A2 URDF）
```

## 安装

```bash
pip install -r requirements.txt
```

## 快速开始

```bash
# 模型验证
python run_simulation.py --test T1

# 壁面四足吸附静止
python run_simulation.py --test T2

# 三足支撑（抬起FL腿）
python run_simulation.py --test T3 --leg FL

# 单腿完整运动
python run_simulation.py --test T7 --leg FL

# 连续爬行（4步）
python run_simulation.py --test T10 --steps 4

# 180Nm限幅爬行
python run_simulation.py --test T12 --tlim 180 --steps 8

# 带负载爬行
python run_simulation.py --test T13 --payload 15 --steps 8

# 参数扫描
python run_simulation.py --sweep adhesion
```

## 测试工况

| 编号 | 工况 | 命令 |
|------|------|------|
| T1 | 水平站立 | `--test T1` |
| T2 | 壁面四足吸附静止 | `--test T2` |
| T3-T6 | 三足支撑 | `--test T3 --leg FL/FR/RL/RR` |
| T7-T9 | 单腿运动 | `--test T7 --leg FL` |
| T10 | 完整四步爬行 | `--test T10 --steps 4` |
| T11 | 连续爬行 | `--test T11 --steps 20` |
| T12 | 180Nm限幅 | `--test T12 --tlim 180` |
| T13 | 带等效负载 | `--test T13 --payload 15` |
| T14 | 带机械臂 | 手动配置模型文件 |

## 配置

所有仿真参数在 `config/controller.yaml` 中修改：
- 仿真步长、控制频率
- 步态参数（步长、摆动时间）
- 吸附力
- PD增益
- 力矩限幅

## 注意事项

- 需要在 `assets/a2_urdf/` 中放置A2的URDF模型文件
- 如果没有A2 URDF，可以使用 `models/robots/a2.xml` 中的简化MJCF模型进行初步验证
- 简化模型仅供参考运动学和控制逻辑，实际力矩分析需使用准确质量/惯量参数的URDF
