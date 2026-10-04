# pfdenoise

**物理先验驱动的点云 / 网格去噪基准与工具箱** —— 零训练、纯 numpy、可复现。

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)

---

## 为什么做这个

近两年这个方向发生了两件对评测方式有直接影响的事：

**1. 深度学习把去噪的 SOTA 整个拿走了，但评测方式有问题。**
2025 年的点云去噪综述（[arXiv:2508.17011](https://arxiv.org/abs/2508.17011)）用一套功能分类法
（reconstruction / displacement / distribution / filter / classification）梳理了现状，
并明确指出两个缺口：

- 领先方法几乎全是**有监督**的，依赖成对的「脏—净」数据，而真实传感器造不出这种配对；
- 方法多在**局部 patch** 上操作，全局一致性差；综述把「cloud-level denoising」列为未来方向。

**2. 决定无监督方法成败的，是噪声模型而不是损失函数。**
2026 年的 Noise2Noise 重访（[arXiv:2609.16788](https://arxiv.org/abs/2609.16788)）给出的结论很直接：
**起作用的是训练对的分布（the training pair distribution），不是损失函数。**
实测数据：合成高斯噪声训出来的 N2N 模型在真实相机噪声上只涨 0.8~3.7 dB；
换成该域自己的噪声对训练（全程不碰真值）则涨 9.4~11.0 dB。

也就是说：**噪声模型是这件事里最重要的设计变量，却最常被当成实现细节。**
而当拿不到真值时，社区也缺少一个可复现的无标签评测协议。

本包就是针对这两点：**把噪声模型提升为一等公民（带统计假设声明），
并提供经典方法与相场方法共用的可复现基准 + 无标签协议。**

### 相关的其它动向（本包刻意不覆盖的部分）

| 动向 | 代表工作 | 与本包的关系 |
|---|---|---|
| 可微 / GPU 相场求解器 | [JAX-PF](https://arxiv.org/abs/2601.06079)（2025-12，比 PRISMS-PF+MPI 快约 5×，AD 做反问题标定） | 本包**不**做可微前向模拟；那是材料方向的地盘，工具链已成熟 |
| 可微 FEM / 神经算子 | [jax-fem](https://github.com/duchoanguyen/jax-fem)、[equation-aware neural operator](https://arxiv.org/abs/2511.09729) | 同上 |
| 神经符号 / 智能体发现方程 | [MAGE](https://arxiv.org/abs/2608.27869)（4 智能体 + 置信度治理的假设验证环）、[NMIPS](https://arxiv.org/abs/2602.11630) | 属于「发现方程」，与本包的「给定噪声模型、评测去噪器」正交，可叠加 |
| 网格去噪的扩散模型 | [DMESH](https://doi.org/10.1109/TNNLS.2024.3367327)（TNNLS 2025，结构保持扩散） | 本包提供它的**物理基线对照**与无标签评测协议 |

> 本包与上述工作不冲突：那些工作解决「怎么做得更好」，本包解决「在同一个协议下比较」。

---

## 安装

```bash
pip install -e .            # 只依赖 numpy
pip install -e ".[plot]"    # 附带 matplotlib（画图用）
```

## 快速开始

```bash
# 看看有什么
python -m pfdenoise list
python -m pfdenoise noise          # 各噪声模型的统计假设 + N2N 理论是否适用

# 跑有标签基准，直接出 markdown 表
python -m pfdenoise run --shape bumpy --noise realistic -n 6000

# 无标签协议：只给两份独立噪声观测，不给真值
python -m pfdenoise n2n --noise gaussian -n 6000
```

Python API：

```python
import pfdenoise as pfd

rows = pfd.run_benchmark(shape="torus", noise="realistic", n=6000)
print(pfd.make_table(rows))
print(pfd.make_table(rows, fmt="csv"))          # 贴进论文用

# 自己拼噪声模型
noise = pfd.Composite(pfd.Gaussian(0.01), pfd.Protrusion(0.05, 0.02))
print(noise.assumption_text())                   # 显式告诉你统计假设
print(noise.n2n_compatible)                      # → False（有偏，N2N 理论不适用）
```

---

## ⚠️ 我们实测跑出来的结果（含一个负面结论）

`python -m pfdenoise run --shape bumpy --noise realistic -n 6000` 的真实输出：

| solver | CD↓ | P2M↓ | HD↓ | nErr↓° | rDrift% | sec |
|---|---|---|---|---|---|---|
| none（噪声基线） | 0.0125 | 0.0150 | 0.2643 | 6.78 | −0.00 | 0.0 |
| laplacian | **0.0160** | **0.0135** | **0.1458** | 9.25 | −0.03 | 12.7 |
| bilateral | 0.0196 | **0.0135** | 0.2643 | 14.84 | −0.05 | 15.9 |
| pf_ac | 0.0861 | 0.0840 | 0.3033 | 34.86 | −0.97 | 1.5 |
| pf_ac_fidelity | 0.0807 | 0.0960 | 0.3205 | 36.40 | **+0.05** | 1.4 |
| pf_ac_barrier | 0.0882 | 0.0859 | 0.3027 | 39.52 | −1.21 | 1.4 |
| cahn_hilliard | 0.0657 | 0.0832 | 0.2814 | 40.90 | **+0.25** | 1.7 |

**结论 1（负面，但重要）：在 6000 点的密度下，相场方法打不过经典滤波。**
CD 差 5~7 倍。原因不是求解器实现有问题，而是**体素化分辨率**：
`res_floor = h/2 ≈ 0.027`，而噪声本身才 0.0125 —— 相场能恢复的精度上限
被 `h` 卡死，而拉普拉斯/双边滤波是直接在点云上工作的，没有这一层量化。

这个负面结论恰恰是本包想传达的核心信息之一：
**评价相场去噪时，必须同时报 `res_floor`，否则会把「分辨率不够」误读成「方法不行」。**
想在真实密度上让相场方法有意义，需要 `n ≳ A/h²` 的点密度（代价见下）。

**结论 2：相场的优势在保体积，不在这个设置下的 CD。**
`pf_ac_fidelity` 与 `cahn_hilliard` 的半径漂移是 **+0.05% / +0.25%**，
而 `pf_ac`（弱保真）是 −0.97%、`pf_ac_barrier` 是 −1.21%。
Cahn–Hilliard 的 `∂∫φ/∂t = 0` 是**格式级**保证，确实测到了。

---

## ⚠️ N2N 一致性作为质量代理是**不够用**的（一个方法论警告）

`python -m pfdenoise n2n --shape bumpy --noise gaussian -n 6000` 的真实输出：

| solver | 去噪后一致性↓ |
|---|---|
| **none** | **0.0187** ← 最低 |
| laplacian | 0.0212 |
| bilateral | 0.0226 |
| pf_ac | 0.0324 |
| pf_ac_barrier | 0.0338 |
| pf_ac_fidelity | 0.0396 |
| cahn_hilliard | 0.0411 |

**一致性越低越好 —— 而「什么都不做」的最低。** 也就是说，
N2N 一致性这个量**被恒等变换单调地奖励**：任何真正修改了数据的方法都会显得更差。
它测的是「两次观测经过同一个估计器后有多接近」，而不是「离真值有多近」。

所以本包**不**把它当作质量指标使用，只作为**协议演示**与辅助信号：
- 它的正确用法是**相对排序**（同一族方法内比较），且必须配合「方法确实在做功」的检查；
- 单独用它选模型会被「越保守越好」的退化解骗到；
- 想在无标签下真正比较质量，需要多次独立观测 + 偏差/方差分解，
  这正是本包 `Contribute` 里列的方向。

---

## 三件东西

### 1. 噪声模型库 —— 噪声是一等公民

每种噪声都**显式声明统计假设**，因为这直接决定 Noise2Noise 那一套能不能用：

| 噪声 | 假设 | N2N 理论 |
|---|---|---|
| `Gaussian` | 零均值、各向同性、逐点独立 | ✅ 适用 |
| `Protrusion` | **有偏**（沿法向系统外推） | ❌ N2N 要求 E[noise]=0 |
| `Outlier` | 重尾、密度不均、强有偏 | ❌ |
| `Aniso` | 零均值但**空间相关**（低频偏移） | ❌ |
| `Density` | 点数变化，非一一对应 | ❌ 点级配对不可用 |
| `Composite` | 逐项声明 | 全部兼容才为真 |

这个区分不是学术洁癖：2026 那篇的实测结论就是「噪声对分布决定成败」，
用有偏噪声去验证一个假设零均值的 N2N 方法，结论必然不可迁移。

### 2. 求解器库 —— 经典基线 + 相场族

**经典基线**（必须有对照组，否则「相场更好」没有意义）
- `none` 噪声基线 · `laplacian` 拉普拉斯（收缩对照） · `bilateral` 迭代双边滤波

**相场族**（本包重点）
- `pf_ac` Allen–Cahn + 弱保真项
- `pf_ac_fidelity` AC + 保真项 λ(T−φ) —— **核心基线**，ε 与 λ 对抗 accuracy / volume
- `pf_ac_barrier` AC + 边缘指示函数屏障 g=1/(1+|∇(G_σ·T)|^α)（论文①路线）
- `cahn_hilliard` 守恒型，**体积是格式级保证**（∂∫φ/∂t = 0），不依赖调参

### 3. 评测协议

**有标签**：CD、P2M、P2M95、HD、法向误差（k-NN PCA 估计）、**半径漂移**
**无标签**：N2N 一致性（两次独立观测各自去噪后的一致性）

`radius_drift` 是经典文献的卖点、深度学习论文普遍不报的一项 —— 这正是相场方法的用武之地。

---

## ⚠️ 三个必须知道的实现坑

这三条都在本包代码里踩过、也都在注释里写明了原因。**如果你要自己写相场去噪，这三条能省你一周。**

### 坑 1：体素化相场，分辨率被点密度卡死

体素化相场成败取决于 `h` 与点间距 `s` 的比值。实测（bumpy 曲面，3000 点）：

| h | n_vox | 占据体素 | 6-邻域连通率 | S_max (grow=0/1) |
|---|---|---|---|---|
| 2.0·s | 35 | 2135 | 0.90 | 0.80 / 1.00 |
| **3.0·s** | **24** | **1541** | **0.99** | **0.87 / 1.00** |
| 5.0·s | 14 | 694 | 1.00 | 0.92 / 1.00 |

- `h ≲ 1.5·s`：每格约 0.3 个点，壳是**孤立散点**；膨胀补连通会让正区比原占据大十几倍，几何完全走样。
- `h ≈ 2~3·s`：壳连通、S_max 接近 1，目标场才有真正的正相。

**硬约束：`n_vox ∝ √n`。** 想要更细的网格必须给更多点（`n ≈ A/h²`）。
所以本包把 `res_floor = h/2` 作为指标一起报出，表格里带 `⚠` 的行表示
**CD 已经触到体素量化下限，那是在测分辨率、不是求解器精度**，别再拿来比方法。

> 顺带一个调试陷阱：早期版本里 `_extract_surface` 在正相塌缩时**静默返回输入点**，
> 于是所有相场求解器的指标和「不做处理」一模一样 —— 看起来像「相场没用」，
> 实际是分辨率不对。现在塌缩会直接抛错并说明原因。

### 坑 2：`Δt` 的量纲随 Laplacian 前面有没有 `ε²` 而变

```
PDE 写成  ε²Δφ  →  Δt = cfl·h²/ε²      稳定条件 Δt·ε²·(4 或 6)/h² ≤ 1
PDE 写成  Δφ    →  Δt = cfl·h²/2
```

很多论文印的 `Δt = 0.05h²` 对应的是**没有** `ε²` 前缀的写法。
照抄到 `ε²Δφ` 上，界面几百步只动 0.01 —— 现象是「程序在跑但什么都不发生」。

**Cahn–Hilliard 还要更严**：化学势 μ 里有 Δφ，μ 外面又套一层 Δ，
格式实际含**四阶**算子 `ε²∇⁴φ`，限制是 `Δt ≤ h⁴/(6Mε²)`，比 AC 严格得多。
直接套 AC 的 Δt 会在第 3~5 步就炸（实测确实如此）。

本包每个相场求解器都返回它用的 `dt` 和 `cfl`，建议每次都检查。

### 坑 3：纯 AC 会腐蚀有限液滴

纯 Allen–Cahn（λ=0）是双阱势能的梯度流，有限大小的正相受毛细压力**必然收缩到零测度**。
所以作为去噪器必须给一点保真项。本包 `pf_ac` 默认 `lam=0.25`；
想看腐蚀过程可以显式传 `lam=0`。

---

## 设计取舍与已知局限

诚实地列出来，免得你踩：

- **P2M 用密集重采样代理，不是真正的点到三角形距离。** 本包不引入网格资产，
  用 4× 密度重采样近似。数值会略偏小，但**所有基线用同一代理，方法间比较是公平的**。
  报绝对数值时请注明用了代理。
- **`radius_drift` 是尺度代理，不是真体积。** 点云没有三角剖分就算不出封闭体积
  （需要凸包/三角化库）。它用「质心到点的平均距离」，对相对质心星形的曲面
  （sphere / torus / bumpy）有效；非星形曲面下会失真。需要真体积请自行接
  `scipy.spatial.ConvexHull` 或 `trimesh`。
- **N2N 一致性是代理不是真值。** 一致性高可能意味着「两个结果都停在同一个错的解上」。
  必须和有标签指标一起看，或只用于选超参时的稳定性排序。
- **纯 numpy 意味着慢。** 3000 点 / 24³ 体素 / 150 步大约几秒；
  要上 10⁵ 点以上请改用 JAX（参考 [JAX-PF](https://arxiv.org/abs/2601.06079)）
  或自己接 `scipy.ndimage`。
- **曲面是解析生成的**（sphere / torus / superquadric / bumpy），没有真实扫描资产。
  这是为了自包含与真值精确；接真实 `.ply` 时把 `make_shape` 换掉即可。

---

## 引用

如果你在论文里用了本包的基准或协议，请同时引用下面这些工作
（本包是**评测基础设施**，不替代被评测的方法）：

```bibtex
@article{pointclouddenoise-survey2025,
  title   = {A Survey of Deep Learning-based Point Cloud Denoising},
  journal = {arXiv preprint arXiv:2508.17011},
  year    = {2025}
}
@article{noise2noise-revisited2026,
  title   = {Noise2Noise Revisited: Training Pair Distributions Dominate
             Loss Choice in Self-Supervised Denoising},
  journal = {arXiv preprint arXiv:2609.16788},
  year    = {2026}
}
@article{noise2noise,
  title   = {Noise2Noise: Learning Image Restoration without Clean Data},
  author  = {Lehtinen, J{\"a}rvi and Laine, Miika and Karras, Tero and
             Aila, Timo and Laine, Samuli},
  journal = {ICML},
  year    = {2018}
}
```

## 贡献

欢迎。几个当前比较需要的方向：

- 加真实扫描数据集（`.ply`）的 loader 与法向标定流程
- 接 `scipy`/`trimesh` 做真正的 P2M 与凸包体积
- 加 GPU 后端（JAX / CuPy）
- 把 N2N 协议扩展到「三次以上独立观测」与异方差噪声

```bash
pip install -e ".[dev]"
pytest
ruff check .
```

## License

MIT，见 [LICENSE](LICENSE)。
