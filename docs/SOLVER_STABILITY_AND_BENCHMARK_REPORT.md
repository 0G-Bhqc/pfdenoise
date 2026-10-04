# 数值稳定性条件审查与相场去噪 Benchmark 科研实验规范报告

**评审专家**：Codex 深度算法与数值偏微分方程（PDE）特聘专家  
**审查对象**：`pfdenoise` v0.1.0-alpha 新增求解器：
1. **拉格朗日体积守恒 Allen–Cahn 相场求解器** (`phase_field_ac_conserved` / `pf_ac_conserved`)
2. **各向异性几何扩散张量相场求解器** (`phase_field_ac_anisotropic` / `pf_ac_aniso`)
**对标基线**：标准曲率流 AC (`pf_ac`)、保真项 AC (`pf_ac_fidelity`)、Cahn–Hilliard (`cahn_hilliard`)、拉普拉斯平滑 (`laplacian`)、网格双边滤波 (`bilateral`)

---

## 摘要 (Executive Summary)

本报告针对在 `pfdenoise/solvers.py` 中实现的拉格朗日乘子全局体积守恒 Allen–Cahn（L-AC）以及切向增强/法向抑制各向异性扩散张量 Allen–Cahn（A-AC）两种新型求解器开展严格的离散泛函分析与对抗性代码审计。

**主要结论**：
1. **离散质量守恒性**：L-AC 采用空间均值补偿标量 $\beta(t)$，在中心差分离散拉普拉斯算子与齐次 Neumann 边界的伸缩对消（telescoping cancellation）配合下，在离散代数层面**严格满足机器精度级质量守恒**（$\left|\sum \phi^{n+1} - \sum \phi^n\right| < 10^{-15}$）。其克服了四阶 Cahn–Hilliard 方程对显式时间步长 $\mathcal{O}(h^4 / (M \varepsilon^2))$ 的极度苛刻刚性限制，在二阶扩散 CFL 条件（$\Delta t \le h^2 / (6\varepsilon^2)$）下即可稳定运行。
2. **离散能量有界性与最大值原理**：连续 L-AC 严格服从约束梯度流的李雅普诺夫能量单调衰减（$dE/dt = -\|\partial_t \phi\|_{L^2}^2 \le 0$）。但在显式欧拉时步推进下，能量衰减仅为**条件稳定**。由于标量 $\beta(t)$ 的空间全局常数平移，离散最大值原理（Discrete Maximum Principle, $|\phi| \le 1$）在非对称两相体积分数下被微扰破坏（$\mathcal{O}(\beta)$ 阱位偏移），需设定 CFL 安全裕度 $\text{CFL} \le 1/12$。
3. **各向异性张量频域耗散性**：A-AC 的切向平滑与法向投影通量在交错网格差分展开下具有严格正定的连续扩散矩阵 $\mathbf{D} \ge c_{\text{normal}}\mathbf{I} > 0$。经离散傅里叶分析（von Neumann 分析），其 1D/3D 离散符号在所有非零波数域内恒负（$\widehat{\mathcal{L}}_h(\boldsymbol{\theta}) \le 0$），不存在反向扩散不稳定性；但双重中心差分带来跨度 $4h$ 的扩展模板，对 Nyquist 极限频率阻尼较弱。
4. **Benchmark 实验协议设计**：制定了涵盖 4 类代表性解析几何（球、环、超二次曲面、高频谐波凹凸曲面）、4 种非理想噪声（各向同性高斯、有偏凸起、空间低频各向异性、真实复合）以及 7 维几何保真与物理不变量指标的标准实验规范。

---

# 第一部分：数学与算法公式推导 (Mathematical & Algorithmic Formulation)

## 1.1 拉格朗日体积守恒 Allen–Cahn (L-AC)

### 1.1.1 变分原理与连续系统
标准 Ginzburg–Landau 自由能泛函包含界面梯度能、双阱势能以及 $L^2$ 保真势：
$$
\mathcal{E}(\phi) = \int_{\Omega} \left[ \frac{\varepsilon^2}{2} |\nabla \phi|^2 + G(\phi) + \frac{\lambda}{2} (\phi - T)^2 \right] d\mathbf{x}
$$
其中 $G(\phi) = \frac{1}{4}(\phi^2 - 1)^2$ 为对称双阱势，$T \in [-1, 1]$ 为数据目标场，$\varepsilon$ 为界面厚度参数，$\lambda$ 为保真权重。

在全局体积（质量）守恒约束 $\int_{\Omega} \phi(\mathbf{x}, t) d\mathbf{x} = \int_{\Omega} \phi(\mathbf{x}, 0) d\mathbf{x} \equiv \mathcal{M}_0$ 下，引入时间依赖的拉格朗日乘子 $\beta(t)$，构造增广作用量泛函：
$$
\mathcal{L}(\phi, \beta) = \mathcal{E}(\phi) - \beta(t) \left( \int_\Omega \phi d\mathbf{x} - \mathcal{M}_0 \right)
$$
对 $\phi$ 求 $L^2$ 变分梯度：
$$
\frac{\delta \mathcal{L}}{\delta \phi} = -\varepsilon^2 \Delta \phi + G'(\phi) + \lambda(\phi - T) - \beta(t)
$$
对应的约束梯度流演化方程为：
$$
\begin{cases}
\displaystyle \frac{\partial \phi}{\partial t} = \varepsilon^2 \Delta \phi - (\phi^3 - \phi) + \lambda(T - \phi) + \beta(t), & \mathbf{x} \in \Omega, \; t > 0 \\[2mm]
\displaystyle \nabla \phi \cdot \mathbf{n}_{\partial\Omega} = 0, & \mathbf{x} \in \partial\Omega
\end{cases}
$$

### 1.1.2 乘子 $\beta(t)$ 的闭式解析解与能量耗散恒等式
两端对计算域 $\Omega$ 做积分，并利用齐次 Neumann 边界条件与高斯散度定理 $\int_\Omega \Delta \phi d\mathbf{x} = \int_{\partial\Omega} \nabla \phi \cdot \mathbf{n} dS = 0$：
$$
\frac{d}{dt} \int_\Omega \phi d\mathbf{x} = \int_\Omega \left[ -(\phi^3 - \phi) + \lambda(T - \phi) \right] d\mathbf{x} + \beta(t) |\Omega| \equiv 0
$$
直接导出标量乘子 $\beta(t)$ 的空间均值积分恒等式：
$$
\beta(t) = \frac{1}{|\Omega|} \int_\Omega \left[ (\phi^3 - \phi) + \lambda(\phi - T) \right] d\mathbf{x} = \langle \phi^3 - \phi + \lambda(\phi - T) \rangle_\Omega
$$

**能量耗散定理**：
记零均值变分空间 $V_0 = \{ v \in L^2(\Omega) : \int_\Omega v d\mathbf{x} = 0 \}$，$L^2$ 正交投影算子为 $\mathcal{P}_{V_0}(w) = w - \langle w \rangle_\Omega$。演化方程可写为：
$$
\frac{\partial \phi}{\partial t} = -\mathcal{P}_{V_0}\left( \frac{\delta \mathcal{E}}{\delta \phi} \right)
$$
系统关于原能泛函 $\mathcal{E}(\phi)$ 的时间导数满足：
$$
\frac{d\mathcal{E}}{dt} = \int_\Omega \frac{\delta \mathcal{E}}{\delta \phi} \frac{\partial \phi}{\partial t} d\mathbf{x} = \int_\Omega \frac{\delta \mathcal{E}}{\delta \phi} \left[ -\mathcal{P}_{V_0}\left( \frac{\delta \mathcal{E}}{\delta \phi} \right) \right] d\mathbf{x} = -\int_\Omega \left| \mathcal{P}_{V_0}\left( \frac{\delta \mathcal{E}}{\delta \phi} \right) \right|^2 d\mathbf{x} = -\int_\Omega \left( \frac{\partial \phi}{\partial t} \right)^2 d\mathbf{x} \le 0
$$
**推论**：连续体积守恒 Allen–Cahn 方程在能量空间内具备无条件 Lyapunov 单调耗散性，全局能下有界。

---

## 1.2 各向异性扩散张量相场 (A-AC)

### 1.2.1 张量构造与几何流分解
传统各向同性拉普拉斯平滑 $\varepsilon^2 \Delta \phi$ 的主要缺陷在于：在平滑切向高频几何波动的同等强度下，沿法向 $\mathbf{n} = \frac{\nabla \phi}{|\nabla \phi|}$ 发生剧烈的等值面收缩与锐棱钝化。

引入自适应扩散张量 $\mathbf{D}(\nabla \phi) \in \mathbb{R}^{3 \times 3}$：
$$
\mathbf{D} = c_{\text{tangent}} \mathbf{P}_{\boldsymbol{\tau}} + c_{\text{normal}} \mathbf{P}_{\mathbf{n}}
$$
其中法向投影算子 $\mathbf{P}_{\mathbf{n}} = \mathbf{n} \mathbf{n}^T$，切向投影算子 $\mathbf{P}_{\boldsymbol{\tau}} = \mathbf{I} - \mathbf{n}\mathbf{n}^T$。代入得：
$$
\mathbf{D} = c_{\text{tangent}} \mathbf{I} - (c_{\text{tangent}} - c_{\text{normal}}) \mathbf{n} \mathbf{n}^T
$$
设置切向高扩散 $c_{\text{tangent}} \approx 1.0$，法向低扩散 $c_{\text{normal}} \in (0, 0.2]$。
为消除平坦区 $|\nabla \phi| \to 0$ 处的极点奇异性，引入梯度模平滑正则化：
$$
\mathbf{n}_\sigma = \frac{\nabla \phi_\sigma}{\sqrt{|\nabla \phi_\sigma|^2 + \eta^2}}, \quad \phi_\sigma = G_\sigma * \phi, \quad w(|\nabla \phi_\sigma|) = \frac{|\nabla \phi_\sigma|^2}{|\nabla \phi_\sigma|^2 + \kappa^2}
$$
其各向异性扩散通量为：
$$
\mathbf{J} = \mathbf{D} \nabla \phi = c_{\text{tangent}} \nabla \phi - c_{\text{diff}} w (\mathbf{n}_\sigma \cdot \nabla \phi) \mathbf{n}_\sigma, \quad c_{\text{diff}} = c_{\text{tangent}} - c_{\text{normal}}
$$
对应的散度形式控制方程为：
$$
\frac{\partial \phi}{\partial t} = \varepsilon^2 \left[ c_{\text{tangent}} \Delta \phi - \nabla \cdot \mathbf{v} \right] - \frac{1}{2}\phi(\phi^2 - 1) + \lambda(T - \phi)
$$
其中各向异性修正通量为 $\mathbf{v} = c_{\text{diff}} w (\mathbf{n}_\sigma \cdot \nabla \phi) \mathbf{n}_\sigma$。

---

# 第二部分：离散格式与稳定性条件审查 (Numerical Scheme & Stability Audit)

## 2.1 离散格式实施审查

在 `pfdenoise/solvers.py` 中，三维离散均布交错体素网格步长为 $h$，显式一阶时间步长为 $\Delta t$。

### 2.1.1 L-AC 的离散空间求和与代数质量守恒
离散拉普拉斯算子采用标准 7 点中心差分：
$$
(\Delta_d \phi)_{ijk} = \frac{1}{h^2} \left( \phi_{i+1,j,k} + \phi_{i-1,j,k} + \phi_{i,j+1,k} + \phi_{i,j-1,k} + \phi_{i,j,k+1} + \phi_{i,j,k-1} - 6\phi_{i,j,k} \right)
$$
边界处采用 `_neumann` 模式（`mode="edge"`，一阶 Ghost 镜像填充）。
**伸缩对消分析**：
对于任意三维阵列 $u \in \mathbb{R}^{N_x \times N_y \times N_z}$，沿 $x$ 轴考察内点与边界：
- 边界 $i=0$：Ghost 格点设为 $\phi_{-1, j, k} = \phi_{0, j, k}$，差分项为 $(\phi_{1} - \phi_0)/h^2$。
- 边界 $i=N_x-1$：Ghost 格点设为 $\phi_{N_x, j, k} = \phi_{N_x-1, j, k}$，差分项为 $(\phi_{N_x-2} - \phi_{N_x-1})/h^2$。
- 沿 $x$ 轴求和：
$$
\sum_{i=0}^{N_x-1} (\phi_{i+1} - 2\phi_i + \phi_{i-1}) = (\phi_1 - \phi_0) + \sum_{i=1}^{N_x-2}(\phi_{i+1} - 2\phi_i + \phi_{i-1}) + (\phi_{N_x-2} - \phi_{N_x-1}) \equiv 0
$$
三维全空间离散积分严格恒等于零：
$$
\sum_{i,j,k} (\Delta_d \phi)_{ijk} \equiv 0
$$
同时，代码中乘子计算为：
$$
\beta^n = \frac{1}{N} \sum_{i,j,k} \left[ (\phi_{ijk}^n)^3 - \phi_{ijk}^n + \lambda (\phi_{ijk}^n - T_{ijk}) \right]
$$
显式时间步为：
$$
\phi_{ijk}^{n+1} = \phi_{ijk}^n + \Delta t \left\{ \varepsilon^2 (\Delta_d \phi^n)_{ijk} - \left[ (\phi_{ijk}^n)^3 - \phi_{ijk}^n \right] + \lambda (T_{ijk} - \phi_{ijk}^n) + \beta^n \right\}
$$
两端对全网格 $N$ 个体素求和：
$$
\sum_{i,j,k} \phi_{ijk}^{n+1} - \sum_{i,j,k} \phi_{ijk}^n = \Delta t \left\{ \varepsilon^2 \cdot 0 - \sum_{i,j,k} R_{ijk}^n + N \cdot \left( \frac{1}{N}\sum_{i,j,k} R_{ijk}^n \right) \right\} = 0
$$
**审查评定**：离散格式在浮点舍入误差容限内（$|\Delta M| \sim \mathcal{O}(N \epsilon_{\text{mach}}) \approx 10^{-15}$）保证离散质量绝对守恒。

---

## 2.2 稳定性与 CFL 条件推导 (CFL Condition)

### 2.2.1 空间扩散算子谱半径与显式时步上界
三维 7 点差分离散拉普拉斯算子的 Fourier 本征模为 $e^{I(k_x x + k_y y + k_z z)}$，本征值为：
$$
\mu(\theta_x, \theta_y, \theta_z) = -\frac{4}{h^2} \left( \sin^2\frac{\theta_x}{2} + \sin^2\frac{\theta_y}{2} + \sin^2\frac{\theta_z}{2} \right)
$$
最大谱半径在 Nyquist 极限频率 $\theta_x = \theta_y = \theta_z = \pi$ 处取得：
$$
\rho(\Delta_d) = |\mu_{\max}| = \frac{12}{h^2}
$$
显式欧拉时步对高斯扩散项 $\partial_t \phi = \varepsilon^2 \Delta \phi$ 的 $L^2$ 稳定性条件为：
$$
|1 - \Delta t \varepsilon^2 \rho(\Delta_d)| \le 1 \implies \Delta t \le \frac{2}{\varepsilon^2 \rho(\Delta_d)} = \frac{h^2}{6 \varepsilon^2}
$$
在 `pfdenoise/solvers.py` 中：
$$
\Delta t = \text{cfl} \cdot \frac{h^2}{\varepsilon^2}, \quad \text{默认 } \text{cfl} = \frac{1}{12} \implies \text{cfl\_actual} = \frac{6 \Delta t \varepsilon^2}{h^2} = 6 \times \frac{1}{12} = 0.5 \le 1.0
$$
其具备 2 倍的线性稳定性安全裕度。

### 2.2.2 非线性反应项的 CFL 扰动分析
反应项 Jacobian 为 $J(\phi) = \frac{\partial}{\partial \phi} [-(\phi^3 - \phi) - \lambda \phi] = 1 - 3\phi^2 - \lambda$。
当相场处于稳定双阱稳态 $\phi \approx \pm 1$ 时：
$$
J(\pm 1) = 1 - 3(1) - \lambda = -(2 + \lambda) < 0
$$
局部反应稳定性要求：
$$
|1 + \Delta t J(\pm 1)| \le 1 \implies \Delta t \le \frac{2}{2 + \lambda}
$$
由于典型网格尺度下 $\varepsilon = 2h$，$\Delta t = \frac{1}{12 \times 4} = \frac{1}{48} \approx 0.0208$，而 $\frac{2}{2 + \lambda} = \frac{2}{2.8} \approx 0.714$。扩散算子刚性严格主导非线性反应刚性（$\Delta t_{\text{diff}} \ll \Delta t_{\text{react}}$），因此显式欧拉步在稳态附近无需针对局部反应截断时步。

---

## 2.3 各向异性算子的频域稳定性分析 (Fourier Symbol of A-AC)

A-AC 引入了双重中心差分构造散度项 $\nabla \cdot \mathbf{v}$。我们审查该格式在极端法向取向下的连续与离散稳定性。
设局部法向沿 $x$ 轴正向 $\mathbf{n} = (1, 0, 0)^T$，且权重处于界面饱和区 $w \approx 1$。
此时连续算子沿 $x$ 轴退化为：
$$
\mathcal{L}_{\text{aniso}} \phi = c_{\text{tangent}} \frac{\partial^2 \phi}{\partial x^2} - (c_{\text{tangent}} - c_{\text{normal}}) \frac{\partial^2 \phi}{\partial x^2} = c_{\text{normal}} \frac{\partial^2 \phi}{\partial x^2}
$$
因为 $c_{\text{normal}} > 0$，算子严格保持强椭圆性（Strong Ellipticity）。

考察其离散差分格式：
$$
(\mathcal{L}_h \phi)_i = c_{\text{tangent}} \frac{\phi_{i+1} - 2\phi_i + \phi_{i-1}}{h^2} - c_{\text{diff}} \frac{\phi_{i+2} - 2\phi_i + \phi_{i-2}}{4h^2}
$$
对其进行离散 Fourier 变换，令 $\phi_i = e^{I \theta i}$，$\theta = k h \in [0, \pi]$：
$$
\begin{aligned}
\widehat{\mathcal{L}}_h(\theta) &= -\frac{4 c_{\text{tangent}}}{h^2} \sin^2\frac{\theta}{2} + \frac{c_{\text{diff}}}{h^2} \sin^2\theta \\
&= -\frac{4 \sin^2(\theta/2)}{h^2} \left[ c_{\text{tangent}} - c_{\text{diff}} \cos^2\frac{\theta}{2} \right]
\end{aligned}
$$
注意到 $c_{\text{diff}} = c_{\text{tangent}} - c_{\text{normal}}$，代入方括号内：
$$
\mathcal{B}(\theta) = c_{\text{tangent}} - (c_{\text{tangent}} - c_{\text{normal}}) \cos^2\frac{\theta}{2}
$$
由于对于任意实数波数 $\theta \in [0, \pi]$，$\cos^2(\theta/2) \in [0, 1]$：
- 低频极限 $\theta \to 0$ 时，$\cos^2(\theta/2) \to 1 \implies \mathcal{B}(0) = c_{\text{normal}} > 0$；
- 高频极限 $\theta \to \pi$ 时，$\cos^2(\theta/2) \to 0 \implies \mathcal{B}(\pi) = c_{\text{tangent}} > 0$。
因为 $c_{\text{tangent}} \ge c_{\text{normal}} > 0$，$\mathcal{B}(\theta)$ 在全频段上恒满足：
$$
0 < c_{\text{normal}} \le \mathcal{B}(\theta) \le c_{\text{tangent}}
$$
由此导出：
$$
\widehat{\mathcal{L}}_h(\theta) \le -\frac{4 c_{\text{normal}}}{h^2} \sin^2\frac{\theta}{2} \le 0, \quad \forall \theta \in [0, \pi]
$$
**稳定性结论**：
1. **耗散性保证**：离散各向异性扩散算子在全频域符号恒负，**绝不会产生伪反向扩散（Anti-diffusion / 负扩散爆破）**。
2. **时步谱约束**：最大幅值满足 $|\widehat{\mathcal{L}}_h(\theta)| \le \frac{4 c_{\text{tangent}}}{h^2}$。三维全谱上界为 $\frac{12 c_{\text{tangent}}}{h^2}$。只要 $c_{\text{tangent}} \le 1.0$，该算子的 CFL 稳定性要求与标准 AC 完全同一，默认 $\text{cfl} = 1/12$ 充分保证数值绝对稳定。

---

## 2.4 对抗性代码审查缺陷与补丁清单 (Defect & Patch Table)

经深度逆向走查 `pfdenoise/solvers.py`，发现以下 3 处处于中高风险区的问题，列出严谨的判定与修复方案：

| 编号 | 严重度 | 所在文件与行号 | 缺陷描述 | 潜在风险与物理后果 | 改进方案 / 补丁建议 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **SEC-01** | **Medium** | `solvers.py:471-475` | CFL 实际值检查写死各向同性扩散系数，未将 $c_{\text{tangent}}$ 纳入扩散乘子 | 若用户配置 $c_{\text{tangent}} > 1.0$（如强切向平滑），`cfl_actual` 将低估实际谱半径，引发高频发散 | `cfl_actual = 6.0 * dt * eps2 * max(c_tangent, 1.0) / (h * h)` |
| **SEC-02** | **Medium** | `solvers.py:501` | 梯度截断阈值 $\kappa^2=0.25$ 为带量纲硬编码常数，未按网格尺寸归一化 | $|\nabla \phi|^2 \sim \mathcal{O}(1/\varepsilon^2) = \mathcal{O}(1/(4h^2))$。当 $h$ 极端变化时截断灵敏度漂移 | 改为尺度自适应阈值：$\kappa_0 = 0.25 / \varepsilon^2$ 或在梯度计算前归一化 |
| **SEC-03** | **Low** | `solvers.py:444` | $\beta(t)$ 全局均值补偿未设幅值上限防饱和机制 | 在极端偏态噪声或极度远离平衡态初始场下，$\beta$ 过大会导致双阱势非对称偏移 | 建议增加软截断：`beta = np.clip(beta, -1.0, 1.0)` |

---

# 第三部分：科研论文 Benchmark 实验设计规范 (Benchmark Protocol Specification)

为在国际顶级几何处理与图像计算期刊（如 ACM TOG, IEEE TVCG, SICON, SIAM Imaging Sciences, Computer-Aided Design）中充分论证 `pf_ac_conserved` 与 `pf_ac_aniso` 相对传统 AC、拉普拉斯平滑和双边滤波的核心优势，设计如下标准 Benchmark 实验。

## 3.1 实验对比矩阵与测试曲面集 (Test Shapes)

选取的 4 类基准形状覆盖从平凡拓扑到尖锐棱角、从恒定曲率到非局部高频振荡的全谱特征：

```
+----------------------------------------------------------------------------------------------------+
|                                    4 大核心基准测试曲面拓扑与几何特征                              |
+----------------------+--------------------+--------------------------------+-----------------------+
| 曲面标识 (Shape)     | 拓扑亏格 (Genus)   | 主曲率分布与几何特征           | 物理/算法挑战定位     |
+----------------------+--------------------+--------------------------------+-----------------------+
| Sphere (球体)        | $g = 0$ (凸)       | 恒定正曲率 $\kappa_1=\kappa_2$ | 检验纯毛细收缩与体积  |
|                      |                    | 解析真体积 $V = \frac{4}{3}\pi$| 守恒相场无漂移特性    |
+----------------------+--------------------+--------------------------------+-----------------------+
| Torus (圆环面)       | $g = 1$ (非单连通) | 正/负高斯曲率并存              | 检验局部自交与颈部收缩|
|                      |                    | 最小环颈处有极强收缩应力       | (Pinching) 拓扑稳定性 |
+----------------------+--------------------+--------------------------------+-----------------------+
| Superquadric         | $g = 0$ (拟立方体) | 大面积平坦区 ($\kappa \approx 0$) 检验各向异性张量在尖锐  |
| (超二次曲面)         | $e_1=e_2=0.6$      | 与极高曲率锐利边缘 ($C^0$ 棱线)| 棱边处的法向保真能力  |
+----------------------+--------------------+--------------------------------+-----------------------+
| Bumpy (多频凹凸曲面) | $g = 0$ (星形)     | 球谐多频波动叠加               | 检验细节保持与噪声去除|
|                      |                    | 高频几何纹理                   | 的频域分离帕累托前沿  |
+----------------------+--------------------+--------------------------------+-----------------------+
```

---

## 3.2 噪声注入模型与统计假设 (Noise Models)

本设计拒绝仅测试单一白高斯噪声，选取 4 组具有严格真实传感器物理背景的扰动分布：

1. **加性白高斯噪声 (`Gaussian`, $\sigma = 0.015, 0.035$)**：
   - *物理假设*：零均值、各向同性、空间独立同分布（i.i.d.）。
   - *评测目的*：建立无偏去噪下的理论极限，作为与 Noise2Noise 协议对照的黄金基准。
2. **单向定向凸起噪声 (`Protrusion`, $p = 0.08, \text{amp} = 0.04$)**：
   - *物理假设*：**有偏分布**（$\mathbb{E}[\mathbf{n} \cdot \Delta \mathbf{x}] > 0$），在局部区域单侧隆起。
   - *评测目的*：传统拉普拉斯与标准 AC 在单侧有偏噪声下会产生伪形态偏移；检验拉格朗日体积守恒对全局体积平衡的强约束能力。
3. **空间相关各向异性噪声 (`Aniso`, $\sigma_{\text{low}} = 0.002, \sigma_{\text{high}} = 0.015$)**：
   - *物理假设*：低频沿法向波动 + 高频抖动，模拟激光雷达（LiDAR）与结构光扫描仪的条纹漂移与机械震动。
   - *评测目的*：检验各向异性扩散张量对条纹状与法向相关误差的抑制能力。
4. **真实复合噪声 (`Realistic`)**：
   - *物理假设*：复合 $\text{Gaussian}(0.012) + \text{Protrusion}(0.04) + \text{Outlier}(0.01)$。
   - *评测目的*：多重非理想扰动下的工程级鲁棒性验证。

---

## 3.3 定量评价指标体系 (Quantitative Metrics Suite)

从**几何贴合度**、**特征法向保持**、**物理不变量维持**与**计算经济性** 4 个正交维度量化：

1. **几何重建精度**：
   - **双向 Chamfer 距离 ($\text{CD}\downarrow$)**：
     $$
     \text{CD}(P, Q) = \frac{1}{2|P|}\sum_{p \in P} \min_{q \in Q}\|p - q\|_2 + \frac{1}{2|Q|}\sum_{q \in Q} \min_{p \in P}\|q - p\|_2
     $$
   - **点到曲面密集重采样代理距离 ($\text{P2M}\downarrow$) 与 95 分位值 ($\text{P2M}_{95}\downarrow$)**：
     采用 $4\times \sim 8\times$ 解析重采样代理真值，量化非对称极端偏离误差。
   - **双向 Hausdorff 最坏截断误差 ($\text{HD}\downarrow$)**：
     $$
     \text{HD}(P, Q) = \max \left\{ \max_{p \in P} \min_{q \in Q}\|p - q\|_2, \; \max_{q \in Q} \min_{p \in P}\|q - p\|_2 \right\}
     $$
2. **尖锐特征与法向一致性**：
   - **法向角误差度数 ($\text{nErr}^\circ\downarrow$)**：
     $$
     \text{nErr} = \frac{1}{|P|} \sum_{i=1}^{|P|} \arccos\left( |\mathbf{n}_{\text{est}}(p_i) \cdot \mathbf{n}_{\text{gt}}(p_i)| \right) \times \frac{180^\circ}{\pi}
     $$
     利用 $k\text{-NN}$ 协方差 PCA 恢复法向，度量去噪后曲面法向场与解析法向场的一致性。
3. **物理保真与不变量守恒**：
   - **相对尺度/体积漂移率 ($\text{rDrift}\% \to 0$)**：
     $$
     \text{rDrift} = \frac{\bar{r}(P_{\text{est}}) - \bar{r}(P_{\text{ref}})}{\bar{r}(P_{\text{ref}})} \times 100\%
     $$
     在星形曲面上监测等效半径漂移；在封闭 Sphere/Torus 上，计算三维相场正相体积分数：
     $$
     \Delta V\% = \frac{h^3 \sum \mathbb{I}(\phi > 0) - V_0}{V_0} \times 100\%
     $$
4. **体素下限触底标志 (`at_floor`)**：
   - 检验 CD 是否已达到空间体素量化极限 $h/2$。若触底，表明任何剩余误差源于离散网格分辨率而非算法松弛。

---

## 3.4 论文对比实验结果推演与预期性能分析 (Hypothesis & Performance Table)

根据算法数学机制，预期各求解器在评测矩阵中的优势对比如下：

```
+----------------------------------------------------------------------------------------------------+
|                     各求解器在标准 Benchmark 下的理论特性与预期性能对比表                         |
+-------------------+---------------------+--------------------+--------------------+----------------+
| 求解器 (Solver)   | 空间阶数 / CFL 刚性 | 体积漂移 (rDrift%) | 棱角钝化 (nErr°)   | 核心优势与短板 |
+-------------------+---------------------+--------------------+--------------------+----------------+
| Laplacian         | 2阶 / 无刚性        | 严重收缩 (-5%~-15%)| 严重钝化 (>25°)    | 计算极快，但收 |
|                   |                     |                    |                    | 缩严重，抹杀角 |
+-------------------+---------------------+--------------------+--------------------+----------------+
| Bilateral         | 非局部积分 / O(N·27)| 较小 (-0.1%~-1.0%) | 良好 (<15°)        | 特征保持好，但 |
|                   |                     |                    |                    | 复杂度高拒绝大点|
+-------------------+---------------------+--------------------+--------------------+----------------+
| pf_ac (标准 AC)   | 2阶 / CFL=1/12      | 毛细腐蚀 (< -1.0%) | 中等 (20°~35°)     | 曲率流平滑强， |
|                   |                     | (λ=0 时整相消失)   |                    | 但易腐蚀尖锐处 |
+-------------------+---------------------+--------------------+--------------------+----------------+
| cahn_hilliard     | 4阶 / 极端刚性      | 严格守恒 (~0.0%)   | 欠平滑，噪声残留   | 理论保体积，但 |
|                   | Δt ~ O(h⁴/Mε²)      |                    |                    | 时步极小很难跑 |
+-------------------+---------------------+--------------------+--------------------+----------------+
| pf_ac_conserved   | 2阶 / CFL=1/12      | 绝对守恒 (< 0.05%) | 良好 (15°~25°)     | 完美替代 CH：  |
| (本工作贡献 1)    | (无四阶刚性)        | (即使 λ=0 亦不缩)  |                    | 大时步+严格保体积|
+-------------------+---------------------+--------------------+--------------------+----------------+
| pf_ac_aniso       | 2阶 / CFL=1/12      | 极小漂移 (< 0.2%)  | 最优 (显著下降 50%)| 沿切向快速去噪 |
| (本工作贡献 2)    | (切向平滑+法向抑制) |                    | 棱线角点保持极佳   | 法向抑制保护特征|
+-------------------+---------------------+--------------------+--------------------+----------------+
```

---

## 3.5 论文呈现建议与消融实验设计 (Ablation Studies for Paper)

为使论文达到顶刊规范要求，建议在实验章节安排以下三组正交消融实验：

### 实验 1：长时间演化下的毛细腐蚀崩溃实验（Capillary Coarsening Immunity）
- **实验目的**：对比 `pf_ac` 与 `pf_ac_conserved` 在保真项系数 $\lambda \to 0$ 或长演化迭代（$N_{\text{iter}} = 10 \to 1000$）下的收缩演变轨迹。
- **作图方案**：绘制 $t \in [0, T]$ 期间的 $\Delta V(t)$ 曲线与三维等值面演化图。标准 AC 呈现特征时间 $t^* \sim R_0^2$ 下的不可逆收缩坍塌至 0 测度；而 L-AC 的体积保持率稳定在 $100\% \pm 0.01\%$。

### 实验 2：超二次曲面尖锐特征保持对比实验（Edge Sharpness Preservation）
- **实验目的**：在 `superquadric`（参数 $e_1=e_2=0.6$）上对比各向同性 AC、双边滤波与各向异性 A-AC 的棱边保真度。
- **作图方案**：提取截面等值线（Cross-section profile）并叠加展示；统计法向误差累积直方图（Normal error histogram）。A-AC 的法向误差应呈现紧密围绕 $0^\circ$ 的尖峰分布，在转折棱线上无圆角化过度平滑。

### 实验 3：网格分辨率敏感度分析（Resolution Floor Scaling）
- **实验目的**：检验求解器在体素分辨率 $n_{\text{vox}} \in \{32, 48, 64, 96, 128\}$ 下的 Chamfer 距离与 P2M 渐近线。
- **论据支撑**：证明在 $h \ge 2 \times \text{点间距}$ 时相场求解器稳定收敛至体素物理下限 $\text{res\_floor} = h/2$，证明本包自动推导网格分辨率策略（`auto_resolution`）的严谨有效性。

---

# 结论与下一步工程建议

`phase_field_ac_conserved` 与 `phase_field_ac_anisotropic` 构成了 `pfdenoise` 求解器库中兼具**理论深度**与**工程实用性**的核心算子。建议合并入主分支，并在下一步中：
1. 在 `pfdenoise/solvers.py` 中合入本报告提出的 `SEC-01`（各向异性张量谱半径 CFL 校正）及 `SEC-02`（自适应梯度截断）；
2. 运行预设的完整 Benchmark 命令 `python -m pfdenoise run --shape superquadric --solvers laplacian bilateral pf_ac pf_ac_conserved pf_ac_aniso --out benchmark_quadric.csv`，将产出数据固化为论文图表。
