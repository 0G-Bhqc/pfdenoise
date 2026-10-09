"""pfdenoise — 物理先验驱动的点云/网格去噪基准与工具箱

设计立场（为什么是这个包）
--------------------------
2025 年的点云去噪综述（arXiv:2508.17011）指出：SOTA 几乎全是有监督方法，
依赖成对脏-净数据，真实传感器造不出来；且方法多在**局部 patch** 上操作，
全局一致性差。2026 年 Noise2Noise 重访（arXiv:2609.16788）进一步给出结论：
起作用的是**训练对的分布**而不是损失函数。

也就是说：**噪声模型是这件事里最重要的设计变量，却最常被当成实现细节**；
而且当拿不到真值时，社区缺少一个可复现的无标签评测协议。

本包因此提供三件东西，全部零训练、纯 numpy：
  1. 噪声模型库   —— 每种噪声显式声明统计假设，判断 N2N 理论是否成立
  2. 求解器库     —— 经典基线 + 相场族（AC / AC+保真 / AC+屏障 / CH）
  3. 评测协议     —— 有标签全指标 + 无标签 N2N 一致性协议

用法
----
    import pfdenoise as pfd
    rows = pfd.run_benchmark(shape="bumpy", noise="realistic")
    print(pfd.make_table(rows))
"""

__version__ = "0.1.1"

import sys as _sys

# Windows 控制台默认 GBK，打印中文/箭头会 UnicodeEncodeError。
# 这是本工作区（AGENTS.md 编码契约）刻意要求的导入期副作用：
# CLI 与脚本统一在包入口完成编码兜底，调用方无需各自处理。
for _s in (_sys.stdout, _sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:                                # pragma: no cover
        pass

from .shapes import make_shape, SHAPES
from .noise import (NoiseModel, Gaussian, Protrusion, Outlier, Aniso, Density,
                    Composite, NOISE_MODELS, make_noise)
from .solvers import (SOLVERS, DESCRIPTIONS, get_solver,
                      phase_field_ac, phase_field_ac_fidelity,
                      phase_field_ac_conserved, phase_field_ac_anisotropic,
                      phase_field_ac_convex, phase_field_ac_convex_adaptive,
                      project_to_zero_level)
from . import metrics, solvers, noise, shapes, mesh_io
from .benchmark import (run_benchmark, run_n2n, run_bias_variance,
                        make_table, make_bv_table, DEFAULT_SOLVERS)
from .mesh_io import (load_point_cloud, save_point_cloud, normalize_cloud, denoise_file,
                      load_ply, save_ply, load_obj, save_obj, load_mat_volume)
from . import benchmark
from . import cli

__all__ = [
    "__version__",
    "make_shape", "SHAPES",
    "NoiseModel", "Gaussian", "Protrusion", "Outlier", "Aniso", "Density",
    "Composite", "NOISE_MODELS", "make_noise",
    "SOLVERS", "DESCRIPTIONS", "get_solver",
    "phase_field_ac", "phase_field_ac_fidelity",
    "phase_field_ac_conserved", "phase_field_ac_anisotropic",
    "phase_field_ac_convex", "phase_field_ac_convex_adaptive",
    "project_to_zero_level",
    "run_benchmark", "run_n2n", "run_bias_variance", "make_table", "make_bv_table",
    "DEFAULT_SOLVERS",
    "load_point_cloud", "save_point_cloud", "normalize_cloud", "denoise_file",
    "load_ply", "save_ply", "load_obj", "save_obj", "load_mat_volume",
    "metrics", "solvers", "noise", "shapes", "mesh_io",
    "benchmark", "cli",
]
