#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""run_comparative_benchmark.py — 真实端到端科研基准测试跑表脚本

对比六大求解器：
1. none (噪声基线)
2. laplacian (经典平滑收缩基线)
3. pf_ac (标准显式 Allen-Cahn)
4. pf_ac_conserved (拉格朗日乘子保体积 Allen-Cahn)
5. pf_ac_aniso (各向异性扩散张量 Allen-Cahn)
6. pf_ac_convex (半隐式 Eyre 凸分裂无条件稳定 Allen-Cahn)
"""

import sys
from pathlib import Path

# Force UTF-8 on Windows
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import pfdenoise as pfd

def main():
    print("=" * 70)
    print("  pfdenoise 真实物理评测跑表：Bumpy 复杂含噪曲面去噪实验")
    print("=" * 70)
    
    solvers_to_test = [
        "none",
        "laplacian",
        "pf_ac",
        "pf_ac_conserved",
        "pf_ac_aniso",
        "pf_ac_convex",
    ]
    
    rows = pfd.run_benchmark(
        shape="bumpy",
        n=2500,
        noise="gaussian",
        seed=42,
        solver_names=solvers_to_test,
        solver_kwargs={
            "pf_ac": {"iters": 80, "n_vox_max": 48},
            "pf_ac_conserved": {"iters": 80, "n_vox_max": 48},
            "pf_ac_aniso": {"iters": 80, "n_vox_max": 48},
            "pf_ac_convex": {"iters": 50, "dt_mult": 1.5, "n_vox_max": 48},
        },
        verbose=True,
    )
    
    print("\n" + "=" * 70)
    print("  实验结果 Markdown 汇总表 (可直接引用进论文)")
    print("=" * 70)
    table_md = pfd.make_table(rows, fmt="markdown")
    print(table_md)
    print("=" * 70)

if __name__ == "__main__":
    main()
