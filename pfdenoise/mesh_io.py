"""io.py — 3D 点云与网格数据 I/O 引擎（纯 Python / 零依赖 + 格式适配）

支持格式
--------
* PLY (.ply)：支持标准 ASCII 与常见 Binary Little Endian 点云与三角网格读写
* OBJ (.obj)：支持标准 Wavefront 顶点 (v)、法向 (vn)、面片 (f) 解析与导出
* MAT (.mat)：支持读取 MATLAB 3D 占据网格 (如 Image and Vision Computing 语料)
* XYZ / CSV / PTS：通配坐标散点格式

功能
----
* load_point_cloud(path) -> (P, N, faces)
* save_point_cloud(path, P, N=None, faces=None)
* normalize_cloud(P, scale=1.0) -> (P_norm, centroid, scale_factor)
* denoise_file(in_path, out_path, solver="pf_ac_conserved", **kw)
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import numpy as np

from .metrics import estimate_normals
from .solvers import get_solver

__all__ = [
    "load_point_cloud",
    "save_point_cloud",
    "normalize_cloud",
    "denoise_file",
    "load_ply",
    "save_ply",
    "load_obj",
    "save_obj",
    "load_mat_volume",
]


def normalize_cloud(
    P: np.ndarray, target_scale: float = 1.0
) -> tuple[np.ndarray, np.ndarray, float]:
    """将点云平移至质心并各向同性缩放至 [-target_scale, target_scale] 包围盒。

    返回 (P_norm, centroid, scale_factor)，用于在去噪后逆变换还原真实空间尺度。
    """
    P = np.asarray(P, dtype=np.float64)
    centroid = P.mean(axis=0)
    shifted = P - centroid
    ext = np.ptp(shifted, axis=0)  # max - min
    max_ext = float(np.max(ext))
    if max_ext < 1e-12:
        return shifted, centroid, 1.0
    scale_factor = (2.0 * target_scale) / max_ext
    P_norm = shifted * scale_factor
    return P_norm, centroid, scale_factor


# --------------------------------------------------------------------------
# PLY 读写器
# --------------------------------------------------------------------------
def load_ply(path: str | Path) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """读取 PLY 文件，支持 ASCII 和 binary_little_endian 格式。"""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"PLY 文件不存在：{p}")

    with open(p, "rb") as f:
        header_lines: list[str] = []
        while True:
            line = f.readline().decode("latin-1", errors="replace").strip()
            header_lines.append(line)
            if line == "end_header":
                break

    # 解析头部信息
    format_type = "ascii"
    n_vertices = 0
    n_faces = 0
    properties: list[tuple[str, str]] = []
    current_element = ""

    for h in header_lines:
        tokens = h.split()
        if not tokens:
            continue
        if tokens[0] == "format":
            format_type = tokens[1]
        elif tokens[0] == "element":
            current_element = tokens[1]
            if tokens[1] == "vertex":
                n_vertices = int(tokens[2])
            elif tokens[1] == "face":
                n_faces = int(tokens[2])
        elif tokens[0] == "property" and current_element == "vertex":
            properties.append((tokens[1], tokens[2]))

    prop_names = [prop[1] for prop in properties]
    x_idx = prop_names.index("x") if "x" in prop_names else 0
    y_idx = prop_names.index("y") if "y" in prop_names else 1
    z_idx = prop_names.index("z") if "z" in prop_names else 2
    has_nx = "nx" in prop_names and "ny" in prop_names and "nz" in prop_names
    nx_idx = prop_names.index("nx") if has_nx else -1
    ny_idx = prop_names.index("ny") if has_nx else -1
    nz_idx = prop_names.index("nz") if has_nx else -1

    if format_type == "ascii":
        with open(p, "r", encoding="utf-8", errors="replace") as f:
            # 跳过 header
            for line in f:
                if line.strip() == "end_header":
                    break
            v_data: list[list[float]] = []
            for _ in range(n_vertices):
                parts = f.readline().strip().split()
                if not parts:
                    continue
                v_data.append([float(x) for x in parts])

            v_arr = np.array(v_data, dtype=np.float64)
            P = v_arr[:, [x_idx, y_idx, z_idx]]
            N = v_arr[:, [nx_idx, ny_idx, nz_idx]] if has_nx else None

            faces = None
            if n_faces > 0:
                f_data: list[list[int]] = []
                for _ in range(n_faces):
                    parts = f.readline().strip().split()
                    if len(parts) >= 4 and int(parts[0]) == 3:
                        f_data.append([int(parts[1]), int(parts[2]), int(parts[3])])
                if f_data:
                    faces = np.array(f_data, dtype=np.int64)

            return P, N, faces

    elif "binary_little_endian" in format_type:
        # Binary little endian 格式快速解析
        dtype_map = {
            "float": "<f4",
            "float32": "<f4",
            "double": "<f8",
            "float64": "<f8",
            "int": "<i4",
            "int32": "<i4",
            "uchar": "u1",
            "uint8": "u1",
        }
        dt_spec = [
            (name, dtype_map.get(dtype_name, "<f4"))
            for dtype_name, name in properties
        ]
        vertex_dt = np.dtype(dt_spec)

        with open(p, "rb") as f:
            while True:
                line = f.readline().decode("latin-1", errors="replace").strip()
                if line == "end_header":
                    break
            raw_v = np.fromfile(f, dtype=vertex_dt, count=n_vertices)
            P = np.column_stack([raw_v["x"], raw_v["y"], raw_v["z"]]).astype(np.float64)
            N = (
                np.column_stack([raw_v["nx"], raw_v["ny"], raw_v["nz"]]).astype(np.float64)
                if has_nx
                else None
            )
            return P, N, None

    raise ValueError(f"不支持的 PLY 格式：{format_type}")


def save_ply(
    path: str | Path,
    P: np.ndarray,
    N: np.ndarray | None = None,
    faces: np.ndarray | None = None,
) -> None:
    """以标准 ASCII PLY 格式保存点云或三角网格。"""
    P = np.asarray(P, dtype=np.float64)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n_v = len(P)
    n_f = len(faces) if faces is not None else 0
    has_n = N is not None and len(N) == n_v

    header = [
        "ply",
        "format ascii 1.0",
        "comment Created by pfdenoise scientific engine",
        f"element vertex {n_v}",
        "property float x",
        "property float y",
        "property float z",
    ]
    if has_n:
        header.extend([
            "property float nx",
            "property float ny",
            "property float nz",
        ])
    if n_f > 0:
        header.extend([
            f"element face {n_f}",
            "property list uchar int vertex_indices",
        ])
    header.append("end_header\n")

    with open(p, "w", encoding="utf-8") as f:
        f.write("\n".join(header))
        if has_n and N is not None:
            for pt, norm in zip(P, N):
                f.write(
                    f"{pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f} "
                    f"{norm[0]:.6f} {norm[1]:.6f} {norm[2]:.6f}\n"
                )
        else:
            for pt in P:
                f.write(f"{pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f}\n")

        if n_f > 0 and faces is not None:
            for fc in faces:
                f.write(f"3 {fc[0]} {fc[1]} {fc[2]}\n")


# --------------------------------------------------------------------------
# OBJ 读写器
# --------------------------------------------------------------------------
def load_obj(path: str | Path) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """读取 Wavefront OBJ 格式点云或网格。"""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"OBJ 文件不存在：{p}")

    verts: list[list[float]] = []
    norms: list[list[float]] = []
    faces: list[list[int]] = []

    with open(p, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            tokens = line.split()
            tag = tokens[0]
            if tag == "v":
                verts.append([float(tokens[1]), float(tokens[2]), float(tokens[3])])
            elif tag == "vn":
                norms.append([float(tokens[1]), float(tokens[2]), float(tokens[3])])
            elif tag == "f":
                # 处理 v/vt/vn 复合索引
                v_indices: list[int] = []
                for tok in tokens[1:4]:
                    idx = int(tok.split("/")[0])
                    v_indices.append(idx - 1 if idx > 0 else len(verts) + idx)
                if len(v_indices) == 3:
                    faces.append(v_indices)

    P = np.array(verts, dtype=np.float64)
    N = np.array(norms, dtype=np.float64) if len(norms) == len(verts) else None
    F = np.array(faces, dtype=np.int64) if faces else None
    return P, N, F


def save_obj(
    path: str | Path,
    P: np.ndarray,
    N: np.ndarray | None = None,
    faces: np.ndarray | None = None,
) -> None:
    """写入 Wavefront OBJ 文件。"""
    P = np.asarray(P, dtype=np.float64)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    has_n = N is not None and len(N) == len(P)

    with open(p, "w", encoding="utf-8") as f:
        f.write("# Wavefront OBJ created by pfdenoise\n")
        for pt in P:
            f.write(f"v {pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f}\n")
        if has_n and N is not None:
            for nm in N:
                f.write(f"vn {nm[0]:.6f} {nm[1]:.6f} {nm[2]:.6f}\n")
        if faces is not None:
            for fc in faces:
                # 1-based index
                if has_n:
                    f.write(
                        f"f {fc[0]+1}//{fc[0]+1} {fc[1]+1}//{fc[1]+1} {fc[2]+1}//{fc[2]+1}\n"
                    )
                else:
                    f.write(f"f {fc[0]+1} {fc[1]+1} {fc[2]+1}\n")


# --------------------------------------------------------------------------
# MATLAB 3D Volume 读写器
# --------------------------------------------------------------------------
def load_mat_volume(
    path: str | Path, var_name: str = "ou", n_sample: int = 4000
) -> tuple[np.ndarray, np.ndarray]:
    """从 MATLAB .mat 3D 占据矩阵提取边界曲面点云。

    专用于加载课题组 IMAVIS / Surface Reconstruction 论文实验中的 .mat 数据。
    """
    try:
        import scipy.io as sio
    except ImportError as e:
        raise ImportError("加载 .mat 文件需要安装 scipy: pip install scipy") from e

    mat = sio.loadmat(str(path))
    if var_name not in mat:
        keys = [k for k in mat if not k.startswith("__")]
        if not keys:
            raise KeyError(f".mat 中没有可用变量：{path}")
        var_name = keys[0]

    vol = mat[var_name]
    if vol.ndim != 3:
        raise ValueError(f"期望 3D 矩阵，实测 shape: {vol.shape}")

    # 提取内部占据和外边界的等值面体素
    occ = vol > 0.5 if vol.dtype != bool else vol
    # 6 邻域差分找表面
    p_occ = np.pad(occ, 1, mode="constant", constant_values=False)
    has_empty_nb = (
        (~p_occ[2:, 1:-1, 1:-1])
        | (~p_occ[:-2, 1:-1, 1:-1])
        | (~p_occ[1:-1, 2:, 1:-1])
        | (~p_occ[1:-1, :-2, 1:-1])
        | (~p_occ[1:-1, 1:-1, 2:])
        | (~p_occ[1:-1, 1:-1, :-2])
    )
    surface_idx = np.argwhere(occ & has_empty_nb)
    if len(surface_idx) == 0:
        surface_idx = np.argwhere(occ)

    # 归一化体素中心到 [-1, 1]
    nx, ny, nz = vol.shape
    h = 2.0 / max(nx, ny, nz)
    pts = (surface_idx - np.array([nx, ny, nz]) * 0.5) * h

    if len(pts) > n_sample:
        sel = np.linspace(0, len(pts) - 1, n_sample).astype(np.int64)
        pts = pts[sel]

    norms = estimate_normals(pts, k=16)
    return pts, norms


# --------------------------------------------------------------------------
# 通用顶层分发
# --------------------------------------------------------------------------
def load_point_cloud(
    path: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """统一接口：从文件加载点云 (P, N, faces)。

    若源文件未提供法向 N，自动基于局部 PCA 估计出一致法向。
    """
    p = Path(path)
    ext = p.suffix.lower()
    if ext == ".ply":
        P, N, F = load_ply(p)
    elif ext == ".obj":
        P, N, F = load_obj(p)
    elif ext == ".mat":
        P, N = load_mat_volume(p)
        F = None
    elif ext in (".xyz", ".pts", ".txt", ".csv"):
        data = np.loadtxt(p, delimiter="," if ext == ".csv" else None)
        P = data[:, :3]
        N = data[:, 3:6] if data.shape[1] >= 6 else None
        F = None
    else:
        raise ValueError(f"不支持的文件扩展名：{ext} (支持 .ply, .obj, .mat, .xyz, .csv)")

    if N is None or len(N) != len(P):
        N = estimate_normals(P, k=16)
    return P, N, F


def save_point_cloud(
    path: str | Path,
    P: np.ndarray,
    N: np.ndarray | None = None,
    faces: np.ndarray | None = None,
) -> None:
    """统一保存接口：依据文件扩展名自动分发至 PLY 或 OBJ 写入器。"""
    ext = Path(path).suffix.lower()
    if ext == ".ply":
        save_ply(path, P, N, faces)
    elif ext == ".obj":
        save_obj(path, P, N, faces)
    else:
        # 默认保存为 XYZ 文本
        P = np.asarray(P, dtype=np.float64)
        out = np.hstack([P, N]) if N is not None else P
        np.savetxt(path, out, fmt="%.6f")


def denoise_file(
    in_path: str | Path,
    out_path: str | Path,
    solver: str = "pf_ac_conserved",
    **solver_kwargs: Any,
) -> dict[str, Any]:
    """端到端文件去噪管线：载入 -> 归一化 -> 求解 -> 逆尺度变换 -> 导出文件。"""
    P_raw, N_raw, faces = load_point_cloud(in_path)
    P_norm, centroid, scale_factor = normalize_cloud(P_raw, target_scale=1.0)
    solver_fn = get_solver(solver)

    out = solver_fn(P_norm, N_raw, **solver_kwargs)
    P_est_norm = out[0] if isinstance(out, tuple) else out
    info = out[1] if isinstance(out, tuple) else {}

    # 还原真实空间尺度
    P_est = (P_est_norm / scale_factor) + centroid
    N_est = estimate_normals(P_est, k=16)

    save_point_cloud(out_path, P_est, N_est, faces=faces)
    return {
        "input_points": len(P_raw),
        "output_points": len(P_est),
        "solver": solver,
        "scale_factor": scale_factor,
        "centroid": centroid.tolist(),
        "info": info,
    }
