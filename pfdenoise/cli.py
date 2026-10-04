"""命令行入口：python -m pfdenoise <子命令>"""

import argparse
import sys

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

from .benchmark import DEFAULT_SOLVERS, make_table, run_benchmark, run_n2n
from .noise import NOISE_MODELS
from .shapes import SHAPES
from .solvers import DESCRIPTIONS, SOLVERS


def _listing():
    print("曲面 :", ", ".join(SHAPES))
    print("噪声 :", ", ".join(NOISE_MODELS))
    print("求解器:")
    for k in SOLVERS:
        print(f"   {k:<18} {DESCRIPTIONS.get(k, '')}")


def cmd_list(args):
    _listing()


def cmd_run(args):
    rows = run_benchmark(shape=args.shape, n=args.n, noise=args.noise,
                         seed=args.seed, solver_names=args.solvers,
                         solver_kwargs={s: {} for s in args.solvers},
                         dense_mult=args.dense)
    print()
    print(make_table(rows, title=f"{args.shape} / {args.noise} / n={args.n}"))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(make_table(rows, fmt="csv"))
        print(f"\nCSV 已写入 {args.out}")


def cmd_n2n(args):
    rows = run_n2n(shape=args.shape, n=args.n, noise=args.noise,
                   seed=args.seed, solver_names=args.solvers,
                   solver_kwargs={s: {} for s in args.solvers})
    print()
    print("| solver | 一致性↓ |")
    print("|---|---|")
    for r in rows:
        if "error" in r:
            print(f"| {r['solver']} | FAILED |")
        else:
            print(f"| {r['solver']} | {r['consistency']:.4f} |")


def cmd_noise(args):
    print("噪声模型与其统计假设：\n")
    for name, ctor in NOISE_MODELS.items():
        m = ctor()
        ok = "✓ 适用" if m.n2n_compatible else "✗ 不适用"
        print(f"[{name}]  N2N: {ok}")
        print(f"   {m.assumption_text()}\n")


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="pfdenoise",
        description="物理先验驱动的点云/网格去噪基准与工具箱（零训练，纯 numpy）")
    sub = p.add_subparsers(dest="cmd", required=True)

    pl = sub.add_parser("list", help="列出可用的曲面/噪声/求解器")
    pl.set_defaults(func=cmd_list)

    pr = sub.add_parser("run", help="跑有标签基准并出表")
    pr.add_argument("--shape", default="bumpy", choices=list(SHAPES))
    pr.add_argument("--noise", default="realistic", choices=list(NOISE_MODELS))
    pr.add_argument("-n", type=int, default=6000)
    pr.add_argument("--seed", type=int, default=0)
    pr.add_argument("--dense", type=int, default=4, help="P2M 代理的参考点密度倍数")
    pr.add_argument("--solvers", nargs="+", default=DEFAULT_SOLVERS,
                    choices=list(SOLVERS))
    pr.add_argument("--out", default=None, help="额外导出 CSV 到该路径")
    pr.set_defaults(func=cmd_run)

    pn = sub.add_parser("n2n", help="跑无标签 N2N 一致性协议")
    pn.add_argument("--shape", default="bumpy", choices=list(SHAPES))
    pn.add_argument("--noise", default="gaussian", choices=list(NOISE_MODELS))
    pn.add_argument("-n", type=int, default=6000)
    pn.add_argument("--seed", type=int, default=0)
    pn.add_argument("--solvers", nargs="+", default=DEFAULT_SOLVERS,
                    choices=list(SOLVERS))
    pn.set_defaults(func=cmd_n2n)

    pz = sub.add_parser("noise", help="打印各噪声模型的统计假设")
    pz.set_defaults(func=cmd_noise)

    args = p.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
