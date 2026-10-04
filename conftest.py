"""让测试与**运行目录无关**。

问题
----
跑 `pytest tests/` 一律先 `cd` 到项目根，所以全过。
换个目录用绝对路径跑就 ImportError —— 包没安装，而 Python 只把
**当前工作目录**加进 `sys.path`。

为什么要在意
------------
这类故障只报 ImportError，不提示「这是 cwd 问题」，
读的人会以为包坏了。修法是 pytest 的标准做法：conftest.py 显式加 sys.path。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
