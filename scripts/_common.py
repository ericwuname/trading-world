"""各阶段脚本共用的工具（路径、真实基准、指标落盘）。

为什么要抽这一层：四个阶段脚本都要"取真实基准 → 算指标 → 存 json → 打表"。
如果各写一份，某天改了口径（比如把方差比的 q 从 5 改成 10）就必然漏改其中一处，
于是报告里的数字和 json 里的数字不一致——而且没人会发现。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tw import realdata  # noqa: E402
from tw.analyzer import SeriesMetrics, analyze, format_compare_table  # noqa: E402

OUT = ROOT / "out"
FIG = OUT / "figs"

REAL_KEYS = ["BTCUSDT_1h", "ETHUSDT_1h", "SOLUSDT_1h"]
REAL_PRIMARY = "BTCUSDT_1h"


def ensure_dirs() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)


def save_json(obj, name: str) -> Path:
    ensure_dirs()
    path = OUT / name
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, default=_jsonable),
        encoding="utf-8",
    )
    return path


def load_json(name: str) -> dict:
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        v = float(o)
        return None if not np.isfinite(v) else v
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


def real_series(key: str = REAL_PRIMARY):
    return realdata.load_builtin(key)


def real_metrics(keys: list[str] | None = None) -> list[SeriesMetrics]:
    keys = keys or REAL_KEYS
    out = []
    for k in keys:
        s = realdata.load_builtin(k)
        out.append(analyze(s.close, k, vol_window=24))
    return out


def real_flat_map() -> dict[str, dict]:
    """{数据集名: 指标字典}，供报告按名字取数。"""
    return {m.label: m.flat() for m in real_metrics()}


def metrics_of(log, label: str, vol_window: int = 24) -> SeriesMetrics:
    """对一次模拟的中间价序列算全套指标。"""
    return analyze(
        log.mid,
        label,
        vol_window=vol_window,
        n_trades=len(log.trades),
    )


def print_table(metrics: list[SeriesMetrics]) -> None:
    print(format_compare_table(metrics))


def relative_position(sim: dict, real: dict) -> dict[str, float]:
    """模拟指标 / 真实指标 的比值。1.0 = 完全一致。"""
    out = {}
    for k in (
        "excess_kurtosis",
        "acf_abs_lag1",
        "acf_abs_mean_1_10",
        "n_sig_lags_abs",
        "p999_bp",
        "sigma_bp",
        "hill_alpha_left",
        "vr5",
    ):
        r = real.get(k)
        s = sim.get(k)
        if r is None or s is None:
            continue
        try:
            rf, sf = float(r), float(s)
        except (TypeError, ValueError):
            continue
        if rf == 0 or not np.isfinite(rf) or not np.isfinite(sf):
            continue
        out[k] = sf / rf
    return out


def banner(text: str) -> None:
    line = "=" * 74
    print(f"\n{line}\n{text}\n{line}")


def ensure_scripts_on_path() -> None:
    """保证 ``scripts/`` 在 ``sys.path`` 上——**延迟 import 之前请调用它**。

    ⚠️ 为什么不能只靠"模块级插入一次"（本轮真实踩到）：
      · 测试为了避免 ``scripts/gui.py`` 遮蔽仓库根的 ``gui/`` 包，
        会在 import 完之后**把 scripts 从 sys.path 里摘掉**；
      · 于是那些写在各函数体里的延迟 import（``from run_stage3 import ...``）
        就找不到模块了 —— 实测：一份 13 条测试里 5 条直接 ERROR。
    ``_common`` 本身在模块级已被成功 import（缓存在 ``sys.modules``），
    所以拿它的这个函数**不需要** path 先可见 —— 这正是它放在这里的原因。
    """
    p = str(Path(__file__).resolve().parent)
    if p not in sys.path:
        sys.path.insert(0, p)


# ----------------------------------------------------------------------
# 变异体清单：**唯一事实源**是 mutation_check.py 里注册的那些编号。
# 报告、README 对账、小结都要"这个数是多少"，但**不许各自解析一份**——
# 本轮真实踩到：同一个正则被写在两处（make_workstream_summary 与 selfcheck），
# 而且两处**都只抓单引号**，于是 M1~M33（写成 "M1"）全被漏掉，
# 解析结果只剩 9 个。两处同时错，症状却是"数字看着挺正常"。
# 所以这里只留一份实现，别人 import 它。
# ----------------------------------------------------------------------
def mutation_ids() -> list[int]:
    """从 ``scripts/mutation_check.py`` 里解析出全部变异体编号（升序）。

    只读源码文本，不 import 那个模块（避免副作用与沙箱复制的开销）。
    ⚠️ 正则必须**同时认单引号和双引号**：早期条目写成 ``"M1",``，
    新增的写成 ``'M40',``。只认一种就会静默少数一半以上。
    """
    import re
    src = (Path(__file__).resolve().parent / "mutation_check.py").read_text(
        encoding="utf-8")
    return sorted({int(m) for m in re.findall(r"""["']M(\d+)["']\s*,""", src)})


def mutation_range() -> str:
    """``"M1~M42"`` 这样的编号范围；解析不出来时返回 ``"M?"``。"""
    ids = mutation_ids()
    return f"M{ids[0]}~M{ids[-1]}" if ids else "M?"


# ======================================================================
# ⭐⭐ 调用失败率守卫（2026-09-22 由**配额耗尽**逼出来）
# ======================================================================
def check_call_fail_rate(record_path, *, max_rate: float = 0.01,
                         where: str = "") -> dict:
    """扫录制文件，统计 ``ok=False`` 的比例；**超阈值就抛异常**。

    ⚠️⚠️ 为什么这一条必须"会失败"而不是"打印一个数"：

    当天额度用完后，`HTTPClient` 按 ``on_exhausted="hold"`` 把 429
    **回退成一个合法的"弃权"决策** ⇒ 后面 1030/1336 条**根本不是模型做的**，
    而在场率/换手全变成 0 —— **看起来像"模型很保守"**，
    脚本照常跑完、照常出数字、报告照常生成。

    ⇒ 判据：**「跑完了」不等于「数据是用模型跑出来的」。**
    凡是"外部依赖可能失败、而失败会退化成**合法输出**"的地方，
    失败率就必须是一个**会失败的断言**。
    （同型守卫：回放覆盖率、KPI 接线自检。）

    返回统计字典；超阈值时 ``raise RuntimeError``（调用方转成 SystemExit）。
    """
    import json as _json
    from pathlib import Path as _P

    p = _P(record_path)
    n_all = n_bad = 0
    errs: dict = {}
    if p.exists():
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                try:
                    row = _json.loads(line)
                except ValueError:
                    continue
                n_all += 1
                if row.get("ok") is False:
                    n_bad += 1
                    e = str(row.get("error") or "?")[:60]
                    errs[e] = errs.get(e, 0) + 1
    rate = (n_bad / n_all) if n_all else 0.0
    stat = {"n_all": n_all, "n_bad": n_bad, "rate": rate, "errors": errs}
    if rate > max_rate:
        top = max(errs.items(), key=lambda x: x[1])[0] if errs else "?"
        raise RuntimeError(
            f"❌{(' ' + where) if where else ''} 调用失败率 {rate:.2%} "
            f"超过阈值 {max_rate:.2%} ⇒ **这一组数据不可用于任何结论**。\n"
            f"   最常见原因：**额度耗尽**（HTTP 429）或代理不通。\n"
            f"   最严重的后果不是「跑失败」，而是**静默回退**：\n"
            f"   on_exhausted=\"hold\" 会把失败决策写成一个合法的「弃权」，\n"
            f"   于是在场率/换手全变 0，**看起来像模型很保守**，\n"
            f"   而脚本照常跑完、照常出数字。\n"
            f"   最高频错误：{top}\n"
            f"   如确需接受这些失败，请显式放宽 max_rate。")
    return stat
