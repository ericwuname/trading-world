"""实验脚本共用的磁盘缓存与**可证明完备的缓存键**。

为什么单独抽一层
----------------
阶段5 与阶段6 各自有一份几乎相同的 ``_Cache``：读 JSON、``get_or_run``、
``flush``。重复本身不是问题，问题是**两份的缓存键都不是完备的**。

真实事故（本文件存在的直接原因）
--------------------------------
阶段6 的缓存键原来是::

    CACHE.key("lmf_run", tag=tag, seed=seed, n_ticks=n_ticks,
              mix=sorted(mix.items()))

看起来"该有的都有了"。但被缓存的那个量 —— 一场模拟的长记忆指标 ——
实际还依赖**市场是怎么被装配的**：``Stage6Market.decorate_agent`` 决定
要不要给 ``MetaOrderMixin`` 传一个惰性配置。这段代码当时有个 bug：
对照组本应"机制全关"，结果混进默认的 ``p_meta_start=0.02``，
**对照组根本不是对照**。

修掉那段代码之后重跑，键**一个字符都没变** → 全部命中陈旧缓存 →
产出的数字与修 bug 之前**逐位相同**。如果没有事后核对，
这个"看起来完全正常"的陈旧基线就会被当成修复后的证据写进报告。

教训：缓存键必须包含**所有能改变被缓存量的输入**，其中最难想到的一类是
**"把配置翻译成行为"的那段代码本身**。所以本模块提供三层签名：

``lib_sig``        —— ``tw/`` 整棵源码树的内容哈希（库改了必须失效）
``source_sig``     —— 指定函数/类的源码哈希（**行为定义代码**改了必须失效）
``consts_sig``     —— 模块级常量的哈希（常量不在参数里，容易漏）

用法::

    from _cache import Cache, lib_sig, source_sig, consts_sig

    B = source_sig(Stage6Market, factory) + lib_sig()
    key = CACHE.key("lmf_run", tag=..., seed=..., behavior=B)
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


# ----------------------------------------------------------------------
def sha1_hex(parts) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update(p.encode("utf-8", errors="replace"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


def file_sig(paths) -> str:
    """一组文件的内容哈希（按相对路径排序，保证跨机器/跨次运行稳定）。"""
    pairs = []
    for p in sorted(Path(x) for x in paths):
        try:
            pairs.append((str(p), hashlib.sha1(p.read_bytes()).hexdigest()))
        except OSError:
            pairs.append((str(p), "<missing>"))
    return sha1_hex([f"{a}:{b}" for a, b in pairs])


def lib_sig(root: Path | None = None) -> str:
    """``tw/`` 源码树的内容哈希。库改了 → 所有缓存失效。

    ⚠️ **故意不加 ``functools.lru_cache``。**
    第一版加了，理由是"一次进程里算一次就够、省点 IO"。结果被自己的测试
    抓住了：函数的**返回值取决于磁盘上的文件内容**，而 ``lru_cache``
    只按**参数**记忆 → 参数没变（还是同一个 ``root``）就永远返回第一次的结果。
    一个"用来发现文件变了"的函数，自己却看不见文件变化。

    代价核算：目录里 20 个左右的 ``.py``，全量读一遍 + sha1 是毫秒级；
    每个脚本在模块级调用一次，或经 ``LIB_SIG`` 常量调用。
    为这点开销换回"签名真的反映当前代码"，是明显划算的。
    """
    r = Path(root) if root else ROOT
    files = [p for p in (r / "tw").rglob("*.py")
             if "__pycache__" not in p.parts]
    files.append(r / "scripts" / "_common.py")
    return file_sig(files)


def source_sig(*objs) -> str:
    """指定函数/类的**源码**哈希——"行为定义代码"。

    用 ``inspect.getsource`` 而不是整个文件：改一句 print 不该让缓存失效，
    改一行决策逻辑必须失效。后者才是这里要防的。
    """
    parts = []
    for o in objs:
        try:
            parts.append(f"{o.__qualname__}:{inspect.getsource(o)}")
        except (OSError, TypeError):
            parts.append(f"{o!r}:<unavailable>")
    return sha1_hex(parts)


def consts_sig(**kw) -> str:
    """模块级常量的哈希。

    常量**不在**函数参数里，所以既不会被 ``source_sig`` 覆盖，
    也不会被调用方显式传进来——典型的"少一个输入"。
    这里用 ``repr`` 而非 ``json``，好让它接受任意可表示对象。
    """
    return sha1_hex([f"{k}={v!r}" for k, v in sorted(kw.items())])


# ----------------------------------------------------------------------
class Cache:
    """极简磁盘缓存（JSON）：只缓存**每场的汇总**，不缓存行情序列。

    与旧版本的关键差别：``get_or_run`` 会记录命中/未命中，
    ``report()`` 把它们打出来。**静默命中陈旧缓存**是这类事故的共同特征，
    所以命中情况必须可见。
    """

    def __init__(self, path: Path, *, label: str = "") -> None:
        self.path = Path(path)
        self.label = label or self.path.stem
        self.data: dict[str, dict] = {}
        self.hits = 0
        self.misses = 0
        self.stored_sig: str | None = None
        self.sig = ""
        if self.path.exists():
            try:
                raw = json.loads(self.path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                print(f"  ⚠️ 缓存 {self.path.name} 解析失败，重建")
                raw = None
            if not isinstance(raw, dict):
                raw = None
            if raw is not None and "_sig" in raw and "_entries" in raw:
                self.data = raw["_entries"]
                self.stored_sig = raw["_sig"]
            elif raw:
                # 旧格式（没有签名外壳）——**一律丢弃**。
                # 用没有签名机制的代码写出来的缓存，无法判断是否陈旧；
                # 宁可重算一次，也不要拿一个"看起来正常"的旧数字当证据。
                print(f"  ♻️ {self.path.name}：旧格式（无签名），丢弃 "
                      f"{len(raw)} 条缓存并重算")
                self.data = {}

    # ------------------------------------------------------------------
    def bind(self, sig: str) -> None:
        """绑定本轮的**全局签名**（库 + 行为代码）。

        签名与缓存文件里记录的不一致 → 整个缓存作废。
        这是最后一道保险：即使某个 ``key()`` 调用忘了传某个输入，
        只要这个缓存文件是被**别的代码版本**写出来的，就不会被误用。
        """
        self.sig = sig
        if self.stored_sig is not None and self.stored_sig != sig:
            n = len(self.data)
            if n:
                print(f"  ♻️ {self.path.name}：代码已变（签名 {self.stored_sig} → "
                      f"{sig}），丢弃 {n} 条陈旧缓存")
            self.data = {}
        self.stored_sig = sig

    def key(self, kind: str, **kw) -> str:
        """缓存键。⚠️ 第一个参数**不能**叫 ``tag``——调用方常常也要传
        ``tag=...``，重名会直接 ``TypeError: got multiple values``。"""
        blob = json.dumps({"kind": kind, **kw}, sort_keys=True, default=str)
        return hashlib.sha1(blob.encode()).hexdigest()[:16]

    def get_or_run(self, key: str, fn):
        if key in self.data:
            self.hits += 1
            return self.data[key]
        self.misses += 1
        v = fn()
        self.data[key] = v
        return v

    def flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"_sig": self.sig, "_entries": self.data}
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self.path)   # 原子替换：中途崩了不会留下半个文件

    def report(self) -> None:
        tot = self.hits + self.misses
        if not tot:
            return
        pct = 100.0 * self.hits / tot
        print(f"  📦 {self.label} 缓存：命中 {self.hits} / 共 {tot} "
              f"（{pct:.0f}%），未命中 {self.misses}")
        if self.misses:
            print(f"     未命中说明缓存键或代码签名变了——"
                  f"重跑出来的是新结果，不是旧数字")
