"""变异测试编号的权威分配器。

为什么需要它
------------
本项目已经**两次**出现"任务书里预写的编号与实际对不上"：
  · 三线深挖任务书写 M40~M44，实际只能从 M34 起（编号早被占了）；
  · 分辨力危机任务书写 M45~M49，实际是 M44~M50（差了一位）。
根因是**编号被写在了任务书的文字里**，而写任务书的人看不到源码的当前状态。

⇒ 从本文件起：**编号由本工具在运行时动态领取**，
  任务书只写"通过 claim_mutation_ids 领取"，不再写死数字。

设计
----
· 权威文件：``docs/mutation_registry.json``，结构 ``{"next_available": N, "history": [...]}``
· **原子性**：用 ``O_CREAT|O_EXCL`` 创建锁文件实现互斥（多进程并发也不会发重号）
· ``--init`` **自动探测**当前最大编号（从 ``mutation_check.py`` 实读），
  不信任任何文字里写的数字

用法::

    python scripts/claim_mutation_ids.py --init          # 初始化/校正 next_available
    python scripts/claim_mutation_ids.py --claim 2 --by 工作线L
    python scripts/claim_mutation_ids.py --list
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import ROOT, banner, mutation_ids  # noqa: E402

REGISTRY = ROOT / "docs" / "mutation_registry.json"
LOCK = ROOT / "docs" / ".mutation_registry.lock"

#: 进程内互斥（见 ``_file_lock`` 的说明：环境的安全删除护栏不允许并发 trash 锁文件）
_THREAD_LOCK = threading.Lock()

#: 锁文件超过这个年龄视为「陈旧」（上个进程崩了），允许接管
STALE_SECONDS = 60.0

#: 锁文件超过这个年龄视为「陈旧」（上个进程崩了），允许接管
STALE_SECONDS = 60.0


class RegistryError(Exception):
    pass


@contextmanager
def _file_lock(lock_path: Path = LOCK, timeout: float = 15.0):
    """跨进程互斥（``O_CREAT|O_EXCL`` 语义）。

    ⚠️ 任务书要求"原子地分配"。读-改-写**不是**原子的——
    两个进程同时领号会拿到同一批编号，而那种错误**当时看不出来**
    （两个工作线各自以为拿到了 M51，直到合并时才撞号）。
    所以这里用锁文件把临界区圈起来。

    ⚠️⚠️ 还要额外套一层进程内 ``threading.Lock``（实测踩到）：
    本环境把 ``Path.unlink()`` 实现成了**安全删除（移到回收站）**。
    8 个线程并发抢文件锁时，多个线程会同时去 trash 同一个锁文件，
    直接报 ``OSError: [safe-delete] ... Error during a trash operation``。
    那不是逻辑错，是**环境护栏不允许并发 trash**。
    ⇒ 同一进程内的多线程先串行（这把锁），文件锁只负责跨进程。
    两层各管一段，合起来才是「原子」。
    """
    with _THREAD_LOCK:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.time() + timeout
        fd = None
        # ⚠️ Windows 上加 ``O_TEMPORARY``：文件在**最后一个句柄关闭时自动删除**，
        #    完全不依赖 ``unlink``。本环境把 ``unlink`` 实现成了"安全删除（移到回收站）"，
        #    实测并发 trash 会报错、且失败后锁文件**永久残留**让后续全部超时
        #    （这正是第一次并发测试 8 个线程只成功 2 个的原因）。
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
        if hasattr(os, "O_TEMPORARY"):
            flags |= os.O_TEMPORARY
        while True:
            try:
                fd = os.open(lock_path, flags)
                break
            except FileExistsError:
                # 陈旧锁接管：上个进程崩了、或环境删不掉锁文件时，
                # 不能让一把僵尸锁把整条工作线卡死。
                try:
                    age = time.time() - lock_path.stat().st_mtime
                except OSError:
                    age = 0.0
                if age > STALE_SECONDS:
                    try:
                        os.remove(lock_path)
                    except OSError:
                        # 连 remove 都不行（护栏）⇒ 直接覆盖它的时间戳，
                        # 让下一次判断不再认为它"陈旧"以外的状态
                        try:
                            lock_path.write_text(str(os.getpid()),
                                                 encoding="utf-8")
                        except OSError:
                            pass
                    continue
                if time.time() > deadline:
                    raise RegistryError(
                        f"拿不到锁 {lock_path}（等 {timeout:.0f}s，"
                        f"锁文件年龄 {age:.0f}s）。"
                        f"若确定没有别的进程在跑，删掉这个锁文件即可。")
                time.sleep(0.05)
        try:
            yield
        finally:
            try:
                os.close(fd)
            finally:
                try:
                    lock_path.unlink(missing_ok=True)
                except OSError:
                    # 删不掉不影响正确性：``O_TEMPORARY`` 已经让它在句柄关闭时
                    # 自动消失；即便没有该标志，上面的陈旧锁接管也会兜住。
                    pass


def load_registry(path: Path = REGISTRY) -> dict:
    if not path.exists():
        raise RegistryError(
            f"找不到 {path}。先跑：python scripts/claim_mutation_ids.py --init")
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise RegistryError(f"{path} 不是合法 JSON：{e}") from e
    if "next_available" not in d:
        raise RegistryError(f"{path} 缺 next_available 字段")
    d.setdefault("history", [])
    return d


def save_registry(d: dict, path: Path = REGISTRY) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(d, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")


def current_max_in_code() -> int:
    """从 ``mutation_check.py`` **实读**当前最大编号（不信任文字里的数字）。"""
    ids = mutation_ids()
    return max(ids) if ids else 0


def init_registry(path: Path = REGISTRY, *, force: bool = False) -> dict:
    """初始化（或校正）``next_available`` = **代码里实际最大编号 + 1**。

    ⚠️ 不硬编码任何数字：任务书写 51 也好、写 45 也好，
    都以 ``mutation_check.py`` 里的实际编号为准。
    （这正是上一轮两次撞号的教训。）
    """
    real_next = current_max_in_code() + 1
    with _file_lock():
        if path.exists() and not force:
            d = load_registry(path)
            if d["next_available"] != real_next:
                raise RegistryError(
                    f"registry 里写的是 next_available={d['next_available']}，"
                    f"但代码里实际最大编号 +1 = {real_next}。"
                    "两者不一致说明**有编号被写进了代码却没登记**。"
                    f"确认后加 --force 校正。")
            return d
        old = None
        if path.exists():
            old = load_registry(path)
        d = {
            "next_available": real_next,
            "history": (old or {}).get("history", []),
            "note": ("变异测试编号的权威来源。新增变异体前用 "
                     "`python scripts/claim_mutation_ids.py --claim N --by <工作线>` 领取；"
                     "next_available 的初值由 `--init` 从 mutation_check.py 实读推导，"
                     "**不采用任何文档里预先写死的数字**。")
        }
        if old is None:
            d["history"].append({
                "ids": [], "claimed_by": "初始化",
                "claimed_at": datetime.now().isoformat(timespec="seconds"),
                "note": f"从 mutation_check.py 实读，最大编号 {real_next - 1} ⇒ "
                        f"next_available={real_next}",
            })
        save_registry(d, path)
        return d


def claim_next_ids(n: int, claimed_by: str, path: Path = REGISTRY) -> list[int]:
    """原子地领取 ``n`` 个连续编号。"""
    if n < 1:
        raise RegistryError(f"n 必须 ≥ 1，收到 {n}")
    if not claimed_by.strip():
        raise RegistryError("必须写 claimed_by（哪条工作线领的）——history 要可追溯")
    with _file_lock():
        d = load_registry(path)
        start = int(d["next_available"])
        ids = list(range(start, start + n))
        d["next_available"] = start + n
        d["history"].append({
            "ids": ids, "claimed_by": claimed_by,
            "claimed_at": datetime.now().isoformat(timespec="seconds"),
        })
        save_registry(d, path)
    return ids


def list_history(path: Path = REGISTRY) -> dict:
    d = load_registry(path)
    return {"next_available": d["next_available"], "history": d["history"]}


def main() -> int:
    ap = argparse.ArgumentParser(description="变异测试编号权威分配器")
    ap.add_argument("--init", action="store_true", help="初始化/校正 next_available")
    ap.add_argument("--force", action="store_true", help="与 --init 搭配：不一致时强制校正")
    ap.add_argument("--claim", type=int, default=None, help="领取 n 个编号")
    ap.add_argument("--by", default="", help="领取者标识（工作线名）")
    ap.add_argument("--list", action="store_true", help="打印编号账本")
    args = ap.parse_args()

    banner("变异测试编号分配器")
    try:
        if args.init:
            d = init_registry(force=args.force)
            print(f"  ✅ next_available = {d['next_available']}"
                  f"（代码里实际最大编号 {current_max_in_code()}）")
            return 0
        if args.claim is not None:
            ids = claim_next_ids(args.claim, args.by)
            print(f"  ✅ 领到编号：{ids}")
            print(f"     下一批从 M{ids[-1] + 1} 开始")
            print(f"     领取者：{args.by or '（未写）'}")
            return 0
        if args.list:
            d = list_history()
            print(f"  next_available = {d['next_available']}")
            for h in d["history"]:
                rng = (f"M{h['ids'][0]}~M{h['ids'][-1]}" if h["ids"] else "（无编号）")
                print(f"    {h['claimed_at']}  {rng:<14} {h['claimed_by']}")
            return 0
        args.print_help()
        return 0
    except RegistryError as e:
        print(f"  ❌ {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
