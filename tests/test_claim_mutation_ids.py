"""变异测试编号分配器的回归测试（工作线N.2）。

为什么这份测试重要
------------------
编号分配器是**跨轮次的基础设施**：它一旦发重号，
两个工作线会各自以为拿到了同一批编号，
而**直到合并时才暴露**（那时两边的测试文件都已经写完了）。

所以这里测三件事：
  ① 编号必须连续且不重复（含**并发**领取）；
  ② history 必须可追溯（哪条工作线、什么时候领的）；
  ③ ``--init`` 必须从**代码实读**推导 next_available，而不是信任文字里的数字。

⚠️ 全部用临时 registry 路径，**绝不碰** ``docs/mutation_registry.json``。
"""

from __future__ import annotations

import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.append(str(ROOT / "scripts"))  # 追加：插到前面会遮蔽同名包（scripts/gui.py vs gui/）

from scripts.claim_mutation_ids import (  # noqa: E402
    RegistryError,
    claim_next_ids,
    current_max_in_code,
    init_registry,
    list_history,
    load_registry,
)

_SCRIPTS = str(ROOT / "scripts")
while _SCRIPTS in sys.path:
    sys.path.remove(_SCRIPTS)

#: 真实的 registry 路径 —— 测试**必须**避开它
REAL_REGISTRY = ROOT / "docs" / "mutation_registry.json"


class _Sandbox:
    def __init__(self) -> None:
        self._td = tempfile.TemporaryDirectory(prefix="reg_test_")
        self.path = Path(self._td.name) / "registry.json"

    def cleanup(self) -> None:
        self._td.cleanup()


class TestInitDerivesFromCode(unittest.TestCase):
    """③ next_available 必须从代码实读推导。"""

    def setUp(self) -> None:
        self.sb = _Sandbox()

    def tearDown(self) -> None:
        self.sb.cleanup()

    def test_初始化等于代码最大编号加一(self) -> None:
        d = init_registry(self.sb.path)
        self.assertEqual(d["next_available"], current_max_in_code() + 1,
                         "next_available 不是从代码实读推导的")

    def test_与实际代码一致(self) -> None:
        """跨轮次的真实约束：不许把编号写死在文档里。"""
        d = init_registry(self.sb.path)
        self.assertGreaterEqual(d["next_available"], 51,
                                "本项目的编号已经用到 M50 以上了")
        self.assertEqual(d["next_available"] - 1, current_max_in_code())

    def test_重复初始化不报错且不改变值(self) -> None:
        d1 = init_registry(self.sb.path)
        d2 = init_registry(self.sb.path)
        self.assertEqual(d1["next_available"], d2["next_available"])

    def test_不一致时报错_force可校正(self) -> None:
        init_registry(self.sb.path)
        # 人为把 next_available 改错
        raw = json.loads(self.sb.path.read_text(encoding="utf-8"))
        raw["next_available"] = 999
        self.sb.path.write_text(json.dumps(raw), encoding="utf-8")
        with self.assertRaises(RegistryError):
            init_registry(self.sb.path)
        d = init_registry(self.sb.path, force=True)
        self.assertEqual(d["next_available"], current_max_in_code() + 1)


class TestClaimIsConsecutiveAndAtomic(unittest.TestCase):
    """① 连续、不重复，并发也安全。"""

    def setUp(self) -> None:
        self.sb = _Sandbox()
        init_registry(self.sb.path)

    def tearDown(self) -> None:
        self.sb.cleanup()

    def test_领取的编号连续(self) -> None:
        start = load_registry(self.sb.path)["next_available"]
        ids = claim_next_ids(3, "测试", self.sb.path)
        self.assertEqual(ids, [start, start + 1, start + 2])

    def test_两次领取不重叠(self) -> None:
        a = claim_next_ids(2, "工作线A", self.sb.path)
        b = claim_next_ids(2, "工作线B", self.sb.path)
        self.assertEqual(set(a) & set(b), set(), "两次领取出现了重复编号")
        self.assertEqual(b[0], a[-1] + 1)

    def test_并发领取不发重号(self) -> None:
        """⭐ 这是"原子"这个承诺的实测。读-改-写不加锁时这里会撞号。"""
        results: list[list[int]] = []
        errors: list[str] = []
        lock = threading.Lock()

        def work() -> None:
            try:
                r = claim_next_ids(2, "并发测试", self.sb.path)
                with lock:
                    results.append(r)
            except Exception as e:  # noqa: BLE001
                with lock:
                    errors.append(f"{type(e).__name__}: {e}")

        threads = [threading.Thread(target=work) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [], f"有线程领取失败：{errors[:3]}")

        flat = [i for r in results for i in r]
        self.assertEqual(len(flat), len(set(flat)),
                         f"并发领取出现了重复编号：{sorted(flat)}")
        # 16 个编号必须是连续的一段
        self.assertEqual(sorted(flat), list(range(min(flat), min(flat) + 16)))

    def test_非法参数报错(self) -> None:
        with self.assertRaises(RegistryError):
            claim_next_ids(0, "x", self.sb.path)
        with self.assertRaises(RegistryError):
            claim_next_ids(-1, "x", self.sb.path)

    def test_必须写领取者(self) -> None:
        """history 要可追溯 ⇒ 领取者不能空。"""
        with self.assertRaises(RegistryError):
            claim_next_ids(1, "   ", self.sb.path)


class TestHistoryIsAuditable(unittest.TestCase):
    """② history 必须能看出"哪条工作线、什么时候领的"。"""

    def setUp(self) -> None:
        self.sb = _Sandbox()
        init_registry(self.sb.path)

    def tearDown(self) -> None:
        self.sb.cleanup()

    def test_history记录编号范围与领取者(self) -> None:
        ids = claim_next_ids(2, "工作线X-某功能", self.sb.path)
        h = list_history(self.sb.path)
        last = h["history"][-1]
        self.assertEqual(last["ids"], ids)
        self.assertEqual(last["claimed_by"], "工作线X-某功能")
        self.assertIn("claimed_at", last)
        self.assertTrue(last["claimed_at"])

    def test_多批次可追溯(self) -> None:
        claim_next_ids(1, "N", self.sb.path)
        claim_next_ids(2, "L", self.sb.path)
        claim_next_ids(1, "M", self.sb.path)
        h = list_history(self.sb.path)["history"]
        claimers = [x["claimed_by"] for x in h if x["claimed_by"] != "初始化"]
        self.assertEqual(claimers, ["N", "L", "M"])

    def test_初始化也留痕(self) -> None:
        h = list_history(self.sb.path)["history"]
        self.assertTrue(any(x["claimed_by"] == "初始化" for x in h),
                        "初始化没有留痕——之后就说不清 next_available 是怎么来的")


class TestDoesNotPolluteRealRegistry(unittest.TestCase):
    """测试**绝不能**改到真实账本（否则会把项目编号搞乱）。"""

    def test_真实账本在这次测试前后一致(self) -> None:
        before = (REAL_REGISTRY.read_text(encoding="utf-8")
                  if REAL_REGISTRY.exists() else None)
        sb = _Sandbox()
        try:
            init_registry(sb.path)
            claim_next_ids(3, "污染测试", sb.path)
        finally:
            sb.cleanup()
        after = (REAL_REGISTRY.read_text(encoding="utf-8")
                 if REAL_REGISTRY.exists() else None)
        self.assertEqual(before, after, "测试污染了真实的编号账本！")

    def test_真实账本存在且结构正确(self) -> None:
        if not REAL_REGISTRY.exists():
            # ⚠️ 兜底：mutation 沙箱现在会复制 docs/（见 prepare_sandbox），
            #    但换个环境仍可能缺——缺文件时**跳过**而不是断言存在，
            #    否则沙箱基线会红，而基线一红所有变异体都会"变红"（假证据）。
            self.skipTest(f"账本不在：{REAL_REGISTRY}")
        d = json.loads(REAL_REGISTRY.read_text(encoding="utf-8"))
        self.assertIn("next_available", d)
        self.assertIn("history", d)
        self.assertIsInstance(d["history"], list)


if __name__ == "__main__":
    unittest.main(verbosity=2)
