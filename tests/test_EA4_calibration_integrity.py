"""EA.4 标定完整性校验的回归测试（任务书 §2.1）。

为什么这份测试必须存在
----------------------
前置校验本身是**整轮工作的门**：它一旦失效（比如校验器恒返回 clean=True），
后面 D/E/F 全部建立在一个没被验证的起点上——**而失败是静默的**。
这与项目里反复出现的形态完全一致（自检自己变脆/自己误报）。

所以这里测三件事：
  ① **同源**：标定侧与实验侧的配置确实从**同一份 spec**派生（不是两份手抄）；
  ② **能发现**：人为制造的不一致必须被报出来（反面参照——证明校验器有分辨力）；
  ③ **指纹有效**：物化指纹对参数敏感，不是恒等函数。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from scripts.verify_EA4_calibration import (  # noqa: E402
    _effective_experiment_side,
    _market_fingerprint,
    diff_configs,
    verify_EA4_lambda_calibration,
)

_SCRIPTS = str(ROOT / "scripts")
while _SCRIPTS in sys.path:
    sys.path.remove(_SCRIPTS)


class TestEA4ConfigSingleSource(unittest.TestCase):
    """标定侧与实验侧必须从**同一份 spec**派生。"""

    def _specs(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "_wsA_for_verify", str(ROOT / "scripts" / "run_workstream_A.py"))
        mod = importlib.util.module_from_spec(spec)
        sys.path.insert(0, _SCRIPTS)
        try:
            spec.loader.exec_module(mod)
            return mod
        finally:
            while _SCRIPTS in sys.path:
                sys.path.remove(_SCRIPTS)

    def test_两侧从同一spec派生(self) -> None:
        mod = self._specs()
        market = mod.ea4_market_spec()
        win = mod.ea4_windows()["experiment"]
        cal = mod.ea4_calibration_kwargs(mod.SEEDS, win)
        exp = mod.ea4_experiment_kwargs(win)
        # 两侧的"市场部分"必须是同一份值（同一个对象或逐字段相等）
        self.assertEqual(cal["n_agents"], exp["n_agents"])
        self.assertIs(cal["mix"], exp["mix"],
                      "两侧的 mix 不是同一个对象——说明有人复制了一份配置")
        self.assertIs(cal["sim_kw"], exp["sim_kw"],
                      "两侧的 sim_kw 不是同一个对象——说明有人复制了一份配置")
        self.assertEqual(cal["n_agents"], market["n_agents"])

    def test_标定侧不夹带实验侧没有的字段(self) -> None:
        """标定多传一个参数、实验侧没传，就是"看起来同步、实际不同步"的一种。"""
        mod = self._specs()
        win = mod.ea4_windows()["experiment"]
        cal = mod.ea4_calibration_kwargs(mod.SEEDS, win)
        exp = mod.ea4_experiment_kwargs(win)
        # 标定侧独有的字段只能是"标定才需要的"：seeds / warmup / n_ticks
        self.assertEqual(
            set(cal) - set(exp), {"seeds", "warmup", "n_ticks"},
            f"标定侧出现了意料之外的字段：{set(cal) - set(exp)}")

    def test_实验侧实际生效值与spec一致(self) -> None:
        """⚠️ 最要紧的一条：``make_market`` 把 ``N_AGENTS``/``MM_KW``
        **写死在函数体里**，所以 spec 改了、实验侧不一定跟着改。
        这条测试直接核对"实验侧真正会用的值"。
        """
        mod = self._specs()
        market = mod.ea4_market_spec()
        eff = _effective_experiment_side()
        self.assertEqual(eff["n_agents"], market["n_agents"],
                         "spec 里的主体数与 make_market 实际用的不一致")
        self.assertEqual(
            diff_configs({"mix": market["mix"], "sim_kw": market["sim_kw"]},
                         {"mix": eff["mix"], "sim_kw": eff["sim_kw"]}),
            [], "spec 的做市配置与 make_market 实际用的不一致")


class TestCheckerHasResolution(unittest.TestCase):
    """反面参照：**校验器必须抓得住人为制造的不一致**。

    没有这一组，上面那些"通过"什么都不说明——一个恒返回 [] 的 diff 也能通过。
    """

    def test_能抓到数值差异(self) -> None:
        d = diff_configs({"n_agents": 300}, {"n_agents": 301})
        self.assertEqual(len(d), 1)
        self.assertIn("n_agents", d[0])

    def test_能抓到嵌套字典里的差异(self) -> None:
        d = diff_configs({"sim_kw": {"mm_quote_qty": 2.0}},
                         {"sim_kw": {"mm_quote_qty": 2.5}})
        self.assertEqual(len(d), 1, f"嵌套差异没被抓到：{d}")
        self.assertIn("mm_quote_qty", d[0])

    def test_能抓到缺失字段(self) -> None:
        d = diff_configs({"a": 1}, {"a": 1, "b": 2})
        self.assertEqual(len(d), 1)
        self.assertIn("缺失于标定侧", d[0])

    def test_一致时不报任何差异(self) -> None:
        self.assertEqual(diff_configs({"a": 1, "b": {"c": 2.0}},
                                      {"a": 1, "b": {"c": 2.0}}), [])

    def test_浮点容差不报假差异(self) -> None:
        self.assertEqual(diff_configs({"x": 1e-15}, {"x": 0.0}), [])


class TestFingerprintIsSensitive(unittest.TestCase):
    """物化指纹必须**对参数敏感**——否则"指纹一致"这句话没有信息量。"""

    def test_不同主体构成给出不同指纹(self) -> None:
        a = _market_fingerprint({"zero_intel": 1.0}, 30, {})
        b = _market_fingerprint({"chartist": 1.0}, 30, {})
        self.assertNotEqual(a["kinds"], b["kinds"],
                            "两种完全不同的主体构成给出同一指纹——指纹是恒等的")

    def test_同参数同指纹(self) -> None:
        mix = {"zero_intel": 0.5, "chartist": 0.5}
        self.assertEqual(_market_fingerprint(mix, 40, {}),
                         _market_fingerprint(mix, 40, {}))

    def test_主体数真的生效(self) -> None:
        mix = {"zero_intel": 1.0}
        self.assertEqual(_market_fingerprint(mix, 20, {})["n_agents"], 20)
        self.assertEqual(_market_fingerprint(mix, 60, {})["n_agents"], 60)


class TestVerifyResultShape(unittest.TestCase):
    """验收标准（任务书 §2.3）：``mismatch_fields`` 必须**公开可读**。"""

    def test_返回结构完整(self) -> None:
        v = verify_EA4_lambda_calibration()
        for key in ("clean", "mismatch_fields", "clean_market_config",
                    "clean_window", "fingerprint_match",
                    "experiment_window", "legacy_calibration_window"):
            self.assertIn(key, v, f"校验结果缺字段 {key}")
        self.assertIsInstance(v["mismatch_fields"], list)

    def test_窗口差异的描述必须说清两侧取值(self) -> None:
        """不合格的东西必须**说得出哪里不合格**，不能只给一个 False。"""
        v = verify_EA4_lambda_calibration()
        if not v["clean_window"]:
            self.assertTrue(v["mismatch_fields"], "窗口不干净却没给出差异说明")
            txt = " ".join(v["mismatch_fields"])
            self.assertIn("window", txt)
            self.assertIn(str(v["experiment_window"][0]), txt)
            self.assertIn(str(v["legacy_calibration_window"][0]), txt)


if __name__ == "__main__":
    unittest.main(verbosity=2)
