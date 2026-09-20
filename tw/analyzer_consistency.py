"""一致性审计：把「点估计方向变了」和「结论真的变了」分开。

为什么需要这个模块
------------------
本项目一路在纠正的正是「**点估计冒充结论**」这个错误。
但在「回填标准配置」那一轮，它**又犯了一次**：
报告里写了两条「历史结论被推翻」——

    · 「+基本面派改善最大」→ 变成「比基线还差」
    · 「J2 优于 J3」→ 反转成「J3 优于 J2」

这两条都只依据**点估计的方向变化**就下了「推翻」的判断，
**没有做** EL.2 已经建立的那套「区间重叠度量」检验。
而做一遍就会发现：它们的区间重叠是 **100%**（一个完全包住另一个），
连「方向变化」都不该被当成结论。

⇒ 第十条纪律：**任何「此前 A 更好 → 现在 B 更好」这类方向反转的陈述，
   必须用区间重叠度量正式检验**；点估计换了方向只能说「新读数如此」。

三个函数（分工明确，别混用）
---------------------------
· ``ci_overlap_length``   —— **绝对**重叠长度（EL.2 报的 0.668 就是这个）
· ``ci_overlap_fraction`` —— 重叠长度 / 较窄区间宽度（EL.2 报的 99.8% 是这个）
· ``pairwise_verdict``    —— 三选一判定 + 「还需要多少样本」外推

⚠️ 前提核对：任务书 §2.5 要求「用 EL.2 已知的『taker vs both 重叠 0.668』
做回归测试，验证函数复现这个已知值」——但 0.668 是**绝对长度**，
而 §2.2 定义的 ``ci_overlap_fraction`` 返回的是**比例**（那组数据的比例是 0.998）。
两者不是同一个量。本模块把**两个都提供**，测试里用同一组真实数字分别锚定，
这样任务书那句「复现 0.668」和「fraction」的定义就都满足了。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

#: 重叠比例超过这个值 ⇒ 判「依然无法判定」。
#:
#: ⚠️ **这是一个约定，不是推导出来的**——必须写明，否则读者会以为它有理论依据。
#: 选 0.3 的理由：
#:   ① 参照 EL.2 的已被判定为「无法判定」的先例（重叠 99.8%）；
#:   ② 它的反面意义是「较窄区间至少要有 **70%** 是独有的」才敢谈方向差别——
#:      这是保守的：即便 70% 独有，也还要看两个区间中心离得多远。
#:   ③ 真正决定「能不能分辨」的是**中心距与半宽和之比**，
#:      所以本模块在任何判定里都**同时给出 required_n 外推**，
#:      不让读者只看一个「无法判定」的标签。
DEFAULT_OVERLAP_THRESHOLD = 0.30

#: 「显著」的参照锚点：重叠比例低于这个值才算得上干净分离。
#: 任务书 §2.4 要求给出一个反面参照（<20% 可作「显著」的锚点）。
SIGNIFICANT_MAX_OVERLAP = 0.20


# ======================================================================
def ci_overlap_length(ci_a: tuple[float, float],
                      ci_b: tuple[float, float]) -> float:
    """两个区间的**绝对**重叠长度（不相交时为 0）。

    EL.2 报的「taker vs both 重叠 0.668」就是这个量。
    """
    lo = max(float(ci_a[0]), float(ci_b[0]))
    hi = min(float(ci_a[1]), float(ci_b[1]))
    return max(0.0, hi - lo)


def ci_overlap_fraction(ci_a: tuple[float, float],
                        ci_b: tuple[float, float]) -> float:
    """重叠长度 / **较窄那个区间**的宽度。

    返回值域 ``[0, 1]``：0 = 完全不重叠；1 = 较窄区间被完全包含。
    较窄区间宽度为 0 时返回 ``nan``（退化输入，不给假装有意义的数）。

    ⚠️ 分母是**较窄的那个**，不是两者之和、也不是较宽的那个。
    用「之和」做分母会把比例系统性压低（把「100% 包含」算成 50%），
    于是本该判「无法判定」的比较会看起来像「显著」——
    方向恰好是危险的（变异体 M55 钉的就是这里）。
    """
    w_a = abs(float(ci_a[1]) - float(ci_a[0]))
    w_b = abs(float(ci_b[1]) - float(ci_b[0]))
    narrower = min(w_a, w_b)
    if narrower <= 0:
        return float("nan")
    return ci_overlap_length(ci_a, ci_b) / narrower


# ======================================================================
def required_n_to_separate(ci_a_width: float, ci_b_width: float,
                           point_diff: float, current_n: float, *,
                           target_overlap: float = 0.0) -> float:
    """按「区间宽度 ∝ 1/√n」外推：要让重叠降到 ``target_overlap`` 需要多少样本。

    推导（两个区间中心距固定为 ``|point_diff|``，半宽各为 ``w/2``）：
        重叠 = (w_a(n) + w_b(n)) / 2 − |Δ|
        令 w(n) = w₀·√(n₀/n)，要求 重叠 = target_overlap：
        (w_a₀ + w_b₀)/2 · √(n₀/n) = |Δ| + target_overlap
        ⇒ n = n₀ · ( (w_a₀ + w_b₀) / (2·(|Δ| + target_overlap)) )²

    ``target_overlap=0`` 即「两区间恰好分开」（EL.2 用的就是这个口径）。

    ⚠️ 分母用的是**两个区间宽度之和**。只用较窄的那个会**低估**所需样本量
    （本项目实测：taker vs both 这组，用较窄宽度算得约 1000，用两者之和算得约 1250）。
    低估的方向是危险的——它会让「不值得做」看起来像「值得做」。
    """
    if current_n <= 0:
        return float("nan")
    denom = 2.0 * (abs(point_diff) + max(0.0, target_overlap))
    if denom <= 0:
        return float("inf")
    ratio = (ci_a_width + ci_b_width) / denom
    return current_n * ratio ** 2


# ======================================================================
@dataclass
class PairwiseVerdict:
    name_a: str
    name_b: str
    point_a: float
    point_b: float
    ci_a: tuple[float, float]
    ci_b: tuple[float, float]
    overlap_length: float
    overlap_fraction: float
    point_diff: float
    verdict: str
    direction: str | None = None
    required_n: float | None = None
    reason: str = ""
    threshold: float = DEFAULT_OVERLAP_THRESHOLD
    extras: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        d = {
            "name_a": self.name_a, "name_b": self.name_b,
            "point_a": self.point_a, "point_b": self.point_b,
            "ci_a": list(self.ci_a), "ci_b": list(self.ci_b),
            "overlap_length": self.overlap_length,
            "overlap_fraction": self.overlap_fraction,
            "point_diff": self.point_diff,
            "verdict": self.verdict, "direction": self.direction,
            "required_n": self.required_n, "reason": self.reason,
            "threshold": self.threshold,
        }
        d.update(self.extras)
        return d


def pairwise_verdict(name_a: str, ci_a: tuple[float, float], point_a: float,
                     name_b: str, ci_b: tuple[float, float], point_b: float,
                     current_n: float, *,
                     alpha_overlap_threshold: float = DEFAULT_OVERLAP_THRESHOLD,
                     ) -> PairwiseVerdict:
    """三选一判定：``显著`` / ``依然无法判定`` / ``边缘``。

    规则（阈值与理由见模块文档）：
      · 重叠比例 ≤ ``SIGNIFICANT_MAX_OVERLAP``（0.20）⇒ **显著**，方向由点估计决定
      · 重叠比例 >  ``alpha_overlap_threshold``（0.30）⇒ **依然无法判定**，附 required_n
      · 其余 ⇒ **边缘**（如实报重叠比例，不强行归类）

    两个阈值之间（0.20~0.30）刻意留成「边缘」而不是硬塞进某一类：
    把不确定的东西强行归类，正是这个项目一路在纠的毛病。
    """
    frac = ci_overlap_fraction(ci_a, ci_b)
    length = ci_overlap_length(ci_a, ci_b)
    diff = float(point_b) - float(point_a)
    w_a = abs(float(ci_a[1]) - float(ci_a[0]))
    w_b = abs(float(ci_b[1]) - float(ci_b[0]))

    v = PairwiseVerdict(
        name_a=name_a, name_b=name_b, point_a=float(point_a),
        point_b=float(point_b), ci_a=(float(ci_a[0]), float(ci_a[1])),
        ci_b=(float(ci_b[0]), float(ci_b[1])),
        overlap_length=length, overlap_fraction=frac,
        point_diff=diff, verdict="", threshold=alpha_overlap_threshold,
    )

    if not (frac == frac):                       # nan
        v.verdict = "无法判定（输入退化）"
        v.reason = "较窄区间宽度为 0，比例没有意义"
        return v

    if frac <= SIGNIFICANT_MAX_OVERLAP:
        v.verdict = "显著"
        v.direction = (f"{name_b} 更高" if diff > 0 else f"{name_a} 更高")
        v.reason = (f"重叠只占较窄区间的 {frac * 100:.1f}%"
                    f"（≤ {SIGNIFICANT_MAX_OVERLAP * 100:.0f}% 才算干净分离）")
        return v

    if frac > alpha_overlap_threshold:
        v.verdict = "依然无法判定"
        v.required_n = required_n_to_separate(w_a, w_b, diff, current_n)
        v.reason = (f"重叠占较窄区间的 {frac * 100:.1f}%"
                    f"（> {alpha_overlap_threshold * 100:.0f}% 判为无法判定）；"
                    f"要让两区间分开约需 {v.required_n:.0f} 个样本"
                    f"（现用 {current_n:.0f} 个）")
        return v

    v.verdict = "边缘"
    v.required_n = required_n_to_separate(w_a, w_b, diff, current_n)
    v.reason = (f"重叠占较窄区间的 {frac * 100:.1f}%，"
                f"落在 {SIGNIFICANT_MAX_OVERLAP * 100:.0f}%~"
                f"{alpha_overlap_threshold * 100:.0f}% 的中间地带——"
                "不强行归类，需人工判断")
    return v


# ======================================================================
def direction_verdict(k_point: float, ci: tuple[float, float]) -> str:
    """**单个**家族的方向性判定（工作线P 的新验收标准）。

    与 ``pairwise_verdict`` 的区别必须说清楚：
    这是**一个区间自己**与世界的关系（是否排除 0.5 / 1.0），
    不是**两个家族之间**的比较。前者不受一致性审计影响，后者才受影响。
    """
    lo, hi = float(ci[0]), float(ci[1])
    excludes_0_5 = not (lo <= 0.5 <= hi)
    excludes_1_0 = not (lo <= 1.0 <= hi)
    if excludes_0_5:
        return ("sqrt_law_excluded_偏线性(k>0.5)" if k_point > 0.5
                else "sqrt_law_excluded_偏超凹(k<0.5)")
    if excludes_1_0:
        return "sqrt_law_consistent"
    return "undetermined"


def format_table(verdicts: list[PairwiseVerdict]) -> list[str]:
    """把判定列表渲成 markdown 表（供报告直接引用）。"""
    out = ["| 比较 | k_A | k_B | 重叠比例 | 判定 | 分开所需样本 |",
           "|---|---|---|---|---|---|"]
    for v in verdicts:
        need = ("—" if v.required_n is None or v.required_n != v.required_n
                else f"{v.required_n:.0f}")
        out.append(f"| {v.name_a} vs {v.name_b} | {v.point_a:.3f} | "
                   f"{v.point_b:.3f} | **{v.overlap_fraction * 100:.1f}%** | "
                   f"**{v.verdict}** | {need} |")
    return out
