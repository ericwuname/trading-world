"""变异验证：证明这套测试**确实能失败**。

为什么需要这一步
----------------
一个全绿的测试套件说明不了任何事。它可能绿是因为实现正确，
也可能绿是因为断言写得太松、根本没在检查东西。两者从"全绿"这个结果上无法区分。

做法（与工业界 mutation testing 同源，但更简单直接）
--------------------------------------------------
1. 把 ``tw/`` 与 ``tests/`` 复制到 ``out/_mutation/``
2. 先跑一遍确认**原始代码全绿**（基线）
3. 然后逐个注入"我们最怕出现的实现 bug"（见 MUTATIONS），每次只改动一处源码
4. 每个变异体都必须让测试套件**变红**；若仍全绿，说明这块没被测到 → 记为漏洞

覆盖的 bug 类型全部来自真实撮合系统事故：
    M1 成交价取主动方报价      —— 最常见也最贵的一类错误（等于给每笔单白送价差）
    M2 破坏价格优先            —— 按插入顺序成交
    M3 破坏时间优先（LIFO）    —— 同价后挂的先成交
    M4 取消 tick 网格          —— 用浮点做价位 key，同档碎片化
    M5 关闭自成交防护          —— 自己和自己成交，制造假成交量
    M6 完全成交后不摘单        —— 零剩余量挂单残留，簿子越跑越脏
    M7 成交后剩余不挂簿        —— 流动性凭空消失
    …（M8~M15 见 MUTATIONS 列表，全部为一期撮合/评估层）

二期新增（阶段5 永续合约 / 资金费率层，M16~M22）：
    M16 费率付款方向反转        —— 多头付钱变收钱
    M17 结算漏乘 mark_price     —— 付款量级失去价格尺度
    M18 溢价通道方向反转        —— 永续贵了反而空头付钱
    M19 结算触发 > 而非 >=      —— 首次结算时点后移
    M20 拥挤度用退化的存量占比   —— 用错信号源，费率变常数偏置
    M21 结算绕过唯一记账路径     —— 直接改 cash，破坏全系统守恒
    M22 结算后不修预留          —— 可用现金变负，市场静默冻结

二期新增（阶段6 长记忆订单流 / 元订单拆分，M23~M26）：
    M23 Pareto 采样漏掉幂次项    —— 重尾退化成有界分布，长记忆的来源被掐断
    M24 子订单方向翻转          —— 长记忆被直接抹掉
    M25 自适应流动性允许负 shift —— 流动性提供者往对手方向让价
    M26 剩余量扣减方向写反       —— 元订单永不终止，重尾被抹成常数
    （编号相对指导书 §3.6 后移 3 位：M20~M23 已被阶段5 占用，
      覆盖的 bug 类型与指导书一一对应。）

二期新增（阶段7 反身性与学习，M27~M29）：
    M27 Q 值学习方向乘反        —— 学出来的"最优"窗口是最差的那个
    M28 已结算的评估又被放回队列 —— 同一条评估被反复结算
    M29 反身性扫描对种子做加工   —— 同种子配对被悄悄破坏

二期新增（阶段8 到达过程 / 阶段9 多资产，M30~M33）：
    M30 放行 α ≥ β（去掉稳定性校验）—— 强度过程发散
    M31 intensity_at 只用最后一个事件 —— 聚集性消失，退化成泊松
    M32 公共冲击重复应用到 0 号资产   —— 该资产波动异常、相关性被破坏
    M33 配对交易两条腿标签搞反       —— 组合方向与 z-score 逻辑相反
    ⚠️ 这四条是**补上来的**：本套件编号相对指导书后移了（阶段5 占了 7 条），
       于是指导书里属于阶段8/9 的四条一直没实现，而报告只写"全绿"——
       **一个只统计自己做了什么的数字，看不出自己漏了什么。**
       指导书终局要求写的是「一期 15 个 + 二期新增 M16-M30 共 15 个」。

三线深挖轮新增（M34~M40）：
    M34 离散分支比写成 n·β           —— 实测事件率偏离 λ̄ 最高 40%
    M35 突发判定方向反了             —— 把低强度标成突发
    M36 元订单执行期没跳过基础决策   —— 同一 tick 出两笔，活跃度翻倍
    M37 基本面派子单方向被重掷       —— 不锚定启动时的 premium 方向
    M38 配对反事实两条路径不同种子   —— 配对失效，降噪消失
    M39 修复前那一臂 short_headroom 没传 0 —— 两组相同，偏差算成 0
    M40 写回合并计划用 "".join        —— 整节被拼成一行，markdown 表格彻底失效
    M41 --only 点名收尾步骤被静默忽略 —— 用户说了，程序没听见，还不吱声
    M42 收尾打印用未定义的裸 skip     —— 全部步骤跑完、对账通过，最后一句崩
    M43 自检豁免只认 Assign 不认 AnnAssign —— 自检对自己误报（假阳性）
    ⚠️ M40 是本轮真实踩到的 bug 的镜像：产品代码里确实写成过 ``"".join``，
       症状是**脚本退出码 0、自检也不看渲染**，所以「跑成功了」与
       「产物能读」之间掉进去了一次。回归测试在
       ``tests/test_workstream_scripts.py::TestPlanDocIsRenderableMarkdown``。
    ⚠️ M41 也是真实踩到的：``--only mutation --with-mutation --list`` 打印
       「共 0 步」。与早先的 ``--skip 8,9,11`` 静默不跳同一形态——
       本套件把「静默忽略用户意图」当成一类**独立的**事故来钉。

用法::

    python scripts/mutation_check.py                      # 全部
    python scripts/mutation_check.py --only M30,M31       # 只跑指定几条
        （新增变异体后**必须**单独跑一次，确认它真的能变红——
          一个永远"变红"的变异体是假的：可能根本没注入进去，
          或者锚点匹配在注释上。--only 让这件事只要两分钟。）
    python scripts/mutation_check.py --workers 8          # 并行度
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / "out" / "_mutation"
PY = sys.executable

# 每项: (编号, 说明, [(相对路径, 原文, 替换)] )
MUTATIONS: list[tuple[str, str, list[tuple[str, str, str]]]] = [
    (
        "M1",
        "成交价取主动方报价（而非被动方）",
        [
            (
                "tw/engine.py",
                '            price = maker_price  # ← 被动方价格优先（蓝图 §3.3 第 1 条）',
                '            price = taker.price if taker.order_type == "limit" else maker_price',
            )
        ],
    ),
    (
        "M2",
        "破坏价格优先：按插入顺序成交",
        [
            (
                "tw/book.py",
                """        levels, ticks = self._side_refs(side)
        if side == "buy":
            candidates = reversed(ticks)
        else:
            candidates = iter(ticks)
        for t in candidates:
            level = levels[t]
            for o in level:
                if exclude_agent is None or o.agent_id != exclude_agent:
                    return o, t
        return None""",
                """        for _sid, (s, t, o) in self._index.items():
            if s != side:
                continue
            if exclude_agent is None or o.agent_id != exclude_agent:
                return o, t
        return None""",
            )
        ],
    ),
    (
        "M3",
        "破坏时间优先：同价后挂先成交（LIFO）",
        [
            (
                "tw/book.py",
                """            for o in level:
                if exclude_agent is None or o.agent_id != exclude_agent:
                    return o, t""",
                """            for o in reversed(level):
                if exclude_agent is None or o.agent_id != exclude_agent:
                    return o, t""",
            )
        ],
    ),
    (
        "M4",
        "取消 tick 网格：直接用浮点做价位 key",
        [
            (
                "tw/book.py",
                """    def to_tick(self, price: float) -> int:
        return int(round(price / self.tick_size))

    def price_of(self, tick: int) -> float:
        return tick * self.tick_size""",
                """    def to_tick(self, price: float) -> int:
        return price

    def price_of(self, tick: int) -> float:
        return tick""",
            )
        ],
    ),
    (
        "M5",
        "关闭自成交防护",
        [
            (
                "tw/book.py",
                """                if exclude_agent is None or o.agent_id != exclude_agent:
                    return o, t
        return None""",
                """                return o, t
        return None""",
            )
        ],
    ),
    (
        "M6",
        "完全成交后不摘除挂单",
        [
            (
                "tw/engine.py",
                """            taker.remaining -= qty
            maker.remaining -= qty
            if maker.remaining <= EPS:
                book.remove(maker)""",
                """            taker.remaining -= qty
            maker.remaining -= qty""",
            )
        ],
    ),
    (
        "M7",
        "成交后剩余部分不挂入订单簿",
        [
            (
                "tw/engine.py",
                """        if order.remaining > EPS and order.order_type == "limit":
            self.book.add(order)
        return trades""",
                """        return trades""",
            )
        ],
    ),
    (
        "M8",
        "被动方预留额度不随成交更新（曾导致市场静默冻结）",
        [
            (
                "tw/market.py",
                """        self._refresh_party_after_fill(buyer, trade.buy_order_id)
        self._refresh_party_after_fill(seller, trade.sell_order_id)""",
                """        _ = (buyer, seller)""",
            )
        ],
    ),
    (
        "M9",
        "成交不写入日志（价格序列正常但成交数为零）",
        [
            (
                "tw/market.py",
                """        if trades:
            self.log.trades.extend(trades)""",
                """        _ = trades""",
            )
        ],
    ),
    (
        "M10",
        "价格不对齐 tick 网格（预留与结算口径不一致）",
        [
            (
                "tw/market.py",
                """        if order.order_type == "limit":
            snapped = self.book.price_of(self.book.to_tick(order.price))
            if snapped <= 0:
                return None
            order.price = snapped""",
                """        if order.order_type == "limit":
            if order.price <= 0:
                return None""",
            )
        ],
    ),
    (
        "M11",
        "清算滑点的基准取错（用成交后中间价代替成交 VWAP）",
        [
            (
                "tw/market.py",
                """        vwap = sum(t.notional for t in swept) / qty if qty > 0 else None""",
                """        vwap = p_after if qty > 0 else None""",
            )
        ],
    ),
    (
        "M12",
        "清算滑点符号反转（把「付出代价」写成「获得改善」）",
        [
            (
                "tw/market.py",
                """            "slippage_bp": (
                (vwap / p_before - 1.0) * 1e4
                if vwap is not None and p_before and p_before > 0
                else None
            ),""",
                """            "slippage_bp": (
                (1.0 - vwap / p_before) * 1e4
                if vwap is not None and p_before and p_before > 0
                else None
            ),""",
            )
        ],
    ),
    (
        "M13",
        "markout 方向符号取反（把「接刀」读成「赚了」）",
        [
            (
                "tw/eval.py",
                """            mh = mid[j]
            if not (mh > 0):
                continue
            drift = s * (mh - m) / m * 1e4""",
                """            mh = mid[j]
            if not (mh > 0):
                continue
            drift = -s * (mh - m) / m * 1e4""",
            )
        ],
    ),
    (
        "M14",
        "PnL 分解漏掉底仓重估项（inv0 非零时恒等式破裂）",
        [
            (
                "tw/eval.py",
                """    initial_mark = float(inv0) * float(mid[-1] - base)""",
                """    _ = base
    initial_mark = 0.0""",
            )
        ],
    ),
    (
        "M15",
        "成交时不回填中间价（评估工具失去基准）",
        [
            (
                "tw/market.py",
                """        mid_now = self._state.mid
        if mid_now is not None and mid_now > 0:
            trade.mid_at_fill = float(mid_now)
        else:
            trade.mid_at_fill = self._last_mid""",
                """        _ = self._state.mid""",
            )
        ],
    ),
    # ==================================================================
    # 二期阶段5：永续合约 / 资金费率层（M16~M22）
    #
    # 这七个 bug 的共同点：**都不会让程序报错或崩溃**，只会让
    # 「费率方向」「payment 量级」「记账归属」悄悄错掉，
    # 而主指标（年化、正占比）看起来仍然合理。
    #
    # ⭐ 与指导书 §2.6 的差异（三处，都已核对过源码）
    # ------------------------------------------------------------------
    # · 指导书 M17「守恒测试失败」**不成立**：本架构里交易所虚拟账户承担对手方，
    #   付款乘不乘 mark_price，`Σ主体payment + 交易所变动` 都恒为 0，
    #   守恒测试**抓不到**。错的是量级，抓住它的是「自记与账本一致」那条断言。
    # · 指导书 M18 的预期现象「tick 间隔不均匀」**不成立**：实现用的是计数器
    #   `_since_settle`，把 `>=` 改成 `>` 只会让**首次**结算从第 8 tick 推到第 9 tick，
    #   之后的间隔仍然是 8。真正抓住它的是「首次结算时点」断言。
    #   （这正是指导书修正④「不用 tick 取模」的副产品：取模写法才会间隔不均。）
    # · 指导书 M19 在这里**不可注入**：`net_long_ratio` 因交易不改变 Σ持仓而恒为
    #   常数（实测 500 次结算全部 1.0），它根本不是费率信号，改它什么也不会发生。
    #   已改写成等价意图的 M20：把拥挤度通道换成这个退化量（= 用错信号源）。
    # 另补两条指导书没列、但按本项目纪律必须覆盖的（M21/M22）。
    # ==================================================================
    (
        "M16",
        "资金费率付款方向反（多头付钱写成多头收钱）",
        [
            (
                "tw/perpetual.py",
                "            nominal = -a.inventory * rate * mark",
                "            nominal = a.inventory * rate * mark",
            )
        ],
    ),
    (
        "M17",
        "结算时漏乘 mark_price（付款量级失去价格尺度）",
        [
            (
                "tw/perpetual.py",
                "            nominal = -a.inventory * rate * mark",
                "            nominal = -a.inventory * rate",
            )
        ],
    ),
    (
        "M18",
        "溢价通道方向反（永续贵了反而空头付钱）",
        [
            (
                "tw/perpetual.py",
                "        return float((mid - anchor) / anchor)",
                "        return float((anchor - mid) / anchor)",
            )
        ],
    ),
    (
        "M19",
        "结算触发用 > 而非 >=（首次结算时点后移一格）",
        [
            (
                "tw/perpetual.py",
                "        if self._since_settle >= self.funding_config.settle_interval_ticks:",
                "        if self._since_settle > self.funding_config.settle_interval_ticks:",
            )
        ],
    ),
    (
        "M20",
        "拥挤度通道用退化的存量占比（= 用错信号源，费率变成常数偏置）",
        [
            (
                "tw/perpetual.py",
                "        return float(self._flow_imbalance(window=self.funding_config.crowding_window))",
                "        return self._compute_net_long_ratio()",
            )
        ],
    ),
    (
        "M21",
        "费率结算绕过唯一记账路径（直接改 cash，破坏全系统守恒）",
        [
            (
                "tw/perpetual.py",
                '            self.apply_cash_delta(a, pay, reason="funding_settlement")',
                "            a.cash += pay",
            )
        ],
    ),
    (
        "M22",
        "结算后不修预留（可用现金变负 → 市场静默冻结）",
        [
            (
                "tw/perpetual.py",
                """            if a.reserved_cash > a.cash + 1e-9:
                self.reconciled_orders += self.reconcile_reservations(a)""",
                """            if False:
                pass""",
            )
        ],
    ),
    # ==================================================================
    # 二期阶段6：长记忆订单流 / 元订单拆分（M23~M26）
    #
    # ⭐ 编号相对指导书 §3.6 后移了 3 位：指导书写的是 M20~M23，
    # 但那四个编号已被阶段5 的实现在用（阶段5 需要 7 个变异体而不是 4 个，
    # 见上面的说明）。编号只是索引，**覆盖的 bug 类型与指导书一一对应**。
    #
    # 这四个 bug 的共同点：都是"机制静默失效"型——
    # 程序不报错、不崩溃，只是长记忆永远不会出现（或方向被抹掉），
    # 而结论会变成"这个机制没用"。
    # ==================================================================
    (
        "M23",
        "Pareto 采样漏掉幂次项（重尾退化成有界分布，长记忆的来源被掐断）",
        [
            (
                "tw/order_flow/meta_order.py",
                "        base = cfg.pareto_xmin * (1.0 - u) ** (-1.0 / cfg.pareto_alpha)",
                "        base = cfg.pareto_xmin * (1.0 - u)",
            )
        ],
    ),
    (
        "M24",
        "子订单方向在拆分过程中翻转（长记忆被直接抹掉）",
        [
            (
                "tw/order_flow/meta_order.py",
                """        if side == "buy":
            return self.buy_order(state, qty)
        return self.sell_order(state, qty)""",
                """        if side == "buy":
            return self.sell_order(state, qty)
        return self.buy_order(state, qty)""",
            )
        ],
    ),
    (
        "M25",
        "自适应流动性允许负的 shift（流动性提供者主动往对手方向让价）",
        [
            (
                "tw/agents/adaptive_liquidity.py",
                """        if pressure <= 0.0:
            return base_offset

        mult = min(self.max_stretch, 1.0 + self.flow_sensitivity * pressure)""",
                """        mult = min(self.max_stretch, 1.0 + self.flow_sensitivity * pressure)""",
            ),
            (
                "tw/agents/adaptive_liquidity.py",
                "        out = max(base_offset, base_offset * mult)",
                "        out = base_offset * mult",
            ),
        ],
    ),
    (
        "M26",
        "剩余量扣减方向写反（元订单永不终止，重尾被抹成常数）",
        [
            (
                "tw/order_flow/meta_order.py",
                "        st.remaining_qty -= qty           # ⚠️ 必须是**减**。写成 += 会让",
                "        st.remaining_qty += qty           # ⚠️ 必须是**减**。写成 += 会让",
            )
        ],
    ),
    # ==================================================================
    # 二期阶段7：自适应主体与反身性（M27~M29）
    #
    # 这三个 bug 的形态是"实验装置坏了，而不是被测机制坏了"——
    # 它们产出的是一条**看起来合理但归因错误**的曲线：
    #   · 方向学反 → "最优窗口"是最差的那个
    #   · 重复结算 → 某些窗口的 Q 更新次数异常偏多，学习速率被偷偷放大
    #   · 配对破掉 → 衰减曲线里出现不合理的大幅震荡，统计检验通不过
    # 指导书 §4.6 的编号是 M24~M26，这里后移 3 位（M24~M26 已被阶段6 占用）。
    # ==================================================================
    (
        "M27",
        "Q 值学习方向乘反（学到的『最优』窗口是最差的那个）",
        [
            (
                "tw/agents/adaptive_trend.py",
                "                    realized = direction * (state.mid - entry) / entry",
                "                    realized = -direction * (state.mid - entry) / entry",
            )
        ],
    ),
    (
        "M28",
        "待评估队列把已结算的条目又放回去（同一条评估被反复结算）",
        [
            (
                "tw/agents/adaptive_trend.py",
                """                    n += 1
            else:
                still.append(ev)
        self.pending_evals = still""",
                """                    n += 1
            still.append(ev)
        self.pending_evals = still""",
            )
        ],
    ),
    (
        "M29",
        "反身性扫描对种子做了加工（同种子配对被悄悄破坏）",
        [
            (
                "tw/experiments/reflexivity.py",
                """    m = Market(SimConfig(
        seed=seed, n_ticks=cfg.total_ticks(), population=pop,
        zi_p_buy=0.5,
    ))""",
                """    m = Market(SimConfig(
        seed=seed + adopter_count * 1000, n_ticks=cfg.total_ticks(),
        population=pop, zi_p_buy=0.5,
    ))""",
            )
        ],
    ),
    # ------------------------------------------------------------------
    # 阶段8 / 阶段9（指导书 §5.6 要求的 M27~M30 四条）。
    # ⚠️ 这四条是**补上来的**：本套件的编号相对指导书后移了（阶段5 占了 7 条而不是 4 条），
    #    于是指导书里属于阶段8/9 的四条一直没被实现，而报告只写"29/29 全绿"——
    #    **一个只统计自己做了什么的数字，看不出自己漏了什么。**
    #    指导书的终局要求写的是「一期15个 + 二期新增 M16-M30 共15个」。
    # ------------------------------------------------------------------
    (
        "M30",
        "Hawkes 放行 alpha >= beta（稳定性条件不再被校验，强度过程会发散）",
        [
            (
                "tw/order_flow/hawkes.py",
                """        if self.branching >= self.max_branching:
            raise ValueError(""",
                """        if False and self.branching >= self.max_branching:
            raise ValueError(""",
            )
        ],
    ),
    (
        "M31",
        "intensity_at 遗漏历史事件的衰减项（只用最后一个事件 → 退化成泊松到达）",
        [
            (
                "tw/order_flow/hawkes.py",
                """        exc = 0.0
        for ti in self.event_times:
            if ti < t:
                exc += self.alpha * np.exp(-self.beta * (t - ti))
        return self.mu + exc""",
                """        exc = 0.0
        past = [ti for ti in self.event_times if ti < t]
        if past:
            ti = past[-1]
            exc = self.alpha * np.exp(-self.beta * (t - ti))
        return self.mu + exc""",
            )
        ],
    ),
    (
        "M32",
        "common_shock 被重复应用到 0 号资产（该资产波动异常、相关性被破坏）",
        [
            (
                "tw/multi_asset.py",
                """        ret = self.betas * common + idio
        # 向初始锚点弱回归""",
                """        ret = self.betas * common + idio
        ret[0] = ret[0] + self.betas[0] * common
        # 向初始锚点弱回归""",
            )
        ],
    ),
    (
        "M33",
        "PairsTrader 的两条腿标签搞反（组合方向与 z-score 逻辑相反）",
        [
            (
                "tw/agents/pairs_trader.py",
                """        for idx, leg, qsign in ((self.i, self.legs[0], +1.0),
                                (self.j, self.legs[1], -1.0)):""",
                """        for idx, leg, qsign in ((self.i, self.legs[0], -1.0),
                                (self.j, self.legs[1], +1.0)):""",
            )
        ],
    ),
    # ------------------------------------------------------------------
    # 三线深挖（2026-09-18）：工作线A/B/C 新增的六条
    # ------------------------------------------------------------------
    (
        'M34',
        'Hawkes 用连续分支比（α=n·β 而不是 n·(e^{β·dt}−1)）——实测率偏离 λ̄ 最高 40%',
        [
            (
                'tw/order_flow/hawkes.py',
                '        return float(self.branching / self.geometric)',
                '        return float(self.branching * self.beta)',
            ),
        ],
    ),
    (
        'M35',
        'classify_burst_vs_background 把**低**强度标成突发（方向反了）',
        [
            (
                'tw/impact.py',
                '    return x > thr',
                '    return x < thr',
            ),
        ],
    ),
    (
        'M36',
        '元订单执行期没跳过基础决策（同一 tick 出两笔，活跃度翻倍）',
        [
            (
                'tw/order_flow/meta_order.py',
                '        if st.active:\n            child = self.next_child()\n            if child is None:\n                return None\n            side, qty = child\n            return self._child_order(state, side, qty)',
                '        if st.active:\n            child = self.next_child()\n            if child is None:\n                return None\n            side, qty = child\n            self._child_order(state, side, qty)\n            return super().decide(state)',
            ),
        ],
    ),
    (
        'M37',
        '基本面派子单方向被反复重掷（不锚定启动时的 premium 方向）',
        [
            (
                'scripts/run_stage6.py',
                'class MetaFundamentalist(MetaOrderMixin, Fundamentalist):\n    """基本面派 + 元订单拆分执行（代表"基于估值的大宗建仓/减仓"）。"""',
                'class MetaFundamentalist(MetaOrderMixin, Fundamentalist):\n    """基本面派 + 元订单拆分执行（代表"基于估值的大宗建仓/减仓"）。"""\n\n    def next_child(self):\n        # 变异：每个子单按 50% 概率重掷方向，而不是锚定启动时的方向。\n        # 观察量：订单符号 ACF 的"均值回归味道"被抹平。\n        r = super().next_child()\n        if r is not None and float(self.rng.random()) < 0.5:\n            r = ("sell" if r[0] == "buy" else "buy", r[1])\n        return r',
            ),
        ],
    ),
    (
        'M38',
        '配对反事实的两条路径用了不同种子（配对失效，降噪消失）',
        [
            (
                'scripts/run_workstream_C.py',
                '        m, tr, _ = build(2, cv, 4e-4, seed, pairs=(0, 1), pairs_cfg=PAIR_CFG,\n                         short_headroom=short_headroom)\n        run_market(m, tr, WARMUP)',
                '        m, tr, _ = build(2, cv, 4e-4, seed + (1 if trigger else 0), pairs=(0, 1), pairs_cfg=PAIR_CFG,\n                         short_headroom=short_headroom)\n        run_market(m, tr, WARMUP)',
            ),
        ],
    ),
    (
        'M39',
        '修复前后对比里 short_headroom=0 没生效（两组相同，偏差算成 0）',
        [
            (
                'scripts/run_workstream_C.py',
                '    broken = pairs_pnl(0.0, seeds)\n    fixed = pairs_pnl(3.0, seeds)',
                '    broken = pairs_pnl(3.0, seeds)\n    fixed = pairs_pnl(3.0, seeds)',
            ),
        ],
    ),
    (
        'M40',
        '写回合并计划时用 "".join 而不是 "\\n".join（整节被拼成一行，表格彻底失效）',
        [
            (
                'scripts/run_workstream_joint.py',
                '    PLAN.write_text(head + "\\n".join(L), encoding="utf-8")',
                '    PLAN.write_text(head + "".join(L), encoding="utf-8")',
            ),
        ],
    ),
    (
        'M41',
        '--only 点名收尾步骤（tests/mutation）时被静默忽略（算出「共 0 步」也不报）',
        [
            (
                'scripts/reproduce_all.py',
                '        elif name in only:',
                '        elif False:  # 变异：--only 点名收尾步骤时静默丢掉',
            ),
        ],
    ),
    (
        'M42',
        '收尾打印用未定义的裸 skip（--only 路径跑完全部步骤后抛 NameError，退出码 1）',
        [
            (
                'scripts/reproduce_all.py',
                '    skipped = normalize_skip(args.skip)\n    if skipped:\n'
                '        print(f"   （跳过的步骤：{sorted(skipped)} —— 未做新鲜度要求）")',
                '    if skip:\n'
                '        print(f"   （跳过的步骤：{sorted(skip)} —— 未做新鲜度要求）")',
            ),
        ],
    ),
    (
        'M43',
        '自检的豁免只认 Assign 不认带类型注解的 AnnAssign'
        '（MUTATIONS 写成 MUTATIONS: list[...] = [...]，豁免静默失效 → 自检自己误报）',
        [
            (
                'scripts/selfcheck.py',
                '        if isinstance(node, ast.Assign):\n'
                '            targets = node.targets\n'
                '        elif isinstance(node, ast.AnnAssign):\n'
                '            targets = [node.target]\n'
                '        else:\n'
                '            continue',
                '        if isinstance(node, ast.Assign):\n'
                '            targets = node.targets\n'
                '        else:\n'
                '            continue',
            ),
        ],
    ),
    # ------------------------------------------------------------------
    # 分辨力危机那一轮新增（M44~M50）。
    # ⚠️ 编号相对任务书**顺延**：任务书写的是 M45~M49，
    #    但它没看到 M40~M43 已被上一轮（渲染/--only/skip/AnnAssign）占用，
    #    而且 E 线的两条（任务书沿用旧编号 M42/M43）也已被占用。
    # ------------------------------------------------------------------
    (
        'M44',
        '滚动同步率的窗口漏掉当前 tick（区间写成 [lo, i) 而不是 [lo, i]）'
        '——同步率系统性偏低',
        [
            (
                'tw/analyzer_asymmetry.py',
                '        total = csum[i + 1] - csum[lo]',
                '        total = csum[i] - csum[lo]',
            ),
        ],
    ),
    (
        'M45',
        '四象限里 taker_only 与 maker_only 的掩码写反'
        '（结果与 H4 预期方向相反，看起来像「假说被否证」）',
        [
            (
                'tw/analyzer_asymmetry.py',
                '        "taker_only": t & ~m,\n        "maker_only": ~t & m,',
                '        "taker_only": ~t & m,\n        "maker_only": t & ~m,',
            ),
        ],
    ),
    (
        'M46',
        '饱和判定阈值方向反了（判 fill_ratio > 0.95 为饱和）'
        '——所有档位都被标成饱和，未饱和集为空',
        [
            (
                'scripts/run_workstream_H.py',
                '    return bool(fill_ratio < threshold)',
                '    return bool(fill_ratio > threshold)',
            ),
        ],
    ),
    (
        'M47',
        '局部弹性的 log 分子分母写反（size_big/small 或 |滑点| 比值颠倒）'
        '——弹性方向或量级明显不合理',
        [
            (
                'tw/analyzer_concavity.py',
                '        mean_e = math.log(abs(float(mean_big)) / abs(float(mean_small))) / log_ratio',
                '        mean_e = math.log(abs(float(mean_small)) / abs(float(mean_big))) / log_ratio',
            ),
        ],
    ),
    (
        'M48',
        '凹度判决的边界写成 hi <= 1.0（上界恰好 1.0 时误判为「显著凹」）',
        [
            (
                'tw/analyzer_concavity.py',
                '    if hi < 1.0:\n        return VERDICT_CONCAVE, (lo, hi)',
                '    if hi <= 1.0:\n        return VERDICT_CONCAVE, (lo, hi)',
            ),
        ],
    ),
    (
        'M49',
        '阶段3 审计的种子数守卫方向写反（变成「必须不少于原始值」）'
        '——重跑时偷偷扩种子将不再被拦住',
        [
            (
                'scripts/run_workstream_J.py',
                '    assert max(n_seeds) <= ORIGINAL_N_SEEDS, (',
                '    assert max(n_seeds) >= ORIGINAL_N_SEEDS, (',
            ),
        ],
    ),
    (
        'M50',
        '数值陈述扫描的正则退回窄版本（漏掉「从 A 拉到 B」「带负号」两类常见表述）',
        [
            (
                'scripts/audit_numeric_claims.py',
                '    ("区间变化", r"从\\s*[-+]?[\\d.]+[^。；\\n]{0,10}?[-+]?[\\d.]+"),',
                '    ("区间变化", r"从\\s*[\\d.]+\\s*(?:到|→|->)\\s*[\\d.]+"),',
            ),
        ],
    ),
    (
        'M51',
        '安全替换工具校验了**原文**而不是替换后的文本（等于没校验）'
        '——破坏语法的替换会被放行并写回真实文件',
        [
            (
                'scripts/safe_batch_replace.py',
                '                try:\n                    ast.parse(modified)',
                '                try:\n                    ast.parse(original)',
            ),
        ],
    ),
    (
        'M52',
        '回填时不再检查「标定/饱和检查/拟合三处种子数一致」'
        '——某一处偷偷用了别的种子集合，数字会悄悄变得不可比',
        [
            (
                'scripts/run_workstream_L.py',
                '    bad = [r.get("n_seeds") for r in rows if r.get("n_seeds") != expected]',
                '    bad = []   # 变异：不检查种子一致性',
            ),
        ],
    ),
    (
        'M53',
        'λ̄ 标定窗口退回历史口径（含瞬态）而不是实验窗口'
        '——复现第六条纪律要防的那类「标定 ≠ 实验」问题',
        [
            (
                'scripts/run_workstream_L.py',
                '    return (WARMUP, WARMUP + HORIZON)',
                '    return (500, 3_000)',
            ),
        ],
    ),
    (
        'M54',
        '功效分析的 ratio 方向写反（n2 = n1 / ratio 而不是 n1 * ratio）'
        '——ratio=1 时完全看不出来，只在不等样本量下暴露',
        [
            (
                'scripts/power_analysis_H4.py',
                '    n2 = n1 * ratio\n    if n2 < 2:',
                '    n2 = n1 / ratio\n    if n2 < 2:',
            ),
        ],
    ),

]


def _required_packages() -> set[str]:
    """扫 tests/ 里的 import，推导出沙箱必须复制哪些顶层包。

    为什么不写死一份清单：清单会随着新增模块而过时，而过时的**表现**是
    "基线 ImportError → 每个变异体都变红 → 证据全是假的"，
    看起来像满分，其实什么都没验证。自动推导让这个失效模式消失。
    """
    import re

    found: set[str] = {"tw", "tests"}
    pat = re.compile(r"^\s*(?:from|import)\s+([A-Za-z_]\w*)", re.M)
    for tf in sorted((ROOT / "tests").glob("*.py")):
        try:
            src = tf.read_text(encoding="utf-8")
        except OSError:
            continue
        for name in pat.findall(src):
            if (ROOT / name).is_dir() and not name.startswith("_"):
                found.add(name)
    return found


def run_suite(cwd: Path) -> tuple[int, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = "."
    env["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.run(
        [PY, "-m", "unittest", "discover", "-s", "tests", "-t", ".", "-v"],
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def first_failure_names(output: str, limit: int = 3) -> list[str]:
    names = []
    for line in output.splitlines():
        s = line.strip()
        if s.startswith("FAIL: ") or s.startswith("ERROR: "):
            names.append(s.split(":", 1)[1].strip().split(" ")[0])
        if len(names) >= limit:
            break
    return names


# ----------------------------------------------------------------------
# 并行执行
#
# 为什么需要：一轮变异验证要跑 len(MUTATIONS) 次**完整测试套件**。
# 二期把套件扩到 490+ 项之后，单次约 200 秒 —— 全部变异体串行就是**近两小时**。
# 实测第一版串行跑了 16 分钟才做完基线 + 1 个变异体。
#
# 并行的前提是**每个 worker 有自己的沙箱**：不能共享一个目录，
# 否则一个 worker 注入的改动会污染另一个 worker 的基线，
# 于是"变红"的原因变成"别人改的"，证据全废。
# 本机 32 核，取 8 个 worker 是"够快"与"不把机器打满"的折中。
# ----------------------------------------------------------------------
def prepare_sandbox(work_dir: Path, pkgs: set[str]) -> None:
    work_dir.mkdir(parents=True, exist_ok=True)
    # ⚠️ ``docs/`` 也要复制（它只有几 MB）。
    #   理由：**sandbox 里缺 docs/ 会让基线变红**，而基线一红，
    #   每个变异体都会"变红"——整份变异验证就变成假证据。
    #   本项目为此栽过两次（一次是产物依赖、一次是编号账本依赖），
    #   两次的修法都是"给测试加 skip"；这次改成**从根上补上 docs/**，
    #   让那些测试能真正跑（而不是被跳过）——跳过等于没测。
    #   ``out/``（约 500MB）仍不复制：那里的产物可以由脚本重建，
    #   依赖它的测试继续用显式 skip。
    for sub in sorted(pkgs) + ["data", "docs"]:
        src_dir = ROOT / sub
        if src_dir.is_dir():
            shutil.copytree(src_dir, work_dir / sub, dirs_exist_ok=True)


def check_anchors() -> list[str]:
    """先校验**全部**锚点，再开始跑。返回有问题的编号。

    锚点失效（原文找不到、或匹配到多处）必须在跑之前就发现：
    否则会先花几分钟跑完前面的变异体，最后才发现后面几个根本注入不进去。
    """
    bad: list[str] = []
    for mid, _desc, patches in MUTATIONS:
        for rel, old, _new in patches:
            p = ROOT / rel
            if not p.exists():
                print(f"  {mid}: 文件缺失 {rel}")
                bad.append(mid)
                break
            n = p.read_text(encoding="utf-8").count(old)
            if n != 1:
                print(f"  {mid}: 锚点失效（{n} 处匹配）{rel}")
                bad.append(mid)
                break
    return bad


def run_one_mutation(job: tuple[int, str, str, list]) -> dict:
    """在**自己的沙箱**里注入一个变异、跑一遍套件、还原。

    顶层函数（不是闭包）—— ``ProcessPoolExecutor`` 要把参数 pickle 过去，
    闭包与 lambda 都不可序列化。
    """
    idx, mid, desc, patches = job
    work = WORK / f"w{idx}"
    prepare_sandbox(work, _required_packages())
    backups: dict[Path, str] = {}
    try:
        for rel, old, new in patches:
            f = work / rel
            src = f.read_text(encoding="utf-8")
            backups.setdefault(f, src)
            f.write_text(src.replace(old, new), encoding="utf-8")
        rc, out = run_suite(work)
        names = first_failure_names(out) if rc != 0 else []
        ran = next((l for l in out.splitlines() if l.startswith("Ran ")), "")
        return {"mid": mid, "desc": desc, "ok": rc != 0, "ran": ran,
                "first_failures": names, "detail": ""}
    except Exception as e:  # pragma: no cover - 环境异常
        return {"mid": mid, "desc": desc, "ok": False, "ran": "",
                "first_failures": [], "detail": f"⚠️ 执行异常：{e!r}"}
    finally:
        for f, src in backups.items():
            f.write_text(src, encoding="utf-8")


def _selected() -> list[tuple[str, str, list]]:
    """按 ``--only M30,M31`` 过滤要跑的变异体（不给就是全部）。

    为什么要这个开关：**新增一个变异体之后，必须单独确认它真的能变红。**
    一个永远"变红"的变异体是假的（比如注入的代码根本没被执行到、
    或者锚点匹配在注释上），而只有单独跑它、看它具体挂在哪条断言上，
    才能判断这条证据是不是真的。

    锚点校验**始终对全部变异体做**（不因为过滤而跳过）——
    锚点是最便宜的一致性检查，没有理由只查一部分。
    """
    ids: set[str] | None = None
    if "--only" in sys.argv:
        raw = sys.argv[sys.argv.index("--only") + 1]
        ids = {x.strip() for x in raw.split(",") if x.strip()}
    if ids is None:
        return list(MUTATIONS)
    out = [m for m in MUTATIONS if m[0] in ids]
    unknown = ids - {m[0] for m in MUTATIONS}
    if unknown:
        raise SystemExit(f"❌ --only 里有不存在的编号：{sorted(unknown)}")
    return out


def main() -> int:
    """变异验证入口。``--workers N`` 控制并行度（默认 8）。
    ``--only M30,M31`` 只跑指定变异体（用于验证新增的变异体真的能变红）。"""
    workers = 8
    if "--workers" in sys.argv:
        workers = max(1, int(sys.argv[sys.argv.index("--workers") + 1]))
    selected = _selected()

    print("=" * 74)
    print("变异验证：这套测试能不能失败？")
    if len(selected) != len(MUTATIONS):
        print(f"（本次只跑 {len(selected)}/{len(MUTATIONS)} 个："
              f"{[m[0] for m in selected]}）")
    print("=" * 74)

    # ① 先校验全部锚点（失败快）
    print("\n  校验锚点…")
    bad = check_anchors()
    if bad:
        print(f"  ❌ 有 {len(bad)} 个变异体的锚点失效：{bad}")
        print("     锚点失效时**不能**继续——否则那几个变异体会被误报成"
              "「仍全绿」（其实是根本没注入进去）。")
        return 2
    print(f"  ✅ {len(MUTATIONS)} 个变异体的锚点全部有效（各匹配 1 处）")

    # ② 基线：先确认原始代码全绿
    prepare_sandbox(WORK / "base", _required_packages())
    print("\n  跑基线（未注入 bug）…")
    base_rc, base_out = run_suite(WORK / "base")
    total_line = next(
        (l for l in base_out.splitlines() if l.startswith("Ran ")), "Ran ? tests")
    if base_rc != 0:
        print("  ❌ 基线不通过 —— 原始代码就有问题，变异验证无意义。")
        print(base_out[-3000:])
        return 1
    print(f"  ✅ 基线：{total_line} 全绿")
    copied = ", ".join(sorted(p for p in _required_packages() if (ROOT / p).is_dir()))
    print(f"     沙箱已复制：{copied}、data")

    # ③ 并行跑选中的变异体
    print(f"\n  并行跑 {len(selected)} 个变异体（{workers} 个 worker，"
          f"每个 worker 一个独立沙箱）…")
    jobs = [(i, mid, desc, patches)
            for i, (mid, desc, patches) in enumerate(selected)]
    results: dict[str, dict] = {}
    t0 = time.time()
    # ⚠️ 这里**不能**用 ``ex.map``。``map`` 是按**提交顺序**产出结果的：
    #    本套件的 M1 要跑近十分钟，于是前 8 个变异体里只要 M1 没完，
    #    后面的结果一个都打印不出来——**十几分钟完全没有输出**，
    #    "在算"和"卡死"看起来一模一样（本轮实测踩到）。
    #    ``as_completed`` 谁先完谁先报，进度是真的进度。
    #    结果的**顺序**不受影响：下面出总表时是按 ``selected`` 的编号顺序读
    #    ``results`` 的，所以表格仍然稳定可读。
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(run_one_mutation, j): j[1] for j in jobs}
        for fut in as_completed(futs):
            mid = futs[fut]
            try:
                r = fut.result()
            except Exception as e:                        # pragma: no cover
                # 一个 worker 崩了不该让整轮没结果——把它记成"无结果"，
                # 汇总时会被列进 holes 并让返回码非零。
                r = {"mid": mid, "ok": False, "detail": f"worker 异常：{e!r}",
                     "ran": "", "first_failures": []}
            results[r["mid"]] = r
            done = len(results)
            mark = "✅ 变红" if r["ok"] else "❌ 仍全绿"
            print(f"    [{done:>2}/{len(selected)}] {r['mid']:<5}{mark}"
                  f"  ({time.time() - t0:.0f}s)", flush=True)
    print(f"\n  并行用时 {time.time() - t0:.0f}s")

    # ④ 按编号顺序出总表
    print("\n" + "-" * 74)
    print(f"{'编号':<5}{'注入的 bug':<40}{'结果':<10}证据")
    print("-" * 74)
    holes: list[str] = []
    for mid, desc, _patches in selected:
        r = results.get(mid)
        if r is None:
            print(f"{mid:<5}{desc:<40}⚠️ 无结果")
            holes.append(mid)
            continue
        if r["detail"]:
            print(f"{mid:<5}{desc:<40}⚠️ {r['detail']}")
            holes.append(mid)
        elif r["ok"]:
            names = ", ".join(r["first_failures"]) or "-"
            print(f"{mid:<5}{desc:<40}✅ 变红   {r['ran']} 首挂: {names}")
        else:
            print(f"{mid:<5}{desc:<40}❌ 仍全绿")
            holes.append(mid)
    print("-" * 74)
    if holes:
        print(f"\n⚠️ 有 {len(holes)} 个变异体没有被测试抓到：{holes}")
        print("   这些位置的测试是摆设，需要补断言。")
        return 2
    n = len(selected)
    suffix = "" if n == len(MUTATIONS) else f"（本次只跑了 {n} 个，未跑全量）"
    print(f"\n✅ {n}/{n} 个注入的 bug 全部被测试抓到。{suffix}")
    print("   结论：这套测试有检出能力，全绿是有信息量的。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
