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
                '                    compile(modified, key, "exec")',
                '                    compile(original, key, "exec")',
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
    (
        'M55',
        '区间重叠比例的分母用了两区间宽度**之和**而不是较窄的那个'
        '——100% 包含会被算成约 50%，「无法判定」会看起来像「显著」',
        [
            (
                'tw/analyzer_consistency.py',
                '    narrower = min(w_a, w_b)',
                '    narrower = w_a + w_b',
            ),
        ],
    ),
    (
        'M56',
        '一致性判定的阈值方向写反（重叠 **大** 时反而判「显著」）',
        [
            (
                'tw/analyzer_consistency.py',
                '    if frac <= SIGNIFICANT_MAX_OVERLAP:',
                '    if frac >= SIGNIFICANT_MAX_OVERLAP:',
            ),
        ],
    ),
    (
        'M57',
        '方向性判定的两个排除条件写反（excludes_0_5 用 1.0 的判据、反之亦然）'
        '——把「偏线性」的家族标成「偏超凹」或「无法判断」',
        [
            (
                'tw/analyzer_consistency.py',
                '    excludes_0_5 = not (lo <= 0.5 <= hi)\n'
                '    excludes_1_0 = not (lo <= 1.0 <= hi)',
                '    excludes_0_5 = not (lo <= 1.0 <= hi)\n'
                '    excludes_1_0 = not (lo <= 0.5 <= hi)',
            ),
        ],
    ),
    (
        'M58',
        '行情库的重复写入改成**覆盖**（ON CONFLICT DO UPDATE）'
        '——「入库后不覆盖」那条约束被悄悄取消，'
        '「当时的决策看到哪版数据」这个问题从此没有答案',
        [
            (
                'tw/marketdb.py',
                '                INSERT OR IGNORE INTO candles',
                '                INSERT OR REPLACE INTO candles',
            ),
        ],
    ),
    (
        'M59',
        '读 K 线时不再过滤未确认的根（confirmed_only 默认改 False）'
        '——**未走完的 K 线 = 未来数据**会流进策略，'
        '这是本项目测量纪律里最不可接受的一类错误',
        [
            (
                'tw/marketdb.py',
                '        confirmed_only: bool = True,\n        name: str = "",',
                '        confirmed_only: bool = False,\n        name: str = "",',
            ),
        ],
    ),
    (
        'M60',
        'OKX 分页游标不减 1ms（after=oldest 而不是 oldest−1）'
        '——边界那根会同时出现在两批里，'
        '行数虚高而**看不出异常**（主键会挡掉重复写入，所以数字不会明显错）',
        [
            (
                'tw/okx_data.py',
                '                cursor = oldest - 1  # ⚠️ 减 1ms，否则边界那根会重复出现',
                '                cursor = oldest  # ⚠️ 减 1ms，否则边界那根会重复出现',
            ),
        ],
    ),
    (
        'M61',
        '分页「没有前进」的守卫被去掉（不死循环而是静默转圈）'
        '——静默死循环比报错危险得多：调用方以为在拉数据，实际原地打转',
        [
            (
                'tw/okx_data.py',
                '                if oldest >= cursor:\n',
                '                if False:\n',
            ),
        ],
    ),
    (
        'M62',
        '拉取失败不写账本（except 分支里不再调用 log_fetch）'
        '——失败不留痕 = 半年后不知道「当时是不是拉失败了」，'
        '而账本存在的全部意义就是回答这个',
        [
            (
                'tw/okx_data.py',
                '            from_ts=want_from, n_rows=total, pages=pages, started_at=started,\n'
                '            finished_at=int(time.time() * 1000), ok=False, error=str(exc)[:400],\n',
                '            from_ts=want_from, n_rows=total, pages=pages, started_at=started,\n'
                '            finished_at=int(time.time() * 1000), ok=True, error=str(exc)[:400],\n',
            ),
        ],
    ),
    (
        'M63',
        '合成数据的 high/low 直接取 max/min(open, close)（无影线）'
        '——K 线退化成折线，蜡烛图画不出形态，'
        '「窗口内极值」这个 OHLC 的定义被废掉',
        [
            (
                'tw/synthetic.py',
                '        hi = max(hi, p0, p1) * (1.0 + abs(rng.normal(0.0, 2e-4)))\n',
                '        hi = max(p0, p1)\n',
            ),
        ],
    ),
    # ==================================================================
    # A1：OKX 风格订单模型 + 保证金账户（2026-09-20 第八轮）
    # ==================================================================
    (
        'M64',
        '止盈止损触发方向不再按开仓方向取反（空头 tp 也写成 >=）'
        '——空头的止盈会在价格**上涨**时触发，等于把止盈做成了止损；'
        '而成品系统里这种单会立即成交，比没有止盈止损更糟',
        [
            (
                'tw/order_model.py',
                '            if self.tp_trigger_px is not None:\n'
                '                p = pick(self.tp_trigger_px_type)\n'
                '                if p <= self.tp_trigger_px:\n'
                '                    return "tp"\n',
                '            if self.tp_trigger_px is not None:\n'
                '                p = pick(self.tp_trigger_px_type)\n'
                '                if p >= self.tp_trigger_px:\n'
                '                    return "tp"\n',
            ),
        ],
    ),
    (
        'M65',
        '市价单不再用 ±inf 而带上一个有限价格（1e18）'
        '——撮合层无法把它识别成"吃穿全盘"，市价单退化成"挂在很远处的限价单"，'
        '本应立即成交的单永远不成交（静默失效，报错都没有）',
        [
            (
                'tw/order_model.py',
                '            px = math.inf if self.side == "buy" else -math.inf\n',
                '            px = 1e18 if self.side == "buy" else -1e18\n',
            ),
        ],
    ),
    (
        'M66',
        'post_only 交叉检查反向（买用 <= 而不是 >=）'
        '——post_only 的安全阀失效：本该被拒的"一定会吃单"的挂单被放行，'
        '名义上的 maker 单实际付了 taker 费率（费差 2.5 倍），'
        '而且它仍然以 post_only 记账，策略开发者完全看不出来',
        [
            (
                'tw/order_model.py',
                '            if best_ask is not None and req.px is not None and req.px >= best_ask:\n'
                '                raise _reject("TW-1001")\n',
                '            if best_ask is not None and req.px is not None and req.px <= best_ask:\n'
                '                raise _reject("TW-1001")\n',
            ),
        ],
    ),
    (
        'M67',
        'reduce_only 的超量**不再裁剪**而是原样放行'
        '——平仓单能平出反向仓位，「只减仓」的语义被废掉；'
        '而留痕里看不出量被放大了（Agent 说平 100 手，实际平了 300 手并反向持仓）',
        [
            (
                'tw/order_model.py',
                '    if req.sz > abs(position_qty) + EPS:\n'
                '        return req.with_size(abs(position_qty))\n',
                '    if False:\n'
                '        return req.with_size(abs(position_qty))\n',
            ),
        ],
    ),
    (
        'M68',
        '强平价公式里多空用了同一个分母（1 - m）'
        '——空头强平价被算低，账户在真正该被强平时看起来还安全；'
        '这是最典型的「抄错一行公式」型缺陷',
        [
            (
                'tw/account.py',
                '        return (self.avg_px * q + c) / (q * (1.0 + m))\n',
                '        return (self.avg_px * q + c) / (q * (1.0 - m))\n',
            ),
        ],
    ),
    (
        'M69',
        '权益估值时缺标记价的仓位按 0 计而不是按开仓均价计'
        '——缺一个标记价就把整仓价值算成 0，账户凭空亏光并触发假强平；'
        '这类"缺失值当 0"是研究代码里最常见的系统性偏差来源',
        [
            (
                'tw/account.py',
                '            m = marks.get(p.inst_id)\n'
                '            if m is None or not (math.isfinite(m) and m > 0):\n'
                '                m = p.avg_px\n',
                '            m = marks.get(p.inst_id)\n'
                '            if m is None or not (math.isfinite(m) and m > 0):\n'
                '                m = 0.0\n',
            ),
        ],
    ),
    (
        'M70',
        '强平判据用严格小于（equity < mm）而不是 <='
        '——恰好踩在强平线上的账户不被强平，'
        '而这个边界恰恰是数值上最常出现的位置',
        [
            (
                'tw/account.py',
                '        return self.equity(marks) <= self.maintenance_margin(marks)\n',
                '        return self.equity(marks) < self.maintenance_margin(marks)\n',
            ),
        ],
    ),
    (
        'M71',
        '开仓能力检查把"新仓带来的未实现盈亏"当成 0（upl 强制置 0）'
        '——只看开仓前已有仓位的盈亏，于是"在亏损仓位上继续加仓"永远查不出问题：'
        '加仓那一刻模拟权益虚高，等价格再动一点就穿仓。'
        '这正是 can_open 要用"开仓**后**"模拟权益的原因',
        [
            (
                'tw/account.py',
                '        upl = (m - avg_after) * q_after if abs(q_after) > EPS else 0.0\n'
                '        eq_after = self.cash + self.total_upl(marks) - p.upl(m) + upl\n',
                '        upl = (m - avg_after) * q_after if abs(q_after) > EPS else 0.0\n'
                '        eq_after = self.cash + self.total_upl(marks) - p.upl(m)\n',
            ),
        ],
    ),
    # ==================================================================
    # A2：风控闸门 + 决策留痕（2026-09-20 第八轮）
    # ==================================================================
    (
        'M72',
        '止盈止损不再校验「偏离 mid 是否过大」'
        '——直接放行第 6 轮实测到的 100 倍量级错误（tp=10350 vs mid=102.4），'
        'TP 挂在 100 倍远、SL 永不触发，账户带着裸仓一直跑而留痕显示"已设止盈止损"',
        [
            (
                'tw/risk.py',
                '        if dev > lim.max_tp_sl_pct:\n'
                '            d.rule = "tp_sl_off_market"\n',
                '        if False:\n'
                '            d.rule = "tp_sl_off_market"\n',
            ),
        ],
    ),
    (
        'M73',
        '止盈止损不再校验方向自洽（tp 在上/sl 在下）'
        '——"止盈在下方"的单会立即触发，等于下单瞬间以亏损平仓；'
        '风控放行了一条比不下单更糟的决策',
        [
            (
                'tw/risk.py',
                '        if should_be == "above" and vf <= mid:\n',
                '        if False:\n',
            ),
        ],
    ),
    (
        'M74',
        '风控被拒时返回原意图而不是退回 hold'
        '——证据链第 ⑤ 项断裂：留痕里"风控拒了"与"Agent 本就要这么干"'
        '无法区分，Agent 的决策质量会被错误地算在风控头上',
        [
            (
                'tw/risk.py',
                '    if action not in ("buy", "sell"):\n'
                '        d.rule = "bad_action"\n'
                '        d.code = "TW-1007"\n'
                '        d.note(f"action 非法: {action!r}")\n'
                '        return _hold(f"风控拒绝：action 非法（{action}）"), d\n',
                '    if action not in ("buy", "sell"):\n'
                '        d.rule = "bad_action"\n'
                '        d.code = "TW-1007"\n'
                '        d.note(f"action 非法: {action!r}")\n'
                '        return dict(parsed), d\n',
            ),
        ],
    ),
    (
        'M75',
        '裁量步被取消（量超上限直接按原量放行）'
        '——「风控改了多少量」这个数字消失，'
        '而它正是量化"风控贡献"的唯一来源：Agent 要 1000 手、实际下 30 手，'
        '不改量就等于把风控的贡献记成 0',
        [
            (
                'tw/risk.py',
                '    if notional > cap:\n'
                '        new_sz = cap / mid\n',
                '    if False:\n'
                '        new_sz = cap / mid\n',
            ),
        ],
    ),
    (
        'M76',
        'decision_id 不再包含 prompt_template 版本'
        '——A/B 换 prompt 的实验里两条决策会算出同一个 ID，'
        '归因彻底失效；而日志看起来完全正常'
        '（"看起来对"的静默错误：ID 照样是 32 位十六进制）',
        [
            (
                'tw/decision_log.py',
                '            str(self.agent_id),\n'
                '            str(self.prompt_template),\n'
                '            str(self.model),\n',
                '            str(self.agent_id),\n'
                '            str(self.model),\n',
            ),
        ],
    ),
    (
        'M77',
        '可见状态的 extra 守卫被去掉'
        '——`build_visible_state(**extra)` 的 **extra 是一个不受限入口，'
        '拆掉守卫后调用方就能把 `future_close` / `outcome_pnl` 塞进 ① 可见状态，'
        '回放会读到未来数据（答案泄漏）。'
        '危险之处在于：签名上"没有 future_* 参数"的承诺看起来仍然成立，'
        '守卫被绕过的表现是完全静默的——这条测试就是那个承诺的唯一载体',
        [
            (
                'tw/decision_log.py',
                '    bad = [k for k in extra if _is_leaky_key(k)]\n'
                '    if bad:\n',
                '    bad = []\n'
                '    if bad:\n',
            ),
        ],
    ),
    # ==================================================================
    # A3：LLM 接入 + prompt 模板 + 解析（2026-09-21 第九轮）
    # 这批的目标全是**静默失效**：不报错、结果看起来正常、
    # 只有逐条核对或变异测试才分得出来。
    # ==================================================================
    (
        'M78',
        '缺 LLM key 时静默返回 hold 而不是报错'
        '——这是最难发现的一类：跑起来完全正常（每条决策都合规、都留痕、'
        '都带理由），只是这个 Agent **永远不交易**。它把"系统是坏的"'
        '伪装成"Agent 很保守"，而收益率曲线看起来只是"没机会"',
        [
            (
                'tw/llm.py',
                '        if self.config.api_key_env and not self.api_key:\n'
                '            raise RuntimeError(\n',
                '        if False:\n'
                '            raise RuntimeError(\n',
            ),
        ],
    ),
    (
        'M79',
        '解析层"帮模型把量级改回来"（10350 → 103.5）'
        '——看起来像是善意修正，实际把"模型在数值上不可靠"这个事实'
        '从数据里抹掉；而那正是最该被量化的东西。'
        '把关的责任在风控层，不在解析层：解析层只翻译，不审校',
        [
            (
                'tw/parse.py',
                '    for key in ("sz", "px", "tp", "sl"):\n'
                '        if key in parsed:\n'
                '            parsed[key] = _to_float(parsed[key])\n',
                '    for key in ("sz", "px", "tp", "sl"):\n'
                '        if key in parsed:\n'
                '            parsed[key] = _to_float(parsed[key])\n'
                '            if key in ("tp", "sl") and parsed[key]:\n'
                '                parsed[key] = parsed[key] / 100.0\n',
            ),
        ],
    ),
    (
        'M80',
        '回放按 prompt 只存**最后一条**响应（覆盖式索引）'
        '——多采样时同一 prompt 调用 N 次，覆盖后只剩最后 1 条，'
        '回放时 N 次全发那一条 ⇒ 真实的"2 hold / 1 buy"变成"3 buy"，'
        '**多数派方向整个反过来**。而覆盖率仍显示 100%、零报错。'
        '这个 bug 真的活过一轮，靠逐条比对 live/replay 才发现',
        [
            (
                'tw/llm.py',
                '                        self._by_hash.setdefault(str(h), []).append(rec)\n',
                '                        self._by_hash[str(h)] = [rec]\n',
            ),
        ],
    ),
    (
        'M81',
        '可见状态用"序列末尾 N 根"而不是"截至 i 的 N 根"'
        '——`close[-N:]` 取的是序列末尾；调用方把整条序列传进来、'
        'i 停在中间（回放历史某一天）时，它会取到**未来**的根。'
        '不报错，只会让那一天的决策"神奇地准"',
        [
            (
                'tw/agent.py',
                '    lo = max(0, i - int(n_closes) + 1)\n'
                '    rc = [float(x) for x in closes[lo:i + 1]]\n',
                '    rc = [float(x) for x in closes[-int(n_closes):]]\n',
            ),
        ],
    ),
    (
        'M82',
        'mid 取 (high+low)/2 而不是收盘价'
        '——high/low 是**事后才知道**的极值，用它们当 mid 等于把'
        '"这根 K 线内最高能到哪"提前告诉模型。'
        '数据全在同一根 K 线里，看起来"不过分"，其实是最隐蔽的一类泄漏',
        [
            (
                'tw/agent.py',
                '    mid = float(closes[i])\n',
                '    _hl = getattr(series, "high", None), getattr(series, "low", None)\n'
                '    mid = (float(_hl[0][i]) + float(_hl[1][i])) / 2.0\n',
            ),
        ],
    ),
    (
        'M83',
        '多采样全失败时把 `majority_sample` 造的兜底 hold 写进 `parsed`'
        '——那条兜底的 action 也是 "hold"，于是"全部解析失败"'
        '被读成"解析成功"（判据看最终 parsed 而不是采样列表）。'
        '造一个看起来合法的兜底值，是"把故障伪装成正常"的标准形态',
        [
            (
                'tw/agent.py',
                '            n_ok = sum(1 for p in parsed_list if p.get("action"))\n'
                '            vote = consistency(parsed_list)\n',
                '            n_ok = sum(1 for p in parsed_list if p.get("action"))\n'
                '            vote = consistency(parsed_list)\n'
                '            if n_ok == 0:\n'
                '                n_ok = 1\n',
            ),
        ],
    ),
    (
        'M84',
        '给模型的"建议最大量"用全精度而不是向下取整'
        '——模型会照抄但那串数字有 17 位有效数字，它写 6 位小数就**比上限大**，'
        '于是触发 size_cap。`resized_frac` 是给风控算功劳的指标，'
        '被纯四舍五入抬高 = 给风控记了一笔假功劳',
        [
            (
                'tw/agent.py',
                '        exp = math.floor(math.log10(raw)) - 1\n'
                '        step = 10.0 ** exp\n'
                '        return max(step, math.floor(raw / step) * step)\n',
                '        return raw\n',
            ),
        ],
    ),
    # ==================================================================
    # A4：执行模拟 + 规则基线 + 分层评估（2026-09-21 第十轮）
    # 这批的目标是**数字类的静默失效**：回测/收益算错了看不出来。
    # ==================================================================
    (
        'M85',
        '执行器允许订单在**提交的那一根**就成交（决策与成交不错开）'
        '——等于让订单回到过去成交，收益凭空变好，而且完全不报错。'
        '这是 bar 级回测最经典的未来函数',
        [
            (
                'tw/simexec.py',
                '            if i <= oo.submit_bar:\n'
                '                still.append(oo)\n'
                '                continue\n',
                '            if i < oo.submit_bar:\n'
                '                still.append(oo)\n'
                '                continue\n',
            ),
        ],
    ),
    (
        'M86',
        '同根内 TP 与 SL 都触及时按 **TP** 处理（乐观解）'
        '——bar 级回测看不到当根内的价格路径，两个都碰到时选谁是一次'
        '**不可回避的假设**。选 TP 会让回测变好看，而这正是'
        '"偏乐观的回测让人以为策略能上线"的机制。本项目显式选 SL（悲观）',
        [
            (
                'tw/simexec.py',
                '        if hit_sl:\n'
                '            reason, px = "sl", float(a.sl_trigger_px)\n'
                '        else:\n'
                '            reason, px = "tp", float(a.tp_trigger_px)\n',
                '        if hit_tp:\n'
                '            reason, px = "tp", float(a.tp_trigger_px)\n'
                '        else:\n'
                '            reason, px = "sl", float(a.sl_trigger_px)\n',
            ),
        ],
    ),
    (
        'M87',
        '市价单的滑点方向写反（买单向下滑、卖单向上滑）'
        '——成交价比真实更好，回测凭空赚钱。每一笔都多赚一个滑点，'
        '换手越高偏差越大，而日终对账看不出异常（它只是"看起来更赚"）',
        [
            (
                'tw/simexec.py',
                '            px = o * (1.0 + slip) if req.side == "buy" else o * (1.0 - slip)\n',
                '            px = o * (1.0 - slip) if req.side == "buy" else o * (1.0 + slip)\n',
            ),
        ],
    ),
    (
        'M88',
        '手续费不再走 ExecConfig（退回账户的 MarginConfig）'
        '——`cost_multiplier` 于是**只影响滑点、完全不影响手续费**，'
        '而配置看上去是同时控制两者的。症状：成本 ×5 后收益几乎没变，'
        '看起来像"策略对成本不敏感"——一个漂亮且错误的结论。'
        '这个 bug 真的存在过，是解析式与重跑版的数字对不上才查出来的',
        [
            (
                'tw/simexec.py',
                '        fee = self.config.fee(abs(float(req.sz) * float(px)), is_maker=is_maker)\n',
                '        fee = None\n',
            ),
        ],
    ),
    (
        'M89',
        '逐根收益率漏掉第一根（`zip(eq[1:], eq[2:])`）'
        '——夏普/波动率的样本少一个。收益总额看不出来（它用的是权益首尾），'
        '所以只有夏普悄悄偏了一点。少了这个观测点等于**少记了一根的盈亏**，'
        '而"共几根"这件事没有第二个地方会对账',
        [
            (
                'tw/eval_agent.py',
                '    for a, b in zip(equity, equity[1:]):\n',
                '    for a, b in zip(equity[1:], equity[2:]):\n',
            ),
        ],
    ),
    (
        'M90',
        '与基线对照时**一律**走区间重叠检验（零方差基线也不例外）'
        '——`noop` 的收益恒为 0 ⇒ 区间宽度为 0 ⇒ 重叠比例没有意义 ⇒ '
        '输出「无法判定（输入退化）」。而"能不能打赢 noop"**恰恰是 A4 '
        '最核心的那个问题**：用错检验会让最关键的问题答不出来，'
        '而且看起来像"数据不够"',
        [
            (
                'tw/eval_agent.py',
                '        if width <= scale * 1e-9:\n',
                '        if False:\n',
            ),
        ],
    ),
    (
        'M91',
        '成交率的分子用**全部成交**而不是"由订单产生的成交"'
        '——TP/SL 是挂在仓位上的触发单，**没有对应的已提交订单**。'
        '把它们算进分子会让比率**超过 100%**（真机报告里出现过 142%：'
        '19 张订单 27 笔成交）。超过 100% 看起来只是"数字有点怪"，'
        '很容易被放过，但它说明分子分母不同源，'
        '而且这个错误会随 TP/SL 使用率线性放大',
        [
            (
                'tw/eval_agent.py',
                '    n_order_fills = sum(1 for f in fills if f.reason in _ORDER_FILL_REASONS)\n'
                '    n_exit_fills = sum(1 for f in fills if f.reason in ("tp", "sl"))\n',
                '    n_order_fills = len(fills)\n'
                '    n_exit_fills = sum(1 for f in fills if f.reason in ("tp", "sl"))\n',
            ),
        ],
    ),
    (
        'M92',
        '回放覆盖率过低时仍把重跑结果当成本敏感性结论'
        '——LLM 的 prompt 含账户状态（持仓/权益），而账户状态取决于成交、'
        '成交取决于成本 ⇒ 换成本重放会让路径迅速发散。'
        '实测成本 ×1 覆盖率 100%、×2 掉到 **0.8%**，于是 Agent 全程弃权、'
        '交易从 27 笔变 2 笔，账面净收益从 −33 变 −444。'
        '**那个 −444 看起来像成本敏感性结果，其实是回放失败**',
        [
            (
                'tw/eval_agent.py',
                '        if cov is not None and cov < min_coverage:\n',
                '        if False:\n',
            ),
        ],
    ),
    # ==================================================================
    # A5：GUI 集成（GUI 留痕浏览 + JSON 互操作）（2026-09-21 第十一轮）
    # 这批的目标是**自测全绿但产物/页面是坏的**那类问题。
    # ==================================================================
    (
        'M93',
        '写 JSON 时不把 NaN/Inf 换成 null（去掉 json_safe）'
        '——`json.dumps(float("nan"))` 在 Python 里默认输出**裸的 `NaN`**，'
        '而 `NaN` **不是合法 JSON**（RFC 8259 没有它）。'
        'Python 读得回来（解析器是超集）⇒ **自测全绿**；'
        '但 JS 的 `JSON.parse` 直接抛 `Unexpected token N`。'
        '触发源很讽刺：不交易时 `sharpe` 按设计返回 nan（"算不出来"），'
        '于是**最基准的那条配置（noop）**产出了非法 JSON',
        [
            (
                'tw/eval_agent.py',
                '    if isinstance(o, float):\n'
                '        return o if math.isfinite(o) else None\n',
                '    if isinstance(o, float):\n'
                '        return o\n',
            ),
        ],
    ),
    (
        'M94',
        '首屏预载嵌进 HTML 时不转义 `</`'
        '——JSON 里若出现 `</script>`（比如某条决策的理由里恰好写了它），'
        '浏览器会**提前结束脚本块**。那不是转义的小毛病，是**注入**：'
        '后面的内容会被当成 HTML 解析',
        [
            (
                'gui/server.py',
                '    blob = blob.replace("</", "<\\\\/")\n',
                '    blob = blob\n',
            ),
        ],
    ),
    (
        'M95',
        'GUI 的取数接口接受用户给的路径（而不是只按扫描出的 id 取）'
        '——那等于给一个**允许跑代码的本机服务**再开一个**任意文件读**的口子。'
        '而它看起来只是个"按路径取数据"的便利功能（"id 找不到就当成路径"）',
        [
            (
                'gui/agent_api.py',
                '    for e in discover(root):\n'
                '        if e.id == run_id:\n'
                '            return e\n'
                '    raise KeyError(f"没有这个运行 {run_id!r}（先调 /api/agent/runs 看清单）")\n',
                '    for e in discover(root):\n'
                '        if e.id == run_id:\n'
                '            return e\n'
                '    return RunEntry(id=run_id, label=run_id, source_dir="?",\n'
                '                    dec_path=Path(root) / run_id, eval_path=None)\n',
            ),
        ],
    ),
    (
        'M96',
        '「决策依据自报」在两处各写一份实现'
        '——`gui.agent_api` 与 `tw.eval_agent` 各写一遍，'
        '于是两边的字段不一样（一边多 `n_declared`），'
        '而 GUI 用一个、报告用另一个。'
        '本项目的老教训：**同一个量有多份实现，就一定会分叉**',
        [
            (
                'gui/agent_api.py',
                '    return _eval_agent.basis_distribution(recs)\n',
                '    counts: dict[str, int] = {}\n'
                '    for r in recs:\n'
                '        k = str(r.parsed.get("basis") or "")\n'
                '        if k not in BASIS_LABELS:\n'
                '            k = ""\n'
                '        counts[k] = counts.get(k, 0) + 1\n'
                '    return {"n": sum(counts.values()), "counts": counts,\n'
                '            "labels": BASIS_LABELS}\n',
            ),
        ],
    ),
    # ==================================================================
    # A6：多段配对度量（2026-09-21 第十二轮）
    # ==================================================================
    (
        'M97',
        '配对检验在 σ=0（差值完全一致）时仍走一般分支'
        '——`se = s/√n if s > 0 else nan` ⇒ 区间 (nan, nan) ⇒ '
        '`nan > 0` 与 `nan < 0` 都是 False ⇒ 落进「依然无法判定」。'
        '**方向正好相反**：一致性最强（完美一致的效应）的时候说"判不出来"。'
        '而成本倍数这类**确定性**效应正是在 σ≈0 时被观测到的'
        '——「把结论弄反」的静默错误',
        [
            (
                'tw/segmented.py',
                '    if s == 0.0:\n'
                '        se = 0.0\n'
                '        half = 0.0\n'
                '        lo = hi = m\n',
                '    if False:\n'
                '        se = 0.0\n'
                '        half = 0.0\n'
                '        lo = hi = m\n',
            ),
        ],
    ),
    (
        'M98',
        '多段跑批**跨段复用同一个账户**（账户提到段循环外面）'
        '——前一段的持仓会带进下一段 ⇒ 段与段不再独立，'
        '而"段独立"正是这个新度量效能高的**唯一来源**。'
        '症状很隐蔽：数字照样出，只是区间偏窄、更容易"判出显著"',
        [
            (
                'tw/segmented.py',
                '    def _one_segment(k: int, rng: tuple[int, int]\n'
                '                     ) -> tuple[tuple[int, int], dict[str, float],\n'
                '                                dict[str, RunResult]]:\n'
                '        s, e = rng\n',
                '    _hoisted = MarginAccount(cash=float(initial_cash), cfg=MarginConfig())\n'
                '\n'
                '    def _one_segment(k: int, rng: tuple[int, int]\n'
                '                     ) -> tuple[tuple[int, int], dict[str, float],\n'
                '                                dict[str, RunResult]]:\n'
                '        s, e = rng\n',
            ),
            (
                'tw/segmented.py',
                '            # \u26a0\ufe0f 每个 (段, 配置) 一个**新账户** \u2014\u2014 两层都不能共用\n'
                '            acc = MarginAccount(cash=float(initial_cash), cfg=MarginConfig())\n',
                '            acc = _hoisted\n',
            ),
        ],
    ),
    (
        'M100',
        '同一段内**各配置共用一个账户**（账户提到配置循环外面但仍在段内）'
        '——先跑的配置的成交会改变后跑的配置的权益 ⇒ **配对就配错了**：'
        'B 的成绩里混进了 A 的盈亏。'
        '⚠️ 这一条是 M98 第一次**漏网**时暴露出来的真测试缺口：'
        '我当时只检查了"跨段是否新建"，没检查"同段内各配置是否各用各的"',
        [
            (
                'tw/segmented.py',
                '        for name, mk in factories.items():\n'
                '            # \u26a0\ufe0f 每个 (段, 配置) 一个**新账户** \u2014\u2014 两层都不能共用\n'
                '            acc = MarginAccount(cash=float(initial_cash), cfg=MarginConfig())\n',
                '        _seg_acc = MarginAccount(cash=float(initial_cash), cfg=MarginConfig())\n'
                '        for name, mk in factories.items():\n'
                '            acc = _seg_acc\n',
            ),
        ],
    ),
    (
        'M99',
        '切段时第 0 段不留历史（从 0 开始而不是 min_history）'
        '——第 0 段的 `recent_closes` 会比别的段短 ⇒ **各段输入不等价**，'
        '配对的前提（"同一段上两个配置看到同样的东西"）在跨段意义上被破坏，'
        '而它不会被任何单段检查发现',
        [
            (
                'tw/segmented.py',
                '    lo = max(0, int(min_history)) + int(offset)\n',
                '    lo = 0\n',
            ),
        ],
    ),
    (
        'M101',
        '回填时把结果也写进 `visible_state`（或去掉"只动 outcome"的断言）'
        '——`outcome` 含**未来信息**（后续 h 根的涨跌）。一旦它渗进 ① 可见状态，'
        '**复盘学到的经验就会带着未来信息回流到决策里**，整条反馈链路失去意义。'
        '⚠️ 而症状是**收益看起来变好**（它在偷看答案）——这类错误必须机械拦住',
        [
            (
                'tw/outcome.py',
                '    digest_before = state_digest(rec.visible_state)\n',
                '    digest_before = ""\n',
            ),
        ],
    ),
    (
        'M102',
        'KPI 进度**在决策之后**才推进（而不是之前）'
        '——进度里就含了**当前这一根的结果**，而当前这根的结果在决策时'
        '还不知道 ⇒ **答案泄漏**。'
        '症状同样是"收益看起来变好"，且不报错',
        [
            (
                'tw/agent_run.py',
                '        kpi_state = None\n'
                '        if agent.config.kpi is not None:\n'
                '            kpi_state = advance(\n',
                '        kpi_state = None\n'
                '        if False:\n'
                '            kpi_state = advance(\n',
            ),
        ],
    ),
    (
        'M103',
        'KPI 去掉「在场率下限」（只留收益目标）'
        '——单指标一定会被刷（Goodhart）：**不交易就不会亏** ⇒ '
        '"亏了就装死"能刷分。这正是用户点名要防的两种退化之一。'
        '注意 `kpi_verdict` 仍然会跑、仍然会返回结论，只是结论失去了分辨力',
        [
            (
                'tw/kpi.py',
                '        "presence": (st.presence_so_far >= kpi.min_presence,\n'
                '                     st.presence_so_far, kpi.min_presence,\n'
                '                     "在场率 ≥ 下限（防躺平）"),\n',
                '        "presence": (True,\n'
                '                     st.presence_so_far, kpi.min_presence,\n'
                '                     "在场率 ≥ 下限（防躺平）"),\n',
            ),
        ],
    ),
    # ==================================================================
    # A6：复盘 / 归因 / 经验库（2026-09-21 第十二轮，续）
    # ==================================================================
    (
        'M104',
        '经验检索把时间边界从 `<` 放松成 `<=`'
        '——同一根 K 线内可能发生多次决策，**本轮刚生成的经验**'
        '就会喂给**本轮的决策**。'
        '而"复盘看得到结果、决策看不到"正是这套设计的地基：'
        '这一行松掉，未来信息经由"经验"回流到决策，'
        '而症状是**收益看起来变好**（它在偷看答案）',
        [
            (
                'tw/reflect.py',
                '        pool = [e for e in self._items if e.created_tick < int(at_tick)]\n',
                '        pool = [e for e in self._items if e.created_tick <= int(at_tick)]\n',
            ),
        ],
    ),
    (
        'M105',
        '检索结果**不排序**（按插入顺序）'
        '——同一条决策在不同运行/不同插入顺序下会看到**不同的经验**，'
        '于是"经验库为空 vs 只有未来条目"的 prompt 不再逐字节相同，'
        'V2（时间旅行安全）这条最强的等式就永远无法验证。'
        '⚠️ 症状不是报错，而是"验证通不过但查不出为什么"',
        [
            (
                'tw/reflect.py',
                '        pool.sort(key=lambda e: (e.created_tick, e.exp_id))\n',
                '',
            ),
        ],
    ),
    (
        'M106',
        '经验的时间戳采用**模型正文里给的 tick**（而不是外部时钟）'
        '——模型只要在复盘输出里写一个更早的 tick，'
        '就能让"自己刚说的话"在**同一时刻**被自己取回，'
        '整条时间边界被绕过。'
        '⚠️ 这是"把安全边界交给被约束方自己声明"的经典错误',
        [
            (
                'tw/reflect.py',
                '                               lesson=lesson, evidence_ids=ev),\n'
                '            created_tick=int(created_tick),\n',
                '                               lesson=lesson, evidence_ids=ev),\n'
                '            created_tick=int(e.get("created_tick", created_tick)),\n',
            ),
        ],
    ),
    (
        'M107',
        '归因解析把非法 `reason` **静默丢弃**（而不是落到 unclear 并计数）'
        '——被丢掉的条数不留痕，于是"分类分布"看起来比实际干净，'
        '而复盘质量却在下降。'
        '本项目对"静默丢弃"已有一贯判据：凡丢弃都要问「丢的是不是我要的数据」',
        [
            (
                'tw/reflect.py',
                '        if reason not in ATTR_REASONS:\n'
                '            res["n_bad_reason"] += 1\n'
                '            reason = "unclear"\n',
                '        if reason not in ATTR_REASONS:\n'
                '            continue\n',
            ),
        ],
    ),
    (
        'M108',
        '规则归因里 `timing_wrong` 的判据吞掉 `direction_wrong`'
        '（把"曾经有利过"这一条放宽成**恒真**）'
        '——于是"方向看反了"这一类永远判不出来，'
        '`direction_wrong` 变成**不可达代码**：不报错、只是永不命中。'
        '⭐ 这正是我第一版真的写出来的 bug 的等价形态'
        '（我当时用 `markout_h` 与 `signed_move_h` 比符号，而前者已带方向 ⇒ 条件恒假）',
        [
            (
                'tw/reflect.py',
                '    if mfe_f is not None and mfe_f > _COST_EPS:\n',
                '    if True:\n',
            ),
        ],
    ),
    (
        'M109',
        '复盘解析**关掉抢救路径**（输出被截断时直接返回空）'
        '——`max_tokens` 不够是**常态**（实测 24 条归因在 1365 字符处硬截断），'
        '而"截断了"≠"全都不能用"：前面那些条目是完整的。'
        '关掉抢救 ⇒ 16 次复盘「解析成功 0」、归因整批丢失，'
        '而错误信息只写「JSON 解析失败」，**看不出是截断还是模型乱答**',
        [
            (
                'tw/reflect.py',
                '            attrs, exps = _repair_truncated(text)\n'
                '            if attrs or exps:\n',
                '            attrs, exps = [], []\n'
                '            if attrs or exps:\n',
            ),
        ],
    ),
    (
        'M110',
        '`v6` 退化成 `v4`（考核目标段没插进去）'
        '——锚点失效时 `str.replace` 会**静默返回原串**，'
        '于是"单一变量对照"凭空消失，而实验照跑照出数字。'
        '⭐ 本变异体的**看门人**是 `tw/prompts.py` 里 import 期的 assert；'
        '它会在导包时就炸 ⇒ 整份测试变红。'
        '（⚠️ 信号是"崩了"而不是"断言失败"——这仍然是**被抓住了**，'
        '但报告里要写清是哪一种，不能混为一谈。）',
        [
            (
                'tw/prompts.py',
                '_USER_V6 = _USER_V4.replace(_V6_ANCHOR, '
                '_KPI_SECTION + _V6_ANCHOR, 1)\n',
                '_USER_V6 = _USER_V4\n',
            ),
        ],
    ),
    (
        'M111',
        '经验的时间戳从「窗口最后一根 **+ horizon**」退化成「窗口最后一根」'
        '——复盘用到的 `outcome` 延伸到 `last + horizon`，'
        '而经验在 `last + 1` 就可被检索 ⇒ **凭空多出 `horizon − 1` 根前视**'
        '（实测 horizon=4 ⇒ 多 3 根）。'
        '症状：`v7`（带经验）比 `v4` 多"看"了未来 ⇒ **收益看起来变好**，'
        '而且**不报错**。这是"回填延迟 = 复盘可见边界"这条纪律的直接违反',
        [
            (
                'tw/reflect.py',
                '    return max(ticks) + int(horizon)\n',
                '    return max(ticks)\n',
            ),
        ],
    ),
    (
        'M112',
        '`decision_id` 里的经验标记退化成**只拼存在性**（不拼内容哈希）'
        '——于是「空经验库」与「有 1 条可见经验」在同一 run/tick/可见状态下'
        '算出**同一个 ID**，而两者的 prompt **内容不同** ⇒ '
        '留痕里两条"长得一样"、A/B 归因与回放核对同时失效。'
        '⭐ 这正是 KPI 那条教训（模板要唯一标识**实际发出去的** prompt）的复现',
        [
            (
                'tw/agent.py',
                '                _sig = _h.sha256(_el(_exps).encode("utf-8"))'
                '.hexdigest()[:6]\n'
                '                prompt_label = f"{prompt_label}#exp{_sig}"\n',
                '                prompt_label = f"{prompt_label}#exp"\n',
            ),
        ],
    ),
    (
        'M113',
        '经验检索把 `k` 截断去掉（返回**全部**历史经验，而不是最近 k 条）'
        '——段数一多，prompt 会被几十条经验撑爆，'
        '而更隐蔽的后果是：**经验的时间边界还在，但"最近优先"没了**'
        '（旧经验与新经验混在一起），于是测到的是"长记忆"而不是"经验有用"',
        [
            (
                'tw/reflect.py',
                '        return pool[-k:] if k > 0 else []\n',
                '        return pool if k > 0 else []\n',
            ),
        ],
    ),
    (
        'M114',
        '解析层对 `condition` **类型不符时直接强转**（`dict(e.get("condition") or {})`）'
        '——模型把它写成字符串/列表时 `dict("abc")` 会当成键值对序列去解包 ⇒ '
        '抛 `ValueError` ⇒ **整份复盘崩掉**（不是丢一条，是整个 run 死）。'
        '实测于 v5 的 L=50 复盘。'
        '⚠️ 判据要分清：**"解析层宽容"≠"帮模型改数"**——'
        '修正数值是凭空造内容（禁止），类型不符就置空只是不采信（必须）',
        [
            (
                'tw/reflect.py',
                '            "condition": safe_condition(e.get("condition")),\n',
                '            "condition": dict(e.get("condition") or {}),\n',
            ),
        ],
    ),
    (
        'M115',
        '收益目标**退回固定值**（不用基线的净收益分位标定）'
        '——于是它不随窗口长度缩放：L=8 时要求 8 根赚 1% ⇒ '
        '**48/48 段全部 `return` 失败**，模型每一次都被告知"你还差得远"。'
        '⚠️ 症状是"实验照跑照出数字"，而那批数据实际测的是'
        '「**目标不可达**时它会怎么做」，不是「给目标好不好」。'
        '⭐ 这是被 KPI 判定数据自己暴露的：一个"目标"若 100% 不达标，'
        '那就不是目标，是噪音',
        [
            (
                'tw/kpi.py',
                '        out["target_return"] = tgt\n',
                '        out["target_return"] = 0.01\n',
            ),
        ],
    ),
    (
        'M116',
        '分段落偏移**去掉 `[0, seg_len)` 的守卫**——于是 `offset = seg_len` '
        '这类取值会被接受，而它只是把**同一组窗口整体平移**（丢掉开头几段），'
        '**不是"换一组窗口"** ⇒ 项目会以为做了稳健性检验，实际上什么都没验。'
        '⚠️ 这类"看起来验了、其实没验"的假稳健性，比不做检验更危险——'
        '因为它会让人放心地下结论',
        [
            (
                'tw/segmented.py',
                '    if not (0 <= int(offset) < int(seg_len)):\n',
                '    if False:\n',
            ),
        ],
    ),
    (
        'M117',
        '静态检查从 `compile()` **退回 `ast.parse()`**——'
        '`ast.parse` 只建 AST、**不做编译期语义检查** ⇒ '
        '「关键字参数重复」（`f(a=1, a=2)`）这类错误它能**静默通过**。'
        '实测后果：`scripts/agent_review.py` 里重复写了一个 `min_history=` '
        '⇒ **全量 1503 项测试全绿、静态检查也全绿**，而那个脚本**一跑就崩**'
        '（在流水线里跑了两小时才发现）。'
        '⭐ 判据：**「能建 AST」≠「能执行」**——'
        '检查「能不能跑」就要用那个「真正会执行的编译器」',
        [
            (
                'scripts/selfcheck.py',
                '            compile(src, str(p), "exec")\n',
                '            pass\n',
            ),
        ],
    ),
    (
        'M118',
        '异质性检验的临界值写小一个数量级（χ²_{0.95}(df=1) 用 0.384 而不是 3.84）'
        '——于是**几乎什么都判成「异质」**。'
        '⚠️ 后果不是报错，而是**把"噪声大"误判成"真值随组变"** ⇒ '
        '结论从「该补样本」翻成「补多少都没用、必须按组报」，'
        '而两个方向的下一步**正好相反**。'
        '⭐ 这类"判据宽严写错"的错，症状只是**结论更保守**，很容易被当成"我够严谨"',
        [
            (
                'scripts/a9_power_ablation.py',
                '_CHI2_95 = {1: 3.84, 2: 5.99, 3: 7.81, 4: 9.49, 5: 11.07,',
                '_CHI2_95 = {1: 0.384, 2: 0.599, 3: 0.781, 4: 0.949, 5: 1.107,',
            ),
        ],
    ),
    (
        'M119',
        '调用失败率守卫**永远不触发**（`rate > max_rate` 改成 `if False`）'
        '——于是额度耗尽（HTTP 429）时，`on_exhausted="hold"` 把失败决策'
        '回退成一个**合法的「弃权」**，脚本照常跑完、照常出数字，'
        '而在场率/换手全变 0 **看起来像"模型很保守"**。'
        '⭐ 实测踩过：`v6_k300o4` 那一臂 **1030/1336（77%）调用是 429**，'
        '只有 306 条是模型真实决策。'
        '判据：**「跑完了」≠「数据是用模型跑出来的」**——'
        '外部依赖失败会退化成合法输出的地方，失败率必须是**会失败的断言**',
        [
            (
                'scripts/_common.py',
                '    if rate > max_rate:\n',
                '    if False:\n',
            ),
        ],
    ),
    (
        'M120',
        'σ=0 被当成"输入不合法/信息不足"（`sd <= 0 ⇒ nan`）'
        '——而 **σ=0 是信息最强的情形**：每段差值一模一样 ⇒ '
        'se=0、t=±inf ⇒ 任意小的效应都判得出。'
        '报成 nan/None 会被报告印成「**判不出来**」，**方向正好反了**。'
        '⚠️ 同一个错 A6 在 `paired_verdict` 里修过一次，'
        '但漏了 `min_detectable_effect` 与 `power_analysis` 两个兄弟函数；'
        'A10 的标定测试才把它照出来。'
        '⭐ 判据：**"最强证据"与"最弱证据"看起来一样（都是空/未定义）**，'
        '所以必须专门测 σ=0 这一档',
        [
            (
                'tw/segmented.py',
                '    if sd != sd or sd < 0 or n_segs < 2:\n',
                '    if sd != sd or sd <= 0 or n_segs < 2:\n',
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
    #   本项目为此栽过三次（产物依赖 / 编号账本依赖 / M 线观测值依赖），
    #   前两次的修法都是"给测试加 skip"；这次改成**从根上补上目录**，
    #   让那些测试能真正跑（而不是被跳过）——跳过等于没测。
    for sub in sorted(pkgs) + ["data", "docs"]:
        src_dir = ROOT / sub
        if src_dir.is_dir():
            shutil.copytree(src_dir, work_dir / sub, dirs_exist_ok=True)
    # ⚠️ ``out/`` 整体约 500MB（含变异沙箱自身、快照、日志），不能复制；
    #   但**顶层的 *.json 产物**（各阶段 metrics、workstream_*、诊断结果，
    #   合计十几 MB）必须复制——很多测试要读它们，缺了就是基线红。
    #   子目录（_mutation/ 、_archive/、figs/、repro_logs/）仍然不复制。
    out_src = ROOT / "out"
    if out_src.is_dir():
        out_dst = work_dir / "out"
        out_dst.mkdir(parents=True, exist_ok=True)
        for j in sorted(out_src.glob("*.json")):
            try:
                shutil.copy2(j, out_dst / j.name)
            except OSError:
                pass


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
