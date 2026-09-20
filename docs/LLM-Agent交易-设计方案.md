# LLM 交易 Agent + OKX 风格订单：设计方案

> 需求（用户原话）：「接入 agent 的 api，比如 agnes，然后让这个 ai 去交易、
> 去做决策、去下单，然后记录下单的理由原因，入场价格、出场价格等等的订单信息，
> 然后用于数据分析验证；然后交易的环境我希望按照真实的 OKX 交易所的情况进行下单，
> 比如杠杆啊、价格、止盈止损等等订单信息」。
>
> 本文是**先查真实 OKX 的订单模型、再查这个领域公认的评估标准、然后对项目现状**的结论。

---

## 0. 先说一件必须先定的事

**「按照真实 OKX 的情况下单」有两条完全不同的路**：

| | A. 模拟（用 OKX 的规则，不连网） | B. 真实（连 OKX 下单） |
|---|---|---|
| 撮合 | 本项目引擎，但**用 OKX 的订单语义**（杠杆/保证金/强平/TP-SL） | OKX 服务器 |
| 钱 | 无风险 | **真钱** |
| 可复现 | ✅ 完全（同种子可重放） | ❌ 不可复现 |
| 能做的事 | 所有研究、对照、留痕、评估 | 只能做"最终验证" |

**建议：先做 A**。理由不是保守，而是——
搜索到的这个领域的综述（arXiv 2605.19337）统计了 **77 篇** LLM 交易 Agent 研究，
其中只有 **19 篇**达到"有动作输出 + 闭环评估"的最低标准，而这 19 篇里
**只有 2 篇**报告了时间一致的划分协议、**各只有 1 篇**报告了明确的交易成本模型
与标的池处理，**没有任何一篇**达到可复现级别。

⇒ 这个领域当前的主要状态是「**回测表演**」。在把真钱接上之前，
先把"A 这条路"的证据链做扎实，是唯一不亏钱的做法。

**B 可以后置**：A 的订单模型如果按 OKX 语义实现，将来换成真实网关是**替换一个执行器**的事。

---

## 0.5 实测：Agnes 到底能不能用（3 次调用）

在写方案之前先跑了一次连通性自检（`scripts/probe_llm.py`），**结果决定了方案的形状**：

| 次数 | action | TP | SL | 延迟 |
|---|---|---|---|---|
| 1 | sell 0.15 | 102.9 | 101.8 | 2.26s |
| 2 | hold | **10350** ⚠️ | **10180** ⚠️ | 6.01s |
| 3 | hold | 103.5 | 101.95 | 1.59s |

**结论**：

✅ **可以用**：3/3 调用成功，直接给出合法 JSON，**延迟中位 2.26s**
（最慢 6.01s）⇒ 只适合「每个决策窗口一次」，不适合高频。
理由质量也不错（会提趋势、浮盈、保证金率、资金费率）。

⚠️ **但第 2 次的止盈止损量级错了 100 倍**（`10350` 而不是 `103.5`）。
如果不过校验就直接下单，TP 挂在 100 倍远的地方、SL 永远不会触发。
**这不是模型的偶发 bug，而是"LLM 不该决定数值"的实证**——
它印证了搜索里的判断：「LLM 可能擅长总结宏观，却是个糟糕的仓位引擎」。

⚠️ **三次的决策并不一致**（1 次 sell、2 次 hold，尽管 temperature=0.2）。
所以**决策稳定性本身**也要作为指标记录（同一上下文重复问，
结论是否稳定；不稳定的话，"这一次的收益"有多少是运气）。

⇒ **这两条直接决定了本方案的两条硬约束**：
1. **风控层不是可选项**：所有数值（价格、数量、杠杆）必须过**合理性校验**
   （TP/SL 必须在 mid 的合理区间内、size 不得超过上限），
   校验不过就**退回 hold 并记录原因**；
2. **同上下文要重复采样**：记录 `n_samples` 与一致性，
   不一致时按保守规则（如取多数票或不动作）。

---

## 1. 真实 OKX 的订单模型（查证结果）

### 下单参数（`POST /api/v5/trade/order`）

| 参数 | 取值 | 我们的现状 |
|---|---|---|
| `instId` | 如 `BTC-USDT-SWAP` | 有品种概念，需补 |
| `tdMode` | `cash` / `cross`（全仓）/ `isolated`（逐仓） | **缺** |
| `side` | `buy` / `sell` | ✅ |
| `ordType` | `market` / `limit` / `post_only` / `fok` / `ioc` | 只有 limit/market，**缺后三种** |
| `sz` / `px` | 数量 / 价格 | ✅ |
| `posSide` | `long` / `short` / `net` | **缺** |
| `reduceOnly` | 只减仓 | **缺** |
| `clOrdId` | 客户端订单 ID（≤32 字符） | 有 `order_id`，可对齐 |
| `attachAlgoOrds` | **随主单挂止盈止损** | **缺**（核心） |

### 止盈止损（`attachAlgoOrds` 里）

| 参数 | 说明 |
|---|---|
| `tpTriggerPx` / `slTriggerPx` | 触发价 |
| `tpOrdPx` / `slOrdPx` | 委托价，**`-1` 表示触发后走市价** |
| `tpTriggerPxType` / `slTriggerPxType` | 触发价取哪个：`last` / `index` / **`mark`** |
| `tpTriggerRatio` / `slTriggerRatio` | 按**比例**触发（仅 FUTURES/SWAP） |
| `attachAlgoClOrdId` | 附加单自己的 ID |

⚠️ **`mark`（标记价）这一项很重要**：真实交易所用标记价触发强平与止损，
就是为了防"最后成交价被一笔插针打穿"——这与本项目"锚点必须用同时点的真实盘口价"
那条纪律是同一种思路（别用会被单笔成交污染的价）。

### 算法单（`/api/v5/trade/order-algo`）

七种：`conditional`、`oco`（一撤一）、`trigger`、**`move_order_stop`（移动止损，
带 `callbackRatio`/`callbackSpread`/`activePx`）**、`iceberg`（冰山）、`twap`、`chase`。

### 持仓与保证金

- `GET /api/v5/account/positions` 返回：`pos`、**`avgPx`（入场均价）**、`markPx`、
  `upl`（未实现盈亏）、`imr`/`mmr`（初始/维持保证金）、`lever`
- `setLeverage(instId, lever, mgnMode)` 调杠杆
- 保证金模式：`cross` / `isolated`

---

## 2. LLM 交易 Agent 的评估标准（查证结果）

### ⭐ 最小证据链（六项，缺一项这个评估就是"表演"）

1. 决策时**能看到什么**信息
2. Agent **相信什么**、引用了什么
3. 建议的**动作**
4. 这个建议**是否真的变成了交易**（还是被风控拒了 / 变成观察 / 不动）
5. 什么规则**接受 / 拒绝 / 改了量**
6. 扣掉成本、滑点、风控限制后的**结果**

### 决策日志必须含

输入时间戳 · 数据来源 · **提示词模板版本** · **模型与版本** · 工具调用 ·
检索到的内容 · 建议 + **置信度** + **弃权理由** · 事后施加的风控约束 ·
最终动作与组合状态 · 在**事先声明**的时间窗上的结果

### 三层必须分开评

**预测**（估收益/波动）· **决策**（定方向/仓位/弃权）· **执行**（订单类型/时机/成本）。
「Agent 可能在一层有用、在另一层有害」——混在一起评就看不出来。

### 防"Alpha 幻觉"的四条硬控制

| 控制 | 做法 |
|---|---|
| **让泄漏难发生，而不只是禁止** | point-in-time 上下文（只取 `tick ≤ t`）+ **no-internet 回放模式**（禁外部检索，否则模型会读到答案） |
| **成本模型事先定义** | 手续费/滑点/资金费写死在配置里；**在漂亮回测之后再加成本，是边缘系统活太久的原因** |
| **弃权也要评分** | `no-op` 是决策，要记理由；好系统知道何时不交易 |
| **成本敏感性** | 报告成本 ×2、×5 后的表现 |

### 必报的量

毛收益与**净收益** · turnover · 手续费/价差/滑点假设 · 平均持仓期 ·
**交易数与「不交易」数** · 最大回撤

---

## 3. 项目现状对照

### 已有的（比预想多）

| 已有 | 位置 | 说明 |
|---|---|---|
| **永续合约市场** | `tw/perpetual.py` | **含资金费率结算机制** ✅ |
| 撮合引擎 | `tw/engine.py`、`tw/book.py` | 限价/市价、部分成交 |
| 策略基类 | `tw/strategy.py` | `on_tick / on_start / on_fill / on_end` |
| 决策上下文 | `StrategyContext` | tick / mid / bid / ask / spread / **fundamental** / last / lag / momentum |
| 评估工具 | `tw/eval.py` | **markout（逆向选择）**、PnL 三分解、库存、回撤 |
| **对照基准** | `strategies/noop.py`、`random_taker.py` | ⭐ **现成的"不动手"与"乱来"两个基准** |
| 场景与多种子 | `tw/scenarios.py`、`run_stage3.SEEDS` | 5 场景 × 多种子 |
| 测量纪律 | README 十条 | ⭐ **与搜索到的评估标准高度重合** |

### 缺的

| 缺口 | 影响 |
|---|---|
| **订单模型没有 OKX 语义** | 无杠杆/保证金/强平/TP-SL/`reduceOnly`/`posSide` |
| **没有持仓与账户账本** | 无法算保证金率、强平价、未实现盈亏 |
| **没有决策留痕层** | 这是整个需求的核心，也是现在完全没有的 |
| **没有 LLM 接入层** | 无 prompt 模板、无解析、无版本化 |
| **没有风控闸门** | 无头寸规模、无合规检查、无熔断 |
| **`StrategyContext` 看不到盘口深度** | LLM 决策最需要的信息之一 |

⭐ **一个重要的好消息**：项目那十条测量纪律（配对、同种子、
标定窗口=实验窗口、先算 CI 再报差、方向反转走区间重叠检验…）
**几乎逐条对上了**搜索里这个领域公认的评估标准。
也就是说：**这个项目已有的方法论，比这个领域大多数论文还严**。
缺的不是"怎么评"，是"要评的那个 LLM Agent 还不存在"。

---

## 4. 建议架构（六层）

```
⓪ 数据层（新增，见 §4.5）——「真实 / 历史 / 自生成」三源，统一进 SQLite
   └─ 三种来源、一个 schema；按需拉取；缓存优先、联网兜底

① 数据层（Point-in-time，防泄漏的第一道）
   └─ 市场快照：盘口深度 / 近期成交 / 资金费率 / 标记价
      严格只取 tick ≤ t 的数据；上下文构造器带 hash 便于事后核对

② 决策层（LLM）
   ├─ Prompt 模板（**版本化存储**，改模板要留痕）
   ├─ 上下文构造器（把 ① 变成模型能读的文本，含"你能看到什么"的显式声明）
   └─ 输出解析器（结构化，容错解析 + 解析失败也留痕）

③ 风控层（LLM 说了不算）
   ├─ 头寸规模（按波动率与保证金，不是模型报的 sz）
   ├─ 合规检查（杠杆上限 / 单品种上限 / 回撤熔断 / 只减仓模式）
   └─ 人工否决开关（GUI 上一个按钮）

④ 执行层（OKX 语义）
   ├─ 订单模型：tdMode / posSide / reduceOnly / attachAlgoOrds / clOrdId
   ├─ 账户账本：保证金、杠杆、强平价、未实现盈亏、资金费
   └─ 撮合：复用现有引擎，外加触发单（TP/SL）的 tick 级检查

⑤ 留痕层（⭐ 需求的中心）
   └─ 每个决策一条完整记录（见下面 §5），落 JSONL，可回放
```

### 关键设计决策

- **LLM 只做"决策"，不做"算数"**：仓位、保证金、强平价全由确定性代码算。
  （搜索里明确指出：**LLM 可能擅长总结宏观、却是个糟糕的仓位引擎**。）
- **弃权是一等公民**：`no-op` 与买卖一样进留痕，且评估时要单独看弃权质量。
- **回放模式默认关外部访问**：LLM 只能看我们给的上下文——
  这是防"读到答案"的唯一可靠做法（不是靠嘱咐它别上网）。
- **每次决策都记 prompt 模板版本与模型名**：否则半年后不知道哪版的成绩。

---

## 4.5 数据层（2026-09-20 用户新增需求）

> 用户原话：*「实时拉取 OKX 或其他交易所数据…建立一个数据库 sqlite…
> 不必所有币种都拉，在选择准备交易的时候才去拉…也可以下载历史数据，
> 还有自己生成数据，这样子 agent 可以利用不同来源的数据进行自我测试、真实数据测试。」*

### 三种数据来源，一个 schema

| 来源 | 用途 | 获取方式 | 关键属性 |
|---|---|---|---|
| `live` | **真实盘面**（当前） | OKX 公开端点，按需拉 | 有网络依赖；**不可复现**（数据会变） |
| `history` | **真实历史**（回测） | OKX 历史端点，一次拉完存库 | 落库后**可复现**；有 3 个月/3 年上限 |
| `synthetic` | **自生成**（压力测试） | 复用本项目 ABM 引擎 | **完全可复现**；已知 ground truth |

⭐ **三者用同一张表**，只多一列 `source`。这是刻意的设计：
Agent 的**同一套策略代码**应该在三源上跑出可对比的结果；
如果三源走三套接口，那么「策略在真实数据上表现不同」到底是策略的问题
还是接口口径的问题，就分不清了——**这正是本项目「同一个量不要有两条路径」
那条纪律的同一形态**。

### SQLite 库结构（草案）

```sql
-- 行情：三源共用。source + inst_id + bar + ts 是唯一键
CREATE TABLE candles (
  source TEXT NOT NULL,          -- 'okx' | 'synthetic'
  inst_id TEXT NOT NULL,         -- 'BTC-USDT' / 'BTC-USDT-SWAP'
  bar TEXT NOT NULL,             -- '1m' '15m' '1H' '1D'
  ts INTEGER NOT NULL,           -- 毫秒（开盘时间，与 OKX 一致）
  open REAL, high REAL, low REAL, close REAL, volume REAL,
  confirm INTEGER DEFAULT 1,     -- OKX 的 confirm 字段：0=未走完
  fetched_at INTEGER,            -- 本地写入时刻（区分「数据时间」与「拿到时间」）
  PRIMARY KEY (source, inst_id, bar, ts)
);

-- 快照类（无需 OHLC 的量）：资金费率 / 未平仓量 / 标记价
CREATE TABLE metrics (
  source TEXT, inst_id TEXT, kind TEXT,   -- 'funding_rate'|'open_interest'|'mark_price'
  ts INTEGER, value REAL, extra TEXT,      -- extra 存 JSON（如 nextFundingTime）
  PRIMARY KEY (source, inst_id, kind, ts)
);

-- 盘口快照（实时拉取时选存；全存会很占空间）
CREATE TABLE books (
  source TEXT, inst_id TEXT, ts INTEGER, sz INTEGER,
  bids TEXT, asks TEXT,        -- JSON: [[px, sz, n], ...]
  PRIMARY KEY (source, inst_id, ts, sz)
);

-- ⭐ 拉取账本：回答「这份数据是什么时候、用什么参数拉下来的」
CREATE TABLE fetches (
  id INTEGER PRIMARY KEY,
  source TEXT, inst_id TEXT, bar TEXT,
  from_ts INTEGER, to_ts INTEGER, n_rows INTEGER,
  started_at INTEGER, finished_at INTEGER,
  endpoint TEXT,               -- 哪个端点（candles vs history-candles）
  ok INTEGER, error TEXT, pages INTEGER
);
```

**为什么 `fetches` 表不是多余的**：本项目最贵的一课是「结果对 ≠ 有能力知道
结果对不对」。行情库也一样——**没有拉取账本，半年后没人知道某根 K 线是
从哪个端点、什么参数拿的**，而 OKX 的 `candles`（热缓存，近 3 个月）与
`history-candles`（冷存储，上限 100 条）**可能返回不同的修正后数据**。

### 按需拉取（用户明确要求）

**不预下载全市场**。触发条件是「某标的一次即将开始的交易会话」：

```
用户/Agent 说「我要交易 BTC-USDT」
   ↓
① 查库：这个标的有多少数据？覆盖到什么时间？
   ↓ 不足
② 拉：candles 补近端 + history-candles 倒推补远端（分页）
   ↓
③ 存库（幂等 UPSERT），写 fetches 账本
   ↓
④ 会话开始了，之后全程只读库（决策路径不碰网络）
```

⭐ **第 ④ 步是硬约束，不是优化**：决策期间联网 = 引入不可复现性 +
可能读到未来数据。**库是决策路径的唯一数据源。**

### OKX 公开端点（查证结果，全部**无需鉴权**）

| 端点 | 用途 | 单次上限 | 限频 |
|---|---|---|---|
| `/api/v5/market/candles` | 近期 K 线（**近 3 个月**，热缓存） | **300** | 40 次/2s |
| `/api/v5/market/history-candles` | 历史 K 线（**主流币约 3 年**，冷存储） | **100** | 20 次/2s |
| `/api/v5/market/books` | 盘口深度 | `sz` 5/10/20/50/100/400 档 | — |
| `/api/v5/market/trades` | 最近成交 | — | — |
| `/api/v5/public/funding-rate` | 资金费率（**仅永续**） | — | 20 次/2s |
| `/api/v5/public/open-interest` | 未平仓量 | — | 20 次/2s |
| `/api/v5/public/mark-price` | **标记价**（强平计算基准） | — | — |
| `/api/v5/public/instruments` | 合约规格（tick/lot/最小下单） | — | 10 次/2s |

⚠️ **两个坑（查证时发现，不踩第二次）**：

1. **分页参数命名反直觉**：`after` 传时间戳返回的是**比它更早**的数据、
   `before` 返回**更新**的。倒推历史要用 `after` 翻页，且两者**不能同时传**。
   翻页时取上批最旧一条 `ts − 1ms` 作为下一次 `after`——
   **减 1ms 是必须的**，否则边界那根会出现在两批里（重复写入会被主键挡掉，
   但会污染 `n_rows` 统计）。
2. **`confirm` 字段**：`'0'` 表示这根 K 线还没走完、价格仍在跳。
   **决策时必须只用 `confirm='1'` 的根**，否则就是读未来的数据
   （与项目「路径覆盖事件全程但不越界」同源）。

### 自生成数据（用户明确要求「自己生成数据」）

**这是本项目最有优势的一块——引擎已经在那儿了。**

价值不在于「造更多数据」，而在于造**已知 ground truth 的对照**：

| 生成方式 | 能验证什么 | 为什么真实数据做不到 |
|---|---|---|
| 改 `scenarios.py` 参数 | 策略在极端行情下的行为 | 真实数据里很少出现 liquidation 场景 |
| 注入已知冲击 | Agent 能否识别并正确反应 | 真实市场的事件归因本身有内生性问题 |
| **同种子重跑** | Agent 决策的可复现性 | 真实数据只有一条路径，无法重放 |
| 调主体构成（零智能/基本面/图表派比例） | 策略对市场微结构的敏感性 | 无法控制真实市场的主体构成 |

⭐ **最关键的一条**：自生成数据能让「**Agent 的收益是本事还是行情**」这个问题
有一个**答案已知的对照组**——因为我们可以造出「基本面随机游走、
完全没有可预测性」的市场，此时任何正收益都只能是运气。
**这类对照在真实数据上永远做不了。**

### 与测量纪律的接口

- **`live` 数据进库后就不再更新**（同一 `(source,inst,bar,ts)` 不覆盖）
  → 保证「当时的决策」事后可复现，哪怕交易所之后修正了历史
- 每条决策留痕里记 `data_snapshot_id`（库里数据的最大 ts + 拉取时间）
  → 半年后能确认「那次决策看到的是哪一版数据」
- **`synthetic` 源必须记构造参数**（种子 + 场景名），否则无法重造

---

## 5. 决策留痕的结构（需求的中心）

一条记录 = 一次决策的完整因果链：

```jsonc
{
  "decision_id": "sha256(tick|role|params|time)",   // 确定性 ID，可复现
  "run_id": "…", "tick": 3120, "wall_ms": 1758...,
  "context_hash": "…",            // 当时可见数据的哈希（防事后篡改）
  "prompt_template": "v1.2",      // ⭐ 模板版本
  "model": "agnes-2.5-flash", "model_params": {"temperature": 0.2},
  "visible_state": {              // ⭐「决策时能看到什么」
    "mid": 102.44, "spread_bp": 8.1, "mark": 102.5,
    "depth_bid": [...], "depth_ask": [...],
    "funding_rate": 0.0001, "inventory": 0.3, "margin_ratio": 0.42,
    "recent_closes": [...]
  },
  "llm_raw": "……模型原始输出，不截断",     // 解析失败也要留
  "parsed": {                              // 建议 + 置信度 + 理由
    "action": "buy", "sz": 0.5, "px": null, "ordType": "limit",
    "tp": 103.5, "sl": 101.2, "confidence": 0.72,
    "reason": "……自然语言理由（用户明确要求记录的）"
  },
  "risk": {                                // ⭐ 什么规则改/拒了它
    "accepted": true, "resized_to": 0.35, "rule": "vol_target",
    "notes": ["杠杆 3x ≤ 上限 5x", "单品种敞口 12% ≤ 20%"]
  },
  "order": {                               // ⭐ 最终落成的订单（OKX 语义）
    "tdMode": "isolated", "posSide": "long", "reduceOnly": false,
    "attachAlgoOrds": [{"tpTriggerPx": 103.5, "tpOrdPx": -1,
                        "slTriggerPx": 101.2, "slOrdPx": -1,
                        "tpTriggerPxType": "mark", "slTriggerPxType": "mark"}],
    "clOrdId": "tw-3120-01"
  },
  "outcome": { ... }   // 事后回填：成交价、滑点、h 后 markout、平仓盈亏
}
```

**「入场价 / 出场价」在 OKX 语义下不是订单上的字段，而是持仓账本的产物**：
- 入场价 = 该持仓的 `avgPx`
- 出场价 = 平仓成交价
⇒ 所以记录要**关联到持仓**，而不是只看单笔订单。

---

## 6. 评估体系（分层，且必须与基准对照）

| 层 | 指标 | 对照 |
|---|---|---|
| **决策层** | 弃权率、置信度校准（说 0.7 的是不是真 70% 对）、理由可用性 | — |
| **执行层** | 实际滑点 vs 预期、成交率、下单延迟 | — |
| **结果层（毛）** | 收益、Sharpe、最大回撤 | — |
| **结果层（净）** | **扣手续费+滑点+资金费**后的收益 | ⭐ **与 `noop` 对照** |
| **成本敏感性** | 成本 ×2 / ×5 后还活着吗 | — |
| **稳健性** | 跨 5 个场景 × 多种子 | 同种子配对 |

⭐ **最关键的一条**：**LLM 必须显著优于 `noop`（什么都不做）**。
项目里 `noop` 的 PnL 恰好是 0，`random_taker` 是稳定亏损——
这两条基准线**已经现成**。如果 Agent 打不过 noop，
那它做的所有"分析"都只是在给自己找理由交易。

---

## 6.5 MCP 服务（2026-09-20 用户新增需求）

> 用户原话：*「还有也要做 MCP，方便 agent 调用」*

### 先分清「谁的 Agent」

这一节有个容易混的地方，必须写清楚：

| 场景 | 谁是 MCP 的「客户端」 | 什么在调工具 |
|---|---|---|
| **本项目的 LLM 交易 Agent** | 项目自己 | ❌ **不需要 MCP**——项目直接 import `tw/*` 更快更可控 |
| **外部 Agent**（Claude Desktop / Cursor / 其他） | 那些宿主 | ✅ **需要 MCP** |

⇒ **MCP 的价值是「让外部 Agent 能操作这个项目」**，不是给内部 Agent 用。
内部 Agent 走 Python 函数调用；MCP 是**对外的一扇门**。

### 三种消费者（各自的用法不同）

1. **外部 AI 助手**（Claude/Cursor）→ 让它帮你查行情、看实验、跑对照
2. **本项目自己的开发**（我）→ 用 MCP 校验接口设计是否自洽
3. **未来的多 Agent 协作**（§7 的 C 路线）→ Agent 之间通过 MCP 共享能力

### 工具设计（按「读写分离 + 危险度分级」）

⚠️ 关键安全原则：**能读的和能下单的不能是同一批工具**，
且**能下单的工具默认不启用**。

**只读类（默认启用，无副作用）**

| 工具 | 参数 | 返回 |
|---|---|---|
| `list_instruments` | `source`（okx/synthetic） | 库里有数据的标的 + 时间范围 |
| `get_candles` | `inst_id, bar, from_ts, to_ts, limit` | OHLCV 数组（**列式**，省 token） |
| `get_market_snapshot` | `inst_id` | 当前 bid/ask/mid/spread/资金费率/标记价 |
| `list_scenarios` | — | 5 个场景名 + 描述 |
| `list_strategies` | — | 可用策略名 |
| `get_run_summary` | `run_id` | 该次实验的指标汇总 |
| `get_decision_log` | `run_id, limit` | 决策留痕（含理由、入场出场价） |
| `compare_runs` | `run_id_a, run_id_b` | 两两对照 + **区间重叠判定**（复用第十条纪律） |

**写入类（需显式开启）**

| 工具 | 说明 | 护栏 |
|---|---|---|
| `fetch_data` | 按需拉 OKX 数据入库 | 只允许白名单 `inst_id`；限频；写 `fetches` 账本 |
| `run_backtest` | 跑一次实验 | 上限 tick 数（复用 `MAX_LAB_TICKS` 思路） |

**交易类（⭐ 默认关闭，`TW_MCP_ALLOW_TRADE=1` 才启）**

| 工具 | 说明 |
|---|---|
| `place_order` | 下单（A 路线 = 写进模拟账本；B 路线 = 真钱） |
| `cancel_order` / `get_positions` | — |

⚠️ **A 路线（模拟）下这些工具安全**（只改内存账本）；
**B 路线（真钱）下必须**：独立进程 + 独立 token + 单笔上限 + 日累计上限 +
人工确认开关。**在做 B 之前，这一组不放出去。**

### 技术选型（查证结果 + **实测修正**）

- **官方 Python SDK 已到 v2**（`mcp` **2.2.0**）；v2 是大重写，v1.x 仍在维护。
  **新项目直接用 v2。**
- ⚠️ **v2 把 `FastMCP` 改名成了 `MCPServer`**（`from mcp.server.mcpserver import
  MCPServer`）。网上绝大多数教程仍是 v1 的 `from mcp.server.fastmcp import FastMCP`，
  **照抄会在 v2 上直接 `ModuleNotFoundError`** —— 本轮实测踩到。
  实现里 `_require_mcp()` 兼容两者，优先 v2。
- ⚠️ v2 字段风格从 camelCase 转到 snake_case：`serverInfo`→`server_info`、
  `inputSchema`→`input_schema`、`structuredContent`→`structured_content`、
  `isError`→`is_error`。实测协商到的协议版本是 **2025-11-25**。
- **`@tool()` 装饰器风格不变**：函数签名 + docstring 自动变成 JSON Schema，
  **零样板代码** —— 这一点 v1/v2 一致。
- **transport 选 `stdio`**：客户端把服务作为子进程拉起，走 stdin/stdout 的
  JSON-RPC。**不绑端口、不需要 TLS、单机默认安全**——与本项目 GUI 那套
  「只绑 127.0.0.1」是同一个安全哲学
- 备选 `streamable-http`：仅当要跨机时用（未来多 VM 场景）
- ⚠️ **`python -m 包名` 需要 `__main__.py`**：光有 `__init__.py` 里的
  `if __name__ == "__main__"` 不会执行。与打包那次「入口与包同名」是同一类问题
  （运行方式与文件结构不匹配）。

**教训**：只读文档不试跑，会在「类名」这种最表层的地方翻车。
**写 MCP 服务前先 `pip install` 再 `import` 一次，比读十篇教程有用。**

### ⚠️ 一条必须提前说的依赖风险

**`mcp` 是第三方包，而本项目现在的依赖只有 numpy / scipy / matplotlib**
（`tw/realdata.py` 的注释里明确写过「多一个 pandas 就多一个『环境不对所以
没跑』的借口」）。

⇒ 处理方式：
1. **MCP 服务作为可选组件**，放在 `mcp_server/` 目录，**不进核心测试链**
2. `import mcp` 失败时**给出明确指引**（`pip install "mcp>=2.2.0"`），不静默
3. 核心库 `tw/*` **绝不 import mcp**——保证「不装 mcp 也能跑全部研究」
4. 在 `packaging/` 的 exe 里**默认不打进去**（体积 + 用不上）

### 目录结构（建议）

```
mcp_server/
├── __init__.py
├── server.py        # FastMCP 实例 + 工具注册
├── tools_read.py    # 只读工具（行情/实验/留痕）
├── tools_write.py   # 写入工具（拉数据/跑实验）
├── tools_trade.py   # 交易工具（默认关闭）
└── README.md        # 如何挂到 Claude Desktop / Cursor
```

**验收方式**：用 `mcp dev server.py`（MCP Inspector）逐个调通工具，
并**用一个真实外部客户端连一次**——自己写的客户端测不出兼容性问题。

---

## 7. 阶段划分（建议顺序，每阶段可独立验收）

| 阶段 | 内容 | 验收 | 量级 |
|---|---|---|---|
| **A0** | ⭐ **数据层**：SQLite schema + 三源适配 + 按需拉取 + `fetches` 账本 | 单测：幂等 UPSERT、分页边界不重不漏、`confirm` 过滤、离线读库 | **中** |
| **A1** | OKX 风格订单模型 + 账户账本（保证金/杠杆/强平/TP-SL 触发） | 单测：强平价、保证金率、TP/SL 触发、`reduceOnly` 生效 | 中~大 |
| **A2** | 决策留痕层（记录结构 + JSONL 落盘 + **回放**） | 单测：同种子回放逐位一致；记录字段完整性 | 中 |
| **A3** | LLM 接入（Agnes）+ prompt 模板 + 解析 + 风控闸门 | 用 **no-internet 回放**跑通一次；解析失败也留痕 | 中 |
| **A4** | 评估报告（分层 + 与 noop/random 对照 + 成本敏感性） | 出一份真实报告 | 中 |
| **A5** | GUI 集成（决策时间线、留痕浏览器、订单/持仓面板） | 截图确认 | 中 |
| **A6** | ⭐ **MCP 服务**（只读工具优先） | MCP Inspector 逐个调通 + **一个真实外部客户端连一次** | 中 |
| **B**（可选，后置） | 真实 OKX 网关（**真钱，需单独评估**） | — | 小（替换执行器） |
| **C**（可选） | 多 Agent 圆桌（多个模型辩论后再决策） | — | 未知 |

**顺序理由**：
- **A0 排最前**：它是「数据从哪来」的问题，**没有它后面全部无源**。
  而且它**完全不依赖 LLM**，可以立刻开工、独立验收。
- **A1 + A2 是任何后续方案的地基**，且与「用不用 LLM」无关。
- **A6 放最后但独立**：MCP 是对外的门，**门后的东西得先有**才有意义。
  不过它的只读工具（`get_candles` / `get_run_summary`）其实**只要 A0 完成
  就能做**——所以也可以**把 A6 的只读部分提前到 A0 之后**，
  用来给外部 Agent 做第一轮体验。

---

## 8. 诚实边界（必须先说清楚）

1. **LLM Agent 的真实收益，几乎肯定远低于演示**。这是该领域公认现状
   （77 篇研究里 0 篇可复现）。
2. **LLM 延迟不适合高频**：单次调用几百 ms 到数秒，只适合
   「每个决策窗口决策一次」（如每 N 个 tick 或每个 K 线收盘）。
3. **不能因为它说得头头是道就信**：`reason` 字段是**审计材料**，不是证据。
   证据是"扣掉成本后还赢 noop"。
4. **真钱（B 路）需要有充分验证之后才谈**，且需要单独确认（权限/额度/风控）。
5. **Agnes 通道的可用性要实测**：模型名、配额、并发限制、失败重试，
   都要在 A3 里先验证再设计流程。

---

## 9. 需要用户拍板的事

### ✅ 已定（2026-09-20 用户回复「A吧」）

1. **路线**：**A（模拟，用 OKX 规则不连网）**。B（真钱）作为 A 验证通过后的
   最后一小步。
2. **数据层**：实时拉 OKX 公开行情 + SQLite + 按需拉取 + 历史下载 +
   自生成数据 + MCP 服务（已写入 §4.5 / §6.5）。

### ⬜ 仍待定

| # | 问题 | 影响 | 我的建议 |
|---|---|---|---|
| 1 | **决策频率**：每 tick / 每 N tick / 每根 K 线收盘 | LLM 调用量、成本、`reason` 的信息量 | **每根 K 线收盘**——理由：① 与 OKX 数据天然的粒度对齐，不用自己聚合；② LLM 延迟 2~6s，tick 级根本来不及；③ 「收盘」是真实交易员的标准决策时点 |
| 2 | **哪些标的** | 库的规模、拉取时间 | 先 **BTC-USDT-SWAP + ETH-USDT-SWAP** 两个——主流、数据全、有资金费率（永续专属） |
| 3 | **Agnes key 是否轮换** | 安全 | ⚠️ **建议换一个**：现有 key 已在本会话命令行里出现过。换完只放环境变量，不写进任何文件 |
| 4 | **MCP 只读部分是否提前**（排到 A0 之后而非 A6） | 影响你何时能用自己的助手探索 | 建议**提前**——成本低，且能早点看到「外部 Agent 用起来什么感觉」 |

## 附：本次查证的来源

### 订单与交易所（第六轮）
- OKX API v5 官方文档（下单参数、`attachAlgoOrds`、算法单、持仓字段）
- okx-api / okx_rs 的 SDK 端点清单（订单生命周期、批量操作）
- arXiv 2605.19337《Agentic Trading: When LLM Agents Meet Financial Markets》
  （77 篇研究的证据审计；「协议不可比」与可复现性缺失的统计）
- OpenWisdom《How to Evaluate LLM Trading Agents Without Backtest Theater》
  （最小证据链六项、三层分离、成本先行、弃权评分）
- Libertify《Financial Agents Orchestration Framework》
  （Memory Agent 的确定性 UUID、A2A 消息、风险闸门 `vol_ok/beta_ok/dd_ok`）
- 中文实践总结《LLM 交易代理的 Alpha 幻觉》（五层架构、输出解析器 + 风险过滤器）

### 行情数据与 MCP（第七轮新增）
- OKX 公开行情端点（`/market/candles`、`/market/history-candles`、
  `/market/books`、`/market/trades`、`/public/funding-rate`、
  `/public/open-interest`、`/public/mark-price`、`/public/instruments`）
  ——**全部免鉴权**；限频与单次上限见 §4.5 表格
- python-okx 端点封装拆分说明（**热缓存 vs 冷存储**导致两个 K 线端点
  上限不同：300 vs 100）
- OKX **API 协议第 3.2(b) 条**提到官方有「Agent Trade Kit（MCP Server /
  Skills / CLI）」——**说明交易所自己也认为 MCP 是标准接法**
- Model Context Protocol 官方站（Tools / Resources / Prompts 三原语、
  stdio vs Streamable HTTP）
- **MCP Python SDK v2（`mcp` 2.2.0）**：对应 2026-07-28 规范，
  `FastMCP` 装饰器风格，`pip install "mcp[cli]>=2.2.0"`
