# A3 LLM 运行证据（2026-09-21）

> **为什么把运行结果提交进仓库**：本项目的纪律是「报告里的数字必须能追溯到
> 某一次运行」（见 `research-artifact-integrity`）。A3 的交付里有一个
> **真机 A/B 结论**（"只换 prompt 版本，成交率差 3~5 倍"），
> 如果只留下结论而没有原始记录，那个结论就只是一句话，无法复核。

## 文件

| 文件 | 内容 |
|---|---|
| `records_<模板>_<标的>.jsonl` | **LLM 原始响应记录**（真机调用的逐条留痕） |
| `decisions_<模板>_<标的>.jsonl` | **决策留痕**（`DecisionRecord`，证据链六项） |

模板 `v1` / `v2`，标的 `BTC` / `ETH`（均为 USDT 永续 1H）。
每格 8 次决策 × 3 采样 = 24 次调用，四格共 **96 次调用**。

## ⚠️ 安全说明

**这批文件里没有密钥，也没有任何认证头。**
`Recorder` 只记录：`prompt_hash` / `messages` / `model` / `temperature` /
响应正文 / 延迟 / token 用量。
密钥在那个 `Authorization: Bearer` 头里，**不进录制文件**。

已用 `grep -l "cpk-\|sk-proj\|ark-\|Bearer\|Authorization"` 扫过，零命中。

## 怎么用

```bash
# 1) 看模型当时到底看到了什么（离线，不需要 key）
python scripts/agent_cli.py --inst BTC-USDT-SWAP --n 8 --template v2 --show-prompt

# 2) 回放这批记录（离线，不需要 key，零 API 成本）
python scripts/agent_cli.py --inst BTC-USDT-SWAP --n 8 --samples 3 \
    --template v2 --replay docs/llm-run-20260921/records_v2_BTC.jsonl
```

回放必须**逐条**与 `decisions_*.jsonl` 一致（投票 / 成交 / 风控规则）。
⚠️ 注意 `decision_id` 取决于 `run_id`，回放时要用生成这批 decision 时的
`--run-id`（当时用的是 `R-<模板>-<标的>`，如 `R-v2-BTC`）。

## 已知边界（不要过度解读这批数据）

1. **样本极小**：每格 8 次决策。成交率 12.5% vs 75% 的差距虽然很大，
   **不足以下统计结论**。按本项目测量纪律，这类比较要走**区间重叠检验**
   才算数——留给 A4。
2. **未接账户**：跑批时 `run_session` 的 `on_record` 钩子没用上，
   所以每次决策看到的都是"空仓 / equity 不变"。
   ⇒ **这批数字不能当作策略收益**，只能当作"管线的行为特征"。
3. **只覆盖了 4 个格子**（2 模板 × 2 标的），且都取自同一段行情
   （2026-08-18 ~ 09-21）。换一段行情结论可能不同。
