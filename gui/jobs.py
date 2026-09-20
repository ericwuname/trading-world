"""后台任务：把"跑一场模拟"变成可查进度、可取消的异步作业。

为什么要这一层
--------------
一场模拟要跑几秒到几分钟（预热 4000 + 观测 8000 tick）。
如果在 HTTP 请求里同步跑完：

  · 页面会像卡死一样（浏览器等不到响应）
  · 没有任何进度可看，长任务不知道还要多久
  · 一个请求就能把整个服务占住，其他请求全部排队

所以：**创建请求只登记作业并立刻返回 id，真正的计算在工作线程里分块推进**。

分块推进换来三件事
------------------
1. **进度**：每块结束更新一次，UI 能画进度条
2. **取消**：块与块之间检查取消标志——参数跑错了不用等它跑完
3. **可复现的耗时**：每块用同一个 ``Market.run(chunk)`` 接口推进，
   而不是自己循环 ``step()``——后者会跳过 ``run()`` 结尾的
   ``log.trim()`` / ``_collect_agent_stats()``，日志长度和主体统计会不对。

工作线程数
----------
默认取 ``min(4, CPU//2)``：模拟是纯 CPU 密集，开太多只会互相抢核，
让每个作业都变慢。留一半核给 HTTP 线程和系统。
"""

from __future__ import annotations

import queue
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

#: 每块推进多少 tick。太小 → 进度回调与日志开销占比过高；
#: 太大 → 进度条一跳一大格、取消响应变慢。
#: 250 tick 在 300 主体下约 0.15~0.2 秒，是进度流畅度与开销的平衡点。
CHUNK = 250

#: 单个作业最多保留多少条日志（防止长时间跑把内存吃满）
MAX_LOG = 400


@dataclass
class Job:
    """一次后台计算。"""

    id: str
    kind: str                    # market | strategy | lab
    spec: dict                   # 请求参数原样回显（便于复现）
    state: str = "queued"        # queued | running | done | error | cancelled
    progress: float = 0.0        # 0..1
    phase: str = "排队中"
    error: str | None = None
    traceback: str | None = None
    result: dict | None = None   # 汇总（小，直接回给前端）
    series: dict | None = None   # 时序（大，单独取）
    log: list[str] = field(default_factory=list)
    created: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None
    _cancel: threading.Event = field(default_factory=threading.Event, repr=False)

    # --- 给作业函数用的小工具 -------------------------------------------
    def say(self, msg: str) -> None:
        """往作业日志里追加一行（UI 上能看到，排查用）。"""
        self.log.append(msg)
        if len(self.log) > MAX_LOG:
            del self.log[: len(self.log) - MAX_LOG]

    def cancelled(self) -> bool:
        return self._cancel.is_set()

    @property
    def elapsed(self) -> float:
        end = self.finished or time.time()
        return max(0.0, end - (self.started or self.created))

    def public(self, *, with_result: bool = True) -> dict:
        out: dict[str, Any] = {
            "id": self.id,
            "kind": self.kind,
            "spec": self.spec,
            "state": self.state,
            "progress": round(self.progress, 4),
            "phase": self.phase,
            "elapsed": round(self.elapsed, 2),
            "created": self.created,
            "log": self.log[-40:],
        }
        if self.error:
            out["error"] = self.error
        if with_result and self.result is not None:
            out["result"] = self.result
        return out


#: 作业函数签名：``fn(job) -> tuple[result, series]``
JobFn = Callable[[Job], "tuple[dict | None, dict | None]"]


class JobManager:
    """作业登记表 + 工作线程池。"""

    def __init__(self, workers: int = 1) -> None:
        self._jobs: dict[str, Job] = {}
        self._order: list[str] = []
        self._lock = threading.Lock()
        self._q: queue.Queue[str | None] = queue.Queue()
        self._workers: list[threading.Thread] = []
        self._stop = threading.Event()
        #: 作业 id → 执行函数。放在实例里而不是 Job 上：
        #: Job 是要被序列化给前端的，塞函数进去会破坏这一点。
        self._pending: dict[str, JobFn] = {}
        for i in range(max(1, workers)):
            t = threading.Thread(target=self._worker, name=f"tw-job-{i}", daemon=True)
            t.start()
            self._workers.append(t)

    # --- 对外接口 -------------------------------------------------------
    def submit(self, kind: str, spec: dict, fn: JobFn) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], kind=kind, spec=dict(spec))
        with self._lock:
            self._jobs[job.id] = job
            self._order.append(job.id)
            # 只保留最近 60 个作业，其余从登记表里清掉（series 可能很大）
            while len(self._order) > 60:
                old = self._order.pop(0)
                self._jobs.pop(old, None)
        self._pending[job.id] = fn
        self._q.put(job.id)
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> list[Job]:
        with self._lock:
            return [self._jobs[i] for i in reversed(self._order) if i in self._jobs]

    def cancel(self, job_id: str) -> bool:
        job = self.get(job_id)
        if job is None or job.state in ("done", "error", "cancelled"):
            return False
        job._cancel.set()
        if job.state == "queued":
            job.state = "cancelled"
            job.phase = "已取消"
        return True

    def stats(self) -> dict:
        jobs = self.list()
        return {
            "total": len(jobs),
            "running": sum(1 for j in jobs if j.state == "running"),
            "queued": sum(1 for j in jobs if j.state == "queued"),
            "workers": len(self._workers),
        }

    # --- 工作线程 -------------------------------------------------------
    def _worker(self) -> None:
        while not self._stop.is_set():
            try:
                job_id = self._q.get(timeout=0.5)
            except queue.Empty:
                continue
            if job_id is None:
                return
            job = self.get(job_id)
            fn = self._pending.pop(job_id, None)
            if job is None or fn is None:
                continue
            if job.cancelled():
                job.state = "cancelled"
                job.phase = "已取消"
                continue
            job.state = "running"
            job.started = time.time()
            job.phase = "启动中"
            try:
                result, series = fn(job)
                job.result, job.series = result, series
                if job.cancelled():
                    job.state = "cancelled"
                    job.phase = "已取消（结果不完整）"
                else:
                    job.state = "done"
                    job.progress = 1.0
                    job.phase = "完成"
            except Exception as exc:  # noqa: BLE001
                # 作业失败不能把工作线程带走——它是池里共享的资源，
                # 死了之后所有后续作业都会永远排队。
                job.state = "error"
                job.error = f"{type(exc).__name__}: {exc}"
                job.traceback = traceback.format_exc()[-2000:]
                job.phase = "出错"
            finally:
                job.finished = time.time()

    def shutdown(self) -> None:
        self._stop.set()
        for _ in self._workers:
            self._q.put(None)


def step_until(
    job: Job,
    run_chunk: Callable[[int], None],
    total: int,
    *,
    offset: int = 0,
    work: int | None = None,
    phase: str = "",
) -> bool:
    """把 ``total`` 个 tick 分块推进，边推边更新进度。返回是否被取消。

    参数
    ----
    run_chunk  接受"本块 tick 数"的可调用对象，通常直接是 ``market.run``
    total      本次推进的总 tick 数
    offset     已经完成的 tick 数（用于把"预热 + 观测"拼成一个总进度）
    work       总工作量（默认 = offset + total），算 progress 用
    """
    total_work = max(1, work if work is not None else offset + total)
    done = 0
    while done < total:
        if job.cancelled():
            return True
        n = min(CHUNK, total - done)
        run_chunk(n)
        done += n
        job.progress = min(1.0, (offset + done) / total_work)
        if phase:
            job.phase = f"{phase} {offset + done:,}/{total_work:,}"
    return job.cancelled()
