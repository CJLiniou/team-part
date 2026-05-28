"""可观测性模块 — Agent 和团队状态实时汇报。

提供干净的控制台输出，让用户能看到每个 Agent 在做什么。
"""

from __future__ import annotations

import shutil
import textwrap
import time
from typing import Optional


class TeamConsole:
    """团队状态控制台输出。

    用法:
        console = TeamConsole(width=80)
        console.agent_step("Developer", "Analyzing task...")
        console.tool_call("Developer", "read_file", {"path": "utils.py"}, "def add(a,b):...")
        console.task_completed("Developer", "Create utils.py", "Created file successfully")
        console.team_dashboard(agents, tasks)
    """

    def __init__(self, enabled: bool = True, width: Optional[int] = None):
        self.enabled = enabled
        self.width = width or min(shutil.get_terminal_size().columns, 120)
        self._start_time = time.time()

    # ── Agent events ──────────────────────────────────────────────

    def agent_start(self, name: str, model: str) -> None:
        if not self.enabled:
            return
        self._print(f"  [{name}] started ({model})")

    def agent_step(self, name: str, action: str) -> None:
        if not self.enabled:
            return
        self._print(f"  [{name}] {action}")

    def tool_call(self, name: str, tool: str, args: dict, result: str = "") -> None:
        if not self.enabled:
            return
        args_str = ", ".join(f"{k}={self._trunc(str(v), 40)}" for k, v in args.items() if v)
        r = ""
        if result:
            r = f" -> {self._trunc(result, 80)}"
        self._print(f"  [{name}] -> {tool}({args_str}){r}")

    def tool_result(self, name: str, tool: str, result: str) -> None:
        if not self.enabled:
            return
        self._print(f"  [{name}] <- {tool}: {self._trunc(result, 100)}")

    def task_claimed(self, name: str, task_name: str) -> None:
        if not self.enabled:
            return
        self._print(f"  [{name}] CLAIMED: {task_name}")

    def task_completed(self, name: str, task_name: str, result: str = "") -> None:
        if not self.enabled:
            return
        r = f" — {self._trunc(result, 60)}" if result else ""
        self._print(f"  [{name}] COMPLETED: {task_name}{r}")

    def task_failed(self, name: str, task_name: str, reason: str = "") -> None:
        if not self.enabled:
            return
        r = f" — {reason}" if reason else ""
        self._print(f"  [{name}] FAILED: {task_name}{r}")

    def error(self, name: str, msg: str) -> None:
        if not self.enabled:
            return
        self._print(f"  [{name}] ERROR: {self._trunc(msg, 100)}")

    def rate_limited(self, name: str, retry: int, delay: float) -> None:
        if not self.enabled:
            return
        self._print(f"  [{name}] rate-limited, retry #{retry} in {delay:.0f}s")

    # ── Team events ───────────────────────────────────────────────

    def team_dashboard(self, agents: dict, tasks: list) -> None:
        """打印团队仪表盘。"""
        if not self.enabled:
            return

        lines = []
        lines.append("")
        lines.append("=" * min(self.width, 60))
        lines.append(f"  TEAM STATUS ({time.time() - self._start_time:.0f}s elapsed)")
        lines.append("-" * min(self.width, 60))

        # Agent 状态
        lines.append("  Agents:")
        for a in agents.values():
            task_info = f" [{a.current_task_id[:8]}...]" if a.current_task_id else " idle"
            lines.append(f"    {a.name:20s} | {a.state.value:8s}{task_info}")

        # 任务统计
        pending = sum(1 for t in tasks if t.status.value == "pending")
        in_progress = sum(1 for t in tasks if t.status.value == "in_progress")
        completed = sum(1 for t in tasks if t.status.value == "completed")
        failed = sum(1 for t in tasks if t.status.value == "failed")
        lines.append(f"  Tasks: {pending} pending | {in_progress} running | "
                     f"{completed} done | {failed} failed")
        lines.append("-" * min(self.width, 60))

        for line in lines:
            print(line)

    def leader_tick(self, msg: str) -> None:
        if not self.enabled:
            return
        self._print(f"  [LEADER] {msg}")

    def separator(self, title: str = "") -> None:
        if not self.enabled:
            return
        if title:
            print(f"\n{'─' * min(self.width, 60)}")
            print(f"  {title}")
            print(f"{'─' * min(self.width, 60)}")
        else:
            print(f"{'─' * min(self.width, 60)}")

    # ── Helpers ───────────────────────────────────────────────────

    def _print(self, msg: str) -> None:
        print(msg)

    @staticmethod
    def _trunc(s: str, max_len: int) -> str:
        s = " ".join(str(s).split())
        if len(s) <= max_len:
            return s
        return s[:max_len - 3] + "..."
