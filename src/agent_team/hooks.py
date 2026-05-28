"""Hooks 系统 — 质量门禁机制。

在关键事件点触发用户注册的钩子函数，钩子可以放行（返回 None）或阻止（返回字符串原因）。
对应 team-define.md 中的 TeammateIdle / TaskCreated / TaskCompleted 钩子。
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class HookEvent(str, Enum):
    TASK_CREATED = "task_created"
    TASK_COMPLETED = "task_completed"
    TEAMMATE_IDLE = "teammate_idle"


@dataclass
class HookContext:
    """传递给钩子函数的上下文数据。"""
    event: HookEvent
    agent_id: str = ""
    agent_name: str = ""
    task_id: str = ""
    task_name: str = ""
    task_description: str = ""
    task_result: str = ""
    data: Dict[str, Any] = field(default_factory=dict)


# 钩子函数签名: async (HookContext) -> Optional[str]
# 返回 None = 放行, 返回 str = 阻止并以此字符串作为原因
HookCallback = Callable[..., Any]


class HookRegistry:
    """管理事件钩子的注册和触发。

    用法:
        registry = HookRegistry()
        registry.register(HookEvent.TASK_CREATED, my_hook)
        allowed, reason = await registry.trigger(HookEvent.TASK_CREATED, context)
    """

    def __init__(self):
        self._hooks: Dict[HookEvent, List[HookCallback]] = {
            e: [] for e in HookEvent
        }

    def register(self, event: HookEvent, callback: HookCallback) -> None:
        """注册一个钩子函数。

        Args:
            event: 触发钩子的事件类型
            callback: async (HookContext) -> Optional[str] 函数
        """
        self._hooks[event].append(callback)
        logger.info(f"Hook registered: {event.value} -> {getattr(callback, '__name__', str(callback))}")

    def unregister(self, event: HookEvent, callback: HookCallback) -> None:
        """取消注册一个钩子函数。"""
        if callback in self._hooks[event]:
            self._hooks[event].remove(callback)

    async def trigger(self, event: HookEvent, context: HookContext) -> tuple[bool, str]:
        """触发某个事件的所有钩子（异步版本）。

        Args:
            event: 事件类型
            context: 传递给钩子的上下文

        Returns:
            (allowed, reason) — allowed=True 表示所有钩子放行，
            任一钩子返回字符串 reason 表示阻止。
        """
        for hook in self._hooks[event]:
            try:
                result = hook(context)
                if asyncio.iscoroutine(result):
                    result = await result
                if result is not None and isinstance(result, str):
                    logger.warning(f"Hook {event.value} BLOCKED: {result}")
                    return False, result
            except Exception:
                logger.exception(f"Hook for {event.value} raised exception, skipping")
        return True, ""

    def trigger_sync(self, event: HookEvent, context: HookContext) -> tuple[bool, str]:
        """触发某个事件的所有钩子（同步安全版本）。

        适用于从同步代码（如 create_task）中调用。
        异步钩子在运行中事件循环里无法阻塞，会自动跳过并警告。
        """
        try:
            asyncio.get_running_loop()
            # 运行中的事件循环 — 不能 await，仅执行同步钩子
            for hook in self._hooks[event]:
                try:
                    result = hook(context)
                    if asyncio.iscoroutine(result):
                        logger.warning(
                            f"Async hook for {event.value} skipped in sync context. "
                            "Use sync functions for TASK_CREATED/TASK_COMPLETED hooks."
                        )
                        continue
                    if result is not None and isinstance(result, str):
                        logger.warning(f"Hook {event.value} BLOCKED: {result}")
                        return False, result
                except Exception:
                    logger.exception(f"Hook for {event.value} raised exception, skipping")
            return True, ""
        except RuntimeError:
            return asyncio.run(self.trigger(event, context))

    async def trigger_teammate_idle(self, agent_id: str, agent_name: str) -> tuple[bool, str]:
        """触发 TEAMMATE_IDLE 钩子。"""
        ctx = HookContext(event=HookEvent.TEAMMATE_IDLE, agent_id=agent_id, agent_name=agent_name)
        return await self.trigger(HookEvent.TEAMMATE_IDLE, ctx)

    async def trigger_task_created(self, task_id: str, task_name: str, task_description: str) -> tuple[bool, str]:
        """触发 TASK_CREATED 钩子。"""
        ctx = HookContext(
            event=HookEvent.TASK_CREATED,
            task_id=task_id,
            task_name=task_name,
            task_description=task_description,
        )
        return await self.trigger(HookEvent.TASK_CREATED, ctx)

    async def trigger_task_completed(self, task_id: str, task_name: str, task_result: str,
                                     agent_id: str = "", agent_name: str = "") -> tuple[bool, str]:
        """触发 TASK_COMPLETED 钩子。"""
        ctx = HookContext(
            event=HookEvent.TASK_COMPLETED,
            task_id=task_id,
            task_name=task_name,
            task_result=task_result,
            agent_id=agent_id,
            agent_name=agent_name,
        )
        return await self.trigger(HookEvent.TASK_COMPLETED, ctx)
