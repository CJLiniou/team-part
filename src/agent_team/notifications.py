"""推送通知系统 — 事件驱动的 agent 唤醒机制。

每个 agent 持有一个异步队列，leader 和 tool handler 通过 NotificationBus 推送事件。
Agent 的主循环阻塞在队列上等待事件，取代忙轮询。

事件类型: new_message, plan_approved, plan_rejected, task_unblocked, shutdown
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class Notification:
    type: str
    data: Dict[str, Any] = field(default_factory=dict)


class NotificationBus:
    """事件驱动的通知总线。

    每个 agent 有自己的 asyncio.Queue，发布者向队列推送事件，
    agent 通过 consume() 阻塞等待。

    用法:
        bus = NotificationBus()
        await bus.publish("dev-1", "new_message", {"from": "reviewer"})
        notif = await bus.consume("dev-1", timeout=10.0)
    """

    def __init__(self):
        self._queues: Dict[str, asyncio.Queue] = {}

    def register(self, agent_id: str) -> None:
        """为 agent 注册一个通知队列。"""
        if agent_id not in self._queues:
            self._queues[agent_id] = asyncio.Queue()

    def unregister(self, agent_id: str) -> None:
        """移除 agent 的通知队列。"""
        self._queues.pop(agent_id, None)

    async def publish(self, agent_id: str, event_type: str, data: Optional[Dict[str, Any]] = None) -> None:
        """向指定 agent 推送通知。

        Args:
            agent_id: 目标 agent ID
            event_type: 事件类型 (new_message, plan_approved, plan_rejected, task_unblocked, shutdown)
            data: 附加数据
        """
        if agent_id not in self._queues:
            self.register(agent_id)
        notif = Notification(type=event_type, data=data or {})
        await self._queues[agent_id].put(notif)
        logger.debug(f"Notification -> {agent_id}: {event_type}")

    async def broadcast(self, event_type: str, data: Optional[Dict[str, Any]] = None,
                        exclude: Optional[str] = None) -> None:
        """向所有已注册 agent 广播通知。"""
        for agent_id in self._queues:
            if agent_id != exclude:
                await self.publish(agent_id, event_type, data)

    async def consume(self, agent_id: str, timeout: Optional[float] = None) -> Optional[Notification]:
        """阻塞等待通知。

        Args:
            agent_id: agent ID
            timeout: 超时秒数，None 表示永不超时

        Returns:
            Notification 或 None（超时）
        """
        if agent_id not in self._queues:
            self.register(agent_id)
        try:
            if timeout is not None:
                return await asyncio.wait_for(self._queues[agent_id].get(), timeout=timeout)
            return await self._queues[agent_id].get()
        except asyncio.TimeoutError:
            return None
