"""Plan 审批系统 — Agent 先规划再执行的两阶段流程。

对应 team-define.md 中的 "Require plan approval for teammates"。
Agent 在 require_plan_approval 模式下：
1. 认领任务后进入 planning 阶段（只有读工具 + submit_plan）
2. 提交计划 → PlanManager 存储，状态为 pending
3. Leader 审批 → approved 或 rejected
4. approved → agent 获得全部工具，进入执行阶段
5. rejected → agent 收到反馈，修改计划重新提交
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)


class PlanStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


@dataclass
class PlanRequest:
    id: str = field(default_factory=lambda: str(uuid4()))
    task_id: str = ""
    agent_id: str = ""
    agent_name: str = ""
    plan_text: str = ""
    status: PlanStatus = PlanStatus.PENDING
    feedback: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class PlanManager:
    """管理计划审批流程。

    用法:
        mgr = PlanManager()
        plan = mgr.submit(task_id, agent_id, agent_name, "I will: 1. Read the code 2. ...")
        mgr.approve(plan.id)
        # 或
        mgr.reject(plan.id, "Please add error handling")
    """

    def __init__(self):
        self._plans: Dict[str, PlanRequest] = {}
        self._approval_callbacks: List[callable] = []
        self._events: Dict[str, asyncio.Event] = {}

    def on_plan_submitted(self, callback) -> None:
        """注册计划提交回调。leader 可以用此来自动审批。

        callback: async (PlanRequest) -> Optional[str]
            返回 None 表示自动放行，返回 str 表示 reject reason。
        """
        self._approval_callbacks.append(callback)

    def submit(self, task_id: str, agent_id: str, agent_name: str, plan_text: str) -> PlanRequest:
        plan = PlanRequest(
            task_id=task_id,
            agent_id=agent_id,
            agent_name=agent_name,
            plan_text=plan_text,
        )
        self._plans[plan.id] = plan
        self._events[plan.id] = asyncio.Event()
        logger.info(f"Plan submitted: {plan.id} by {agent_name} for task {task_id}")

        # 触发自动审批回调
        for cb in self._approval_callbacks:
            try:
                asyncio.create_task(self._auto_approve(cb, plan))
            except Exception:
                pass

        return plan

    async def _auto_approve(self, callback, plan: PlanRequest) -> None:
        try:
            result = callback(plan)
            if asyncio.iscoroutine(result):
                result = await result
            if result is None:
                self.approve(plan.id)
            elif isinstance(result, str):
                self.reject(plan.id, result)
        except Exception as exc:
            logger.exception(f"Auto-approval callback error: {exc}")

    def approve(self, plan_id: str, feedback: str = "") -> bool:
        plan = self._plans.get(plan_id)
        if not plan or plan.status != PlanStatus.PENDING:
            return False
        plan.status = PlanStatus.APPROVED
        plan.feedback = feedback
        logger.info(f"Plan {plan_id} APPROVED")
        self._events[plan_id].set()
        return True

    def reject(self, plan_id: str, reason: str) -> bool:
        plan = self._plans.get(plan_id)
        if not plan or plan.status != PlanStatus.PENDING:
            return False
        plan.status = PlanStatus.REJECTED
        plan.feedback = reason
        logger.info(f"Plan {plan_id} REJECTED: {reason}")
        self._events[plan_id].set()
        return True

    def get(self, plan_id: str) -> Optional[PlanRequest]:
        return self._plans.get(plan_id)

    def get_pending_for_agent(self, agent_id: str) -> Optional[PlanRequest]:
        for plan in self._plans.values():
            if plan.agent_id == agent_id and plan.status == PlanStatus.PENDING:
                return plan
        return None

    def get_pending_for_task(self, task_id: str) -> Optional[PlanRequest]:
        for plan in self._plans.values():
            if plan.task_id == task_id and plan.status == PlanStatus.PENDING:
                return plan
        return None

    def list_pending(self) -> List[PlanRequest]:
        return [p for p in self._plans.values() if p.status == PlanStatus.PENDING]

    async def wait_for_decision(self, plan_id: str, timeout: Optional[float] = None) -> PlanRequest:
        """阻塞等待 plan 被审批或拒绝。"""
        event = self._events.get(plan_id)
        if not event:
            raise ValueError(f"Unknown plan: {plan_id}")
        try:
            if timeout is not None:
                await asyncio.wait_for(event.wait(), timeout=timeout)
            else:
                await event.wait()
        except asyncio.TimeoutError:
            pass
        return self._plans[plan_id]
