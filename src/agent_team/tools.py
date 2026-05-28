"""工具系统 — Anthropic tool-use 格式的工具定义与执行器。

每个工具包含：
- name / description / input_schema（Anthropic API 格式）
- handler 函数，实际调用 TaskManager / Mailbox 等组件
"""

from __future__ import annotations

from typing import Any, Callable, Optional

# ── Tool schema helpers ──────────────────────────────────────────────


def _make_schema(name: str, description: str, properties: dict,
                 required: Optional[list[str]] = None) -> dict:
    return {
        "name": name,
        "description": description,
        "input_schema": {
            "type": "object",
            "properties": properties,
            "required": required or list(properties.keys()),
        },
    }


# ── Tool schemas ─────────────────────────────────────────────────────

CLAIM_TASK_SCHEMA = _make_schema(
    name="claim_task",
    description="认领一个可用的待处理任务。只有依赖已满足、且未被其他 agent 认领的任务才会被返回。",
    properties={
        "task_id": {
            "type": "string",
            "description": "要认领的任务 ID。留空则自动获取优先级最高的可用任务。",
        }
    },
    required=[],
)

COMPLETE_TASK_SCHEMA = _make_schema(
    name="complete_task",
    description="标记当前任务为已完成，并记录执行结果。",
    properties={
        "task_id": {"type": "string", "description": "要完成的任务 ID"},
        "result": {"type": "string", "description": "任务执行结果摘要"},
    },
)

FAIL_TASK_SCHEMA = _make_schema(
    name="fail_task",
    description="标记任务为失败，并记录失败原因。",
    properties={
        "task_id": {"type": "string", "description": "失败的任务 ID"},
        "reason": {"type": "string", "description": "失败原因"},
    },
)

SEND_MESSAGE_SCHEMA = _make_schema(
    name="send_message",
    description="向团队中的另一个 agent 发送消息。用于协调工作、请求帮助或报告发现。",
    properties={
        "recipient": {"type": "string", "description": "接收者的 agent 名称"},
        "subject": {"type": "string", "description": "消息主题"},
        "content": {"type": "string", "description": "消息正文"},
    },
)

CHECK_MAILBOX_SCHEMA = _make_schema(
    name="check_mailbox",
    description="查看自己的未读消息。",
    properties={},
    required=[],
)

LIST_AGENTS_SCHEMA = _make_schema(
    name="list_agents",
    description="列出团队中所有 agent 及其当前状态。",
    properties={},
    required=[],
)

LIST_TASKS_SCHEMA = _make_schema(
    name="list_tasks",
    description="列出所有任务及其当前状态。支持按状态过滤。",
    properties={
        "status": {
            "type": "string",
            "description": "按状态过滤: pending, in_progress, completed, failed, blocked",
        }
    },
    required=[],
)

READ_FILE_SCHEMA = _make_schema(
    name="read_file",
    description="读取项目中的文件内容。用于了解代码库、查看文档或检查其他 agent 的工作成果。",
    properties={
        "path": {"type": "string", "description": "相对于工作目录的文件路径"},
    },
)

WRITE_FILE_SCHEMA = _make_schema(
    name="write_file",
    description="创建或覆盖项目中的文件。",
    properties={
        "path": {"type": "string", "description": "相对于工作目录的文件路径"},
        "content": {"type": "string", "description": "要写入的文件内容"},
    },
)

SUBMIT_PLAN_SCHEMA = _make_schema(
    name="submit_plan",
    description="提交当前任务的执行计划，等待 leader 审批。仅在 require_plan_approval 模式下可用。",
    properties={
        "task_id": {"type": "string", "description": "当前任务 ID"},
        "plan": {"type": "string", "description": "详细的执行计划：你要做什么、用什么工具、预期产出"},
    },
)

CHECK_PLAN_STATUS_SCHEMA = _make_schema(
    name="check_plan_status",
    description="查看自己已提交计划的审批状态。",
    properties={},
    required=[],
)

RESPOND_TO_SHUTDOWN_SCHEMA = _make_schema(
    name="respond_to_shutdown",
    description="响应 leader 的关闭请求。如果你可以安全停止，接受；如果正在处理关键任务，拒绝并说明原因。",
    properties={
        "accept": {"type": "boolean", "description": "true=同意关闭, false=拒绝关闭"},
        "reason": {"type": "string", "description": "接受/拒绝的原因"},
    },
)

# 所有工具 schema 的集合
ALL_TOOL_SCHEMAS = [
    CLAIM_TASK_SCHEMA,
    COMPLETE_TASK_SCHEMA,
    FAIL_TASK_SCHEMA,
    SEND_MESSAGE_SCHEMA,
    CHECK_MAILBOX_SCHEMA,
    LIST_AGENTS_SCHEMA,
    LIST_TASKS_SCHEMA,
    READ_FILE_SCHEMA,
    WRITE_FILE_SCHEMA,
    SUBMIT_PLAN_SCHEMA,
    CHECK_PLAN_STATUS_SCHEMA,
    RESPOND_TO_SHUTDOWN_SCHEMA,
]

# ── Tool registry ─────────────────────────────────────────────────────

ToolHandler = Callable[..., str]


class ToolRegistry:
    """管理工具 schema 与 handler 的注册和调用。"""

    def __init__(self):
        self._handlers: dict[str, ToolHandler] = {}
        self._schemas: dict[str, dict] = {}

    def register(self, schema: dict, handler: ToolHandler) -> None:
        name = schema["name"]
        self._schemas[name] = schema
        self._handlers[name] = handler

    def get_schemas(self) -> list[dict]:
        return list(self._schemas.values())

    def get_schema(self, name: str) -> Optional[dict]:
        return self._schemas.get(name)

    async def execute(self, name: str, args: dict) -> str:
        handler = self._handlers.get(name)
        if not handler:
            return f"错误: 未知工具 '{name}'"
        import asyncio
        result = handler(**args)
        if asyncio.iscoroutine(result):
            result = await result
        return str(result)


def create_default_registry(task_manager, mailbox, agents_dict,
                            work_dir: str = ".",
                            plan_manager=None,
                            notification_bus=None) -> ToolRegistry:
    """创建包含所有默认工具的 ToolRegistry。

    Args:
        task_manager: TaskManager 实例
        mailbox: AgentMailbox 实例
        agents_dict: {agent_id: Agent} 字典，由 Orchestrator 维护
        work_dir: 文件操作的工作目录
        plan_manager: 可选的 PlanManager，用于 submit_plan/check_plan_status 工具
        notification_bus: 可选的 NotificationBus，用于推送通知
    """
    registry = ToolRegistry()

    # ── claim_task ──
    def _claim_task(task_id: str = "") -> str:
        agent = _claim_task._agent_id or _claim_task._agent_name
        if task_id:
            ok = task_manager.claim_task(task_id, agent)
            if ok:
                _claim_task._current_task_id = task_id
                return f"已认领任务 {task_id}"
            return f"无法认领任务 {task_id}（可能已被认领或依赖未满足）"

        available = task_manager.get_available_tasks(agent)
        if not available:
            return "当前没有可认领的任务"
        task = available[0]
        ok = task_manager.claim_task(task.id, agent)
        if ok:
            _claim_task._current_task_id = task.id
            return f"已认领任务: {task.name} ({task.id})"
        return "认领失败"
    _claim_task._agent_name = ""
    _claim_task._agent_id = ""
    _claim_task._current_task_id = ""

    registry.register(CLAIM_TASK_SCHEMA, _claim_task)

    # ── complete_task ──
    async def _complete_task(task_id: str, result: str) -> str:
        from .tasks import TaskStatus
        ok = task_manager.update_task_status(task_id, TaskStatus.COMPLETED, result)
        if ok:
            _claim_task._current_task_id = ""
            return f"任务 {task_id} 已标记为完成"
        return f"无法完成任务 {task_id}"
    registry.register(COMPLETE_TASK_SCHEMA, _complete_task)

    # ── fail_task ──
    def _fail_task(task_id: str, reason: str) -> str:
        from .tasks import TaskStatus
        ok = task_manager.update_task_status(task_id, TaskStatus.FAILED, reason)
        if ok:
            _claim_task._current_task_id = ""
            return f"任务 {task_id} 已标记为失败: {reason}"
        return f"无法将任务 {task_id} 标记为失败"
    registry.register(FAIL_TASK_SCHEMA, _fail_task)

    # ── send_message ──
    def _send_message(recipient: str, subject: str, content: str) -> str:
        msg = mailbox.send(_send_message._agent_name, recipient, subject, content)
        return f"消息已发送给 {recipient} (id: {msg.id})"
    _send_message._agent_name = ""
    registry.register(SEND_MESSAGE_SCHEMA, _send_message)

    # ── check_mailbox ──
    def _check_mailbox() -> str:
        msgs = mailbox.receive(_check_mailbox._agent_name, limit=10, unread_only=True)
        if not msgs:
            return "没有未读消息"
        lines = []
        for m in msgs:
            lines.append(f"[{m.sender}] {m.subject}: {m.content}")
            mailbox.mark_read(m.id)
        return "\n".join(lines)
    _check_mailbox._agent_name = ""
    registry.register(CHECK_MAILBOX_SCHEMA, _check_mailbox)

    # ── list_agents ──
    def _list_agents() -> str:
        if not agents_dict:
            return "团队中没有 agent"
        lines = []
        for a in agents_dict.values():
            lines.append(f"- {a.name} ({a.id}): role={a.role.value}, state={a.state.value}")
        return "\n".join(lines)
    registry.register(LIST_AGENTS_SCHEMA, _list_agents)

    # ── list_tasks ──
    def _list_tasks(status: str = "") -> str:
        from .tasks import TaskStatus
        st = None
        if status and status not in ("all", "All", "ALL"):
            try:
                st = TaskStatus(status)
            except ValueError:
                pass  # 无效状态值，视为不过滤
        tasks = task_manager.list_tasks(status=st)
        if not tasks:
            return "没有任务"
        lines = []
        for t in tasks:
            deps = f" (依赖: {', '.join(t.depends_on)})" if t.depends_on else ""
            lines.append(
                f"- [{t.status.value}] {t.name} ({t.id}) 优先级={t.priority}"
                f" 分配给={t.assigned_to or '无人'}{deps}"
            )
        return "\n".join(lines)
    registry.register(LIST_TASKS_SCHEMA, _list_tasks)

    # ── read_file ──
    def _read_file(path: str) -> str:
        import os
        full = os.path.join(work_dir, path)
        try:
            with open(full, encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            return f"文件不存在: {path}"
        except Exception as exc:
            return f"读取错误: {exc}"
    registry.register(READ_FILE_SCHEMA, _read_file)

    # ── write_file ──
    def _write_file(path: str, content: str) -> str:
        import os
        full = os.path.join(work_dir, path)
        os.makedirs(os.path.dirname(full) or ".", exist_ok=True)
        with open(full, "w", encoding="utf-8") as f:
            f.write(content)
        return f"文件已写入: {path}"
    registry.register(WRITE_FILE_SCHEMA, _write_file)

    # ── submit_plan ──
    def _submit_plan(task_id: str, plan: str) -> str:
        if not plan_manager:
            return "Plan approval is not enabled for this team"
        plan_req = plan_manager.submit(task_id, _submit_plan._agent_id,
                                       _submit_plan._agent_name, plan)
        _submit_plan._current_plan_id = plan_req.id
        # 通知 leader 有待审批的计划
        if notification_bus:
            import asyncio
            asyncio.create_task(
                notification_bus.publish("__leader__", "plan_submitted", {
                    "plan_id": plan_req.id,
                    "agent_name": _submit_plan._agent_name,
                    "task_id": task_id,
                    "plan": plan,
                })
            )
        return f"Plan submitted (id: {plan_req.id}). Waiting for approval."
    _submit_plan._agent_name = ""
    _submit_plan._agent_id = ""
    _submit_plan._current_plan_id = ""
    registry.register(SUBMIT_PLAN_SCHEMA, _submit_plan)

    # ── check_plan_status ──
    def _check_plan_status() -> str:
        if not plan_manager:
            return "Plan approval is not enabled"
        plan = plan_manager.get_pending_for_agent(_check_plan_status._agent_id)
        if not plan:
            # Check for recently decided plans
            return "No pending plan. If you submitted one, it may have been decided already."
        if plan.status.value == "pending":
            return f"Plan {plan.id} is still pending approval."
        elif plan.status.value == "approved":
            return f"Plan {plan.id} has been APPROVED. Feedback: {plan.feedback or 'None'}. Proceed with execution."
        elif plan.status.value == "rejected":
            return f"Plan {plan.id} was REJECTED. Reason: {plan.feedback}. Revise and resubmit."
        return f"Unknown plan status: {plan.status.value}"
    _check_plan_status._agent_name = ""
    _check_plan_status._agent_id = ""
    registry.register(CHECK_PLAN_STATUS_SCHEMA, _check_plan_status)

    # ── respond_to_shutdown ──
    def _respond_to_shutdown(accept: bool, reason: str) -> str:
        _respond_to_shutdown._response = (accept, reason)
        if accept:
            return f"Shutdown accepted: {reason}"
        return f"Shutdown rejected: {reason}"
    _respond_to_shutdown._response = None
    registry.register(RESPOND_TO_SHUTDOWN_SCHEMA, _respond_to_shutdown)

    # 存储对闭包变量的引用，以便 LLMAgent 设置 agent_name
    registry._claim_task_fn = _claim_task
    registry._send_message_fn = _send_message
    registry._check_mailbox_fn = _check_mailbox
    registry._submit_plan_fn = _submit_plan
    registry._check_plan_status_fn = _check_plan_status
    registry._respond_to_shutdown_fn = _respond_to_shutdown

    return registry


def set_agent_context(registry: ToolRegistry, agent_name: str, agent_id: str = "") -> None:
    """将 agent 名称和 ID 注入到需要身份识别的工具处理器中。"""
    for attr in ['_claim_task_fn', '_send_message_fn', '_check_mailbox_fn',
                 '_submit_plan_fn', '_check_plan_status_fn']:
        fn = getattr(registry, attr, None)
        if fn:
            fn._agent_name = agent_name
            fn._agent_id = agent_id
