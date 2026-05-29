"""LLMAgent — 由 Anthropic Claude 驱动的自主智能体。

每个 LLMAgent 在独立的异步循环中运行：
1. 检查邮箱中的新消息（通过 NotificationBus 推送）
2. [Plan 模式] 制定执行计划 → 提交审批 → 等待 leader 决策
3. [Execute 模式] 使用 LLM 推理 + 工具调用来执行任务
4. 完成后通过 TokenTracker 记录用量
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from .orchestrator import Agent, AgentState
from .tools import ToolRegistry

logger = logging.getLogger(__name__)

# ── Tool allowlists for plan/execute modes ───────────────────────────

PLAN_MODE_TOOLS = {
    "read_file", "list_tasks", "list_agents",
    "submit_plan", "check_plan_status",
    "check_mailbox", "send_message",
    "respond_to_shutdown",
}

EXECUTE_MODE_TOOLS = {
    "claim_task", "complete_task", "fail_task",
    "send_message", "check_mailbox",
    "list_agents", "list_tasks",
    "read_file", "write_file",
    "respond_to_shutdown",
}

# ── System prompt template ───────────────────────────────────────────

SYSTEM_PROMPT_BASE = """You are an AI agent named {name} with role {role} on team "{team_name}".
You collaborate with {team_size} other agents to complete tasks.

Your capabilities: {capabilities}

## Work process

1. Check messages with check_mailbox
2. If you have no task, claim one with claim_task
3. Understand the task, investigate the codebase with read_file
4. Execute the task using your tools
5. Mark completion with complete_task
6. Communicate with teammates via send_message when needed

## Important rules

- **One task at a time**: claim, execute, complete
- **Investigate before acting**: use read_file and list_tasks first
- **Collaborate**: message teammates if they can help
- **Record results**: describe what you did in complete_task's result field
- **Fail gracefully**: use fail_task with a clear reason if blocked

Working directory: {work_dir}
TEAM.md in the working directory provides project-specific context and guidelines.
"""

PLAN_MODE_PROMPT = """
## PLANNING MODE

You are in PLANNING mode. Before executing, you must submit a plan for approval.

Steps:
1. Use read_file / list_tasks / list_agents to understand the task
2. Create a detailed plan: what files to read/write, what approach to take
3. Call submit_plan with your plan
4. Wait for approval with check_plan_status
5. Once approved, execute your plan

Do NOT call complete_task or write_file until your plan is approved.
"""


# ── LLMAgent ──────────────────────────────────────────────────────────

class LLMAgent:
    """由 LLM 驱动的自主智能体，支持 Plan 审批、推送通知和 Token 追踪。"""

    def __init__(
        self,
        agent: Agent,
        tool_registry: ToolRegistry,
        task_manager,
        mailbox,
        team_name: str = "",
        work_dir: str = ".",
        model: str = "claude-sonnet-4-6",
        require_plan_approval: bool = False,
        plan_manager=None,
        notification_bus=None,
        token_tracker=None,
        hook_registry=None,
        provider=None,
        console=None,
        extra_body=None,
    ):
        self.agent = agent
        self.provider = provider  # LLMProvider instance
        self.console = console
        self.extra_body = extra_body  # 模型特定参数，如 {"enable_thinking": False}
        self.tool_registry = tool_registry
        self.task_manager = task_manager
        self.mailbox = mailbox
        self.team_name = team_name
        self.work_dir = work_dir
        self.model = model

        # 新系统集成
        self.require_plan_approval = require_plan_approval
        self.plan_manager = plan_manager
        self.notification_bus = notification_bus
        self.token_tracker = token_tracker
        self.hook_registry = hook_registry

        self._running = False
        self._stop_event = asyncio.Event()
        self._plan_mode = require_plan_approval  # true = planning, false = executing
        self._plan_id: Optional[str] = None
        self._shutdown_requested = False
        self._shutdown_response: Optional[tuple[bool, str]] = None
        self._rate_limit_retries = 0

    def _get_provider(self):
        if self.provider is None:
            from .llm_provider import create_provider
            self.provider = create_provider(model=self.model)
        return self.provider

    def _build_system_prompt(self, agent_count: int) -> str:
        capabilities = ", ".join(self.agent.capabilities) if self.agent.capabilities else "general task execution"
        prompt = SYSTEM_PROMPT_BASE.format(
            name=self.agent.name, role=self.agent.role.value,
            team_name=self.team_name, team_size=agent_count,
            capabilities=capabilities, work_dir=self.work_dir,
        )

        # 自动加载 TEAM.md（项目上下文）
        team_md = self._load_team_md()
        if team_md:
            prompt += f"\n\n## Project context (TEAM.md)\n\n{team_md}"

        if self._plan_mode:
            prompt += PLAN_MODE_PROMPT
        return prompt

    def _load_team_md(self) -> str:
        """从工作目录加载 TEAM.md 文件。"""
        import os
        md_path = os.path.join(self.work_dir, "TEAM.md")
        try:
            with open(md_path, encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            return ""
        except Exception:
            logger.warning(f"Failed to read TEAM.md from {md_path}")
            return ""

    def _get_active_schemas(self) -> list[dict]:
        """返回当前模式下的工具 schema。"""
        allowlist = PLAN_MODE_TOOLS if self._plan_mode else EXECUTE_MODE_TOOLS
        all_schemas = self.tool_registry.get_schemas()
        return [s for s in all_schemas if s["name"] in allowlist]

    async def run(self, agent_count: int = 1, poll_interval: float = 10.0) -> None:
        """启动自主代理循环。"""
        self._running = True
        self._stop_event.clear()
        system_prompt = self._build_system_prompt(agent_count)

        if self.notification_bus:
            self.notification_bus.register(self.agent.id)

        # 随机启动延迟，避免多 Agent 同时请求触发限流
        stagger = __import__('random').uniform(0.5, 3.0)
        await asyncio.sleep(stagger)

        if self.console:
            self.console.agent_start(self.agent.name, self.model)

        logger.info(f"LLMAgent {self.agent.name} started (plan_mode={self._plan_mode})")

        while self._running and not self._stop_event.is_set():
            try:
                # 检查计划审批状态
                await self._check_plan_decision()

                # 执行推理步骤
                await self._step(system_prompt, agent_count)

                # 检查 shutdown 响应
                if self._shutdown_requested:
                    fn = getattr(self.tool_registry, '_respond_to_shutdown_fn', None)
                    if fn and fn._response is not None:
                        accept, reason = fn._response
                        self._shutdown_response = (accept, reason)
                        fn._response = None
                        if accept:
                            logger.info(f"{self.agent.name} accepted shutdown: {reason}")
                            break
                        else:
                            logger.info(f"{self.agent.name} rejected shutdown: {reason}")
                            self._shutdown_requested = False
                    # 如果 agent 空闲且没有回复（没有 task），自动接受
                    elif not getattr(getattr(self.tool_registry, '_claim_task_fn', None), '_current_task_id', ''):
                        self._shutdown_response = (True, "Agent is idle")
                        logger.info(f"{self.agent.name} auto-accepted shutdown (idle)")
                        break

                # 触发 TEAMMATE_IDLE 钩子
                if self.hook_registry:
                    fn = getattr(self.tool_registry, '_claim_task_fn', None)
                    current = getattr(fn, '_current_task_id', '') if fn else ''
                    if not current:
                        from .hooks import HookEvent, HookContext
                        allowed, reason = await self.hook_registry.trigger(
                            HookEvent.TEAMMATE_IDLE,
                            HookContext(event=HookEvent.TEAMMATE_IDLE,
                                        agent_id=self.agent.id,
                                        agent_name=self.agent.name),
                        )
                        if not allowed:
                            self.agent.state = AgentState.BUSY
                            continue

            except Exception as exc:
                from .tasks import TaskStatus

                # 永久性错误 — 标记当前任务失败，不重试
                permanent = self._is_permanent_error(exc)
                if permanent:
                    fn = getattr(self.tool_registry, '_claim_task_fn', None)
                    current = getattr(fn, '_current_task_id', '') if fn else ''
                    if current:
                        self.task_manager.update_task_status(
                            current, TaskStatus.FAILED, f"Permanent error: {exc}"
                        )
                        fn._current_task_id = ""
                    self.agent.state = AgentState.IDLE
                    if self.console:
                        self.console.error(self.agent.name, str(exc))
                    logger.error(f"Agent {self.agent.name} permanent error, task failed: {exc}")
                    if self._shutdown_requested:
                        self._shutdown_response = (True, "Shutting down after error")
                        break
                    break  # 永久错误直接退出

                # 429 限流 — 指数退避 + 随机 jitter + 上限
                is_rate_limit = "429" in str(exc) or "rate" in str(exc).lower()
                if is_rate_limit:
                    retry_count = getattr(self, '_rate_limit_retries', 0) + 1
                    self._rate_limit_retries = retry_count
                    import random
                    # 上限: 6 次重试后放弃，标记任务失败
                    if retry_count > 6:
                        if self.console:
                            self.console.error(self.agent.name, "Rate limit exceeded 6 retries, giving up")
                        logger.error(f"Agent {self.agent.name} rate limited {retry_count} times, giving up")
                        fn = getattr(self.tool_registry, '_claim_task_fn', None)
                        current = getattr(fn, '_current_task_id', '') if fn else ''
                        if current:
                            self.task_manager.update_task_status(
                                current, TaskStatus.FAILED, "Rate limit exceeded after 6 retries"
                            )
                            fn._current_task_id = ""
                        self.agent.state = AgentState.IDLE
                        self._rate_limit_retries = 0
                        break
                    base = min(5 * (2 ** retry_count), 120)
                    delay = base + random.uniform(0, base * 0.3)  # jitter 防止 agent 同步
                    if self.console:
                        self.console.rate_limited(self.agent.name, retry_count, delay)
                    logger.warning(
                        f"Agent {self.agent.name} rate limited, retry #{retry_count} in {delay:.0f}s"
                    )
                else:
                    self._rate_limit_retries = 0
                    delay = 10
                    if self.console:
                        self.console.error(self.agent.name, str(exc))
                    logger.exception(f"Agent {self.agent.name} error, retrying")

                if self._shutdown_requested:
                    self._shutdown_response = (True, "Shutting down after error")
                    break

                await asyncio.sleep(delay)

            # 等待下一个事件或超时
            await self._wait_for_next_event(poll_interval)

        logger.info(f"LLMAgent {self.agent.name} stopped")

    @staticmethod
    def _is_permanent_error(exc: Exception) -> bool:
        """判断是否是永久性错误（不应重试）。"""
        name = type(exc).__name__
        msg = str(exc).lower()
        # OpenAI / DashScope 错误码
        permanent_names = {"PermissionDeniedError", "AuthenticationError", "BadRequestError"}
        if name in permanent_names:
            return True
        # HTTP 状态码
        for code in ["403", "401", "400"]:
            if f"error code: {code}" in msg or f"status code: {code}" in msg:
                return True
        # 配额相关
        if "quota" in msg or "exhausted" in msg or "free tier" in msg:
            return True
        return False

    async def _wait_for_next_event(self, fallback_timeout: float) -> None:
        """等待通知总线事件或超时。"""
        if self.notification_bus:
            notif = await self.notification_bus.consume(
                self.agent.id, timeout=fallback_timeout
            )
            if notif:
                logger.debug(f"{self.agent.name} woke by: {notif.type}")
                if notif.type == "shutdown_request":
                    self._shutdown_requested = True

    async def _check_plan_decision(self) -> None:
        """检查是否有计划被审批/拒绝。"""
        if not self.plan_manager or not self._plan_id:
            return

        plan = self.plan_manager.get(self._plan_id)
        if not plan:
            return

        if plan.status.value == "approved":
            self._plan_mode = False
            self._plan_id = None
            logger.info(f"{self.agent.name} plan approved, entering execute mode")
        elif plan.status.value == "rejected":
            # 保持 plan mode，在下一个 step 中包含拒绝反馈
            logger.info(f"{self.agent.name} plan rejected: {plan.feedback}")

    async def _step(self, system_prompt: str, agent_count: int = 1) -> None:
        """执行一个推理步骤。"""
        from .llm_provider import ToolUseBlock

        provider = self._get_provider()
        prompt = self._build_turn_prompt()

        messages = [{"role": "user", "content": prompt}]
        schemas = self._get_active_schemas()
        tools = schemas if schemas else None

        response = await provider.create_message(
            model=self.model,
            system_prompt=system_prompt,
            messages=messages,
            tools=tools or None,
            max_tokens=4096,
            extra_body=self.extra_body,
        )

        # 记录 token 用量
        if self.token_tracker:
            self.token_tracker.record(
                self.agent.id, self.agent.name, self.model,
                response.usage.input_tokens, response.usage.output_tokens,
            )

        # 处理工具调用循环
        max_tool_rounds = 5
        for _ in range(max_tool_rounds):
            if response.stop_reason == "end_turn":
                if self.console:
                    thought = ""
                    for b in response.content:
                        if isinstance(b, str):
                            thought = b
                            break
                    if thought:
                        self.console.agent_step(self.agent.name, thought)
                break

            if response.stop_reason == "tool_use":
                tool_blocks = [b for b in response.content if isinstance(b, ToolUseBlock)]
                if not tool_blocks:
                    break

                tool_results = []
                for block in tool_blocks:
                    logger.info(f"{self.agent.name} tool: {block.name}({block.input})")
                    result = await self.tool_registry.execute(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })

                    if self.console:
                        self.console.tool_call(self.agent.name, block.name, block.input, result)

                    # 如果调用了 submit_plan，设置 plan 等待状态
                    if block.name == "submit_plan":
                        fn = getattr(self.tool_registry, '_submit_plan_fn', None)
                        if fn:
                            self._plan_id = getattr(fn, '_current_plan_id', None)

                messages.append({"role": "assistant", "content": response.content})
                messages.append({"role": "user", "content": tool_results})

                response = await provider.create_message(
                    model=self.model,
                    system_prompt=system_prompt,
                    messages=messages,
                    tools=tools or None,
                    max_tokens=4096,
                    extra_body=self.extra_body,
                )

                if self.token_tracker:
                    self.token_tracker.record(
                        self.agent.id, self.agent.name, self.model,
                        response.usage.input_tokens, response.usage.output_tokens,
                    )
            else:
                break

        # 更新 agent 状态
        self._update_state()

    def _update_state(self) -> None:
        fn = getattr(self.tool_registry, '_claim_task_fn', None)
        current_task_id = getattr(fn, '_current_task_id', '') if fn else ''
        if current_task_id:
            self.agent.state = AgentState.BUSY
            self.agent.current_task_id = current_task_id
        else:
            self.agent.state = AgentState.IDLE
            self.agent.current_task_id = None

    def _build_turn_prompt(self) -> str:
        fn = getattr(self.tool_registry, '_claim_task_fn', None)
        current_task_id = getattr(fn, '_current_task_id', '') if fn else ''

        parts = []

        if current_task_id:
            task = self.task_manager.get_task(current_task_id)
            if task:
                parts.append(
                    f"## Current task\n"
                    f"You are working on: **{task.name}** ({task.id})\n"
                    f"Description: {task.description or 'None'}\n"
                )
                if self._plan_mode:
                    parts.append(
                        "**You are in PLANNING mode.** "
                        "Investigate the task, then call submit_plan with your execution plan. "
                        "Do NOT execute yet."
                    )
                else:
                    parts.append("Execute this task and call complete_task when done.")
            else:
                parts.append("Your current task was removed. Find a new one.")
        else:
            parts.append(
                "## Status: IDLE\n"
                "You have no task. Use claim_task to claim an available task, "
                "or use list_tasks to see what's available."
            )

        # Plan 拒绝反馈
        if self._plan_id and self.plan_manager:
            plan = self.plan_manager.get(self._plan_id)
            if plan and plan.status.value == "rejected":
                parts.append(
                    f"\n**Your plan was REJECTED.** Reason: {plan.feedback}\n"
                    "Revise your plan and call submit_plan again."
                )
                self._plan_id = None

        # 未读消息
        unread = self.mailbox.get_unread_count(self.agent.name)
        if unread > 0:
            parts.append(f"\nYou have **{unread}** unread messages. Use check_mailbox.")

        # Shutdown 请求
        if self._shutdown_requested:
            parts.append(
                "\n## SHUTDOWN REQUESTED\n"
                "The team lead has requested that you shut down.\n"
                "- If you are idle or can safely stop, call **respond_to_shutdown(accept=true)** with a brief reason.\n"
                "- If you are in the middle of critical work that cannot be interrupted, "
                "call **respond_to_shutdown(accept=false)** and explain why.\n"
                "- If you are idle with no task, you will be shut down automatically."
            )

        return "\n\n".join(parts)

    def request_shutdown(self) -> None:
        """Leader 调用此方法发起关闭请求。Agent 会在下一个推理步骤中处理。"""
        self._shutdown_requested = True
        logger.info(f"Shutdown requested for {self.agent.name}")

    def get_shutdown_response(self) -> Optional[tuple[bool, str]]:
        """获取 agent 对关闭请求的响应。(accepted, reason) 或 None（尚未响应）。"""
        return self._shutdown_response

    async def shutdown(self) -> None:
        """优雅关闭 agent。"""
        self._running = False
        self._stop_event.set()
        if self.notification_bus:
            self.notification_bus.unregister(self.agent.id)
        logger.info(f"LLMAgent {self.agent.name} shutting down")
