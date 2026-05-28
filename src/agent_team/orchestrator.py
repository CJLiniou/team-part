"""智能体团队编排和协调。

管理多个智能体协同工作、处理任务分配、通信和结果聚合。
支持基于回调函数和基于 LLM 的两种 agent 模式。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional, Callable, Any, Dict, List
import asyncio
import json
import logging
import shutil
from uuid import uuid4

from .tasks import TaskManager, Task, TaskStatus
from .mailbox import AgentMailbox
from .hooks import HookRegistry, HookEvent, HookContext
from .planning import PlanManager
from .notifications import NotificationBus
from .token_tracker import TokenTracker
from .profiles import ProfileRegistry, AgentProfile
from .observability import TeamConsole


logger = logging.getLogger(__name__)


class AgentRole(str, Enum):
    """团队中的智能体角色。"""
    EXECUTOR = "executor"          # 执行任务
    COORDINATOR = "coordinator"    # 协调任务流
    REVIEWER = "reviewer"          # 审核结果
    SPECIALIST = "specialist"      # 领域专家


class AgentState(str, Enum):
    """智能体执行状态。"""
    IDLE = "idle"          # 空闲
    BUSY = "busy"          # 忙碌
    WAITING = "waiting"    # 等待
    ERROR = "error"        # 错误


@dataclass
class Agent:
    """团队中的智能体。"""
    id: str
    name: str
    role: AgentRole
    capabilities: list[str] = field(default_factory=list)
    state: AgentState = AgentState.IDLE
    current_task_id: Optional[str] = None
    last_heartbeat: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    error_message: str = ""
    completed_tasks: int = 0
    failed_tasks: int = 0
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """转换为字典。"""
        return {
            'id': self.id,
            'name': self.name,
            'role': self.role.value,
            'capabilities': self.capabilities,
            'state': self.state.value,
            'current_task_id': self.current_task_id,
            'last_heartbeat': self.last_heartbeat.isoformat(),
            'error_message': self.error_message,
            'completed_tasks': self.completed_tasks,
            'failed_tasks': self.failed_tasks,
            'metadata': self.metadata
        }


@dataclass
class ExecutionResult:
    """任务执行结果。"""
    task_id: str
    agent_id: str
    status: TaskStatus
    output: str = ""
    error: str = ""
    metadata: dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict:
        """转换为字典。"""
        return {
            'task_id': self.task_id,
            'agent_id': self.agent_id,
            'status': self.status.value,
            'output': self.output,
            'error': self.error,
            'metadata': self.metadata,
            'timestamp': self.timestamp.isoformat()
        }


class TeamOrchestrator:
    """编排智能体团队工作。
    
    特性：
    - 智能体注册和状态管理
    - 任务分配和调度
    - 智能体间通信
    - 结果收集和聚合
    - 团队统计和监控
    - 回调函数支持自定义执行逻辑
    """

    def __init__(self, db_path: Path, team_name: str = "default",
                 mailbox_path: Optional[Path] = None):
        """初始化团队编排器。

        Args:
            db_path: 任务数据库路径
            team_name: 团队名称
            mailbox_path: 邮箱数据库路径（默认与 db_path 同目录）
        """
        self.team_name = team_name
        self.hook_registry = HookRegistry()
        self.task_manager = TaskManager(db_path, hook_registry=self.hook_registry)
        self.mailbox_path = mailbox_path or (db_path.parent / f"{team_name}_mailbox.db")
        self.mailbox = AgentMailbox(self.mailbox_path)
        self.plan_manager = PlanManager()
        self.notification_bus = NotificationBus()
        self.token_tracker = TokenTracker()
        self.profile_registry = ProfileRegistry()
        self.agents: Dict[str, Agent] = {}
        self.execution_callbacks: Dict[str, Callable] = {}
        self.results: List[ExecutionResult] = []
        self._llm_agent_tasks: Dict[str, asyncio.Task] = {}
        self._llm_agents: Dict[str, Any] = {}  # LLMAgent instances
        self._lock = asyncio.Lock()
        self._work_dir = "."
        self.console = TeamConsole()

    def register_agent(self, agent_id: str, name: str, role: AgentRole,
                      capabilities: Optional[list[str]] = None,
                      metadata: Optional[dict] = None) -> Agent:
        """注册新智能体。
        
        Args:
            agent_id: 唯一的智能体标识
            name: 人类可读的智能体名称
            role: 智能体角色
            capabilities: 能力名称列表
            metadata: 可选的元数据
            
        Returns:
            已注册的 Agent
        """
        agent = Agent(
            id=agent_id,
            name=name,
            role=role,
            capabilities=capabilities or [],
            metadata=metadata or {}
        )
        self.agents[agent_id] = agent
        logger.info(f"智能体已注册: {name} ({agent_id}) - {role}")
        return agent

    def unregister_agent(self, agent_id: str) -> bool:
        """注销智能体。
        
        Args:
            agent_id: 智能体 ID
            
        Returns:
            成功注销返回 True，未找到返回 False
        """
        if agent_id in self.agents:
            del self.agents[agent_id]
            logger.info(f"智能体已注销: {agent_id}")
            return True
        return False

    def get_agent(self, agent_id: str) -> Optional[Agent]:
        """根据 ID 获取智能体。
        
        Args:
            agent_id: 智能体 ID
            
        Returns:
            找到的 Agent，不存在返回 None
        """
        return self.agents.get(agent_id)

    def list_agents(self, role: Optional[AgentRole] = None,
                   state: Optional[AgentState] = None) -> list[Agent]:
        """列出智能体，支持可选过滤。
        
        Args:
            role: 按角色过滤
            state: 按状态过滤
            
        Returns:
            智能体列表
        """
        agents = list(self.agents.values())

        if role:
            agents = [a for a in agents if a.role == role]

        if state:
            agents = [a for a in agents if a.state == state]

        return agents

    def set_execution_callback(self, capability: str,
                              callback: Callable[[Task], Any]) -> None:
        """为任���执行注册回调。
        
        Args:
            capability: 能力名称
            callback: 异步函数 (task) -> result
        """
        self.execution_callbacks[capability] = callback
        logger.info(f"Execution callback registered: {capability}")

    def set_work_dir(self, path: str) -> None:
        self._work_dir = path

    # ── Subagent profiles ─────────────────────────────────────────

    def spawn_from_profile(self, profile_name: str, agent_id: str,
                           model: str = "", require_plan_approval: bool = False) -> asyncio.Task:
        """从 AgentProfile 创建并启动 LLM agent。

        Args:
            profile_name: ProfileRegistry 中注册的 profile 名称
            agent_id: 新 agent 的唯一 ID
            model: 覆盖 profile 的 model 设置（留空则使用 profile 的 model）
            require_plan_approval: 是否需要计划审批
        """
        profile = self.profile_registry.get(profile_name)
        if not profile:
            raise ValueError(f"Profile '{profile_name}' not found. Available: {self.profile_registry.list()}")

        from .orchestrator import AgentRole
        role = AgentRole(profile.role)

        self.register_agent(
            agent_id=agent_id,
            name=profile.name,
            role=role,
            capabilities=profile.capabilities,
            metadata={"profile": profile_name, **profile.metadata},
        )

        use_model = model or profile.model
        return self.spawn_llm_agent(
            agent_id, model=use_model,
            system_prompt_extra=profile.system_prompt_extra,
            require_plan_approval=require_plan_approval,
        )

    # ── Natural language team creation ─────────────────────────────

    async def create_from_description(self, description: str,
                                      model: str = "claude-sonnet-4-6",
                                      provider=None) -> TeamOrchestrator:
        """从自然语言描述创建团队。

        调用 LLM 解析描述，自动创建 agents 和 tasks。

        Args:
            description: 自然语言描述
            model: 用于解析的模型
            provider: LLMProvider 实例（可选，默认自动创建）
        """
        if provider is None:
            from .llm_provider import create_provider
            provider = create_provider(model=model)

        parse_prompt = f"""Parse the following team description into JSON with "agents" and "tasks" arrays.

Team description: {description}

Output ONLY valid JSON:
{{
  "agents": [
    {{"id": "agent-1", "name": "...", "role": "executor|coordinator|reviewer|specialist", "capabilities": ["..."]}}
  ],
  "tasks": [
    {{"name": "...", "description": "...", "priority": 0|1|2, "depends_on": []}}
  ]
}}

Rules:
- Assign unique IDs like "agent-1", "agent-2"
- roles: executor (does work), coordinator (organizes), reviewer (checks), specialist (expert)
- capabilities: coding, testing, review, analysis, coordination, research
- priority: 2=high, 1=medium, 0=low
- depends_on: list of task indices (0-based) that must complete first
- 3-5 agents and 3-8 tasks is a good range"""

        response = await provider.create_message(
            model=model,
            system_prompt="",
            messages=[{"role": "user", "content": parse_prompt}],
            max_tokens=2048,
        )

        text = response.content[0] if isinstance(response.content[0], str) else str(response.content[0])
        # Extract JSON block
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]

        config = json.loads(text)

        # Register agents
        task_map = {}  # old index -> task id
        for a in config.get("agents", []):
            from .orchestrator import AgentRole as AR
            self.register_agent(
                agent_id=a["id"],
                name=a["name"],
                role=AR(a.get("role", "executor")),
                capabilities=a.get("capabilities", []),
            )

        # Create tasks
        for i, t in enumerate(config.get("tasks", [])):
            deps = []
            for idx in t.get("depends_on", []):
                if idx in task_map:
                    deps.append(task_map[idx])
            created = self.task_manager.create_task(
                name=t["name"],
                description=t.get("description", ""),
                priority=t.get("priority", 0),
                depends_on=deps,
            )
            task_map[i] = created.id

        logger.info(f"Team created from description: {len(self.agents)} agents, "
                     f"{len(task_map)} tasks")
        return self

    async def _review_plan_with_llm(self, plan, criteria: str) -> Optional[str]:
        """使用 LLM 审查计划。返回 None 表示批准，返回字符串表示拒绝原因。"""
        task = self.task_manager.get_task(plan.task_id)
        task_name = task.name if task else "Unknown"
        task_desc = task.description if task else ""

        review_prompt = f"""You are a team lead reviewing an execution plan from agent "{plan.agent_name}".

Task: {task_name}
Task description: {task_desc}

Agent's plan:
---
{plan.plan_text}
---

Review criteria:
{criteria}

Evaluate whether this plan meets the criteria. Respond with exactly ONE of these formats:

APPROVE: <one-line reason for approval>
REJECT: <reason for rejection, with specific feedback for the agent to improve>

Do NOT output anything else. Start your response with APPROVE or REJECT."""

        try:
            from .llm_provider import create_provider
            provider = create_provider(model="claude-haiku-4-5-20251001")
            response = await provider.create_message(
                model="claude-haiku-4-5-20251001",
                system_prompt="",
                messages=[{"role": "user", "content": review_prompt}],
                max_tokens=512,
            )
            text = response.content[0]
            if not isinstance(text, str):
                text = str(text)
            text = text.strip()

            if text.startswith("APPROVE"):
                reason = text[len("APPROVE:"):].strip() if ":" in text else text[len("APPROVE"):].strip()
                logger.info(f"Plan {plan.id} APPROVED by LLM: {reason}")
                return None  # None = approve
            elif text.startswith("REJECT"):
                reason = text[len("REJECT:"):].strip() if ":" in text else text[len("REJECT"):].strip()
                logger.info(f"Plan {plan.id} REJECTED by LLM: {reason}")
                return reason
            else:
                logger.warning(f"LLM review response unclear, auto-approving: {text[:100]}")
                return None
        except Exception as exc:
            logger.error(f"LLM plan review failed: {exc}, auto-approving")
            return None

    # ── Plan approval ─────────────────────────────────────────────

    def approve_plan(self, plan_id: str, feedback: str = "") -> bool:
        """批准一个待处理的计划。"""
        ok = self.plan_manager.approve(plan_id, feedback)
        if ok:
            plan = self.plan_manager.get(plan_id)
            if plan and self.notification_bus:
                asyncio.create_task(
                    self.notification_bus.publish(plan.agent_id, "plan_approved",
                                                   {"plan_id": plan_id, "feedback": feedback})
                )
        return ok

    def reject_plan(self, plan_id: str, reason: str) -> bool:
        """拒绝一个待处理的计划。"""
        ok = self.plan_manager.reject(plan_id, reason)
        if ok:
            plan = self.plan_manager.get(plan_id)
            if plan and self.notification_bus:
                asyncio.create_task(
                    self.notification_bus.publish(plan.agent_id, "plan_rejected",
                                                   {"plan_id": plan_id, "reason": reason})
                )
        return ok

    def list_pending_plans(self) -> list:
        """列出所有待审批的计划。"""
        return self.plan_manager.list_pending()

    # ── LLM Agent management ───────────────────────────────────────

    def spawn_llm_agent(self, agent_id: str, model: str = "claude-sonnet-4-6",
                        system_prompt_extra: str = "",
                        require_plan_approval: bool = False,
                        provider=None) -> asyncio.Task:
        if agent_id not in self.agents:
            raise ValueError(f"Agent {agent_id} not registered")

        from .tools import create_default_registry, set_agent_context
        from .llm_agent import LLMAgent

        if provider is None:
            from .llm_provider import create_provider
            provider = create_provider(model=model)

        registry = create_default_registry(
            self.task_manager, self.mailbox, self.agents, self._work_dir,
            plan_manager=self.plan_manager, notification_bus=self.notification_bus,
        )
        set_agent_context(registry, self.agents[agent_id].name, agent_id)

        llm_agent = LLMAgent(
            agent=self.agents[agent_id],
            tool_registry=registry,
            task_manager=self.task_manager,
            mailbox=self.mailbox,
            team_name=self.team_name,
            work_dir=self._work_dir,
            model=model,
            require_plan_approval=require_plan_approval,
            plan_manager=self.plan_manager,
            notification_bus=self.notification_bus,
            token_tracker=self.token_tracker,
            hook_registry=self.hook_registry,
            provider=provider,
            console=self.console,
        )

        task = asyncio.create_task(
            llm_agent.run(agent_count=len(self.agents)),
            name=f"llm_agent_{agent_id}"
        )
        self._llm_agent_tasks[agent_id] = task
        self._llm_agents[agent_id] = llm_agent
        self.agents[agent_id].metadata["llm_model"] = model
        self.agents[agent_id].metadata["require_plan_approval"] = require_plan_approval
        return task

    async def run_llm_team(self, model: str = "claude-sonnet-4-6",
                           poll_interval: float = 5.0,
                           require_plan_approval: bool = False,
                           plan_review_criteria: Optional[str] = None,
                           provider=None) -> None:
        """以 LLM 模式运行所有 agent。

        Args:
            model: agent 使用的模型 ID（claude-sonnet-4-6 或 gpt-4o 等）
            poll_interval: leader 检查状态的间隔
            require_plan_approval: agent 是否需要先提交计划才能执行
            plan_review_criteria: 计划审查标准。None=手动审批, "auto"=自动放行,
                                  其他字符串=用 LLM 按此标准审查计划。
            provider: LLMProvider 实例（可选，默认根据环境变量自动创建）。
                      所有 agent 共享此 provider。
        """
        if provider is None:
            from .llm_provider import create_provider
            provider = create_provider(model=model)

        for agent_id in self.agents:
            self.spawn_llm_agent(agent_id, model=model,
                                require_plan_approval=require_plan_approval,
                                provider=provider)

        # 配置计划审查策略
        if not require_plan_approval:
            pass  # 不需要审批
        elif plan_review_criteria is None:
            logger.info("Plan review: MANUAL — use approve_plan()/reject_plan()")
        elif plan_review_criteria == "auto":
            self.plan_manager.on_plan_submitted(lambda p: None)  # auto-approve
            logger.info("Plan review: AUTO-APPROVE")
        else:
            orch = self  # capture for closure

            async def llm_review_callback(plan):
                return await orch._review_plan_with_llm(plan, plan_review_criteria)

            self.plan_manager.on_plan_submitted(llm_review_callback)
            logger.info(f"Plan review: LLM — criteria: {plan_review_criteria}")

        self.console.leader_tick(f"LLM team started with {len(self.agents)} agents")
        self.console.team_dashboard(self.agents, self.task_manager.list_tasks())

        self.notification_bus.register("__leader__")

        dash_interval = 15  # 每 15 秒刷新仪表盘
        last_dash = 0
        while True:
            stats = self.task_manager.get_task_stats()
            pending_tasks = stats.get('pending', 0)
            in_progress = stats.get('in_progress', 0)
            pending_plans = len(self.plan_manager.list_pending())

            if pending_tasks == 0 and in_progress == 0 and pending_plans == 0:
                self.console.leader_tick("All tasks completed")
                break

            # 定期打印团队仪表盘
            import time as _time
            now = _time.time()
            if now - last_dash > dash_interval:
                self.console.team_dashboard(self.agents, self.task_manager.list_tasks())
                last_dash = now

            await self.notification_bus.consume("__leader__", timeout=poll_interval)

        await self.shutdown_all()

    async def shutdown_agent(self, agent_id: str, force: bool = False,
                            timeout: float = 30.0) -> tuple[bool, str]:
        """向 agent 发送关闭请求，等待其响应。

        Args:
            agent_id: agent ID
            force: 跳过协商，直接强制关闭
            timeout: 等待 agent 响应的超时秒数

        Returns:
            (success, reason) — success=True 表示已关闭，False 表示被拒绝或超时
        """
        if agent_id not in self._llm_agent_tasks:
            return False, "Agent not found"

        # 强制关闭
        if force:
            task = self._llm_agent_tasks.pop(agent_id, None)
            self._llm_agents.pop(agent_id, None)
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            self.agents[agent_id].state = AgentState.IDLE
            logger.info(f"Agent {agent_id} force-stopped")
            return True, "Force-stopped"

        # 协商关闭
        llm_agent = self._llm_agents.get(agent_id)
        if not llm_agent:
            return False, "Agent instance not found"

        # 发送关闭请求
        llm_agent.request_shutdown()
        if self.notification_bus:
            await self.notification_bus.publish(agent_id, "shutdown_request")

        # 等待响应
        deadline = asyncio.get_event_loop().time() + timeout
        while asyncio.get_event_loop().time() < deadline:
            response = llm_agent.get_shutdown_response()
            if response is not None:
                accept, reason = response
                if accept:
                    # Agent 已接受，等待其自行停止
                    task = self._llm_agent_tasks.get(agent_id)
                    if task:
                        try:
                            await asyncio.wait_for(task, timeout=10)
                        except (asyncio.TimeoutError, asyncio.CancelledError):
                            task.cancel()
                    self._llm_agent_tasks.pop(agent_id, None)
                    self._llm_agents.pop(agent_id, None)
                    self.agents[agent_id].state = AgentState.IDLE
                    logger.info(f"Agent {agent_id} shut down gracefully: {reason}")
                    return True, reason
                else:
                    logger.info(f"Agent {agent_id} rejected shutdown: {reason}")
                    return False, reason

            await asyncio.sleep(0.5)

        # 超时 — 强制关闭
        logger.warning(f"Agent {agent_id} did not respond to shutdown, force-stopping")
        return await self.shutdown_agent(agent_id, force=True)

    async def shutdown_all(self) -> None:
        for agent_id in list(self._llm_agent_tasks):
            ok, reason = await self.shutdown_agent(agent_id, timeout=10.0)
            if not ok:
                logger.warning(f"Force-stopping {agent_id}: {reason}")
                await self.shutdown_agent(agent_id, force=True)

    def cleanup_team(self, remove_data: bool = False) -> None:
        lock_dir = self.task_manager._lock_dir
        if lock_dir.exists() and remove_data:
            shutil.rmtree(lock_dir, ignore_errors=True)

        if remove_data:
            for path in [self.task_manager.db_path, self.mailbox.db_path]:
                try:
                    if path.exists():
                        path.unlink()
                except PermissionError:
                    pass  # 后台连接可能尚未释放，交由 OS 后续清理
            logger.info(f"Team '{self.team_name}' data cleaned up")

    async def heartbeat(self, agent_id: str, state: AgentState = AgentState.IDLE,
                       current_task_id: Optional[str] = None,
                       error_message: str = "") -> bool:
        """记录智能体心跳。
        
        Args:
            agent_id: 智能体 ID
            state: 当前智能体状态
            current_task_id: 当前任务（如有）
            error_message: 错误信息（错误状态时）
            
        Returns:
            成功记录返回 True，智能体不存在返回 False
        """
        async with self._lock:
            agent = self.agents.get(agent_id)
            if not agent:
                return False

            agent.last_heartbeat = datetime.now(timezone.utc)
            agent.state = state
            agent.current_task_id = current_task_id
            agent.error_message = error_message

            return True

    async def claim_task(self, agent_id: str) -> Optional[Task]:
        """为智能体认领可用任务。
        
        Args:
            agent_id: 智能体 ID
            
        Returns:
            成功认领返回 Task，否则返回 None
        """
        async with self._lock:
            agent = self.agents.get(agent_id)
            if not agent:
                return None

            # 获取可用任务
            available = self.task_manager.get_available_tasks(agent_id)

            if not available:
                return None

            # 尝试认领第一个可用任务
            for task in available:
                if self.task_manager.claim_task(task.id, agent_id):
                    agent.state = AgentState.BUSY
                    agent.current_task_id = task.id
                    logger.info(f"智能体 {agent_id} 认领任务 {task.id}")
                    return task

            return None

    async def execute_task(self, agent_id: str, task_id: str) -> ExecutionResult:
        """执行任务（调用已注册的回调）。
        
        Args:
            agent_id: 智能体 ID
            task_id: 任务 ID
            
        Returns:
            ExecutionResult
        """
        agent = self.agents.get(agent_id)
        task = self.task_manager.get_task(task_id)

        if not agent or not task:
            return ExecutionResult(
                task_id=task_id,
                agent_id=agent_id,
                status=TaskStatus.FAILED,
                error="智能体或任务不存在"
            )

        try:
            # 查找匹配的回调
            callback = None
            for capability in agent.capabilities:
                if capability in self.execution_callbacks:
                    callback = self.execution_callbacks[capability]
                    break

            if not callback:
                # 检查默认回调
                if 'default' in self.execution_callbacks:
                    callback = self.execution_callbacks['default']

            if not callback:
                raise ValueError(f"没有可用的执行回调: {agent.capabilities}")

            # 执行任务
            logger.info(f"执行任务 {task_id} 使用智能体 {agent_id}")
            output = await callback(task) if asyncio.iscoroutinefunction(callback) else callback(task)

            # 更新任务
            self.task_manager.update_task_status(
                task_id, TaskStatus.COMPLETED, str(output)
            )

            result = ExecutionResult(
                task_id=task_id,
                agent_id=agent_id,
                status=TaskStatus.COMPLETED,
                output=str(output)
            )

            agent.completed_tasks += 1
            agent.state = AgentState.IDLE
            agent.current_task_id = None

            return result

        except Exception as e:
            logger.error(f"任务执行失败: {e}")
            self.task_manager.update_task_status(task_id, TaskStatus.FAILED)

            result = ExecutionResult(
                task_id=task_id,
                agent_id=agent_id,
                status=TaskStatus.FAILED,
                error=str(e)
            )

            agent.failed_tasks += 1
            agent.state = AgentState.ERROR
            agent.error_message = str(e)

            return result

    async def process_results(self, result: ExecutionResult) -> None:
        """处理和存储任务结果。
        
        Args:
            result: 要处理的 ExecutionResult
        """
        async with self._lock:
            self.results.append(result)
            logger.info(f"结果已记录，任务 {result.task_id}: {result.status.value}")

    async def distribute_work(self, max_concurrent: int = 5) -> None:
        """持续向可用智能体分配工作，当所有任务完成时自动退出。

        Args:
            max_concurrent: 最大并发任务数
        """
        active_tasks: set[str] = set()
        idle_rounds = 0
        max_idle_rounds = 3

        while True:
            # 检查已完成的任务
            for task_id in list(active_tasks):
                task = self.task_manager.get_task(task_id)
                if task and task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED):
                    active_tasks.discard(task_id)

            # 为空闲智能体认领新任务
            pending = self.task_manager.list_tasks(
                status=TaskStatus.PENDING, order_by="priority DESC, created_time ASC"
            )
            pending = [t for t in pending if not t.assigned_to
                       or t.assigned_to in self.agents]

            if pending and len(active_tasks) < max_concurrent:
                for agent in self.list_agents(state=AgentState.IDLE):
                    if len(active_tasks) >= max_concurrent:
                        break
                    task = await self.claim_task(agent.id)
                    if task:
                        active_tasks.add(task.id)
                        asyncio.create_task(self._execute_and_record(agent.id, task.id))

            # 退出条件：无活跃任务且无待处理任务
            if not active_tasks and not pending:
                idle_rounds += 1
                if idle_rounds >= max_idle_rounds:
                    logger.info("所有任务已完成，工作分配结束")
                    return
            else:
                idle_rounds = 0

            await asyncio.sleep(0.5)

    async def _execute_and_record(self, agent_id: str, task_id: str) -> None:
        """执行任务并记录结果。
        
        Args:
            agent_id: 智能体 ID
            task_id: 任务 ID
        """
        result = await self.execute_task(agent_id, task_id)
        await self.process_results(result)

    def get_team_stats(self) -> dict:
        """获取团队统计，包括 token 消耗。"""
        agents_by_state = {}
        for state in AgentState:
            agents_by_state[state.value] = len(self.list_agents(state=state))

        total_completed = sum(a.completed_tasks for a in self.agents.values())
        total_failed = sum(a.failed_tasks for a in self.agents.values())

        token_summary = self.token_tracker.team_summary()

        return {
            'team_name': self.team_name,
            'total_agents': len(self.agents),
            'agents_by_state': agents_by_state,
            'total_completed_tasks': total_completed,
            'total_failed_tasks': total_failed,
            'task_stats': self.task_manager.get_task_stats(),
            'results_recorded': len(self.results),
            'token_usage': token_summary,
        }

    def get_agent_stats(self, agent_id: str) -> Optional[dict]:
        """获取特定智能体的统计。
        
        Args:
            agent_id: 智能体 ID
            
        Returns:
            包含智能体统计的��典
        """
        agent = self.agents.get(agent_id)
        if not agent:
            return None

        return {
            'id': agent.id,
            'name': agent.name,
            'role': agent.role.value,
            'state': agent.state.value,
            'completed_tasks': agent.completed_tasks,
            'failed_tasks': agent.failed_tasks,
            'success_rate': (
                agent.completed_tasks / (agent.completed_tasks + agent.failed_tasks)
                if (agent.completed_tasks + agent.failed_tasks) > 0
                else 0
            ),
            'last_heartbeat': agent.last_heartbeat.isoformat()
        }

    def get_results(self, task_id: Optional[str] = None,
                   agent_id: Optional[str] = None,
                   status: Optional[TaskStatus] = None) -> list[ExecutionResult]:
        """获取已记录的结果。
        
        Args:
            task_id: 按任务 ID 过滤
            agent_id: 按智能体 ID 过滤
            status: 按状态过滤
            
        Returns:
            匹配的结果列表
        """
        results = self.results

        if task_id:
            results = [r for r in results if r.task_id == task_id]

        if agent_id:
            results = [r for r in results if r.agent_id == agent_id]

        if status:
            results = [r for r in results if r.status == status]

        return results

    def export_state(self) -> dict:
        """导出团队状态以便持久化。
        
        Returns:
            包含团队状态的字典
        """
        return {
            'team_name': self.team_name,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'agents': {aid: agent.to_dict() for aid, agent in self.agents.items()},
            'results': [r.to_dict() for r in self.results],
            'task_stats': self.task_manager.get_task_stats()
        }

    def import_state(self, state: dict) -> None:
        """从持久化数据导入团队状态。
        
        Args:
            state: 包含团队状态的字典
        """
        for agent_data in state.get('agents', {}).values():
            agent = Agent(
                id=agent_data['id'],
                name=agent_data['name'],
                role=AgentRole(agent_data['role']),
                capabilities=agent_data['capabilities'],
                state=AgentState(agent_data['state']),
                current_task_id=agent_data['current_task_id'],
                error_message=agent_data['error_message'],
                completed_tasks=agent_data['completed_tasks'],
                failed_tasks=agent_data['failed_tasks'],
                metadata=agent_data['metadata']
            )
            self.agents[agent.id] = agent
