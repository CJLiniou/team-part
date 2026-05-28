# 自动加载 .env 文件（软依赖，未安装 python-dotenv 时静默跳过）
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

"""Agent Team - 多智能体编排框架。

用于管理智能体团队的框架，支持任务依赖、自主任务分配和结果聚合。
支持两种模式：
- 回调模式：Agent 通过预注册的回调函数执行任务
- LLM 模式：每个 Agent 由 Anthropic Claude 驱动，自主推理和工具调用

新增功能（v0.2.0）:
- Hooks 系统 — TaskCreated / TaskCompleted / TeammateIdle 质量门禁
- Plan 审批 — Agent 先规划 → Leader 审批 → 执行
- 自然语言建队 — 用自然语言描述即可创建团队
- Subagent 定义 — 可复用的 Agent 角色模板
- 推送通知 — 事件驱动取代轮询
- Token 追踪 — 按 Agent 统计 token 消耗和成本
"""

from .tasks import Task, TaskStatus, TaskManager
from .orchestrator import (
    Agent, AgentRole, AgentState, TeamOrchestrator, ExecutionResult
)
from .mailbox import Message, MessageStatus, AgentMailbox
from .tools import ToolRegistry, create_default_registry, set_agent_context
from .llm_agent import LLMAgent
from .hooks import HookRegistry, HookEvent, HookContext
from .planning import PlanManager, PlanRequest, PlanStatus
from .notifications import NotificationBus, Notification
from .token_tracker import TokenTracker, TokenRecord, AgentTokenSummary
from .profiles import AgentProfile, ProfileRegistry
from .llm_provider import (
    LLMProvider, AnthropicProvider, OpenAIProvider,
    LLMResponse, ToolUseBlock, TokenUsage, create_provider,
)
from .observability import TeamConsole

__version__ = "0.2.0"
__all__ = [
    # 任务管理
    "Task", "TaskStatus", "TaskManager",
    # 团队编排
    "Agent", "AgentRole", "AgentState", "TeamOrchestrator", "ExecutionResult",
    # 邮箱系统
    "Message", "MessageStatus", "AgentMailbox",
    # 工具系统
    "ToolRegistry", "create_default_registry", "set_agent_context",
    # LLM Agent
    "LLMAgent",
    # Hooks 系统
    "HookRegistry", "HookEvent", "HookContext",
    # Plan 审批
    "PlanManager", "PlanRequest", "PlanStatus",
    # 推送通知
    "NotificationBus", "Notification",
    # Token 追踪
    "TokenTracker", "TokenRecord", "AgentTokenSummary",
    # Subagent 定义
    "AgentProfile", "ProfileRegistry",
    # LLM Provider 抽象层
    "LLMProvider", "AnthropicProvider", "OpenAIProvider",
    "LLMResponse", "ToolUseBlock", "TokenUsage", "create_provider",
    # 可观测性
    "TeamConsole",
]
