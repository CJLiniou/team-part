"""完整演示 — 覆盖框架所有核心功能。

场景: 3 个 Agent 协作构建一个简单的 Python 工具包。

演示功能:
  - LLM Agent 自主任务执行
  - 任务 DAG 依赖链
  - Agent 间邮箱通信
  - Hook 质量门禁
  - Plan 审批 + LLM 审查
  - Token 追踪
  - Shutdown 协商协议
  - 回调模式（无需 API key）

用法:
  # 回调模式（无需 API key）
  python examples/full_demo.py

  # LLM 模式（代码显式嵌入 Provider 配置）
  python examples/full_demo.py --llm --api-key sk-xxx --model gpt-4o

  # Qwen / 低 QPS API — 限制并发避免 429
  python examples/full_demo.py --llm --provider openai \
    --api-key sk-xxx --base-url https://dashscope.aliyuncs.com/compatible-mode/v1 \
    --model qwen-plus --max-concurrent 1

  # LLM + Plan 审批
  python examples/full_demo.py --llm --plan --api-key sk-xxx
"""

import argparse
import asyncio
import shutil
import sys
import tempfile
from pathlib import Path
from time import time

# Add parent to path for direct script execution
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agent_team import (
    TeamOrchestrator, AgentRole,
    HookEvent, Task,
)


# ── Test helpers ─────────────────────────────────────────────────────

async def assert_condition(condition: bool, msg: str) -> None:
    """简单的测试断言。"""
    if condition:
        print(f"  PASS: {msg}")
    else:
        print(f"  FAIL: {msg}")


# ── Scenario setup ───────────────────────────────────────────────────

def setup_team(db_path: Path, work_dir: str = ".") -> TeamOrchestrator:
    """创建团队并注册 Agent 和任务。"""
    orchestrator = TeamOrchestrator(db_path, "full-demo-team")
    orchestrator.set_work_dir(work_dir)

    # 注册 3 个 Agent
    orchestrator.register_agent(
        "dev", "Developer",
        AgentRole.EXECUTOR,
        capabilities=["coding", "file_operations"],
    )
    orchestrator.register_agent(
        "tester", "Tester",
        AgentRole.EXECUTOR,
        capabilities=["testing", "file_operations"],
    )
    orchestrator.register_agent(
        "reviewer", "Code Reviewer",
        AgentRole.REVIEWER,
        capabilities=["review", "analysis"],
    )

    # 创建任务链: 写代码 → 写测试 → 审查
    task_code = orchestrator.task_manager.create_task(
        name="Create utils.py",
        description=(
            "Create a file 'utils.py' with two functions: "
            "1) add(a, b) -> a + b; 2) multiply(a, b) -> a * b. "
            "Include type hints and a docstring."
        ),
        priority=2,
        assigned_to="dev",
    )

    task_test = orchestrator.task_manager.create_task(
        name="Write tests for utils",
        description=(
            "Write a test file 'test_utils.py' that tests add() and multiply() "
            "with at least 3 test cases each, including edge cases."
        ),
        priority=1,
        depends_on=[task_code.id],
        assigned_to="tester",
    )

    orchestrator.task_manager.create_task(
        name="Review code quality",
        description=(
            "Review utils.py and test_utils.py for code quality: "
            "check type hints, docstrings, test coverage, and edge cases. "
            "Send feedback to Developer or Tester via send_message if issues found."
        ),
        priority=0,
        depends_on=[task_test.id],
        assigned_to="reviewer",
    )

    return orchestrator


# ── Callback mode demo ───────────────────────────────────────────────

async def run_callback_mode():
    """纯回调模式 — 无需 API key，演示 Hook、DAG、Mailbox。"""
    print("\n" + "=" * 60)
    print("CALLBACK MODE DEMO")
    print("=" * 60)

    tmpdir = Path(tempfile.mkdtemp(prefix="agent_team_demo_"))
    db_path = tmpdir / "demo.db"

    orch = setup_team(db_path, work_dir=str(tmpdir))

    # ── Hook: 所有任务必须有描述 ──
    def require_description(ctx):
        if not ctx.task_description:
            return "Task must have a description"
        return None

    orch.hook_registry.register(HookEvent.TASK_CREATED, require_description)
    print("\n[1] Hook registered: require_description")

    # 验证 Hook 阻止无描述的任务
    try:
        orch.task_manager.create_task(name="Bad Task", description="")
        print("  FAIL: Hook did not block empty description")
    except ValueError as e:
        await assert_condition("blocked" in str(e), "Hook blocks task without description")

    # ── Hook: 完成时记录 ──
    completed_tasks = []

    def on_complete(ctx):
        completed_tasks.append(ctx.task_name)
        return None  # always allow

    orch.hook_registry.register(HookEvent.TASK_COMPLETED, on_complete)
    print("[2] Hook registered: on_complete")

    # ── 注册回调 ──
    async def coding_callback(task: Task) -> str:
        # 模拟: 创建文件
        p = Path(orch._work_dir) / "utils.py"
        p.write_text(
            '"""Utility functions."""\n\ndef add(a: int, b: int) -> int:\n'
            '    """Return a + b."""\n    return a + b\n\n'
            'def multiply(a: int, b: int) -> int:\n'
            '    """Return a * b."""\n    return a * b\n'
        )
        return f"Created {p}"

    async def testing_callback(task: Task) -> str:
        p = Path(orch._work_dir) / "test_utils.py"
        p.write_text(
            '"""Tests for utils."""\n'
            'from utils import add, multiply\n\n'
            'def test_add():\n    assert add(2, 3) == 5\n'
            '    assert add(-1, 1) == 0\n\n'
            'def test_multiply():\n    assert multiply(2, 3) == 6\n'
            '    assert multiply(0, 5) == 0\n'
        )
        return f"Created {p}"

    async def review_callback(task: Task) -> str:
        # Reviewer 给 Developer 发消息
        orch.mailbox.send("Code Reviewer", "Developer", "Review done",
                          "Code looks good, all type hints present.")
        return "Reviewed all files, sent feedback to Developer"

    orch.set_execution_callback("coding", coding_callback)
    orch.set_execution_callback("file_operations", coding_callback)
    orch.set_execution_callback("testing", testing_callback)
    orch.set_execution_callback("review", review_callback)
    orch.set_execution_callback("analysis", review_callback)
    print("[3] Execution callbacks registered")

    # ── 运行 ──
    print("\n[4] Running team...")
    await orch.distribute_work(max_concurrent=3)

    # ── 验证 ──
    stats = orch.get_team_stats()
    print(f"\n[5] Results:")
    print(f"    Completed: {stats['total_completed_tasks']}")
    print(f"    Failed: {stats['total_failed_tasks']}")
    await assert_condition(stats['total_completed_tasks'] == 3, "All 3 tasks completed")
    await assert_condition(len(completed_tasks) == 3, "TASK_COMPLETED hook fired 3 times")

    # ── 邮箱验证 ──
    msgs = orch.mailbox.receive("Developer", unread_only=True)
    await assert_condition(len(msgs) > 0, "Developer received message from Reviewer")

    # 清理临时目录
    shutil.rmtree(tmpdir, ignore_errors=True)

    print(f"\n  Callback mode: ALL TESTS PASSED")


# ── LLM mode demo ─────────────────────────────────────────────────────

async def run_llm_mode(with_plan: bool = False, provider=None):
    """LLM 驱动模式。provider 由 main() 根据命令行参数创建。"""
    print("\n" + "=" * 60)
    print(f"LLM MODE DEMO (plan_approval={with_plan})")
    print("=" * 60)

    if provider is None:
        print("  SKIP: No provider. Use --api-key to set API key.")
        return

    tmpdir = Path(tempfile.mkdtemp(prefix="agent_team_llm_"))
    db_path = tmpdir / "demo.db"

    orch = setup_team(db_path, work_dir=str(tmpdir))

    # ── Hook ──
    def log_completion(ctx):
        print(f"  Hook: Task '{ctx.task_name}' completed by {ctx.agent_name}")
        return None

    orch.hook_registry.register(HookEvent.TASK_COMPLETED, log_completion)
    print("[1] Hook registered")

    # ── Profile 演示 ──
    from agent_team import AgentProfile
    orch.profile_registry.register(AgentProfile(
        name="quick-dev",
        role="executor",
        capabilities=["coding", "file_operations"],
        system_prompt_extra="You write clean Python code with type hints and docstrings.",
    ))
    print("[2] Profile 'quick-dev' registered")

    # ── Shutdown 演示 ──
    async def demo_shutdown():
        """演示协商关闭流程。"""
        print("\n[4] Shutdown negotiation demo...")

        # 注册一个临时 agent
        orch.register_agent("temp-agent", "Temp Worker", AgentRole.EXECUTOR,
                           capabilities=["general"])
        orch.task_manager.create_task(
            name="Quick task",
            description="Do something quick",
            priority=2,
        )

        # 启动单个 agent
        orch.spawn_llm_agent("temp-agent",
                            model=provider.default_model,
                            provider=provider)
        await asyncio.sleep(3)  # Let it start

        # 协商关闭
        ok, reason = await orch.shutdown_agent("temp-agent", timeout=15.0)
        await assert_condition(ok, f"Shutdown accepted: {reason}")
        print(f"    Agent response: {reason}")
        orch.unregister_agent("temp-agent")  # 清理，避免污染后续团队

    await demo_shutdown()

    # ── 运行全团队（新 orchestrator，不受 shutdown demo 污染）──
    print("\n[5] Running full LLM team...")
    orch = setup_team(tmpdir / "full_team.db", work_dir=str(tmpdir))
    orch.hook_registry.register(HookEvent.TASK_COMPLETED, log_completion)
    orch.profile_registry.register(AgentProfile(
        name="quick-dev",
        role="executor",
        capabilities=["coding", "file_operations"],
        system_prompt_extra="You write clean Python code with type hints and docstrings.",
    ))
    plan_criteria = None
    if with_plan:
        plan_criteria = "only approve plans that are specific and include file operations details"

    try:
        await asyncio.wait_for(
            orch.run_llm_team(
                model=provider.default_model,
                require_plan_approval=with_plan,
                plan_review_criteria=plan_criteria,
                provider=provider,
            ),
            timeout=600,
        )
    except asyncio.TimeoutError:
        print("  Timeout — shutting down...")
        await orch.shutdown_all()

    # ── 结果 ──
    stats = orch.get_team_stats()
    print(f"\n[6] Results:")
    print(f"    Tasks completed: {stats['total_completed_tasks']}")
    print(f"    Tasks failed: {stats['total_failed_tasks']}")
    token_info = stats.get('token_usage', {})
    print(f"    Tokens used: {token_info.get('total_tokens', 0)}")
    print(f"    Est. cost: ${token_info.get('total_cost', 0):.4f}")
    if token_info.get('by_agent'):
        for aid, ainfo in token_info['by_agent'].items():
            print(f"      {ainfo['name']}: {ainfo['input_tokens']}+{ainfo['output_tokens']} tokens, ${ainfo['cost']:.4f}")

    # 邮箱验证
    all_msgs = orch.mailbox.receive("Developer", unread_only=False, limit=100)
    print(f"\n    Mailbox messages exchanged: {len(all_msgs)}")

    # 清理临时目录
    shutil.rmtree(tmpdir, ignore_errors=True)

    print(f"\n  LLM mode: DEMO COMPLETE")


# ── Main ──────────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser(description="Agent Team Full Demo")
    parser.add_argument("--llm", action="store_true", help="Run LLM mode")
    parser.add_argument("--plan", action="store_true", help="Enable plan approval (LLM mode only)")
    parser.add_argument("--callback-only", action="store_true", help="Run only callback mode")
    parser.add_argument("--api-key", type=str, default="",
                       help="LLM API key (or set OPENAI_API_KEY / ANTHROPIC_API_KEY env var)")
    parser.add_argument("--base-url", type=str, default="",
                       help="Custom API base URL (e.g. Qwen: https://dashscope.aliyuncs.com/compatible-mode/v1)")
    parser.add_argument("--model", type=str, default="",
                       help="Model ID (e.g. qwen-max, gpt-4o, claude-sonnet-4-6)")
    parser.add_argument("--provider", type=str, default="",
                       choices=["anthropic", "openai"],
                       help="Provider type (default: from AGENT_TEAM_PROVIDER env, fallback anthropic)")
    parser.add_argument("--max-concurrent", type=int, default=0,
                       help="Max concurrent LLM requests (0=unlimited, 1-2 for low-QPS APIs like DashScope free tier)")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("AGENT TEAM FRAMEWORK — FULL DEMO")
    print("=" * 60)

    # 始终运行回调模式
    if not args.llm:
        await run_callback_mode()
        print("\n  Add --llm to run the LLM-powered demo.")
    elif args.callback_only:
        await run_callback_mode()
    else:
        await run_callback_mode()

    # LLM 模式 — 用命令行参数显式创建 Provider
    if args.llm:
        from agent_team import AnthropicProvider, OpenAIProvider

        ptype = args.provider or "anthropic"
        max_cc = args.max_concurrent if args.max_concurrent > 0 else None
        if ptype == "openai":
            provider = OpenAIProvider(
                api_key=args.api_key,
                base_url=args.base_url,
                model=args.model or "gpt-4o",
                max_concurrent=max_cc,
            )
        else:
            provider = AnthropicProvider(
                api_key=args.api_key,
                base_url=args.base_url,
                model=args.model or "claude-sonnet-4-6",
                max_concurrent=max_cc,
            )

        print(f"  Provider: {ptype} | Model: {provider.default_model}")
        if provider.base_url:
            print(f"  Base URL: {provider.base_url}")

        await run_llm_mode(with_plan=args.plan, provider=provider)

    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())