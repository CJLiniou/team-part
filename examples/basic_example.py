"""基本示例 - 演示回调模式和 LLM 模式两种用法。"""

import asyncio
import os
from pathlib import Path

from agent_team import TeamOrchestrator, AgentRole, Task


# ── 示例 1: 回调模式（无需 LLM）────────────────────────────────────

async def callback_mode():
    """使用预注册的回调函数运行团队。"""
    print("\n=== 回调模式 ===")

    db_path = Path("data/example_callback.db")
    if db_path.exists():
        db_path.unlink()

    orchestrator = TeamOrchestrator(db_path, "callback_team")

    for i in range(3):
        orchestrator.register_agent(
            f"worker-{i}", f"Worker-{i}", AgentRole.EXECUTOR,
            capabilities=["general"]
        )

    async def worker(task: Task) -> str:
        await asyncio.sleep(0.5)
        return f"Task '{task.name}' completed by callback."

    orchestrator.set_execution_callback("general", worker)

    orchestrator.task_manager.create_task(
        name="Task Alpha", description="First task", priority=2
    )
    orchestrator.task_manager.create_task(
        name="Task Beta", description="Second task", priority=1
    )
    orchestrator.task_manager.create_task(
        name="Task Gamma", description="Third task", priority=0
    )

    await orchestrator.distribute_work(max_concurrent=3)

    stats = orchestrator.get_team_stats()
    print(f"Completed: {stats['total_completed_tasks']}, "
          f"Failed: {stats['total_failed_tasks']}")

    orchestrator.cleanup_team(remove_data=True)


# ── 示例 2: LLM 模式 ────────────────────────────────────────────────

async def llm_mode():
    """使用 LLM 驱动的 agent 运行团队。需要 ANTHROPIC_API_KEY。"""
    print("\n=== LLM 模式 ===")

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("跳过: 需要设置 ANTHROPIC_API_KEY 环境变量")
        return

    db_path = Path("data/example_llm.db")
    if db_path.exists():
        db_path.unlink()

    orchestrator = TeamOrchestrator(db_path, "llm_team")
    orchestrator.set_work_dir(".")

    orchestrator.register_agent(
        "dev-1", "Developer Alpha", AgentRole.EXECUTOR,
        capabilities=["coding", "analysis"]
    )
    orchestrator.register_agent(
        "dev-2", "Developer Beta", AgentRole.EXECUTOR,
        capabilities=["coding", "testing"]
    )
    orchestrator.register_agent(
        "reviewer", "Code Reviewer", AgentRole.REVIEWER,
        capabilities=["review", "analysis"]
    )

    # 创建任务
    orchestrator.task_manager.create_task(
        name="Create utils.py",
        description="Create a utils.py file with a function 'add(a, b)' and a function 'multiply(a, b)'.",
        priority=2
    )
    orchestrator.task_manager.create_task(
        name="Write tests for utils",
        description="Write a test file test_utils.py that tests add() and multiply().",
        priority=1,
        depends_on=[]  # Will depend on first task after creation
    )
    # Link dependency
    tasks = orchestrator.task_manager.list_tasks()
    if len(tasks) >= 2:
        test_task = [t for t in tasks if t.name.startswith("Write")][0]
        # Actually let's keep them independent for parallelism
        pass

    print(f"启动 {len(orchestrator.agents)} 个 LLM agent...")
    # Run with a timeout so the example doesn't run forever
    try:
        await asyncio.wait_for(
            orchestrator.run_llm_team(model="claude-sonnet-4-6"),
            timeout=300  # 5 minutes max
        )
    except asyncio.TimeoutError:
        print("示例超时，正在关闭 agent...")
        await orchestrator.shutdown_all()

    stats = orchestrator.get_team_stats()
    print(f"Completed: {stats['total_completed_tasks']}, "
          f"Failed: {stats['total_failed_tasks']}")

    orchestrator.cleanup_team(remove_data=True)


# ── 主入口 ───────────────────────────────────────────────────────────

async def main():
    await callback_mode()
    await llm_mode()


if __name__ == "__main__":
    asyncio.run(main())
