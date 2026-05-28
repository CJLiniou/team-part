"""Agent Team CLI 入口点。

用法:
    python -m agent_team                  # 交互模式
    python -m agent_team run config.json  # 从配置文件运行团队
    python -m agent_team --help           # 显示帮助
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from .orchestrator import TeamOrchestrator, AgentRole


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="agent-team",
        description="LLM 驱动的多智能体团队编排框架",
    )
    sub = parser.add_subparsers(dest="command", help="子命令")

    # run
    run_parser = sub.add_parser("run", help="从配置文件运行团队")
    run_parser.add_argument("config", type=Path, help="团队配置 JSON 文件路径")

    # demo
    sub.add_parser("demo", help="运行回调模式演示")

    # llm
    llm_parser = sub.add_parser("llm", help="运行 LLM 模式演示（需要 ANTHROPIC_API_KEY）")
    llm_parser.add_argument("--model", default="claude-sonnet-4-6", help="Anthropic 模型 ID")
    llm_parser.add_argument("--config", type=Path, help="团队配置 JSON 文件路径")

    return parser.parse_args()


def load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


async def run_team(config: dict) -> None:
    db_path = Path(config.get("db_path", "data/team.db"))
    team_name = config.get("team_name", "agent-team")
    max_concurrent = config.get("max_concurrent", 5)

    orchestrator = TeamOrchestrator(db_path, team_name)

    # 注册智能体
    for a in config.get("agents", []):
        orchestrator.register_agent(
            agent_id=a["id"],
            name=a["name"],
            role=AgentRole(a.get("role", "executor")),
            capabilities=a.get("capabilities", []),
            metadata=a.get("metadata", {}),
        )

    # 创建任务
    for t in config.get("tasks", []):
        orchestrator.task_manager.create_task(
            name=t["name"],
            description=t.get("description", ""),
            priority=t.get("priority", 0),
            depends_on=t.get("depends_on", []),
        )

    # 运行工作分配
    print(f"团队 '{team_name}' 启动，{len(orchestrator.agents)} 个智能体，"
          f"{orchestrator.task_manager.get_task_stats().get('pending', 0)} 个待处理任务")

    try:
        await orchestrator.distribute_work(max_concurrent=max_concurrent)
    finally:
        stats = orchestrator.get_team_stats()
        print(f"\n团队统计: 完成 {stats['total_completed_tasks']}, "
              f"失败 {stats['total_failed_tasks']}")


async def run_demo() -> None:
    """运行一个简单的演示团队。"""
    from . import Task

    db_path = Path("data/demo_team.db")
    if db_path.exists():
        db_path.unlink()

    orchestrator = TeamOrchestrator(db_path, "demo-team")

    orchestrator.register_agent("agent-1", "研究员 Alpha", AgentRole.EXECUTOR,
                                capabilities=["research"])
    orchestrator.register_agent("agent-2", "研究员 Beta", AgentRole.EXECUTOR,
                                capabilities=["research"])
    orchestrator.register_agent("coordinator", "协调者", AgentRole.COORDINATOR,
                                capabilities=["coordination"])

    async def example_callback(task: Task) -> str:
        await asyncio.sleep(1)
        return f"[DEMO] 任务完成: {task.name}"

    orchestrator.set_execution_callback("research", example_callback)
    orchestrator.set_execution_callback("coordination", example_callback)

    orchestrator.task_manager.create_task(name="需求分析", description="分析项目需求", priority=2)
    orchestrator.task_manager.create_task(name="技术调研", description="调研技术方案", priority=1)
    orchestrator.task_manager.create_task(name="编写报告", description="汇总调研结果", priority=0,
                                          depends_on=[
                                              orchestrator.task_manager.list_tasks()[0].id,
                                              orchestrator.task_manager.list_tasks()[1].id,
                                          ])

    print("*** Agent Team Demo ***")
    await orchestrator.distribute_work(max_concurrent=3)

    stats = orchestrator.get_team_stats()
    print(f"\n结果: {stats['total_completed_tasks']} 已完成, "
          f"{stats['total_failed_tasks']} 失败")


async def run_llm_from_config(config_path: Path, model: str) -> None:
    from pathlib import Path as P

    config = load_config(config_path)
    db_path = P(config.get("db_path", "data/team.db"))
    team_name = config.get("team_name", "llm-team")
    work_dir = config.get("work_dir", ".")

    orchestrator = TeamOrchestrator(db_path, team_name)
    orchestrator.set_work_dir(work_dir)

    for a in config.get("agents", []):
        orchestrator.register_agent(
            agent_id=a["id"],
            name=a["name"],
            role=AgentRole(a.get("role", "executor")),
            capabilities=a.get("capabilities", []),
            metadata=a.get("metadata", {}),
        )

    for t in config.get("tasks", []):
        orchestrator.task_manager.create_task(
            name=t["name"],
            description=t.get("description", ""),
            priority=t.get("priority", 0),
            depends_on=t.get("depends_on", []),
        )

    print(f"LLM Team '{team_name}' starting with {len(orchestrator.agents)} agents...")
    await orchestrator.run_llm_team(model=model)

    stats = orchestrator.get_team_stats()
    print(f"Completed: {stats['total_completed_tasks']}, "
          f"Failed: {stats['total_failed_tasks']}")


async def run_llm_demo(model: str) -> None:
    """Run a demo LLM team."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY environment variable is required for LLM mode.")
        return

    db_path = Path("data/demo_llm.db")
    if db_path.exists():
        db_path.unlink()

    orchestrator = TeamOrchestrator(db_path, "llm-demo")
    orchestrator.set_work_dir(".")

    orchestrator.register_agent("dev-1", "Developer Alpha", AgentRole.EXECUTOR,
                                capabilities=["coding"])
    orchestrator.register_agent("dev-2", "Developer Beta", AgentRole.EXECUTOR,
                                capabilities=["coding", "testing"])

    orchestrator.task_manager.create_task(
        name="Create utils.py",
        description="Write a file called utils.py with add(a,b) and multiply(a,b) functions.",
        priority=2
    )
    orchestrator.task_manager.create_task(
        name="Create main.py",
        description="Write a file called main.py that imports from utils and prints add(2,3).",
        priority=1
    )

    print("*** LLM Agent Team Demo ***")
    try:
        await asyncio.wait_for(
            orchestrator.run_llm_team(model=model),
            timeout=300
        )
    except asyncio.TimeoutError:
        print("Demo timed out, shutting down...")
        await orchestrator.shutdown_all()

    stats = orchestrator.get_team_stats()
    print(f"Result: {stats['total_completed_tasks']} completed, "
          f"{stats['total_failed_tasks']} failed")

    orchestrator.cleanup_team(remove_data=True)


def main() -> None:
    args = parse_args()

    if args.command == "run":
        config = load_config(args.config)
        asyncio.run(run_team(config))
    elif args.command == "llm":
        if args.config:
            asyncio.run(run_llm_from_config(args.config, args.model))
        else:
            asyncio.run(run_llm_demo(args.model))
    elif args.command == "demo":
        asyncio.run(run_demo())
    else:
        print("Usage: python -m agent_team [run <config> | llm | demo]")
        print("  run <config>  - Run callback-mode team from JSON config")
        print("  llm [--config] - Run LLM-powered team (requires ANTHROPIC_API_KEY)")
        print("  demo           - Run callback-mode demo")
        print()
        print("Or use the Python API directly:")
        print("  from agent_team import TeamOrchestrator, LLMAgent, AgentRole")
        sys.exit(0)


if __name__ == "__main__":
    main()
