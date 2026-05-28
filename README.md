# Agent Team - 智能体团队编排框架

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 概述

**Agent Team** 是一个轻量级的多智能体编排框架，用于协调多个智能体（Claude Code sessions）协同工作。

### 核心特性

- 🤖 **智能体管理** - 注册、监控和协调多个智能体
- 📋 **共享任务列表** - 中央任务管理，支持依赖关系和优先级
- 💬 **邮箱系统** - 智能体间的点对点和广播消息传递
- 🔄 **自主工作分配** - 智能体自主认领可用任务
- 📊 **结果聚合** - 收集和汇总所有执行结果
- 🔒 **并发安全** - 使用文件锁实现并发访问控制

## 快速开始

### 安装

```bash
cd team
pip install -e .
```

### 基本使用

```python
from pathlib import Path
from agent_team import TeamOrchestrator, AgentRole

# 初始化团队编排器
orchestrator = TeamOrchestrator(
    db_path=Path("data/team.db"),
    team_name="my-team"
)

# 注册智能体
executor = orchestrator.register_agent(
    agent_id="agent-1",
    name="Executor",
    role=AgentRole.EXECUTOR,
    capabilities=["task_execution"]
)

# 创建任务
task = orchestrator.task_manager.create_task(
    name="示例任务",
    description="这是一个示例任务",
    priority=1
)

# 设置执行回调
def execute_task_callback(task):
    return f"已完成: {task.name}"

orchestrator.set_execution_callback(
    "task_execution",
    execute_task_callback
)

# 分配和执行工作
import asyncio
asyncio.run(orchestrator.distribute_work(max_concurrent=5))
```

## 核心概念

### 智能体（Agent）

智能体代表团队中的一个独立工作者。每个智能体有：

- **ID** - 唯一标识
- **名称** - 人类可读的名称
- **角色** - EXECUTOR、COORDINATOR、REVIEWER、SPECIALIST
- **能力** - 支持的任务类型列表
- **状态** - IDLE、BUSY、WAITING、ERROR

### 任务（Task）

任务是要完成的工作单位。每个任务有：

- **名称和描述** - 任务是什么
- **优先级** - 0=低, 1=中, 2=高
- **依赖** - 该任务依赖的其他任务（DAG 支持）
- **状态** - PENDING、IN_PROGRESS、COMPLETED、FAILED、BLOCKED

### 邮箱（Mailbox）

邮箱系统用于智能体间的异步通信。支持：

- 点对点消息
- 广播消息
- 消息持久化
- 已读/已送达状态跟踪

## 架构

```
TeamOrchestrator (团队编排器)
├── Agents (多个智能体)
│   ├── Agent 1 (独立的智能体)
│   ├── Agent 2
│   └── Agent 3
├── TaskManager (任务管理)
│   ├── 共享任务列表 (SQLite)
│   ├── 依赖图管理
│   └── 文件锁（并发控制）
├── AgentMailbox (邮箱系统)
│   ├── 消息存储
│   ├── 状态跟踪
│   └── 对话历史
└── Results (结果聚合)
    └── 执行结果列表
```

## 项目结构

```
team/
├── src/agent_team/
│   ├── __init__.py           # 模块导出
│   ├── tasks.py              # 任务管理（TaskManager, Task）
│   ├── orchestrator.py       # 团队编排（TeamOrchestrator, Agent）
│   └── mailbox.py            # 邮箱系统（AgentMailbox, Message）
├── pyproject.toml            # 项目配置
├── Dockerfile                # Docker 镜像定义
├── docker-compose.yml        # Docker Compose 编排
├── examples/
│   └── basic_example.py      # 基本使用示例
└── README.md                 # 本文件
```

## 使用示例

### 示例 1: 基本的并行任务执行

```python
from pathlib import Path
from agent_team import TeamOrchestrator, AgentRole
import asyncio

# 初始化
orchestrator = TeamOrchestrator(
    db_path=Path("data/team.db"),
    team_name="research-team"
)

# 注册多个智能体
for i in range(3):
    orchestrator.register_agent(
        agent_id=f"researcher-{i}",
        name=f"研究员 {i}",
        role=AgentRole.EXECUTOR,
        capabilities=["research"]
    )

# 创建任务
topics = ["前端架构", "后端设计", "数据库优化"]
for topic in topics:
    orchestrator.task_manager.create_task(
        name=f"研究 {topic}",
        description=f"深入研究 {topic}",
        priority=1
    )

# 执行回调
def research_task(task):
    return f"已完成研究: {task.name}"

orchestrator.set_execution_callback("research", research_task)

# 运行工作分配
asyncio.run(orchestrator.distribute_work(max_concurrent=3))

# 查看统计
stats = orchestrator.get_team_stats()
print(f"团队统计: {stats}")
```

### 示例 2: 任务依赖和协调

```python
# 创建有依赖关系的任务流
task_design = orchestrator.task_manager.create_task(
    name="架构设计",
    priority=2
)

task_backend = orchestrator.task_manager.create_task(
    name="后端实现",
    priority=1,
    depends_on=[task_design.id]
)

task_frontend = orchestrator.task_manager.create_task(
    name="前端开发",
    priority=1,
    depends_on=[task_design.id]
)

task_test = orchestrator.task_manager.create_task(
    name="集成测试",
    priority=1,
    depends_on=[task_backend.id, task_frontend.id]
)

# 自动处理依赖：
# - task_design 立即可认领
# - task_backend 和 task_frontend 等待 task_design 完成
# - task_test 等待 task_backend 和 task_frontend 都完成
```

### 示例 3: 智能体间通信

```python
from agent_team import AgentMailbox

# 创建邮箱
mailbox = AgentMailbox(Path("data/mailbox.db"))

# 前端开发者完成任务后发消息
mailbox.send(
    sender="frontend-dev",
    recipient="test-engineer",
    subject="前端组件完成",
    content="React 组件已完成，请开始测试"
)

# 测试工程师接收消息
messages = mailbox.receive(
    recipient="test-engineer",
    unread_only=True
)

for msg in messages:
    print(f"来自 {msg.sender}: {msg.content}")
    mailbox.mark_read(msg.id)
```

## Docker 支持

### 快速启动

```bash
cd team
docker compose up --build
```

### 使用 shell 脚本启动

```bash
cd team
bash scripts/docker-start.sh
```

## API 参考

### TeamOrchestrator

主编排器类，管理整个团队。

**主要方法：**

| 方法 | 说明 |
|------|------|
| `register_agent()` | 注册新智能体 |
| `unregister_agent()` | 注销智能体 |
| `list_agents()` | 列出智能体（支持过滤） |
| `claim_task()` | 为智能体认领任务 |
| `execute_task()` | 执行任务 |
| `distribute_work()` | 持续分配工作 |
| `get_team_stats()` | 获取团队统计 |

### TaskManager

任务管理器，处理任务的创建、分配和依赖管理。

**主要方法：**

| 方法 | 说明 |
|------|------|
| `create_task()` | 创建新任务 |
| `list_tasks()` | 列出任务（支持过滤） |
| `get_available_tasks()` | 获取可认领的任务 |
| `claim_task()` | 认领任务（文件锁） |
| `update_task_status()` | 更新任务状态 |

### AgentMailbox

邮箱系统，处理智能体间的消息传递。

**主要方法：**

| 方法 | 说明 |
|------|------|
| `send()` | 发送点对点消息 |
| `broadcast()` | 发送广播消息 |
| `receive()` | 接收消息 |
| `get_conversation()` | 获取两个智能体的对话 |

## 最佳实践

### 1. 合理的团队规模

- 推荐 **3-5 个智能体**
- 每个智能体 **5-6 个任务**
- 避免过多协调开销

### 2. 清晰的任务划分

为每个智能体指定明确的责任范围和文件所有权。

### 3. 避免文件冲突

通过清晰的文件所有权避免多个智能体编辑同一文件。

### 4. 使用邮箱进行协调

在邮箱中进行跨智能体通信，而不是依赖外部方式。

## 常见问题

**Q: 智能体如何知道彼此存在？**

A: 通过 `list_agents()` 方法或查询任务列表。

**Q: 如果任务执行失败怎么办？**

A: 任务状态将标记为 FAILED，可以重新分配。

**Q: 消息传递是否保证有序？**

A: 否。消息按时间戳排序，但可能乱序到达。

**Q: 如何清理旧数据？**

A: 使用 `mailbox.clear_old_messages(days=30)` 清理旧消息。

## 许可证

MIT License
