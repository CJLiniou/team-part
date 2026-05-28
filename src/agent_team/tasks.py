"""任务管理 - 支持依赖图和文件锁的持久化任务列表。

提供持久化任务存储，支持 DAG 依赖跟踪、任务分配和并发访问控制。
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional, Set
import json
import sqlite3
from uuid import uuid4

from filelock import FileLock


class TaskStatus(str, Enum):
    """任务执行状态。"""
    PENDING = "pending"        # 待处理
    IN_PROGRESS = "in_progress"  # 进行中
    COMPLETED = "completed"    # 已完成
    FAILED = "failed"          # 失败
    BLOCKED = "blocked"        # 被阻止（依赖未完成）


@dataclass
class Task:
    """工作项目。"""
    id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    description: str = ""
    status: TaskStatus = TaskStatus.PENDING
    assigned_to: str = ""      # 分配给的智能体
    priority: int = 0          # 0=低, 1=中, 2=高
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    depends_on: list[str] = field(default_factory=list)  # 依赖的任务 ID
    result: str = ""           # 完成结果/输出
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """转换为字典（用于存储）。"""
        d = asdict(self)
        d['status'] = self.status.value
        d['created_at'] = self.created_at.isoformat()
        d['started_at'] = self.started_at.isoformat() if self.started_at else None
        d['completed_at'] = self.completed_at.isoformat() if self.completed_at else None
        return d

    @staticmethod
    def from_dict(d: dict) -> Task:
        """从字典创建 Task。"""
        d = d.copy()
        d['status'] = TaskStatus(d['status'])
        d['created_at'] = datetime.fromisoformat(d['created_at'])
        d['started_at'] = datetime.fromisoformat(d['started_at']) if d['started_at'] else None
        d['completed_at'] = datetime.fromisoformat(d['completed_at']) if d['completed_at'] else None
        return Task(**d)


class TaskManager:
    """管理持久化任务列表，支持依赖跟踪和文件锁。
    
    特性：
    - 任务 CRUD 操作
    - 依赖图（DAG）验证
    - 基于依赖的自动阻止
    - 并发访问的文件锁
    - 任务分配给智能体
    """

    def __init__(self, db_path: Path, hook_registry=None):
        """初始化任务管理器。

        Args:
            db_path: SQLite 数据库文件路径
            hook_registry: 可选的 HookRegistry，用于 TASK_CREATED / TASK_COMPLETED 钩子
        """
        self.db_path = db_path
        self.hook_registry = hook_registry
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock_dir = db_path.parent / ".task_locks"
        self._lock_dir.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """初始化数据库模式。"""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS tasks (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    status TEXT NOT NULL DEFAULT 'pending',
                    assigned_to TEXT,
                    priority INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT,
                    result TEXT,
                    metadata TEXT,
                    created_time INTEGER NOT NULL
                );
                
                CREATE TABLE IF NOT EXISTS task_dependencies (
                    task_id TEXT NOT NULL,
                    depends_on_id TEXT NOT NULL,
                    PRIMARY KEY (task_id, depends_on_id),
                    FOREIGN KEY (task_id) REFERENCES tasks(id),
                    FOREIGN KEY (depends_on_id) REFERENCES tasks(id)
                );
                
                CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
                CREATE INDEX IF NOT EXISTS idx_tasks_assigned_to ON tasks(assigned_to);
                CREATE INDEX IF NOT EXISTS idx_tasks_priority ON tasks(priority DESC);
                CREATE INDEX IF NOT EXISTS idx_tasks_created ON tasks(created_time DESC);
            """)

    def create_task(self, name: str, description: str = "", assigned_to: str = "",
                   priority: int = 0, depends_on: Optional[list[str]] = None,
                   metadata: Optional[dict] = None) -> Task:
        """创建新任务。
        
        Args:
            name: 任务名称
            description: 任务描述
            assigned_to: 分配给的智能体
            priority: 优先级（0=低, 1=中, 2=高）
            depends_on: 该任务依赖的任务 ID 列表
            metadata: 可选的元数据
            
        Returns:
            创建的 Task 对象
            
        Raises:
            ValueError: 依赖任务不存在或检测到循环依赖
        """
        task = Task(
            name=name,
            description=description,
            assigned_to=assigned_to,
            priority=priority,
            depends_on=depends_on or [],
            metadata=metadata or {}
        )

        # 验证依赖
        if task.depends_on:
            self._validate_dependencies(task.id, task.depends_on)

        # 触发 TASK_CREATED 钩子
        if self.hook_registry:
            from .hooks import HookEvent, HookContext
            allowed, reason = self.hook_registry.trigger_sync(
                HookEvent.TASK_CREATED,
                HookContext(event=HookEvent.TASK_CREATED, task_id=task.id,
                            task_name=task.name, task_description=task.description),
            )
            if not allowed:
                raise ValueError(f"Task creation blocked by hook: {reason}")

        # 保存任务
        now = int(datetime.now(timezone.utc).timestamp())
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT INTO tasks 
                   (id, name, description, status, assigned_to, priority, 
                    created_at, metadata, created_time)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (task.id, task.name, task.description, task.status.value,
                 task.assigned_to, task.priority, task.created_at.isoformat(),
                 json.dumps(task.metadata), now)
            )

            # 保存依赖关系
            for dep_id in task.depends_on:
                conn.execute(
                    "INSERT INTO task_dependencies (task_id, depends_on_id) VALUES (?, ?)",
                    (task.id, dep_id)
                )

        return task

    def get_task(self, task_id: str) -> Optional[Task]:
        """根据 ID 获取任务。
        
        Args:
            task_id: 任务 ID
            
        Returns:
            找到的 Task，若不存在则返回 None
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
            if not row:
                return None

            task = self._row_to_task(row)
            # 加载依赖
            dep_rows = conn.execute(
                "SELECT depends_on_id FROM task_dependencies WHERE task_id = ?",
                (task_id,)
            ).fetchall()
            task.depends_on = [r['depends_on_id'] for r in dep_rows]
            return task

    def list_tasks(self, status: Optional[TaskStatus] = None,
                  assigned_to: Optional[str] = None,
                  order_by: str = "priority DESC, created_time DESC") -> list[Task]:
        """列出任务，支持可选过滤。
        
        Args:
            status: 按状态过滤
            assigned_to: 按分配者过滤
            order_by: SQL ORDER BY 子句
            
        Returns:
            匹配的任务列表
        """
        query = "SELECT * FROM tasks WHERE 1=1"
        params = []

        if status:
            query += " AND status = ?"
            params.append(status.value)

        if assigned_to:
            query += " AND assigned_to = ?"
            params.append(assigned_to)

        query += f" ORDER BY {order_by}"

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, params).fetchall()

            tasks = []
            for row in rows:
                task = self._row_to_task(row)
                # 加载依赖
                dep_rows = conn.execute(
                    "SELECT depends_on_id FROM task_dependencies WHERE task_id = ?",
                    (task.id,)
                ).fetchall()
                task.depends_on = [r['depends_on_id'] for r in dep_rows]
                tasks.append(task)

            return tasks

    def get_available_tasks(self, agent_name: str) -> list[Task]:
        """获取智能体可以认领的任务（未阻止、未分配）。
        
        Args:
            agent_name: 智能体名称
            
        Returns:
            可认领的任务列表
        """
        # 获取所有待处理/未分配的任务
        candidates = self.list_tasks(
            status=TaskStatus.PENDING,
            order_by="priority DESC, created_time ASC"
        )

        available = []
        for task in candidates:
            if task.assigned_to and task.assigned_to != agent_name:
                continue  # 已分配给其他智能体

            # 检查依赖是否满足
            if self._are_dependencies_satisfied(task.id):
                available.append(task)

        return available

    def claim_task(self, task_id: str, agent_name: str) -> bool:
        """为智能体认领任务（使用跨平台文件锁）。

        Args:
            task_id: 任务 ID
            agent_name: 认领该任务的智能体

        Returns:
            成功认领返回 True，已被认领返回 False
        """
        lock_path = self._lock_dir / f"{task_id}.lock"
        lock = FileLock(lock_path, timeout=0)

        try:
            with lock.acquire(poll_interval=0.01):
                task = self.get_task(task_id)
                if not task or task.status != TaskStatus.PENDING:
                    return False

                if task.assigned_to and task.assigned_to != agent_name:
                    return False

                now = datetime.now(timezone.utc).isoformat()
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute(
                        """UPDATE tasks SET status = ?, assigned_to = ?, started_at = ?
                           WHERE id = ?""",
                        (TaskStatus.IN_PROGRESS.value, agent_name, now, task_id)
                    )
                return True

        except Exception:
            return False

    def update_task_status(self, task_id: str, status: TaskStatus,
                          result: str = "") -> bool:
        """更新任务状态。
        
        Args:
            task_id: 任务 ID
            status: 新状态
            result: 结果/输出（用于已完成的任务）
            
        Returns:
            成功更新返回 True，任务不存在返回 False
        """
        now = datetime.now(timezone.utc).isoformat()
        update_fields = ["status = ?"]
        params = [status.value]

        if status == TaskStatus.COMPLETED:
            # 触发 TASK_COMPLETED 钩子
            if self.hook_registry:
                from .hooks import HookEvent, HookContext
                task = self.get_task(task_id)
                task_name = task.name if task else ""
                allowed, reason = self.hook_registry.trigger_sync(
                    HookEvent.TASK_COMPLETED,
                    HookContext(event=HookEvent.TASK_COMPLETED, task_id=task_id,
                                task_name=task_name, task_result=result),
                )
                if not allowed:
                    return False
            update_fields.append("started_at = ?")
            params.append(now)

        if status == TaskStatus.COMPLETED:
            update_fields.append("completed_at = ?")
            params.append(now)
            if result:
                update_fields.append("result = ?")
                params.append(result)

        params.append(task_id)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                f"UPDATE tasks SET {', '.join(update_fields)} WHERE id = ?",
                params
            )
            return cursor.rowcount > 0

    def get_task_stats(self) -> dict:
        """获取任务统计。
        
        Returns:
            包含按状态分类的任务计数的字典
        """
        with sqlite3.connect(self.db_path) as conn:
            stats = {}
            for status in TaskStatus:
                count = conn.execute(
                    "SELECT COUNT(*) FROM tasks WHERE status = ?",
                    (status.value,)
                ).fetchone()[0]
                stats[status.value] = count
            return stats

    def _validate_dependencies(self, task_id: str, depends_on: list[str]) -> None:
        """验证依赖关系存在且不创建循环。
        
        Args:
            task_id: 任务 ID
            depends_on: 依赖的任务 ID 列表
            
        Raises:
            ValueError: 依赖无效
        """
        with sqlite3.connect(self.db_path) as conn:
            # 检查所有依赖存在
            for dep_id in depends_on:
                exists = conn.execute(
                    "SELECT 1 FROM tasks WHERE id = ?", (dep_id,)
                ).fetchone()
                if not exists:
                    raise ValueError(f"依赖任务 {dep_id} 不存在")

            # 检查循环依赖
            if self._would_create_cycle(task_id, depends_on, conn):
                raise ValueError("检测到循环依赖")

    def _would_create_cycle(self, task_id: str, depends_on: list[str],
                           conn: sqlite3.Connection) -> bool:
        """检查添加依赖是否会创建循环。"""
        visited = set()
        for dep_id in depends_on:
            if self._has_path_to(dep_id, task_id, visited, conn):
                return True
        return False

    def _has_path_to(self, from_id: str, to_id: str, visited: Set[str],
                    conn: sqlite3.Connection) -> bool:
        """检查是否存在从 from_id 到 to_id 的路径。"""
        if from_id == to_id:
            return True
        if from_id in visited:
            return False

        visited.add(from_id)

        # 获取 from_id 依赖的任务
        rows = conn.execute(
            "SELECT depends_on_id FROM task_dependencies WHERE task_id = ?",
            (from_id,)
        ).fetchall()

        for row in rows:
            dep = row[0] if isinstance(row, tuple) else row['depends_on_id']
            if self._has_path_to(dep, to_id, visited, conn):
                return True

        return False

    def _are_dependencies_satisfied(self, task_id: str) -> bool:
        """检查任务的所有依赖是否都已完成。"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """SELECT depends_on_id FROM task_dependencies WHERE task_id = ?""",
                (task_id,)
            ).fetchall()

            for row in rows:
                dep_status = conn.execute(
                    "SELECT status FROM tasks WHERE id = ?",
                    (row['depends_on_id'],)
                ).fetchone()
                if not dep_status or dep_status['status'] != TaskStatus.COMPLETED.value:
                    return False

        return True

    @staticmethod
    def _row_to_task(row: sqlite3.Row) -> Task:
        """将数据库行转换为 Task 对象。"""
        return Task(
            id=row['id'],
            name=row['name'],
            description=row['description'],
            status=TaskStatus(row['status']),
            assigned_to=row['assigned_to'] or "",
            priority=row['priority'],
            created_at=datetime.fromisoformat(row['created_at']),
            started_at=datetime.fromisoformat(row['started_at']) if row['started_at'] else None,
            completed_at=datetime.fromisoformat(row['completed_at']) if row['completed_at'] else None,
            result=row['result'] or "",
            metadata=json.loads(row['metadata']) if row['metadata'] else {}
        )
