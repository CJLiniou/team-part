"""邮箱系统 - 智能体间的异步消息传递。

提供点对点和广播消息传递，支持持久化、传递确认和消息路由。
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional
import json
import sqlite3
from uuid import uuid4


class MessageStatus(str, Enum):
    """消息传递状态。"""
    SENT = "sent"          # 已发送
    DELIVERED = "delivered"  # 已送达
    READ = "read"          # 已读
    FAILED = "failed"      # 失败


@dataclass
class Message:
    """智能体间发送的消息。"""
    id: str = field(default_factory=lambda: str(uuid4()))
    sender: str = ""
    recipient: str = ""    # 空字符串表示广播
    subject: str = ""
    content: str = ""
    status: MessageStatus = MessageStatus.SENT
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    delivered_at: Optional[datetime] = None
    read_at: Optional[datetime] = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """转换为字典（用于存储）。"""
        d = asdict(self)
        d['status'] = self.status.value
        d['created_at'] = self.created_at.isoformat()
        d['delivered_at'] = self.delivered_at.isoformat() if self.delivered_at else None
        d['read_at'] = self.read_at.isoformat() if self.read_at else None
        return d

    @staticmethod
    def from_dict(d: dict) -> Message:
        """从字典创建 Message。"""
        d = d.copy()
        d['status'] = MessageStatus(d['status'])
        d['created_at'] = datetime.fromisoformat(d['created_at'])
        d['delivered_at'] = datetime.fromisoformat(d['delivered_at']) if d['delivered_at'] else None
        d['read_at'] = datetime.fromisoformat(d['read_at']) if d['read_at'] else None
        return Message(**d)


class AgentMailbox:
    """线程安全的邮箱，用于智能体间通信。
    
    支持：
    - 点对点消息传递
    - 广播消息传递
    - 消息持久化
    - 传递确认
    - 消息过滤和路由
    """

    def __init__(self, db_path: Path):
        """初始化邮箱。
        
        Args:
            db_path: SQLite 数据库文件路径
        """
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """初始化数据库模式。"""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS mailbox (
                    id TEXT PRIMARY KEY,
                    sender TEXT NOT NULL,
                    recipient TEXT NOT NULL,
                    subject TEXT NOT NULL,
                    content TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'sent',
                    created_at TEXT NOT NULL,
                    delivered_at TEXT,
                    read_at TEXT,
                    metadata TEXT,
                    created_time INTEGER NOT NULL
                );
                
                CREATE INDEX IF NOT EXISTS idx_mailbox_recipient ON mailbox(recipient);
                CREATE INDEX IF NOT EXISTS idx_mailbox_sender ON mailbox(sender);
                CREATE INDEX IF NOT EXISTS idx_mailbox_status ON mailbox(status);
                CREATE INDEX IF NOT EXISTS idx_mailbox_created ON mailbox(created_time DESC);
            """)

    def send(self, sender: str, recipient: str, subject: str, content: str, 
             metadata: Optional[dict] = None) -> Message:
        """发送消息。
        
        Args:
            sender: 发送者智能体名称
            recipient: 接收者智能体名称（空字符串表示广播）
            subject: 消息主题
            content: 消息内容
            metadata: 可选的元数据字典
            
        Returns:
            创建的 Message 对象
        """
        msg = Message(
            sender=sender,
            recipient=recipient,
            subject=subject,
            content=content,
            metadata=metadata or {}
        )
        self._save_message(msg)
        return msg

    def broadcast(self, sender: str, subject: str, content: str,
                 metadata: Optional[dict] = None) -> Message:
        """发送广播消息给所有智能体。
        
        Args:
            sender: 发送者智能体名称
            subject: 消息主题
            content: 消息内容
            metadata: 可选的元数据字典
            
        Returns:
            创建的 Message 对象
        """
        return self.send(sender, "", subject, content, metadata)

    def receive(self, recipient: str, limit: int = 10, 
                unread_only: bool = False) -> list[Message]:
        """接收智能体的消息。
        
        Args:
            recipient: 接收者智能体名称
            limit: 检索的最大消息数
            unread_only: 若为 True，仅返回未读消息
            
        Returns:
            消息列表
        """
        query = "SELECT * FROM mailbox WHERE (recipient = ? OR recipient = '')"
        params = [recipient]
        
        if unread_only:
            query += " AND status IN ('sent', 'delivered')"
        
        query += " ORDER BY created_time DESC LIMIT ?"
        params.append(limit)

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, params).fetchall()
            return [self._row_to_message(row) for row in rows]

    def mark_delivered(self, message_id: str) -> bool:
        """标记消息为已送达。
        
        Args:
            message_id: 消息 ID
            
        Returns:
            成功更新返回 True，消息不存在返回 False
        """
        now = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "UPDATE mailbox SET status = ?, delivered_at = ? WHERE id = ?",
                (MessageStatus.DELIVERED.value, now, message_id)
            )
            return cursor.rowcount > 0

    def mark_read(self, message_id: str) -> bool:
        """标记消息为已读。
        
        Args:
            message_id: 消息 ID
            
        Returns:
            成功更新返回 True，消息不存在返回 False
        """
        now = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "UPDATE mailbox SET status = ?, read_at = ? WHERE id = ?",
                (MessageStatus.READ.value, now, message_id)
            )
            return cursor.rowcount > 0

    def get_message(self, message_id: str) -> Optional[Message]:
        """根据 ID 获取特定消息。
        
        Args:
            message_id: 消息 ID
            
        Returns:
            找到的 Message，不存在返回 None
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM mailbox WHERE id = ?",
                (message_id,)
            ).fetchone()
            return self._row_to_message(row) if row else None

    def get_conversation(self, agent1: str, agent2: str, limit: int = 50) -> list[Message]:
        """获取两个智能体之间的对话历史。
        
        Args:
            agent1: 第一个智能体名称
            agent2: 第二个智能体名称
            limit: 最大消息数
            
        Returns:
            对话中的消息列表
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """SELECT * FROM mailbox 
                   WHERE (sender = ? AND recipient = ?) 
                      OR (sender = ? AND recipient = ?)
                   ORDER BY created_time DESC LIMIT ?""",
                (agent1, agent2, agent2, agent1, limit)
            ).fetchall()
            return [self._row_to_message(row) for row in rows]

    def get_unread_count(self, recipient: str) -> int:
        """获取智能体的未读消息数。
        
        Args:
            recipient: 接收者智能体名称
            
        Returns:
            未读消息数
        """
        with sqlite3.connect(self.db_path) as conn:
            count = conn.execute(
                """SELECT COUNT(*) FROM mailbox 
                   WHERE (recipient = ? OR recipient = '') 
                   AND status IN ('sent', 'delivered')""",
                (recipient,)
            ).fetchone()[0]
            return count

    def clear_old_messages(self, days: int = 30) -> int:
        """删除早于 N 天的消息。
        
        Args:
            days: 天数阈值
            
        Returns:
            删除的消息数
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """DELETE FROM mailbox 
                   WHERE created_time < datetime('now', ? || ' days')""",
                (f"-{days}",)
            )
            return cursor.rowcount

    def _save_message(self, msg: Message) -> None:
        """保存消息到数据库。"""
        now = int(datetime.now(timezone.utc).timestamp())
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT INTO mailbox 
                   (id, sender, recipient, subject, content, status, 
                    created_at, metadata, created_time)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (msg.id, msg.sender, msg.recipient, msg.subject, msg.content,
                 msg.status.value, msg.created_at.isoformat(),
                 json.dumps(msg.metadata), now)
            )

    @staticmethod
    def _row_to_message(row: sqlite3.Row) -> Message:
        """将数据库行转换为 Message 对象。"""
        return Message(
            id=row['id'],
            sender=row['sender'],
            recipient=row['recipient'],
            subject=row['subject'],
            content=row['content'],
            status=MessageStatus(row['status']),
            created_at=datetime.fromisoformat(row['created_at']),
            delivered_at=datetime.fromisoformat(row['delivered_at']) 
                if row['delivered_at'] else None,
            read_at=datetime.fromisoformat(row['read_at']) 
                if row['read_at'] else None,
            metadata=json.loads(row['metadata']) if row['metadata'] else {}
        )
