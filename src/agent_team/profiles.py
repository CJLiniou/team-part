"""Agent 角色模板 — 可复用的 subagent 定义。

对应 team-define.md 中的 "Use subagent definitions for teammates"。
Profile 可以从项目目录 (.claude/agents/) 或用户目录 (~/.agent_profiles/) 加载。

Profile JSON 格式:
{
    "name": "security-reviewer",
    "role": "reviewer",
    "capabilities": ["security-audit", "code-review"],
    "system_prompt_extra": "You are a security expert focused on OWASP Top 10.",
    "model": "claude-sonnet-4-6",
    "tools_allowlist": ["read_file", "list_tasks", "send_message", "complete_task", "fail_task"]
}
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class AgentProfile:
    name: str
    role: str = "executor"
    capabilities: List[str] = field(default_factory=list)
    system_prompt_extra: str = ""
    model: str = "claude-sonnet-4-6"
    tools_allowlist: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict) -> AgentProfile:
        return cls(
            name=data["name"],
            role=data.get("role", "executor"),
            capabilities=data.get("capabilities", []),
            system_prompt_extra=data.get("system_prompt_extra", ""),
            model=data.get("model", "claude-sonnet-4-6"),
            tools_allowlist=data.get("tools_allowlist", []),
            metadata=data.get("metadata", {}),
        )

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "role": self.role,
            "capabilities": self.capabilities,
            "system_prompt_extra": self.system_prompt_extra,
            "model": self.model,
            "tools_allowlist": self.tools_allowlist,
            "metadata": self.metadata,
        }


class ProfileRegistry:
    """从多个路径加载和管理 AgentProfile。

    加载顺序（后加载的覆盖）:
    1. 用户目录: ~/.agent_profiles/*.json
    2. 项目目录: .claude/agents/*.json
    """

    def __init__(self, project_dir: Optional[Path] = None,
                 user_dir: Optional[Path] = None):
        self._profiles: Dict[str, AgentProfile] = {}
        self._project_dir = project_dir or Path(".claude/agents")
        self._user_dir = user_dir or Path.home() / ".agent_profiles"
        self._load_all()

    def _load_all(self) -> None:
        for directory in [self._user_dir, self._project_dir]:
            if directory.exists():
                for f in sorted(directory.glob("*.json")):
                    try:
                        data = json.loads(f.read_text(encoding="utf-8"))
                        profile = AgentProfile.from_dict(data)
                        self._profiles[profile.name] = profile
                        logger.info(f"Profile loaded: {profile.name} from {f}")
                    except Exception:
                        logger.exception(f"Failed to load profile from {f}")

    def get(self, name: str) -> Optional[AgentProfile]:
        return self._profiles.get(name)

    def list(self) -> List[str]:
        return list(self._profiles.keys())

    def register(self, profile: AgentProfile) -> None:
        self._profiles[profile.name] = profile
        logger.info(f"Profile registered: {profile.name}")

    def unregister(self, name: str) -> bool:
        if name in self._profiles:
            del self._profiles[name]
            return True
        return False
