"""Token 用量追踪 — 按 agent 和团队统计 LLM token 消耗。

对应 team-define.md 中的 token 成本管理。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

# 每百万 token 的参考定价（美元）
MODEL_PRICING: Dict[str, tuple[float, float]] = {
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5-20251001": (1.0, 5.0),
    "claude-opus-4-7": (15.0, 75.0),
    "default": (3.0, 15.0),
}


@dataclass
class TokenRecord:
    """单次 API 调用的 token 记录。"""
    agent_id: str
    agent_name: str
    model: str
    input_tokens: int
    output_tokens: int
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def estimated_cost(self) -> float:
        input_price, output_price = MODEL_PRICING.get(self.model, MODEL_PRICING["default"])
        return (self.input_tokens / 1_000_000) * input_price + (self.output_tokens / 1_000_000) * output_price


@dataclass
class AgentTokenSummary:
    agent_id: str
    agent_name: str
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    call_count: int = 0

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens


class TokenTracker:
    """追踪团队中所有 agent 的 token 消耗。

    用法:
        tracker = TokenTracker()
        tracker.record("dev-1", "Developer", "claude-sonnet-4-6", 1500, 300)
        print(tracker.team_summary())
    """

    def __init__(self):
        self._records: List[TokenRecord] = []

    def record(self, agent_id: str, agent_name: str, model: str,
               input_tokens: int, output_tokens: int) -> None:
        """记录一次 API 调用。"""
        rec = TokenRecord(
            agent_id=agent_id,
            agent_name=agent_name,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        self._records.append(rec)

    def agent_summary(self, agent_id: str) -> Optional[AgentTokenSummary]:
        """获取单个 agent 的汇总。"""
        recs = [r for r in self._records if r.agent_id == agent_id]
        if not recs:
            return None
        return AgentTokenSummary(
            agent_id=agent_id,
            agent_name=recs[0].agent_name,
            total_input_tokens=sum(r.input_tokens for r in recs),
            total_output_tokens=sum(r.output_tokens for r in recs),
            call_count=len(recs),
        )

    def team_summary(self) -> dict:
        """获取全团队的汇总统计。"""
        if not self._records:
            return {
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "total_tokens": 0,
                "total_cost": 0.0,
                "call_count": 0,
                "by_agent": {},
            }

        total_input = sum(r.input_tokens for r in self._records)
        total_output = sum(r.output_tokens for r in self._records)
        total_cost = sum(r.estimated_cost for r in self._records)

        by_agent: dict = {}
        for r in self._records:
            if r.agent_id not in by_agent:
                by_agent[r.agent_id] = {
                    "name": r.agent_name,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "calls": 0,
                    "cost": 0.0,
                }
            by_agent[r.agent_id]["input_tokens"] += r.input_tokens
            by_agent[r.agent_id]["output_tokens"] += r.output_tokens
            by_agent[r.agent_id]["calls"] += 1
            by_agent[r.agent_id]["cost"] += r.estimated_cost

        return {
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_tokens": total_input + total_output,
            "total_cost": round(total_cost, 4),
            "call_count": len(self._records),
            "by_agent": by_agent,
        }

    def reset(self) -> None:
        """重置所有记录。"""
        self._records.clear()
