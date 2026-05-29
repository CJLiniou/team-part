"""LLM Provider 抽象层 — 统一 Anthropic 和 OpenAI 接口。

工具 schema 内部使用 Anthropic 格式（name/description/input_schema），
Provider 负责在调用 OpenAI 时转换格式。

用法:
    provider = AnthropicProvider()          # 默认，需 ANTHROPIC_API_KEY
    provider = OpenAIProvider()             # 需 OPENAI_API_KEY
    provider = OpenAIProvider(model="gpt-4o")

    response = await provider.create_message(
        model="claude-sonnet-4-6",
        system_prompt="You are...",
        messages=[{"role": "user", "content": "Hello"}],
        tools=[{"name": "read_file", "description": "...", "input_schema": {...}}],
        max_tokens=4096,
    )
    # response.content, response.stop_reason, response.usage 统一格式
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ── Unified response types ──────────────────────────────────────────

@dataclass
class ToolUseBlock:
    id: str
    name: str
    input: dict


@dataclass
class TokenUsage:
    input_tokens: int
    output_tokens: int


@dataclass
class LLMResponse:
    """与 provider 无关的 LLM 响应。"""
    content: list            # TextBlock(str) or ToolUseBlock
    stop_reason: str         # "end_turn" or "tool_use"
    usage: TokenUsage


# ── Provider base ────────────────────────────────────────────────────

class LLMProvider(ABC):
    """LLM 供应商的抽象基类。"""

    @abstractmethod
    async def create_message(
        self,
        model: str,
        system_prompt: str,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
        max_tokens: int = 4096,
        extra_body: Optional[dict] = None,
    ) -> LLMResponse:
        """extra_body: 模型特定参数，如 Qwen3 的 enable_thinking=False"""


# ── Anthropic provider ───────────────────────────────────────────────

class AnthropicProvider(LLMProvider):
    """Anthropic Claude API。

    需要环境变量 ANTHROPIC_API_KEY。
    可通过 ANTHROPIC_BASE_URL 或 base_url 参数指定自定义端点。
    max_concurrent: 最大并发请求数（None=不限制，设为 1-2 可避免低 QPS API 限流）
    """

    def __init__(self, api_key: str = "", model: str = "claude-sonnet-4-6",
                 base_url: str = "", max_concurrent: int | None = None):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not self.api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        self.base_url = base_url or os.environ.get("ANTHROPIC_BASE_URL", "")
        self.default_model = model
        self._client = None
        self._semaphore = asyncio.Semaphore(max_concurrent) if max_concurrent else None

    def _get_client(self):
        if self._client is None:
            import anthropic
            kwargs = {"api_key": self.api_key}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._client = anthropic.AsyncAnthropic(**kwargs)
        return self._client

    async def create_message(
        self,
        model: str = "",
        system_prompt: str = "",
        messages: list[dict] | None = None,
        tools: list[dict] | None = None,
        max_tokens: int = 4096,
        extra_body: dict | None = None,
    ) -> LLMResponse:
        if self._semaphore:
            async with self._semaphore:
                return await self._do_create_message(model, system_prompt, messages, tools, max_tokens, extra_body)
        return await self._do_create_message(model, system_prompt, messages, tools, max_tokens, extra_body)

    async def _do_create_message(self, model, system_prompt, messages, tools, max_tokens, extra_body):
        client = self._get_client()
        kwargs = {
            "model": model or self.default_model,
            "system": system_prompt,
            "messages": messages or [],
            "max_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
        if extra_body:
            kwargs["extra_body"] = extra_body
        resp = await client.messages.create(**kwargs)
        return self._to_unified(resp)

    @staticmethod
    def _to_unified(resp) -> LLMResponse:
        content = []
        for block in resp.content:
            if block.type == "tool_use":
                content.append(ToolUseBlock(id=block.id, name=block.name, input=block.input))
            else:
                content.append(getattr(block, 'text', str(block)))

        stop = "tool_use" if resp.stop_reason == "tool_use" else "end_turn"

        return LLMResponse(
            content=content,
            stop_reason=stop,
            usage=TokenUsage(
                input_tokens=resp.usage.input_tokens,
                output_tokens=resp.usage.output_tokens,
            ),
        )


# ── OpenAI provider ──────────────────────────────────────────────────

class OpenAIProvider(LLMProvider):
    """OpenAI 兼容 API（GPT-4o, GPT-4.1, Qwen, DeepSeek 等）。

    需要环境变量 OPENAI_API_KEY。
    通过 OPENAI_BASE_URL 或 base_url 参数指定自定义端点。
    max_concurrent: 最大并发请求数（None=不限制，设为 1-2 可避免低 QPS API 限流）
    """

    def __init__(self, api_key: str = "", model: str = "gpt-4o",
                 base_url: str = "", max_concurrent: int | None = None):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY not set")
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL", "")
        self.default_model = model
        self._client = None
        self._semaphore = asyncio.Semaphore(max_concurrent) if max_concurrent else None

    def _get_client(self):
        if self._client is None:
            import openai
            kwargs = {"api_key": self.api_key}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._client = openai.AsyncOpenAI(**kwargs)
        return self._client

    async def create_message(
        self,
        model: str = "",
        system_prompt: str = "",
        messages: list[dict] | None = None,
        tools: list[dict] | None = None,
        max_tokens: int = 4096,
        extra_body: dict | None = None,
    ) -> LLMResponse:
        if self._semaphore:
            async with self._semaphore:
                return await self._do_create_message(model, system_prompt, messages, tools, max_tokens, extra_body)
        return await self._do_create_message(model, system_prompt, messages, tools, max_tokens, extra_body)

    async def _do_create_message(self, model, system_prompt, messages, tools, max_tokens, extra_body):
        client = self._get_client()

        # 构建 OpenAI 格式的消息列表
        openai_messages = []
        if system_prompt:
            openai_messages.append({"role": "system", "content": system_prompt})

        openai_messages.extend(self._convert_messages(messages or []))

        # 转换工具格式
        openai_tools = None
        if tools:
            openai_tools = [self._convert_tool_schema(t) for t in tools]

        effective_model = model or self.default_model

        # Qwen3 系列默认关掉 thinking（非流式调用要求）
        merged_extra = dict(extra_body) if extra_body else {}
        if effective_model.lower().startswith("qwen3") and "enable_thinking" not in merged_extra:
            merged_extra["enable_thinking"] = False

        kwargs = {
            "model": effective_model,
            "messages": openai_messages,
            "max_tokens": max_tokens,
        }
        if openai_tools:
            kwargs["tools"] = openai_tools
        if merged_extra:
            kwargs["extra_body"] = merged_extra

        logger.debug(f"OpenAI request: model={effective_model}, extra_body={merged_extra}")
        resp = await client.chat.completions.create(**kwargs)
        return self._to_unified(resp)

    @staticmethod
    def _convert_messages(messages: list[dict]) -> list[dict]:
        """将 Anthropic 风格的消息列表转换为 OpenAI 格式。"""
        result = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            # 纯文本消息
            if isinstance(content, str):
                result.append({"role": role, "content": content})
                continue

            # 列表内容
            if isinstance(content, list):
                if role == "assistant":
                    text_parts = []
                    tool_calls = []
                    for block in content:
                        if OpenAIProvider._is_tool_use(block):
                            tool_calls.append({
                                "id": block.id,
                                "type": "function",
                                "function": {
                                    "name": block.name,
                                    "arguments": json.dumps(block.input),
                                },
                            })
                        elif isinstance(block, str):
                            text_parts.append(block)
                        elif hasattr(block, 'text'):
                            text_parts.append(block.text)

                    r = {"role": "assistant"}
                    r["content"] = "\n".join(text_parts) if text_parts else None
                    if tool_calls:
                        r["tool_calls"] = tool_calls
                    result.append(r)

                elif role == "user":
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "tool_result":
                            result.append({
                                "role": "tool",
                                "tool_call_id": block["tool_use_id"],
                                "content": block["content"],
                            })

            else:
                result.append({"role": role, "content": str(content)})
        return result

    @staticmethod
    def _is_tool_use(block) -> bool:
        """检查 block 是否是工具调用（兼容 Anthropic SDK 和 ToolUseBlock）。"""
        if isinstance(block, ToolUseBlock):
            return True
        if hasattr(block, 'type') and getattr(block, 'type') == 'tool_use':
            return True
        return False

    @staticmethod
    def _convert_tool_schema(schema: dict) -> dict:
        """将 Anthropic 工具 schema 转换为 OpenAI 格式。"""
        return {
            "type": "function",
            "function": {
                "name": schema["name"],
                "description": schema.get("description", ""),
                "parameters": schema.get("input_schema", {"type": "object", "properties": {}}),
            },
        }

    @staticmethod
    def _to_unified(resp) -> LLMResponse:
        choice = resp.choices[0]
        content = []
        stop = "end_turn"

        if choice.message.content:
            content.append(choice.message.content)

        if choice.message.tool_calls:
            stop = "tool_use"
            for tc in choice.message.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}
                content.append(ToolUseBlock(
                    id=tc.id,
                    name=tc.function.name,
                    input=args,
                ))

        if choice.finish_reason == "tool_calls":
            stop = "tool_use"

        return LLMResponse(
            content=content,
            stop_reason=stop,
            usage=TokenUsage(
                input_tokens=resp.usage.prompt_tokens,
                output_tokens=resp.usage.completion_tokens,
            ),
        )


# ── Factory ──────────────────────────────────────────────────────────

def create_provider(provider_type: str = "", model: str = "",
                    api_key: str = "", base_url: str = "") -> LLMProvider:
    """根据类型创建 Provider。

    Args:
        provider_type: "anthropic" 或 "openai"（默认从 AGENT_TEAM_PROVIDER 环境变量读取）
        model: 覆盖默认模型
        api_key: 覆盖 API key
        base_url: 自定义 API 端点（如 Qwen: https://dashscope.aliyuncs.com/compatible-mode/v1）
    """
    ptype = provider_type or os.environ.get("AGENT_TEAM_PROVIDER", "anthropic")

    if ptype == "openai":
        return OpenAIProvider(api_key=api_key, model=model or "gpt-4o",
                             base_url=base_url)
    else:
        return AnthropicProvider(api_key=api_key, model=model or "claude-sonnet-4-6",
                                base_url=base_url)
