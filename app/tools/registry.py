"""工具注册表与统一执行事件流。"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from typing import Any, Literal

from langchain_core.tools import BaseTool
from pydantic import ValidationError

logger = logging.getLogger(__name__)

ToolExecutionStatus = Literal[
    "started", "retrying", "completed", "timeout", "failed"
]


@dataclass(frozen=True)
class ToolSpec:
    tool: BaseTool
    label: str
    retryable: bool = True


@dataclass(frozen=True)
class ToolExecutionResult:
    ok: bool
    content: str
    error: str | None = None
    error_code: str | None = None
    retryable: bool = False
    attempts: int = 0
    duration_ms: int = 0


@dataclass(frozen=True)
class ToolExecutionEvent:
    status: ToolExecutionStatus
    tool_call_id: str
    name: str
    label: str
    arguments: dict[str, Any]
    attempt: int = 0
    max_attempts: int = 0
    error_code: str | None = None
    error: str | None = None
    result: ToolExecutionResult | None = None


class ToolRegistry:
    """按名称注册工具，并以事件流形式执行模型发出的 tool call。"""

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(
        self, tool: BaseTool, *, label: str, retryable: bool = True
    ) -> None:
        self._tools[tool.name] = ToolSpec(
            tool=tool, label=label, retryable=retryable
        )

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def bindable_tools(self) -> list[BaseTool]:
        """返回可绑定给模型的工具；InjectedToolArg 不出现在模型 schema 中。"""
        return [spec.tool for spec in self._tools.values()]

    async def execute_stream(
        self,
        tool_call: Mapping[str, Any],
        *,
        injected_args: Mapping[str, Any] | None = None,
        timeout_seconds: float,
        max_retries: int,
        retry_delay_seconds: float,
    ) -> AsyncIterator[ToolExecutionEvent]:
        started_at = time.perf_counter()
        tool_call_id = str(tool_call.get("id") or "")
        name = str(tool_call.get("name") or "")
        arguments = dict(tool_call.get("args") or {})
        max_attempts = 1 + max(0, max_retries)
        attempt = 0

        spec = self.get(name)
        label = spec.label if spec is not None else ""

        if spec is not None and not spec.retryable:
            max_attempts = 1

        yield ToolExecutionEvent(
            status="started",
            tool_call_id=tool_call_id,
            name=name,
            label=label,
            arguments=arguments,
            attempt=1,
            max_attempts=max_attempts,
        )

        try:
            if spec is None:
                yield self._failure_event(
                    status="failed",
                    tool_call_id=tool_call_id,
                    name=name,
                    label=label,
                    arguments=arguments,
                    attempt=0,
                    max_attempts=max_attempts,
                    error_code="unknown_tool",
                    error=f"未知工具: {name}",
                    retryable=False,
                    started_at=started_at,
                )
                return

            merged_args = self._merge_injected_args(
                spec, arguments, injected_args
            )
            validated = self._validate_args(spec, merged_args)
            if validated is None:
                yield self._failure_event(
                    status="failed",
                    tool_call_id=tool_call_id,
                    name=name,
                    label=label,
                    arguments=arguments,
                    attempt=0,
                    max_attempts=max_attempts,
                    error_code="validation_error",
                    error="参数校验失败",
                    retryable=False,
                    started_at=started_at,
                )
                return

            payload = (
                validated.model_dump()
                if hasattr(validated, "model_dump")
                else dict(validated)
            )

            while attempt < max_attempts:
                attempt += 1
                try:
                    raw = await asyncio.wait_for(
                        spec.tool.ainvoke(payload),
                        timeout=timeout_seconds,
                    )
                except asyncio.TimeoutError:
                    if spec.retryable and attempt < max_attempts:
                        yield self._retry_event(
                            tool_call_id=tool_call_id,
                            name=name,
                            label=label,
                            arguments=arguments,
                            next_attempt=attempt + 1,
                            max_attempts=max_attempts,
                            error_code="timeout",
                            error="工具执行超时",
                        )
                        await self._sleep(retry_delay_seconds)
                        continue
                    yield self._failure_event(
                        status="timeout",
                        tool_call_id=tool_call_id,
                        name=name,
                        label=label,
                        arguments=arguments,
                        attempt=attempt,
                        max_attempts=max_attempts,
                        error_code="timeout",
                        error="工具执行超时",
                        retryable=spec.retryable,
                        started_at=started_at,
                    )
                    return
                except Exception:
                    logger.exception("tool %s failed on attempt %s", name, attempt)
                    if spec.retryable and attempt < max_attempts:
                        yield self._retry_event(
                            tool_call_id=tool_call_id,
                            name=name,
                            label=label,
                            arguments=arguments,
                            next_attempt=attempt + 1,
                            max_attempts=max_attempts,
                            error_code="execution_error",
                            error="工具执行失败",
                        )
                        await self._sleep(retry_delay_seconds)
                        continue
                    yield self._failure_event(
                        status="failed",
                        tool_call_id=tool_call_id,
                        name=name,
                        label=label,
                        arguments=arguments,
                        attempt=attempt,
                        max_attempts=max_attempts,
                        error_code="execution_error",
                        error="工具执行失败",
                        retryable=spec.retryable,
                        started_at=started_at,
                    )
                    return

                content = self._stringify_content(raw)
                yield ToolExecutionEvent(
                    status="completed",
                    tool_call_id=tool_call_id,
                    name=name,
                    label=label,
                    arguments=arguments,
                    attempt=attempt,
                    max_attempts=max_attempts,
                    result=ToolExecutionResult(
                        ok=True,
                        content=content,
                        attempts=attempt,
                        duration_ms=self._elapsed_ms(started_at),
                    ),
                )
                return
        except Exception:
            logger.exception("tool %s execution pipeline failed", name)
            yield self._failure_event(
                status="failed",
                tool_call_id=tool_call_id,
                name=name,
                label=label,
                arguments=arguments,
                attempt=attempt,
                max_attempts=max_attempts,
                error_code="internal_error",
                error="工具执行失败",
                retryable=False,
                started_at=started_at,
            )

    async def execute(
        self,
        tool_call: Mapping[str, Any],
        *,
        injected_args: Mapping[str, Any] | None = None,
        timeout_seconds: float,
        max_retries: int,
        retry_delay_seconds: float,
    ) -> ToolExecutionResult:
        result: ToolExecutionResult | None = None
        async for event in self.execute_stream(
            tool_call,
            injected_args=injected_args,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            retry_delay_seconds=retry_delay_seconds,
        ):
            if event.result is not None:
                result = event.result
        if result is None:
            return ToolExecutionResult(
                ok=False,
                content=json.dumps(
                    {"ok": False, "error": "工具执行失败"}, ensure_ascii=False
                ),
                error="工具执行失败",
                error_code="internal_error",
            )
        return result

    @staticmethod
    def _merge_injected_args(
        spec: ToolSpec,
        arguments: Mapping[str, Any],
        injected_args: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        merged = dict(arguments)
        if not injected_args:
            return merged
        fields = getattr(spec.tool.args_schema, "model_fields", {})
        for key, value in injected_args.items():
            if key in fields:
                merged[key] = value
        return merged

    @staticmethod
    def _validate_args(spec: ToolSpec, merged_args: Mapping[str, Any]) -> Any:
        args_schema = spec.tool.args_schema
        if args_schema is None:
            return dict(merged_args)
        try:
            return args_schema.model_validate(dict(merged_args))
        except ValidationError:
            return None

    @staticmethod
    def _stringify_content(raw: Any) -> str:
        if isinstance(raw, str):
            return raw
        if isinstance(raw, (dict, list, int, float, bool)) or raw is None:
            return json.dumps(raw, ensure_ascii=False)
        return json.dumps(raw, ensure_ascii=False, default=str)

    @staticmethod
    async def _sleep(seconds: float) -> None:
        if seconds > 0:
            await asyncio.sleep(seconds)

    @staticmethod
    def _elapsed_ms(started_at: float) -> int:
        return int((time.perf_counter() - started_at) * 1000)

    def _retry_event(
        self,
        *,
        tool_call_id: str,
        name: str,
        label: str,
        arguments: dict[str, Any],
        next_attempt: int,
        max_attempts: int,
        error_code: str,
        error: str,
    ) -> ToolExecutionEvent:
        return ToolExecutionEvent(
            status="retrying",
            tool_call_id=tool_call_id,
            name=name,
            label=label,
            arguments=arguments,
            attempt=next_attempt,
            max_attempts=max_attempts,
            error_code=error_code,
            error=error,
        )

    def _failure_event(
        self,
        *,
        status: ToolExecutionStatus,
        tool_call_id: str,
        name: str,
        label: str,
        arguments: dict[str, Any],
        attempt: int,
        max_attempts: int,
        error_code: str,
        error: str,
        retryable: bool,
        started_at: float,
    ) -> ToolExecutionEvent:
        return ToolExecutionEvent(
            status=status,
            tool_call_id=tool_call_id,
            name=name,
            label=label,
            arguments=arguments,
            attempt=attempt,
            max_attempts=max_attempts,
            error_code=error_code,
            error=error,
            result=ToolExecutionResult(
                ok=False,
                content=json.dumps(
                    {"ok": False, "error": error}, ensure_ascii=False
                ),
                error=error,
                error_code=error_code,
                retryable=retryable,
                attempts=attempt,
                duration_ms=self._elapsed_ms(started_at),
            ),
        )
