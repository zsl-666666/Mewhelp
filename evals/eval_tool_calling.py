"""工具选择评估：让模型只看问题并选择工具，再按名称与参数打分。"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import ValidationError
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.config import get_settings
from app.llm import build_chat_model
from app.prompts.customer_service import (
    TOOL_USAGE_RULES,
    build_customer_service_prompt,
)
from app.tools import build_tool_registry

CASES_PATH = Path(__file__).resolve().parent / "tool_calling_cases.jsonl"

PASS_THRESHOLD = 0.8


def score_tool_call(
    expected_tool: str, expected_args: dict, tool_calls: list[dict]
) -> bool:
    """首个工具调用名称完全匹配，且 expected_args 是实际 args 的子集。"""
    if not tool_calls:
        return False
    first = tool_calls[0]
    if first.get("name") != expected_tool:
        return False
    actual_args = first.get("args") or {}
    if not isinstance(actual_args, dict):
        return False
    return all(
        key in actual_args and actual_args[key] == value
        for key, value in expected_args.items()
    )


def load_cases(path: Path) -> list[dict]:
    cases = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                cases.append(json.loads(line))
    return cases


async def run_eval() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    try:
        settings = get_settings()
    except ValidationError:
        print(
            "请先配置 .env：缺少 LLM_BASE_URL / LLM_API_KEY / LLM_MODEL，"
            "可参考 .env.example 填写后重试。"
        )
        return 2

    cases = load_cases(CASES_PATH)
    model = build_chat_model(settings)
    # 仅用于构造工具 schema；不建表、不写库、不执行工具。
    registry = build_tool_registry(sessionmaker(bind=create_engine("sqlite://")))
    bound = model.bind_tools(registry.bindable_tools())

    prompt_message = build_customer_service_prompt().format_messages(
        company_name=settings.company_name,
        service_hours=settings.service_hours,
        return_window_days=settings.return_window_days,
    )[0]
    system_message = SystemMessage(
        content=f"{prompt_message.content}\n\n{TOOL_USAGE_RULES}"
    )

    print(f"模型：{settings.llm_model}")
    print(f"样例数：{len(cases)}")

    passed = 0
    for index, case in enumerate(cases, start=1):
        question = case["question"]
        expected_tool = case["expected_tool"]
        expected_args = case.get("expected_args") or {}
        try:
            response = await bound.ainvoke(
                [system_message, HumanMessage(content=question)]
            )
        except Exception as error:
            tool_calls: list[dict] = []
            error_text = type(error).__name__
        else:
            tool_calls = list(getattr(response, "tool_calls", []) or [])
            error_text = None

        ok = score_tool_call(expected_tool, expected_args, tool_calls)
        if ok:
            passed += 1

        if tool_calls:
            actual_tool = tool_calls[0].get("name")
            actual_args = tool_calls[0].get("args") or {}
            actual_text = (
                f"{actual_tool} {json.dumps(actual_args, ensure_ascii=False)}"
            )
        elif error_text is not None:
            actual_text = f"无工具（调用失败：{error_text}）"
        else:
            actual_text = "无工具"

        print(f"[{index}/{len(cases)}] {'PASS' if ok else 'FAIL'}")
        print(f"  question={question}")
        print(
            "  expected="
            f"{expected_tool} {json.dumps(expected_args, ensure_ascii=False)}"
        )
        print(f"  actual  ={actual_text}")

    total = len(cases)
    rate = passed / total if total else 0.0
    print(
        f"通过：{passed}/{total}（{rate:.1%}，阈值 >= {PASS_THRESHOLD:.0%}）"
    )
    return 0 if rate >= PASS_THRESHOLD else 1


def main() -> int:
    return asyncio.run(run_eval())


if __name__ == "__main__":
    raise SystemExit(main())
