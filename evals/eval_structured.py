import asyncio
import json
from pathlib import Path

from pydantic import ValidationError

from app.config import get_settings
from app.llm import build_chat_model
from app.services.extract import extract_after_sales

CASES_PATH = Path(__file__).resolve().parent / "after_sales_cases.jsonl"

THRESHOLDS = {
    "order_id": 1.0,
    "request_type": 0.8,
    "expected_solution": 0.8,
}


def load_cases(path: Path) -> list[dict]:
    cases = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                cases.append(json.loads(line))
    return cases


def score_case(case: dict, actual: dict) -> dict[str, bool]:
    expected = case["expected"]
    expected_order = expected.get("order_id")
    actual_order = actual.get("order_id")
    if expected_order is None:
        order_match = actual_order is None or (
            isinstance(actual_order, str) and not actual_order.strip()
        )
    else:
        order_match = (
            isinstance(actual_order, str)
            and actual_order.lower() == str(expected_order).lower()
        )

    request_match = expected.get("request_type") == actual.get("request_type")

    keywords = expected.get("solution_keywords", [])
    solution_text = (actual.get("expected_solution") or "").lower()
    solution_match = all(keyword.lower() in solution_text for keyword in keywords)

    return {
        "order_id": order_match,
        "request_type": request_match,
        "expected_solution": solution_match,
    }


def summarize(results: list[dict[str, bool]]) -> dict[str, float]:
    if not results:
        return {field: 0.0 for field in THRESHOLDS}
    total = len(results)
    return {
        field: round(sum(1 for result in results if result[field]) / total, 4)
        for field in THRESHOLDS
    }


async def main() -> int:
    try:
        settings = get_settings()
    except ValidationError:
        print(
            "请先配置 .env：缺少 LLM_BASE_URL / LLM_API_KEY / LLM_MODEL，"
            "可参考 .env.example 填写后重试。"
        )
        return 2

    model = build_chat_model(settings)
    cases = load_cases(CASES_PATH)
    print(f"模型：{settings.llm_model}")
    print(f"样例数：{len(cases)}")

    results = []
    for case in cases:
        try:
            extraction = await extract_after_sales(case["text"], model, settings)
        except Exception as error:
            print(f"id={case['id']}")
            print(f"  error   ={type(error).__name__}")
            results.append(
                {field: False for field in THRESHOLDS} | {"error": type(error).__name__}
            )
            continue
        actual = extraction.model_dump()
        scores = score_case(case, actual)
        results.append(scores)
        print(f"id={case['id']}")
        print(f"  expected={json.dumps(case['expected'], ensure_ascii=False)}")
        print(f"  actual  ={json.dumps(actual, ensure_ascii=False)}")
        print(f"  pass    ={json.dumps(scores, ensure_ascii=False)}")

    summary = summarize(results)
    print("汇总：")
    for field, threshold in THRESHOLDS.items():
        print(f"  {field}: {summary[field]:.4f}（阈值 >= {threshold}）")

    below = [field for field, threshold in THRESHOLDS.items() if summary[field] < threshold]
    if below:
        print(f"未达标：{', '.join(below)}")
        return 1
    print("全部字段达到阈值。")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
