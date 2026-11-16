from evals.eval_structured import load_cases, score_case, summarize


def test_load_cases_reads_jsonl(tmp_path):
    path = tmp_path / "cases.jsonl"
    path.write_text('{"id":"c1","text":"x","expected":{}}\n', encoding="utf-8")
    assert load_cases(path) == [{"id": "c1", "text": "x", "expected": {}}]


def test_score_case_all_match():
    case = {
        "expected": {
            "order_id": "A1",
            "request_type": "退货",
            "solution_keywords": ["退货", "退款"],
        }
    }
    actual = {
        "order_id": "A1",
        "request_type": "退货",
        "expected_solution": "希望退货退款",
    }
    assert score_case(case, actual) == {
        "order_id": True,
        "request_type": True,
        "expected_solution": True,
    }


def test_score_case_null_order_id():
    case = {"expected": {"order_id": None, "request_type": "咨询", "solution_keywords": []}}
    actual = {"order_id": None, "request_type": "咨询", "expected_solution": None}
    assert score_case(case, actual)["order_id"] is True


def test_score_case_missing_keyword_fails():
    case = {
        "expected": {
            "order_id": "A1",
            "request_type": "退货",
            "solution_keywords": ["退货", "退款"],
        }
    }
    actual = {
        "order_id": "A1",
        "request_type": "退货",
        "expected_solution": "希望退货",
    }
    assert score_case(case, actual)["expected_solution"] is False


def test_summarize_accuracies():
    results = [
        {"order_id": True, "request_type": True, "expected_solution": True},
        {"order_id": False, "request_type": True, "expected_solution": False},
    ]
    summary = summarize(results)
    assert summary["order_id"] == 0.5
    assert summary["request_type"] == 1.0
    assert summary["expected_solution"] == 0.5
