from evals.eval_tool_calling import load_cases, score_tool_call


def test_score_tool_call_accepts_name_and_args_subset():
    assert score_tool_call(
        "query_logistics",
        {"order_id": "1001"},
        [{"name": "query_logistics", "args": {"order_id": "1001", "extra": "ok"}}],
    ) is True


def test_score_tool_call_rejects_wrong_name_or_missing_args():
    assert score_tool_call(
        "query_logistics",
        {"order_id": "1001"},
        [{"name": "query_order", "args": {"order_id": "1001"}}],
    ) is False
    assert score_tool_call(
        "query_logistics",
        {"order_id": "1001"},
        [{"name": "query_logistics", "args": {}}],
    ) is False
    assert score_tool_call("query_logistics", {"order_id": "1001"}, []) is False


def test_score_tool_call_only_considers_first_call():
    assert score_tool_call(
        "query_logistics",
        {"order_id": "1001"},
        [
            {"name": "query_order", "args": {"order_id": "1001"}},
            {"name": "query_logistics", "args": {"order_id": "1001"}},
        ],
    ) is False


def test_load_cases_skips_blank_lines(tmp_path):
    path = tmp_path / "cases.jsonl"
    path.write_text(
        '{"question": "q1", "expected_tool": "query_order", "expected_args": {}}\n'
        "\n"
        '{"question": "q2", "expected_tool": "query_logistics", "expected_args": {}}\n',
        encoding="utf-8",
    )
    assert load_cases(path) == [
        {"question": "q1", "expected_tool": "query_order", "expected_args": {}},
        {"question": "q2", "expected_tool": "query_logistics", "expected_args": {}},
    ]
