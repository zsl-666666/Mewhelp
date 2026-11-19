from evals.eval_tool_calling import score_tool_call


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
