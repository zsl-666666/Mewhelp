from app.sse import format_sse


def test_format_delta_frame_with_chinese():
    assert format_sse({"delta": "你"}) == 'data: {"delta": "你"}\n\n'


def test_format_done_frame():
    assert format_sse({"done": True, "trimmed": False}) == (
        'data: {"done": true, "trimmed": false}\n\n'
    )


def test_format_error_event():
    assert format_sse({"code": "upstream_error"}, event="error") == (
        'event: error\ndata: {"code": "upstream_error"}\n\n'
    )
