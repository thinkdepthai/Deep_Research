import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

import pytest

# Make local package importable without editable install
ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Provide dummy API keys to satisfy client/model initialization
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("TAVILY_API_KEY", "test-key")

from deep_research import utils


def test_get_today_str_formats_date(monkeypatch):
    fake_datetime = MagicMock()
    fake_now = MagicMock()
    fake_now.strftime.return_value = "Thu Jan 01, 2026"
    fake_datetime.now.return_value = fake_now

    with patch("deep_research.utils.datetime", fake_datetime):
        assert utils.get_today_str() == "Thu Jan 01, 2026"

    fake_now.strftime.assert_called_once_with("%a %b %-d, %Y")


def test_get_current_dir_prefers_file_location():
    # Simulate __file__ available
    with patch.object(utils, "__file__", "/tmp/fake_dir/module.py"):
        assert utils.get_current_dir().as_posix() == "/tmp/fake_dir"


def test_get_current_dir_fallbacks_to_cwd(monkeypatch, tmp_path):
    # Simulate __file__ missing
    monkeypatch.delattr(utils, "__file__", raising=False)

    with patch("deep_research.utils.Path.cwd", return_value=tmp_path):
        assert utils.get_current_dir() == tmp_path


def test_deduplicate_search_results_removes_duplicates():
    results = [
        {
            "results": [
                {"url": "http://a", "title": "A", "content": "c1"},
                {"url": "http://b", "title": "B", "content": "c2"},
            ]
        },
        {
            "results": [
                {"url": "http://a", "title": "A2", "content": "c3"},
                {"url": "http://c", "title": "C", "content": "c4"},
            ]
        },
    ]

    unique = utils.deduplicate_search_results(results)

    assert set(unique.keys()) == {"http://a", "http://b", "http://c"}
    assert unique["http://a"]["title"] == "A"  # keeps first seen


@patch("deep_research.utils.summarize_webpage_content")
def test_process_search_results_summarizes_raw(mock_summarize):
    mock_summarize.side_effect = lambda text: f"SUM:{text[:5]}"
    unique = {
        "http://a": {"title": "A", "raw_content": "1234567890"},
        "http://b": {"title": "B", "content": "fallback"},
    }

    processed = utils.process_search_results(unique)

    assert processed["http://a"]["content"].startswith("SUM:12345")
    assert processed["http://b"]["content"] == "fallback"
    mock_summarize.assert_called_once()


def test_format_search_output_handles_empty():
    assert "No valid search results" in utils.format_search_output({})


def test_format_search_output_formats_sources_ordered():
    data = {
        "http://a": {"title": "A", "content": "ca"},
        "http://b": {"title": "B", "content": "cb"},
    }

    out = utils.format_search_output(data)

    assert "SOURCE 1: A" in out
    assert "SOURCE 2: B" in out
    # Order should follow insertion order of dict in Python 3.7+
    assert out.index("SOURCE 1: A") < out.index("SOURCE 2: B")


def test_tavily_search_multiple_calls_client(monkeypatch):
    calls = []

    def fake_search(query, max_results, include_raw_content, topic):
        calls.append((query, max_results, include_raw_content, topic))
        return {"results": [{"url": f"{query}-url", "title": "t", "content": "c"}]}

    monkeypatch.setattr(utils, "search_client", SimpleNamespace(search=fake_search))

    results = utils.tavily_search_multiple(
        ["q1", "q2"], max_results=2, topic="news", include_raw_content=False
    )

    assert len(results) == 2
    assert calls == [("q1", 2, False, "news"), ("q2", 2, False, "news")]
    assert results[0]["results"][0]["url"] == "q1-url"


def test_summarize_webpage_content_formats_output(monkeypatch):
    class FakeStructured:
        def invoke(self, messages):
            return SimpleNamespace(summary="Short summary", key_excerpts="Key points")

    class FakeModel:
        def with_structured_output(self, schema):
            return FakeStructured()

    monkeypatch.setattr(utils, "summarization_model", FakeModel())

    result = utils.summarize_webpage_content("Body text")

    assert "<summary>\nShort summary\n</summary>" in result
    assert "<key_excerpts>\nKey points\n</key_excerpts>" in result


def test_summarize_webpage_content_handles_exception(monkeypatch):
    class BrokenModel:
        def with_structured_output(self, schema):
            raise RuntimeError("boom")

    monkeypatch.setattr(utils, "summarization_model", BrokenModel())

    long_content = "x" * 1100
    result = utils.summarize_webpage_content(long_content)

    assert result.endswith("...")
    assert len(result) == 1003  # 1000 chars + "..."


def test_tavily_search_uses_helpers(monkeypatch):
    calls = {}

    def fake_tavily_search_multiple(search_queries, max_results, topic, include_raw_content):
        calls["search"] = (search_queries, max_results, topic, include_raw_content)
        return ["raw_results"]

    def fake_deduplicate(search_results):
        calls["dedup"] = search_results
        return {"u": {"title": "t", "content": "c"}}

    def fake_process(unique_results):
        calls["process"] = unique_results
        return {"u": {"title": "t", "content": "processed"}}

    def fake_format(processed_results):
        calls["format"] = processed_results
        return "FORMATTED"

    monkeypatch.setattr(utils, "tavily_search_multiple", fake_tavily_search_multiple)
    monkeypatch.setattr(utils, "deduplicate_search_results", fake_deduplicate)
    monkeypatch.setattr(utils, "process_search_results", fake_process)
    monkeypatch.setattr(utils, "format_search_output", fake_format)

    out = utils.tavily_search("query", max_results=5, topic="finance")

    assert out == "FORMATTED"
    assert calls["search"] == (["query"], 5, "finance", True)
    assert calls["dedup"] == ["raw_results"]
    assert calls["process"] == {"u": {"title": "t", "content": "c"}}
    assert calls["format"] == {"u": {"title": "t", "content": "processed"}}


def test_refine_draft_report_uses_writer_model(monkeypatch):
    captured = {}

    def fake_invoke(messages):
        captured["messages"] = messages
        return SimpleNamespace(content="refined report")

    monkeypatch.setattr(utils, "writer_model", SimpleNamespace(invoke=fake_invoke))

    result = utils.refine_draft_report("brief", "findings", "draft text")

    assert result == "refined report"
    prompt = captured["messages"][0].content
    assert "brief" in prompt
    assert "findings" in prompt
    assert "draft text" in prompt
