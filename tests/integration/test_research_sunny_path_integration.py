"""Integration test for sunny-path research flow using real config and APIs.

This test is intentionally real-data-driven (no mocks). It expects:
- CONFIG_PATH pointing to a real config.yml with stages search/LLM credentials
- OPENAI_API_KEY and TAVILY_API_KEY (or equivalent backend creds) set to real values
- Optional STAGE to select the config stage (defaults handled by search_factory/load_config)

The test follows the notebook recipe in thinkdepthai_deepresearch.ipynb:
1) Build an InMemorySaver checkpointer
2) Compile the deep_researcher graph with that checkpointer
3) Invoke the agent asynchronously on a single human message
4) Assert that a non-empty final report is produced

To avoid unintended live calls, the test will skip when required config/keys are missing.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

# Make local package importable without installation
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from modules.util.confighelpers import load_config  # noqa: E402
from deep_research.research_agent_full import deep_researcher_builder  # noqa: E402


@pytest.fixture(scope="session")
def real_config_env():
    """Ensure real config and API keys are present; otherwise skip.

    This guard keeps the test strictly real-data-driven while preventing
    accidental execution without required credentials.
    """

    config_path = Path(os.environ.get("CONFIG_PATH", "config.yml"))
    if not config_path.exists():
        pytest.skip("CONFIG_PATH does not point to an existing config file; required for integration run")

    try:
        load_config(stage_name=os.environ.get("STAGE"))
    except Exception as exc:
        pytest.skip(f"Failed to load config for integration run: {exc}")

    return {
        "config_path": config_path,
        "stage": os.environ.get("STAGE"),
    }


@pytest.mark.asyncio
async def test_research_sunny_path_real_data(real_config_env):
    """Runs the full research graph on a single prompt and checks for output."""

    checkpointer = InMemorySaver()
    agent = deep_researcher_builder.compile(checkpointer=checkpointer)

    prompt = (
        "Write a short research brief on how AI-mediated communication affects "
        "interpersonal trust, noting mechanisms, benefits, and risks."
    )

    thread_config = {"configurable": {"thread_id": "integration-thread", "recursion_limit": 20}}

    result = await agent.ainvoke({"messages": [HumanMessage(content=prompt)]}, config=thread_config)

    assert "final_report" in result, "Agent should return a final_report entry"
    final_report = result["final_report"]
    assert isinstance(final_report, str) and len(final_report.strip()) > 100

    messages = result.get("messages", [])
    assert messages, "Agent should surface user-facing messages"
    assert any(isinstance(m, str) and final_report[:30] in m or "final report" in m.lower() for m in messages)
