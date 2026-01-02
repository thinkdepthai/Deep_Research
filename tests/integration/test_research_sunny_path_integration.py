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
import re
import sys
from datetime import datetime
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


def _slugify_topic(text: str, max_length: int = 48) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text.lower()).strip("-")
    if len(slug) > max_length:
        slug = slug[:max_length].rstrip("-")
    return slug or "topic"


def _coerce_message_text(message):
    """Return a string view of a message when possible."""
    if isinstance(message, str):
        return message
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    return None


def _write_report(final_report: str, prompt: str) -> None:
    reports_dir = ROOT.parent / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    slug = _slugify_topic(prompt)
    report_path = reports_dir / f"{timestamp}-{slug}.md"

    content = (
        "# Research report\n\n"
        f"## Prompt\n{prompt}\n\n"
        "## Final report\n"
        f"{final_report}\n"
    )

    report_path.write_text(content)


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

    if os.getenv("GITLAB_CI"):
        pytest.skip("Skipping live integration in GitLab CI")

    checkpointer = InMemorySaver()
    agent = deep_researcher_builder.compile(checkpointer=checkpointer)

    # prompt = (
    #     "Write a short research brief with risk analysis on how AI-mediated voice communication affects "
    #     "interpersonal trust in finance."
    # )
    
    prompt = (
        "DeepResearch Bench: A Comprehensive Benchmark for Deep Research Agents by Mingxuan Du et al. demonstrated the "
        "https://github.com/thinkdepthai/Deep_Research to reach second place on their benchmark leaderboard. "
        "however all of competitors reach about 50% of the total possible points. "
        "one promintent example has a subcategory score of citation accuracy of over 75%. (gemini 2.5)"
        "research a systematic literature survey of what are recent advancements in approaches, techniques, or methodologies on citation accuracy in research agents. "
        "in the future work section, formulate open problems and researc directions to further improve citation accuracy in research agents."

    )

    thread_config = {"configurable": {"thread_id": "integration-thread", "recursion_limit": 20}}

    result = await agent.ainvoke({"messages": [HumanMessage(content=prompt)]}, config=thread_config)

    assert "final_report" in result, "Agent should return a final_report entry"
    final_report = result["final_report"]
    assert isinstance(final_report, str) and len(final_report.strip()) > 100

    messages = result.get("messages", [])
    assert messages, "Agent should surface user-facing messages"

    def _contains_final_report(msg) -> bool:
        text = _coerce_message_text(msg)
        if not text:
            return False
        prefix_hit = final_report[:30] in text
        phrase_hit = "final report" in text.lower()
        return prefix_hit or phrase_hit

    assert any(_contains_final_report(m) for m in messages)

    _write_report(final_report=final_report, prompt=prompt)
