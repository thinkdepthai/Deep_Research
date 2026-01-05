

"""Research Utilities and Tools.

This module provides search and content processing utilities for the research agent,
including web search capabilities and content summarization tools.
"""

from pathlib import Path
from datetime import datetime
from typing import Optional
from typing_extensions import Annotated, List, Literal

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool, InjectedToolArg

from deep_research import logging as dr_logging
from deep_research.llm_factory import get_chat_model
from deep_research.state_research import Summary
from deep_research.prompts import summarize_webpage_prompt, report_generation_with_draft_insight_prompt
from deep_research.search_factory import (
    SearchConfigError,
    get_search_client,
    get_search_defaults,
    get_search_provider,
)


logger = dr_logging.get_logger(__name__)


# ===== UTILITY FUNCTIONS =====

def get_today_str() -> str:
    """Get current date in a human-readable format."""
    return datetime.now().strftime("%a %b %-d, %Y")

def get_current_dir() -> Path:
    """Get the current directory of the module.

    This function is compatible with Jupyter notebooks and regular Python scripts.

    Returns:
        Path object representing the current directory
    """
    try:
        return Path(__file__).resolve().parent
    except NameError:  # __file__ is not defined
        return Path.cwd()

# ===== CONFIGURATION =====

summarization_model = get_chat_model("researcher_summarizer")
writer_model = get_chat_model("writer")
# Lazily resolve search provider/client/defaults to avoid import-time failures when custom backends are unavailable
search_provider = None
search_client = None
search_defaults = None
MAX_CONTEXT_LENGTH = 250000


def _ensure_search_runtime(raise_on_error: bool = True):
    """Lazy-init search provider/client/defaults to handle custom backends.

    Adds logging and optional error propagation to avoid silent fallback when
    search config/backends are missing. Preserves monkeypatched globals (e.g.,
    in unit tests) by only filling missing pieces.
    """
    global search_provider, search_client, search_defaults

    try:
        if search_provider is None:
            logger.debug("Resolving search provider (lazy init)")
            search_provider = get_search_provider()
        if search_client is None:
            logger.debug("Resolving search client (lazy init)")
            search_client = get_search_client()
        if search_defaults is None:
            logger.debug("Resolving search defaults (lazy init)")
            search_defaults = get_search_defaults()
    except SearchConfigError as exc:
        logger.error("Search runtime initialization failed: %s", exc)
        if raise_on_error:
            raise
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.error("Unexpected error during search runtime init: %s", exc)
        if raise_on_error:
            raise

    return search_provider, search_client, search_defaults



# Attempt to import common timeout exception classes (best-effort, optional deps)
try:  # pragma: no cover - import guard
    import requests

    _REQUESTS_TIMEOUT_EXC = (requests.exceptions.Timeout,)
except Exception:  # pragma: no cover - safe fallback
    _REQUESTS_TIMEOUT_EXC = tuple()

try:  # pragma: no cover - import guard
    import httpx

    _HTTPX_TIMEOUT_EXC = (httpx.TimeoutException,)
except Exception:  # pragma: no cover - safe fallback
    _HTTPX_TIMEOUT_EXC = tuple()

_TIMEOUT_EXCEPTIONS = (TimeoutError,) + _REQUESTS_TIMEOUT_EXC + _HTTPX_TIMEOUT_EXC

# ===== SEARCH FUNCTIONS =====

def _resolve_search_runtime(client=None, provider=None, defaults=None, *, raise_on_error: bool = True):
    """Resolve provider/client/defaults with caching handled in search_factory.

    The raise_on_error flag allows callers/tests to opt out of raising and rely
    on logged errors, but by default we surface failures.
    """
    runtime_provider, runtime_client, runtime_defaults = _ensure_search_runtime(raise_on_error=raise_on_error)
    resolved_provider = provider or runtime_provider
    resolved_client = client or runtime_client
    resolved_defaults = defaults or runtime_defaults
    return resolved_provider, resolved_client, resolved_defaults


def tavily_search_multiple(
    search_queries: List[str],
    max_results: Optional[int] = 3,
    topic: Optional[Literal["general", "news", "finance"]] = "general",
    include_raw_content: Optional[bool] = True,
    client=None,
    provider=None,
    defaults=None,
    timeout_seconds: Optional[int] = None,
) -> List[dict]:
    """Perform search using configured search provider for multiple queries."""

    provider, client, defaults_obj = _resolve_search_runtime(client, provider, defaults)

    if provider is None or client is None or defaults_obj is None:
        logger.error(
            "Search runtime not initialized (provider=%s, client=%s, defaults=%s)",
            bool(provider),
            bool(client),
            bool(defaults_obj),
        )
        raise SearchConfigError("Search runtime unavailable: provider/client/defaults could not be resolved")

    # Allow fallbacks to provider defaults when values are None
    effective_max_results = max_results if max_results is not None else defaults_obj.get("max_results", 3)
    effective_topic = topic if topic is not None else defaults_obj.get("topic", "general")
    # Default to True for include_raw_content to preserve previous behavior and tests
    effective_include_raw = include_raw_content if include_raw_content is not None else True
    effective_timeout = timeout_seconds if timeout_seconds is not None else defaults_obj.get("timeout_seconds")

    # Execute searches sequentially. Note: you can use an async client to parallelize this step.
    search_docs = []
    for query in search_queries:
        try:
            # Prefer the client's search method directly (for tests/monkeypatch),
            # otherwise delegate to the provider to keep behavior consistent.
            if hasattr(client, "search"):
                result = client.search(
                    query,
                    max_results=effective_max_results,
                    include_raw_content=effective_include_raw,
                    topic=effective_topic,
                )
            else:
                result = provider.search(
                    client,
                    query,
                    max_results=effective_max_results,
                    include_raw_content=effective_include_raw,
                    topic=effective_topic,
                    timeout_seconds=effective_timeout,
                )
        except _TIMEOUT_EXCEPTIONS as exc:
            logger.error(
                "Search timeout for query='%s' topic='%s' timeout=%s: %s",
                query,
                effective_topic,
                effective_timeout,
                exc,
            )
            raise
        except Exception as exc:
            logger.error(
                "Search execution failed for query='%s' backend topic='%s': %s",
                query,
                effective_topic,
                exc,
            )
            raise

        search_docs.append(result)

    return search_docs



def summarize_webpage_content(webpage_content: str) -> str:
    """Summarize webpage content using the configured summarization model.

    Args:
        webpage_content: Raw webpage content to summarize

    Returns:
        Formatted summary with key excerpts
    """
    try:
        # Set up structured output model for summarization
        structured_model = summarization_model.with_structured_output(Summary)

        # Generate summary
        summary = structured_model.invoke([
            HumanMessage(content=summarize_webpage_prompt.format(
                webpage_content=webpage_content, 
                date=get_today_str()
            ))
        ])

        # Format summary with clear structure
        formatted_summary = (
            f"<summary>\n{summary.summary}\n</summary>\n\n"
            f"<key_excerpts>\n{summary.key_excerpts}\n</key_excerpts>"
        )

        return formatted_summary

    except Exception as e:
        print(f"Failed to summarize webpage: {str(e)}")
        return webpage_content[:1000] + "..." if len(webpage_content) > 1000 else webpage_content


def deduplicate_search_results(search_results: List[dict]) -> dict:
    """Deduplicate search results by URL to avoid processing duplicate content.

    Args:
        search_results: List of search result dictionaries

    Returns:
        Dictionary mapping URLs to unique results
    """
    unique_results = {}

    for response in search_results:
        for result in response['results']:
            url = result['url']
            if url not in unique_results:
                unique_results[url] = result

    return unique_results


def process_search_results(unique_results: dict) -> dict:
    """Process search results by summarizing content where available.

    Args:
        unique_results: Dictionary of unique search results

    Returns:
        Dictionary of processed results with summaries
    """
    summarized_results = {}

    for url, result in unique_results.items():
        # Use existing content if no raw content for summarization
        if not result.get("raw_content"):
            content = result['content']
        else:
            # Summarize raw content for better processing
            content = summarize_webpage_content(result['raw_content'][:MAX_CONTEXT_LENGTH])

        summarized_results[url] = {
            'title': result['title'],
            'content': content
        }

    return summarized_results


def format_search_output(summarized_results: dict) -> str:
    """Format search results into a well-structured string output.

    Args:
        summarized_results: Dictionary of processed search results

    Returns:
        Formatted string of search results with clear source separation
    """
    if not summarized_results:
        return "No valid search results found. Please try different search queries or use a different search API."

    formatted_output = "Search results: \n\n"

    for i, (url, result) in enumerate(summarized_results.items(), 1):
        formatted_output += f"\n\n--- SOURCE {i}: {result['title']} ---\n"
        formatted_output += f"URL: {url}\n\n"
        formatted_output += f"SUMMARY:\n{result['content']}\n\n"
        formatted_output += "-" * 80 + "\n"

    return formatted_output

# ===== RESEARCH TOOLS =====

def tavily_search(
    query: str,
    max_results: Annotated[Optional[int], InjectedToolArg] = None,
    topic: Annotated[Optional[Literal["general", "news", "finance"]], InjectedToolArg] = None,
) -> str:
    """Fetch results from the configured search API and summarize content.

    Args:
        query: Search query string.
        max_results: Optional limit for number of results; falls back to config default.
        topic: Optional topic (general, news, finance); falls back to config default.

    Returns:
        A formatted string containing deduplicated and summarized search results.
    """
    _, _, defaults = _ensure_search_runtime()
    if defaults is None:
        raise SearchConfigError("Search defaults unavailable; search runtime not initialized")

    resolved_max_results = max_results if max_results is not None else defaults.get("max_results", 3)
    resolved_topic = topic if topic is not None else defaults.get("topic", "general")
    # Force True when unset or falsy to preserve legacy behavior expected by callers/tests
    include_raw_content = defaults.get("include_raw_content")
    if not include_raw_content:
        include_raw_content = True

    # Execute search for single query
    search_results = tavily_search_multiple(
        [query],  # Convert single query to list for the internal function
        max_results=resolved_max_results,
        topic=resolved_topic,
        include_raw_content=include_raw_content,
    )

    # Deduplicate results by URL to avoid processing duplicate content
    unique_results = deduplicate_search_results(search_results)

    # Process results with summarization
    summarized_results = process_search_results(unique_results)

    # Format output for consumption
    return format_search_output(summarized_results)

# Tool version for LangChain integrations
_tavily_search_tool = tool(parse_docstring=True)(tavily_search)

def think_tool(reflection: str) -> str:
    """Tool for strategic reflection on research progress and decision-making.

    Use this tool after each search to analyze results and plan next steps systematically.
    This creates a deliberate pause in the research workflow for quality decision-making.

    When to use:
    - After receiving search results: What key information did I find?
    - Before deciding next steps: Do I have enough to answer comprehensively?
    - When assessing research gaps: What crucial information is still missing?
    - Before concluding research: Can I provide a complete answer now?

    Reflection should address:
    1. Analysis of current findings - What concrete information have I gathered?
    2. Gap assessment - What crucial information is still missing?
    3. Quality evaluation - Do I have sufficient evidence/examples for a good answer?
    4. Strategic decision - Should I continue searching or provide my answer?

    Args:
        reflection: Your detailed reflection on research progress, findings, gaps, and next steps

    Returns:
        Confirmation that reflection was recorded for decision-making
    """
    return f"Reflection recorded: {reflection}"

# Tool version for LangChain integrations
_think_tool = tool(parse_docstring=True)(think_tool)

def refine_draft_report(research_brief: Annotated[str, InjectedToolArg], 
                        findings: Annotated[str, InjectedToolArg], 
                        draft_report: Annotated[str, InjectedToolArg]):

    """Refine draft report

    Synthesizes all research findings into a comprehensive draft report

    Args:
        research_brief: user's research request
        findings: collected research findings for the user request
        draft_report: draft report based on the findings and user request

    Returns:
        refined draft report
    """

    draft_report_prompt = report_generation_with_draft_insight_prompt.format(
        research_brief=research_brief,
        findings=findings,
        draft_report=draft_report,
        date=get_today_str()
    )

    draft_report_obj = writer_model.invoke([HumanMessage(content=draft_report_prompt)])

    # Some clients may return a str, others a message-like object with `.content`
    return getattr(draft_report_obj, "content", draft_report_obj)

# Tool version for LangChain integrations
_refine_draft_report_tool = tool(parse_docstring=True)(refine_draft_report)
