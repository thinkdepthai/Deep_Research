

"""Research Utilities and Tools.

This module provides search and content processing utilities for the research agent,
including web search capabilities and content summarization tools.
"""
import requests
import os
from pathlib import Path
from datetime import datetime
from typing_extensions import Annotated, List, Literal

from langchain.chat_models import init_chat_model 
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool, InjectedToolArg
from tavily import TavilyClient
from deep_research.state_research import Summary
from deep_research.prompts import summarize_webpage_prompt, report_generation_with_draft_insight_prompt

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

summarization_model = init_chat_model(model="openai:gpt-5")
writer_model = init_chat_model(model="openai:gpt-5", max_tokens=32000)
tavily_client = TavilyClient()
MAX_CONTEXT_LENGTH = 250000

# ===== SEARCH FUNCTIONS =====

def perform_you_search(query: str, max_results: int = 3) -> dict:
    url = "https://api.ydc-index.io/v1/search"
    headers = {"x-api-key": os.environ["YOU_API_KEY"]}
    params = {
        "query": query,
        "count": max_results,
        "livecrawl": "all",
        "livecrawl_formats": "markdown",
    }
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error searching You.com: {e}")
        return None
    return response.json()

def ydc_search_multiple(
    search_queries: List[str], 
    max_results: int = 3, 
    include_raw_content: bool = True, 
) -> List[dict]:
    """Perform search using You.com search API for multiple queries.

    Args:
        search_queries: List of search queries to execute
        max_results: Maximum number of results per query
        include_raw_content: Whether to include raw webpage content

    Returns:
        List of search result dictionaries
    """
    # Execute searches sequentially. Note: yon can use AsyncTavilyClient to parallelize this step.
    search_docs = []
    for query in search_queries:
        try:
            result = perform_you_search(
                query=query,
                max_results=max_results,
            )
            search_docs.append(result)
        except Exception as e:
            print(f"Error searching You.com: {e}")
            return None
        
        search_docs.append(result)
    return search_docs

def tavily_search_multiple(
    search_queries: List[str], 
    max_results: int = 3, 
    topic: Literal["general", "news", "finance"] = "general", 
    include_raw_content: bool = True, 
) -> List[dict]:
    """Perform search using Tavily API for multiple queries.

    Args:
        search_queries: List of search queries to execute
        max_results: Maximum number of results per query
        topic: Topic filter for search results
        include_raw_content: Whether to include raw webpage content

    Returns:
        List of search result dictionaries
    """
    # Execute searches sequentially. Note: yon can use AsyncTavilyClient to parallelize this step.
    search_docs = []
    for query in search_queries:
        result = tavily_client.search(
            query,
            max_results=max_results,
            include_raw_content=include_raw_content,
            topic=topic
        )
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

def deduplicate_search_results(search_results: List[dict], search_tool_name: str) -> dict:
    """Deduplicate search results by URL to avoid processing duplicate content.

    Args:
        search_results: List of search result dictionaries

    Returns:
        Dictionary mapping URLs to unique results
    """
    unique_results = {}

    for response in search_results:
        if search_tool_name == "ydc":
            if "news" in response["results"]:
                all_results = response["results"]["web"] + response["results"]["news"]
            else:
                all_results = response["results"]["web"]
        elif search_tool_name == "tavily":
            all_results = response['results']
        
        for result in all_results:
            url = result['url']
            if url not in unique_results:
                unique_results[url] = result  
    return unique_results

def process_search_results(unique_results: dict, search_tool_name: str) -> dict:
    """Process search results by summarizing content where available.

    Args:
        unique_results: Dictionary of unique search results

    Returns:
        Dictionary of processed results with summaries
    """
    summarized_results = {}
    if search_tool_name == "tavily":
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
    elif search_tool_name == "ydc":
        for url, result in unique_results.items():
            # Use existing content if no raw content for summarization
            if not result.get("contents"):
                content = result.get("snippets", "")
                if content and isinstance(content, list):
                    content = " ".join(content)
            else:
                # Summarize raw content for better processing
                content = summarize_webpage_content(result['contents']['markdown'][:MAX_CONTEXT_LENGTH])

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

@tool(parse_docstring=True)
def ydc_search(
    query: str,
    max_results: Annotated[int, InjectedToolArg] = 3,
) -> str:
    """Fetch results from You.com search API with content summarization.

    Args:
        query: A single search query to execute
        max_results: Maximum number of results to return

    Returns:
        Formatted string of search results with summaries
    """
    print("in ydc_search xxx")
    # Execute search for single query
    search_results = ydc_search_multiple(
        [query],  # Convert single query to list for the internal function
        max_results=max_results,
    )


    # Deduplicate results by URL to avoid processing duplicate content
    unique_results = deduplicate_search_results(search_results, "ydc")
    # Process results with summarization
    summarized_results = process_search_results(unique_results, "ydc")
    print(f"length of summarized_results: {len(summarized_results)}")
    # Format output for consumption
    return format_search_output(summarized_results)


@tool(parse_docstring=True)
def tavily_search(
    query: str,
    max_results: Annotated[int, InjectedToolArg] = 3,
    topic: Annotated[Literal["general", "news", "finance"], InjectedToolArg] = "general",
) -> str:
    """Fetch results from Tavily search API with content summarization.

    Args:
        query: A single search query to execute
        max_results: Maximum number of results to return
        topic: Topic to filter results by ('general', 'news', 'finance')

    Returns:
        Formatted string of search results with summaries
    """
    print("in tavily_search")
    # Execute search for single query
    search_results = tavily_search_multiple(
        [query],  # Convert single query to list for the internal function
        max_results=max_results,
        topic=topic,
        include_raw_content=True,
    )

    # Deduplicate results by URL to avoid processing duplicate content
    unique_results = deduplicate_search_results(search_results, "tavily")

    # Process results with summarization
    summarized_results = process_search_results(unique_results, "tavily")
    print(f"length of summarized_results: {len(summarized_results)}")
    # Format output for consumption
    return format_search_output(summarized_results)

@tool(parse_docstring=True)
def think_tool(reflection: str) -> str:
    """Tool for strategic reflection on research progress and decision-making.

    Use this tool after each search to analyze results and plan next steps systematically.
    This creates a deliberate pause in the research workflow for quality decision-making.

    When to use:
    - After receiving search results: What key information did I find?
    - Before deciding next steps: Do I have enough to answer comprehensively?
    - When assessing research gaps: What specific information am I still missing?
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

@tool(parse_docstring=True)
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

    draft_report = writer_model.invoke([HumanMessage(content=draft_report_prompt)])

    return draft_report.content
