"""
Main entry point for ThinkDepth.ai Deep Research.

This script provides a command-line interface to run deep research queries.
"""

import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from rich.console import Console
from rich.panel import Panel


# Initialize console for rich output
console = Console()

# Import the research agent
# Note: This requires the package to be installed: pip install -e .
try:
    from deep_research.research_agent_full import deep_researcher_builder
    from deep_research.usage_tracker import get_tracker
except ImportError as e:
    console.print(
        Panel(
            f"[red]Import Error: {str(e)}[/red]\n\n"
            "The package needs to be installed first.\n\n"
            "[yellow]Install with:[/yellow]\n"
            "  pip install -e .\n\n"
            "This installs the package in editable mode so you can run main.py.",
            title="[bold red]Package Not Installed[/bold red]",
            border_style="red"
        )
    )
    sys.exit(1)


def load_env_file():
    """Load environment variables from .env file if it exists."""
    # Look for .env file in project root (parent of src directory)
    project_root = Path(__file__).parent.parent
    env_file = project_root / ".env"

    if env_file.exists():
        load_dotenv(env_file)
        console.print(f"[dim]Loaded environment variables from {env_file}[/dim]")
        return True
    else:
        # Also try current directory
        current_dir_env = Path(".env")
        if current_dir_env.exists():
            load_dotenv(current_dir_env)
            console.print(f"[dim]Loaded environment variables from {current_dir_env}[/dim]")
            return True
        return False


def check_api_keys():
    """Check if required API keys are set."""
    missing_keys = []

    if not os.getenv("OPENAI_API_KEY"):
        missing_keys.append("OPENAI_API_KEY")

    if not os.getenv("TAVILY_API_KEY"):
        missing_keys.append("TAVILY_API_KEY")

    if missing_keys:
        console.print(
            Panel(
                f"[red]Missing API Keys: {', '.join(missing_keys)}[/red]\n\n"
                "Please set your API keys in one of the following ways:\n\n"
                "[yellow]Option 1: .env file (recommended)[/yellow]\n"
                "  Create a .env file in the project root with:\n"
                "  OPENAI_API_KEY=Your OpenAI API Key\n"
                "  TAVILY_API_KEY=Your Tavily API Key\n\n"
                "[yellow]Option 2: Environment variables[/yellow]\n"
                "[yellow]PowerShell:[/yellow]\n"
                "  $env:OPENAI_API_KEY='Your OpenAI API Key'\n"
                "  $env:TAVILY_API_KEY='Your Tavily API Key'\n\n"
                "[yellow]Linux/macOS:[/yellow]\n"
                "  export OPENAI_API_KEY='Your OpenAI API Key'\n"
                "  export TAVILY_API_KEY='Your Tavily API Key'",
                title="[bold red]API Keys Required[/bold red]",
                border_style="red"
            )
        )
        return False

    return True


async def run_research(query: str, thread_id: str = "default", recursion_limit: int = 50):
    """
    Run the deep research agent with a given query.

    Args:
        query: The research query/question to investigate
        thread_id: Unique identifier for this research session
        recursion_limit: Maximum recursion depth for the agent

    Returns:
        Dictionary containing the final report and other results
    """
    console.print(f"\n[bold cyan]Starting Deep Research...[/bold cyan]")
    console.print(f"[dim]Query: {query}[/dim]\n")

    # Initialize checkpointer for state management
    checkpointer = InMemorySaver()

    # Compile the agent with checkpointer
    full_agent = deep_researcher_builder.compile(checkpointer=checkpointer)

    # Configure thread for state persistence
    thread_config = {
        "configurable": {
            "thread_id": thread_id,
            "recursion_limit": recursion_limit
        }
    }

    # Run the agent
    console.print("[yellow]Research in progress (this may take 10-30 minutes)...[/yellow]\n")

    # Reset and start tracking
    tracker = get_tracker()
    tracker.reset()

    try:
        result = await full_agent.ainvoke(
            {"messages": [HumanMessage(content=query)]},
            config=thread_config
        )

        # Finalize tracking
        tracker.finalize()

        # Add usage stats to result
        result["usage_stats"] = tracker.get_stats().to_dict()

        # Log feedback loop summary if available
        if "red_team_feedback_history" in result:
            feedback_history = result.get("red_team_feedback_history", [])
            if feedback_history:
                print("\n" + "="*80)
                print("BLUE-RED FEEDBACK LOOP SUMMARY")
                print("="*80)
                print(f"Total evaluation cycles: {len(feedback_history)}")
                print(f"Total refinement iterations: {result.get('refinement_iterations', 0)}")

                if len(feedback_history) > 1:
                    initial_score = feedback_history[0].get("score", 0)
                    final_score = feedback_history[-1].get("score", 0)
                    improvement = final_score - initial_score
                    print("\nScore progression:")
                    print(f"  Initial: {initial_score:.1%}")
                    print(f"  Final: {final_score:.1%}")
                    print(f"  Improvement: {improvement:+.1%}")

                print("="*80 + "\n")

        return result

    except Exception as e:  # noqa: BLE001
        console.print(f"[bold red]Error during research:[/bold red] {str(e)}")
        raise


def save_results_to_file(result: dict, output_file: str = None):
    """
    Save research results to a markdown file.

    Args:
        result: Dictionary containing research results
        output_file: Optional path to output file. If None, generates a timestamped filename.
    """
    from datetime import datetime

    # Generate default filename if not provided
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"research_report_{timestamp}.md"

    output_path = Path(output_file)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# Research Report\n\n")
        f.write(f"**Generated:** {datetime.now().isoformat()}\n\n")
        f.write("---\n\n")

        # Usage Statistics
        if "usage_stats" in result:
            stats = result["usage_stats"]
            openai_stats = stats.get("openai", {})
            tavily_stats = stats.get("tavily", {})

            total_in_tokens = openai_stats.get('prompt_tokens', 0)
            total_out_tokens = openai_stats.get('completion_tokens', 0)
            total_tokens = openai_stats.get('total_tokens', 0)
            total_tavily_calls = tavily_stats.get('api_calls', 0)

            f.write("## Usage Statistics\n\n")
            f.write(f"- **Input tokens:** {total_in_tokens:,}\n")
            f.write(f"- **Output tokens:** {total_out_tokens:,}\n")
            f.write(f"- **Total tokens:** {total_tokens:,}\n")
            f.write(f"- **Tavily calls:** {total_tavily_calls:,}\n\n")
            f.write("---\n\n")

        # Red Team Evaluation
        if "red_team_evaluation" in result:
            eval_summary = result["red_team_evaluation"]
            overall_score = eval_summary.get("overall_score", 0.0)

            f.write("## Red Team Evaluation\n\n")
            f.write(f"### Overall Objectivity Score: {overall_score:.1%}\n\n")
            f.write(f"- **Bias Issues:** {eval_summary.get('bias_metrics', {}).get('missing_counter_evidence_count', 0)}\n")
            f.write(f"- **Unsupported Claims:** {eval_summary.get('issues', {}).get('unsupported_claims_count', 0)}\n")
            f.write(f"- **Source Credibility:** {eval_summary.get('source_quality', {}).get('credibility_score', 0.0):.1%}\n\n")

            if "red_team_report" in result:
                f.write("### Detailed Red Team Report\n\n")
                f.write(result["red_team_report"])
                f.write("\n\n")

            f.write("---\n\n")

        # Final Report
        if "final_report" in result:
            f.write("## Final Report\n\n")
            f.write(result["final_report"])
        else:
            f.write("## Warning\n\n")
            f.write("No final report found in results.\n")
            f.write(f"Available keys: {list(result.keys())}\n")

    console.print(f"\n[green]Report saved to: {output_path}[/green]")
    return str(output_path)

def create_research_query():
    """
    Create a research query from a file or use default.

    First tries to read from prompt.txt in the project root.
    If the file doesn't exist or is empty, uses a default query.

    Returns:
        str: The research query to use
    """
    # Default query if file doesn't exist or is empty
    default_query = """I need to write an up-to-date report titled "Notable Companies That Avoid Using MIT-Licensed Code Due to Binary Attribution Requirements".

Please research and report on the existence of cases among large, market-influential companies that have an explicit, documented practice of avoiding MIT-licensed code specifically because of the MIT license's requirement to preserve copyright and license notices in distributed binaries or user-facing products.

Requirements:
- Cross-reference multiple sources before asserting conclusions
- Clearly cite all references with full URLs
- Prioritize companies based on market impact and influence
- Focus on factual, verifiable information
- Include both positive findings (companies that do avoid MIT) and negative findings (no such companies found) if that's what the evidence shows
- Provide a comprehensive list of references at the end of the report

Note: Boost License (BSL-1.0) is convenient because it does not have binary attribution requirements, but this research should focus specifically on MIT license avoidance due to attribution requirements."""

    return default_query

def main():
    """Main entry point for the CLI."""
    # Load .env file first (if it exists)
    load_env_file()

    # Check API keys
    if not check_api_keys():
        sys.exit(1)

    # Get query from command line or prompt user
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        console.print("[bold]ThinkDepth.ai Deep Research[/bold]")
        console.print("[dim]Enter your research query (or press Ctrl+C to exit)[/dim]\n")
        query = create_research_query()

        if not query:
            console.print("[red]No query provided. Exiting.[/red]")
            sys.exit(1)

    # Optional: Get thread_id from environment or use default
    thread_id = os.getenv("RESEARCH_THREAD_ID", "default")

    # Optional: Get recursion limit from environment or use default
    try:
        recursion_limit = int(os.getenv("RESEARCH_RECURSION_LIMIT", "50"))
    except ValueError:
        recursion_limit = 50

    # Run the research
    try:
        result = asyncio.run(run_research(query, thread_id, recursion_limit))

        # Save results to file
        save_file = os.getenv("SAVE_REPORT_TO")
        if not save_file:
            # Generate default filename
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_file = f"research_report_{timestamp}.md"

        save_results_to_file(result, save_file)

    except KeyboardInterrupt:
        console.print("\n[yellow]Research interrupted by user.[/yellow]")
        sys.exit(1)
    except RuntimeError as e:
        # Handle API-related errors with better formatting
        error_msg = str(e)
        if "quota" in error_msg.lower() or "429" in error_msg:
            console.print(
                Panel(
                    error_msg + "\n\n"
                    "[yellow]To resolve this issue:[/yellow]\n"
                    "1. Check your OpenAI account usage: https://platform.openai.com/usage\n"
                    "2. Verify your billing information is set up\n"
                    "3. Consider upgrading your plan if you've hit usage limits\n"
                    "4. Wait for your quota to reset if you're on a free tier",
                    title="[bold red]API Quota Error[/bold red]",
                    border_style="red"
                )
            )
        elif "api key" in error_msg.lower() or "401" in error_msg:
            console.print(
                Panel(
                    error_msg + "\n\n"
                    "[yellow]To resolve this issue:[/yellow]\n"
                    "1. Verify your OPENAI_API_KEY in your .env file\n"
                    "2. Make sure the key is correct and not expired\n"
                    "3. Check that your .env file is in the project root directory",
                    title="[bold red]API Key Error[/bold red]",
                    border_style="red"
                )
            )
        else:
            console.print(
                Panel(
                    error_msg,
                    title="[bold red]Runtime Error[/bold red]",
                    border_style="red"
                )
            )
        sys.exit(1)
    except Exception as e:
        console.print(
            Panel(
                f"[bold red]Unexpected error:[/bold red]\n{str(e)}\n\n"
                "If this is an API error, check your API keys and quota limits.",
                title="[bold red]Fatal Error[/bold red]",
                border_style="red"
            )
        )
        sys.exit(1)


if __name__ == "__main__":
    main()

