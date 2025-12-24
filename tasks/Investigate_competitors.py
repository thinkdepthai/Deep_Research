"""
Library Competitor Investigation Module

This module provides functionality to research competitors for Boost C++ libraries
and analyze recent trends over the last 2-3 years.
"""

import json
import asyncio
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from deep_research.research_agent_full import deep_researcher_builder
from deep_research.usage_tracker import get_tracker, reset_tracker


LIBRARY_FILE="boost_library.json"
OUTPUT_PATH="report_by_library"


class find_library_competitors:
    """
    Class to investigate competitors for Boost C++ libraries.

    For each library in boost_library.json, this class:
    1. Researches top ~10 competitors
    2. Analyzes recent trends (last 2-3 years)
    3. Saves results as JSON
    """

    def __init__(self, json_file: str = "boost_library.json", output_file: str = "library_competitors.json"):
        """
        Initialize the competitor finder.

        Args:
            json_file: Path to the boost_library.json file (currently uses LIBRARY_FILE constant)
            output_file: Path to save the results JSON file
        """
        self.json_file = Path(LIBRARY_FILE)
        self.output_file = Path(output_file)
        self.results = {
            "generated_at": datetime.now().isoformat(),
            "source_file": str(self.json_file),
            "libraries": []
        }

        # Initialize the research agent
        self.checkpointer = InMemorySaver()
        self.research_agent = deep_researcher_builder.compile(checkpointer=self.checkpointer)

    def load_libraries(self) -> List[Dict[str, Any]]:
        """
        Load libraries from boost_library.json.

        Returns:
            List of library dictionaries
        """
        if not self.json_file.exists():
            raise FileNotFoundError(f"Library file not found: {self.json_file}")

        with open(self.json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return data.get("libraries", [])

    def create_research_query(self, library: Dict[str, Any]) -> str:
        """
        Create a research query for a specific library.

        Args:
            library: Library dictionary with name, description, etc.

        Returns:
            Research query string
        """
        name = library.get("name", "")
        description = library.get("description", "").strip()
        cpp_version = library.get("c++_version", "")

        query = f"""Research top 5-6 C++ library competitors for Boost.{name} ({cpp_version}).

Library: {name}
Purpose: {description}

Provide a CONCISE, OBJECTIVE, and QUANTITATIVE report (max 800 words) with SHORT paragraphs (2-4 sentences max).

**CRITICAL: CONCISENESS REQUIREMENTS:**
- Keep paragraphs SHORT (2-4 sentences maximum)
- Use bullet points and tables liberally for better readability
- Avoid long, dense paragraphs - break information into digestible chunks
- Prioritize key insights over exhaustive detail
- Remove redundant information
- Do not use bold formatting and quotation marks within a sentence.

1. **Top 5-6 Competitors** (use bullet points, keep each competitor to 3-4 bullet points):
   For each competitor, provide:
   - Name, GitHub URL, 1-sentence description
   - Main advantage vs Boost {name}
   - Main disadvantage vs Boost {name}
   - GitHub stars, last update date, C++ standard requirement (if available)

2. **Quantitative Metric Comparison Table**:
   Create a concise comparison table with these metrics for Boost.{name} and each competitor:
   - **Performance**: Runtime speed, memory usage, build time impact
   - **Code Quality**: Lines of code (approx), test coverage (%), static analysis findings
   - **Ecosystem**: GitHub stars, contributor count, issue resolution time
   - **Standards & Portability**: C++ standard compliance, platform support
   - **Usability**: API ergonomics (1-5 rating), documentation quality (1-5 rating)

   Use actual numbers when available, or "—" if not available. Be objective and data-driven.

3. **Summary Comparison Table**:
   Create a concise table with columns:
   | Library | Main Advantage vs Boost.{name} | Main Disadvantage vs Boost.{name} | When to Prefer | C++ Standard |

   Include Boost.{name} in the table for reference.

4. **Recent Trends (2023-2025)** - Brief summary (2-3 short paragraphs max):
   - Adoption trends (growing/stable/declining) with evidence
   - Major updates or releases
   - Community activity level

5. **Conclusion** - Objective assessment (2-3 short paragraphs max):
   - When to use Boost {name} vs alternatives
   - Key trade-offs based on quantitative metrics

Focus on objective, quantitative data. Include specific numbers, dates, and metrics when available. Keep the report concise and scannable."""

        return query

    async def research_library_competitors(self, library: Dict[str, Any]) -> Dict[str, Any]:
        """
        Research competitors for a single library using the deep research agent.

        Args:
            library: Library dictionary to research

        Returns:
            Dictionary containing research results
        """
        library_name = library.get("name", "Unknown")
        query = self.create_research_query(library)

        # Configure thread for this library with reduced limits for efficiency
        thread_config = {
            "configurable": {
                "thread_id": f"library_{library_name.lower().replace(' ', '_')}",
                "recursion_limit": 15  # Reduced from 50 to 15
            }
        }

        try:
            # Reset tracker for this library
            reset_tracker()
            tracker = get_tracker()

            # Run the research agent
            result = await self.research_agent.ainvoke(
                {"messages": [HumanMessage(content=query)]},
                config=thread_config
            )

            # Finalize tracking
            tracker.finalize()

            # Extract the final report and usage stats
            final_report = result.get("final_report", "")
            usage_stats = tracker.get_stats().to_dict()

            # Red team evaluation is already performed in research_agent_full.py
            # Just extract it from the result if it exists
            result_dict = {
                "library_name": library_name,
                "library_info": {
                    "name": library.get("name"),
                    "description": library.get("description"),
                    "c++_version": library.get("c++_version"),
                    "url": library.get("name_link")
                },
                "research_report": final_report,
                "research_completed_at": datetime.now().isoformat(),
                "usage_stats": usage_stats,
                "status": "completed"
            }

            # Include red team evaluation if it was performed by research_agent_full
            if "red_team_evaluation" in result:
                result_dict["red_team_evaluation"] = result["red_team_evaluation"]
                result_dict["red_team_report"] = result.get("red_team_report", "")

            return result_dict

        except Exception as e:
            print(f"Error researching {library_name}: {str(e)}")
            return {
                "library_name": library_name,
                "library_info": {
                    "name": library.get("name"),
                    "description": library.get("description"),
                    "c++_version": library.get("c++_version"),
                    "url": library.get("name_link")
                },
                "research_report": f"Error during research: {str(e)}",
                "research_completed_at": datetime.now().isoformat(),
                "status": "error",
                "error": str(e)
            }

    async def process_all_libraries(self, start_index: int = 0, max_libraries: int = None):
        """
        Process all libraries from the JSON file.

        Args:
            start_index: Index to start processing from (for resuming)
            max_libraries: Maximum number of libraries to process (None for all)
        """
        libraries = self.load_libraries()

        if max_libraries:
            libraries = libraries[start_index:start_index + max_libraries]
        else:
            libraries = libraries[start_index:]

        total = len(libraries)
        total_in_tokens = 0
        total_out_tokens = 0
        total_tavily_calls = 0

        # Create progress log file
        progress_log = []
        progress_log.append(f"Processing {total} libraries...\n")

        for idx, library in enumerate(libraries[73:74], start=1):
            library_name = library.get("name", "Unknown")
            progress_log.append(f"\n[{idx}/{total}] Processing library {idx + start_index}: {library_name}")

            result = await self.research_library_competitors(library)
            self.results["libraries"].append(result)

            # Track usage stats for this library
            if "usage_stats" in result:
                stats = result["usage_stats"]
                openai_stats = stats.get("openai", {})
                tavily_stats = stats.get("tavily", {})
                total_in_tokens += openai_stats.get('prompt_tokens', 0)
                total_out_tokens += openai_stats.get('completion_tokens', 0)
                total_tavily_calls += tavily_stats.get('api_calls', 0)
                progress_log.append(f"  Tokens: {openai_stats.get('total_tokens', 0):,} (in: {openai_stats.get('prompt_tokens', 0):,}, out: {openai_stats.get('completion_tokens', 0):,})")
                progress_log.append(f"  Tavily calls: {tavily_stats.get('api_calls', 0)}")

                if "red_team_evaluation" in result:
                    overall_score = result["red_team_evaluation"].get("overall_score", 0.0)
                    progress_log.append(f"  Objectivity Score: {overall_score:.1%}")

            # Save progress after each library (incremental saves)
            self.save_results(library["name"])
            progress_log.append(f"✓ Completed research for: {library_name}")

        # Post-process red team evaluation results
        self.process_red_team_results()

        # Save progress log
        progress_log.append(f"\n{'-'*80}")
        progress_log.append(f"Total libraries processed: {total}")
        progress_log.append(f"Total in tokens: {total_in_tokens:,}")
        progress_log.append(f"Total out tokens: {total_out_tokens:,}")
        progress_log.append(f"Total tavily calls: {total_tavily_calls:,}")

        # Save progress log to file
        output_dir = Path(OUTPUT_PATH)
        output_dir.mkdir(exist_ok=True)
        progress_file = output_dir / "processing_log.md"
        with open(progress_file, 'w', encoding='utf-8') as f:
            f.write("# Processing Log\n\n")
            f.write(f"**Generated:** {datetime.now().isoformat()}\n\n")
            f.write("---\n\n")
            f.write("\n".join(progress_log))

        print(f"Processing complete. Progress log saved to: {progress_file}")

    def save_results(self, library_name: str="unknown"):
        """Save results to markdown file."""
        self.results["last_updated"] = datetime.now().isoformat()
        self.results["total_libraries_researched"] = len(self.results["libraries"])

        # Create output directory if it doesn't exist
        output_dir = Path(OUTPUT_PATH)
        output_dir.mkdir(exist_ok=True)

        output_file = output_dir / f"{library_name.replace('/', '_').replace(' ', '_')}.md"

        # Find the latest research result for this library
        latest_result = None
        for lib_result in self.results["libraries"]:
            if lib_result.get("library_name") == library_name:
                latest_result = lib_result
                break

        if latest_result:
            # Write markdown file with the research report
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(f"# Competitor Analysis: {library_name}\n\n")
                f.write(f"**Generated:** {latest_result.get('research_completed_at', 'N/A')}\n\n")
                f.write("**Library Info:**\n")
                lib_info = latest_result.get("library_info", {})
                f.write(f"- **Name:** {lib_info.get('name', 'N/A')}\n")
                f.write(f"- **C++ Version:** {lib_info.get('c++_version', 'N/A')}\n")
                f.write(f"- **Description:** {lib_info.get('description', 'N/A')}\n")
                f.write(f"- **URL:** {lib_info.get('url', 'N/A')}\n\n")
                f.write("---\n\n")
                f.write(latest_result.get("research_report", "No report available"))

            # Results are saved silently to file

    def process_red_team_results(self):
        """
        Post-process red team evaluation results across all libraries.

        This method:
        1. Calculates aggregate statistics
        2. Identifies libraries with quality issues
        3. Generates summary reports
        4. Tracks common problems
        """
        libraries_with_eval = [
            lib for lib in self.results["libraries"]
            if "red_team_evaluation" in lib and lib.get("status") == "completed"
        ]

        if not libraries_with_eval:
            return

        # Aggregate statistics
        overall_scores = []
        one_sided_scores = []
        source_credibility_scores = []
        quantitative_ratios = []
        total_unsupported_claims = 0
        total_missing_citations = 0
        total_counter_evidence_gaps = 0

        quality_issues = {
            "low_objectivity": [],  # < 60%
            "high_bias": [],  # one_sided > 60%
            "poor_sources": [],  # credibility < 60%
            "low_quantitative": [],  # quantitative_ratio < 30%
            "many_unsupported": [],  # > 3 unsupported claims
        }

        for lib in libraries_with_eval:
            eval_data = lib["red_team_evaluation"]
            lib_name = lib["library_name"]

            overall_score = eval_data.get("overall_score", 0.0)
            overall_scores.append(overall_score)

            bias_metrics = eval_data.get("bias_metrics", {})
            one_sided = bias_metrics.get("one_sided_score", 0.5)
            one_sided_scores.append(one_sided)

            source_quality = eval_data.get("source_quality", {})
            credibility = source_quality.get("credibility_score", 0.5)
            source_credibility_scores.append(credibility)

            quantitative = bias_metrics.get("quantitative_ratio", 0.5)
            quantitative_ratios.append(quantitative)

            issues = eval_data.get("issues", {})
            unsupported = issues.get("unsupported_claims_count", 0)
            counter_evidence = issues.get("counter_evidence_gaps_count", 0)
            total_unsupported_claims += unsupported
            total_counter_evidence_gaps += counter_evidence

            missing_citations = source_quality.get("missing_citations_count", 0)
            total_missing_citations += missing_citations

            # Categorize quality issues
            if overall_score < 0.6:
                quality_issues["low_objectivity"].append((lib_name, overall_score))
            if one_sided > 0.6:
                quality_issues["high_bias"].append((lib_name, one_sided))
            if credibility < 0.6:
                quality_issues["poor_sources"].append((lib_name, credibility))
            if quantitative < 0.3:
                quality_issues["low_quantitative"].append((lib_name, quantitative))
            if unsupported > 3:
                quality_issues["many_unsupported"].append((lib_name, unsupported))

        # Calculate averages
        avg_overall = sum(overall_scores) / len(overall_scores) if overall_scores else 0
        avg_one_sided = sum(one_sided_scores) / len(one_sided_scores) if one_sided_scores else 0
        avg_credibility = sum(source_credibility_scores) / len(source_credibility_scores) if source_credibility_scores else 0
        avg_quantitative = sum(quantitative_ratios) / len(quantitative_ratios) if quantitative_ratios else 0

        # Score distribution
        excellent = sum(1 for s in overall_scores if s >= 0.8)
        good = sum(1 for s in overall_scores if 0.6 <= s < 0.8)
        needs_improvement = sum(1 for s in overall_scores if s < 0.6)

        # Statistics will be saved to file in save_red_team_summary

        # Generate and save summary report
        self.save_red_team_summary(
            libraries_with_eval,
            {
                "avg_overall": avg_overall,
                "avg_one_sided": avg_one_sided,
                "avg_credibility": avg_credibility,
                "avg_quantitative": avg_quantitative,
                "total_unsupported": total_unsupported_claims,
                "total_missing_citations": total_missing_citations,
                "total_counter_evidence": total_counter_evidence_gaps,
                "excellent": excellent,
                "good": good,
                "needs_improvement": needs_improvement,
            },
            quality_issues
        )

    def save_red_team_summary(self, libraries_with_eval, stats, quality_issues):
        """
        Save a comprehensive red team evaluation summary report.

        Args:
            libraries_with_eval: List of libraries with red team evaluations
            stats: Aggregate statistics dictionary
            quality_issues: Dictionary of quality issues by category
        """
        output_dir = Path(OUTPUT_PATH)
        output_dir.mkdir(exist_ok=True)

        summary_file = output_dir / "red_team_evaluation_summary.md"

        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("# Red Team Evaluation Summary\n\n")
            f.write(f"**Generated:** {datetime.now().isoformat()}\n")
            f.write(f"**Total Libraries Evaluated:** {len(libraries_with_eval)}\n\n")
            f.write("---\n\n")

            # Aggregate Statistics
            f.write("## Aggregate Statistics\n\n")
            f.write(f"- **Average Objectivity Score:** {stats['avg_overall']:.1%}\n")
            f.write(f"- **Average One-Sided Score:** {stats['avg_one_sided']:.1%} (lower is better)\n")
            f.write(f"- **Average Source Credibility:** {stats['avg_credibility']:.1%}\n")
            f.write(f"- **Average Quantitative Ratio:** {stats['avg_quantitative']:.1%}\n")
            f.write(f"- **Total Unsupported Claims:** {stats['total_unsupported']}\n")
            f.write(f"- **Total Missing Citations:** {stats['total_missing_citations']}\n")
            f.write(f"- **Total Counter-Evidence Gaps:** {stats['total_counter_evidence']}\n\n")

            # Score Distribution
            f.write("## Score Distribution\n\n")
            f.write(f"- **Excellent (≥80%):** {stats['excellent']} libraries\n")
            f.write(f"- **Good (60-80%):** {stats['good']} libraries\n")
            f.write(f"- **Needs Improvement (<60%):** {stats['needs_improvement']} libraries\n\n")

            # Quality Issues
            f.write("## Quality Issues by Category\n\n")

            if quality_issues["low_objectivity"]:
                f.write("### Low Objectivity (<60%)\n\n")
                for name, score in sorted(quality_issues["low_objectivity"], key=lambda x: x[1]):
                    f.write(f"- **{name}**: {score:.1%}\n")
                f.write("\n")

            if quality_issues["high_bias"]:
                f.write("### High Bias (One-Sided >60%)\n\n")
                for name, score in sorted(quality_issues["high_bias"], key=lambda x: x[1], reverse=True):
                    f.write(f"- **{name}**: {score:.1%}\n")
                f.write("\n")

            if quality_issues["poor_sources"]:
                f.write("### Poor Source Quality (<60%)\n\n")
                for name, score in sorted(quality_issues["poor_sources"], key=lambda x: x[1]):
                    f.write(f"- **{name}**: {score:.1%}\n")
                f.write("\n")

            if quality_issues["low_quantitative"]:
                f.write("### Low Quantitative Data (<30%)\n\n")
                for name, score in sorted(quality_issues["low_quantitative"], key=lambda x: x[1]):
                    f.write(f"- **{name}**: {score:.1%}\n")
                f.write("\n")

            if quality_issues["many_unsupported"]:
                f.write("### Many Unsupported Claims (>3)\n\n")
                for name, count in sorted(quality_issues["many_unsupported"], key=lambda x: x[1], reverse=True):
                    f.write(f"- **{name}**: {count} claims\n")
                f.write("\n")

            # Detailed Library Scores
            f.write("## Detailed Library Scores\n\n")
            f.write("| Library | Objectivity | One-Sided | Source Credibility | Quantitative | Unsupported Claims |\n")
            f.write("|---------|------------|-----------|-------------------|--------------|-------------------|\n")

            for lib in sorted(libraries_with_eval, key=lambda x: x["red_team_evaluation"].get("overall_score", 0), reverse=True):
                lib_name = lib["library_name"]
                eval_data = lib["red_team_evaluation"]
                overall = eval_data.get("overall_score", 0.0)
                one_sided = eval_data.get("bias_metrics", {}).get("one_sided_score", 0.5)
                credibility = eval_data.get("source_quality", {}).get("credibility_score", 0.5)
                quantitative = eval_data.get("bias_metrics", {}).get("quantitative_ratio", 0.5)
                unsupported = eval_data.get("issues", {}).get("unsupported_claims_count", 0)

                f.write(f"| {lib_name} | {overall:.1%} | {one_sided:.1%} | {credibility:.1%} | {quantitative:.1%} | {unsupported} |\n")

            f.write("\n")

            # Recommendations
            f.write("## Recommendations\n\n")
            if stats['avg_overall'] < 0.7:
                f.write("- **Overall quality needs improvement**: Consider reviewing research methodology\n")
            if stats['avg_one_sided'] > 0.5:
                f.write("- **Bias concerns**: Encourage inclusion of alternative perspectives and counter-evidence\n")
            if stats['avg_credibility'] < 0.7:
                f.write("- **Source quality**: Prioritize primary sources and authoritative publications\n")
            if stats['avg_quantitative'] < 0.4:
                f.write("- **Quantitative data**: Increase use of specific numbers, statistics, and measurable metrics\n")
            if stats['total_unsupported'] > len(libraries_with_eval) * 2:
                f.write("- **Unsupported claims**: Add citations and evidence for assertions\n")

            f.write("\n")

    async def run(self, start_index: int = 0, max_libraries: int = None):
        """
        Main method to run the competitor research for all libraries.

        Args:
            start_index: Index to start processing from (for resuming)
            max_libraries: Maximum number of libraries to process (None for all)
        """
        print("="*80)
        print("Boost Library Competitor Research")
        print("="*80)
        print(f"Input file: {self.json_file}")
        print(f"Output file: {OUTPUT_PATH}")
        print("="*80)

        await self.process_all_libraries(start_index, max_libraries)
        self.save_results()

        # Save completion summary
        output_dir = Path(OUTPUT_PATH)
        output_dir.mkdir(exist_ok=True)
        summary_file = output_dir / "completion_summary.md"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("# Research Completion Summary\n\n")
            f.write(f"**Completed:** {datetime.now().isoformat()}\n\n")
            f.write(f"**Total Libraries Researched:** {len(self.results['libraries'])}\n")
            f.write(f"**Output Directory:** {OUTPUT_PATH}\n\n")
            f.write("All individual library reports have been saved to the output directory.\n")

        print(f"Research complete. All results saved to: {OUTPUT_PATH}")


async def main():
    """
    Main entry point for running the competitor research.
    """
    # Initialize the competitor finder
    finder = find_library_competitors()

    # Run the research
    # You can specify start_index and max_libraries to process in batches
    # For example: await finder.run(start_index=0, max_libraries=5)  # Process first 5 libraries
    await finder.run()  # Process all libraries


if __name__ == "__main__":
    asyncio.run(main())

