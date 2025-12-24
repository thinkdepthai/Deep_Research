"""
RAGaaS Platform Investigation Module

This module provides functionality to research RAGaaS (Retrieval-Augmented Generation as a Service)
platforms for C++ Copilot use case, analyzing key features, capabilities, and suitability.
"""

import asyncio
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from deep_research.research_agent_full import deep_researcher_builder
from deep_research.usage_tracker import get_tracker, reset_tracker


OUTPUT_PATH = "report_by_ragaas"


class investigate_ragaas:
    """
    Class to investigate RAGaaS platforms for C++ Copilot use case.

    This class:
    1. Researches top 6-8 RAGaaS platforms
    2. Analyzes features, capabilities, and suitability
    3. Saves results as markdown report
    """

    def __init__(self, output_path: str = OUTPUT_PATH):
        """
        Initialize the RAGaaS investigator.

        Args:
            output_path: Path to save the results
        """
        self.output_path = Path(output_path)
        self.output_path.mkdir(exist_ok=True)

        self.results = {
            "generated_at": datetime.now().isoformat(),
            "use_case": "C++ Copilot",
            "requirements": {
                "data_types": ["HTML", "JSON", "PDF", "MD", "TXT", "ADOC", "code files (C++, header files)"],
                "retrieval_methods": "Hybrid (semantic + keyword/BM25, vector + graph)",
                "customization": "System prompts, retrieval parameters, model selection",
                "modern_rag": "Graph RAG, multi-hop reasoning, structured data handling",
                "data_scale": "~50GB (approximately 3 million documents)",
                "accuracy": "High accuracy for QA and summarization tasks"
            }
        }

        # Initialize the research agent
        self.checkpointer = InMemorySaver()
        self.research_agent = deep_researcher_builder.compile(checkpointer=self.checkpointer)

    def create_query_for_rag(self) -> str:
        """
        Create a research query for RAGaaS (Retrieval-Augmented Generation as a Service) platforms.

        Returns:
            Research query string for RAGaaS platform analysis
        """
        query = """Research top 6-8 RAGaaS (Retrieval-Augmented Generation as a Service) platforms suitable for a C++ Copilot use case.

Use Case: C++ Copilot - AI assistant for C++ development with code understanding, documentation, and technical Q&A capabilities.

Requirements:
- Data types: HTML, JSON, PDF, MD, TXT, ADOC, code files (C++, header files, etc.)
- Hybrid retrieval methods (semantic + keyword/BM25, vector + graph, etc.)
- Sophisticated system prompts with customization
- High accuracy for QA and summarization tasks
- User-customizable configurations (retrieval strategies, prompts, models)
- Modern RAG capabilities: graph RAG, multi-hop reasoning, structured data handling
- Data scale: ~50GB (approximately 3 million documents)
- Support for code-specific understanding and technical documentation

Provide a CONCISE, OBJECTIVE, and QUANTITATIVE report (max 1000 words) with SHORT paragraphs (2-4 sentences max).

**CRITICAL: CONCISENESS REQUIREMENTS:**
- Keep paragraphs SHORT (2-4 sentences maximum)
- Use bullet points and tables liberally for better readability
- Avoid long, dense paragraphs - break information into digestible chunks
- Prioritize key insights over exhaustive detail
- Remove redundant information
- Do not use bold formatting and quotation marks within a sentence.

1. **Top 6-8 RAGaaS Platforms** (use bullet points, keep each platform to 4-5 bullet points):
   For each platform, provide:
   - Platform name, company, website URL, pricing model (if available)
   - Key strengths for C++ Copilot use case
   - Main limitations or concerns
   - Supported data formats and file types
   - Retrieval methods (semantic, keyword, hybrid, graph RAG, etc.)
   - Customization capabilities (system prompts, retrieval parameters, model selection)
   - Data scale limits and pricing tiers (if available)

2. **Quantitative Feature Comparison Table**:
   Create a comprehensive comparison table with these metrics for each platform:
   - **Data Format Support**: HTML, JSON, PDF, MD, TXT, ADOC, code files (C++/header files)
   - **Retrieval Methods**: Semantic search, keyword/BM25, hybrid, graph RAG, multi-hop reasoning
   - **Customization**: System prompt editing, retrieval parameter tuning, model selection, custom embeddings
   - **Scale Capacity**: Max documents, max data size, pricing per document/GB
   - **Accuracy Metrics**: QA accuracy (if available), summarization quality, code understanding capabilities
   - **Modern RAG Features**: Graph RAG, structured data extraction, code-specific parsing, multi-modal support
   - **API & Integration**: REST API, SDK availability, webhook support, real-time updates
   - **Performance**: Query latency, indexing speed, concurrent request limits

   Use actual numbers when available, or "—" if not available. Be objective and data-driven.

3. **Summary Comparison Table**:
   Create a concise table with columns:
   | Platform | Best For | Main Advantage | Main Limitation | Graph RAG Support | Customization Level | Pricing Model |

4. **C++ Copilot Suitability Analysis** (2-3 short paragraphs max):
   - Which platforms are best suited for code understanding and C++ documentation
   - Support for code-specific features (syntax highlighting, code parsing, AST understanding)
   - Integration capabilities with development environments
   - Handling of technical documentation and code examples

5. **Modern RAG Capabilities** (2-3 short paragraphs max):
   - Graph RAG implementation and maturity
   - Multi-hop reasoning and complex query handling
   - Structured data extraction from code and documentation
   - Support for hierarchical and relational data (code dependencies, documentation structure)

6. **Scale and Performance** (2-3 short paragraphs max):
   - How each platform handles ~50GB / 3M documents
   - Indexing performance and update mechanisms
   - Query latency and throughput for large-scale deployments
   - Cost implications for the specified data volume

7. **Recent Trends (2023-2025)** - Brief summary (2-3 short paragraphs max):
   - Platform evolution and new feature releases
   - Industry adoption trends
   - Emerging capabilities in graph RAG and code understanding

8. **Conclusion** - Objective assessment (2-3 short paragraphs max):
   - Top 2-3 recommended platforms for C++ Copilot use case
   - Key trade-offs based on requirements (data types, scale, customization, modern RAG features)
   - Implementation considerations and migration paths

Focus on objective, quantitative data. Include specific numbers, dates, pricing information, and technical metrics when available. Prioritize platforms with strong graph RAG capabilities, code understanding features, and high customization options. Keep the report concise and scannable."""

        return query

    async def research_ragaas_platforms(self) -> Dict[str, Any]:
        """
        Research RAGaaS platforms using the deep research agent.

        Returns:
            Dictionary containing research results
        """
        query = self.create_query_for_rag()

        # Configure thread for RAGaaS research
        thread_config = {
            "configurable": {
                "thread_id": "ragaas_platforms_analysis",
                "recursion_limit": 15
            }
        }

        try:
            # Reset tracker
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

            # Build result dictionary
            result_dict = {
                "research_report": final_report,
                "research_completed_at": datetime.now().isoformat(),
                "usage_stats": usage_stats,
                "status": "completed"
            }

            # Include red team evaluation if it was performed
            if "red_team_evaluation" in result:
                result_dict["red_team_evaluation"] = result["red_team_evaluation"]
                result_dict["red_team_report"] = result.get("red_team_report", "")

            return result_dict

        except Exception as e:
            print(f"Error researching RAGaaS platforms: {str(e)}")
            return {
                "research_report": f"Error during research: {str(e)}",
                "research_completed_at": datetime.now().isoformat(),
                "status": "error",
                "error": str(e)
            }

    def save_results(self, result: Dict[str, Any]):
        """
        Save research results to markdown file.

        Args:
            result: Research result dictionary
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_path / f"ragaas_analysis_{timestamp}.md"

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# RAGaaS Platform Analysis for C++ Copilot\n\n")
            f.write(f"**Generated:** {result.get('research_completed_at', 'N/A')}\n\n")

            f.write("## Use Case Requirements\n\n")
            f.write("- **Use Case:** C++ Copilot - AI assistant for C++ development\n")
            f.write("- **Data Types:** HTML, JSON, PDF, MD, TXT, ADOC, code files (C++, header files)\n")
            f.write("- **Retrieval Methods:** Hybrid (semantic + keyword/BM25, vector + graph)\n")
            f.write("- **Customization:** System prompts, retrieval parameters, model selection\n")
            f.write("- **Modern RAG:** Graph RAG, multi-hop reasoning, structured data handling\n")
            f.write("- **Data Scale:** ~50GB (approximately 3 million documents)\n")
            f.write("- **Accuracy:** High accuracy for QA and summarization tasks\n\n")

            f.write("---\n\n")

            # Write usage statistics
            if "usage_stats" in result:
                stats = result["usage_stats"]
                openai_stats = stats.get("openai", {})
                tavily_stats = stats.get("tavily", {})
                f.write("## Usage Statistics\n\n")
                f.write(f"- **Input tokens:** {openai_stats.get('prompt_tokens', 0):,}\n")
                f.write(f"- **Output tokens:** {openai_stats.get('completion_tokens', 0):,}\n")
                f.write(f"- **Total tokens:** {openai_stats.get('total_tokens', 0):,}\n")
                f.write(f"- **Tavily calls:** {tavily_stats.get('api_calls', 0):,}\n\n")
                f.write("---\n\n")

            # Write red team evaluation if available
            if "red_team_evaluation" in result:
                eval_summary = result["red_team_evaluation"]
                overall_score = eval_summary.get("overall_score", 0.0)
                f.write("## Red Team Evaluation Summary\n\n")
                f.write(f"- **Overall Objectivity Score:** {overall_score:.1%}\n")
                f.write(f"- **Bias Issues (Missing Counter-Evidence):** {eval_summary.get('bias_metrics', {}).get('missing_counter_evidence_count', 0)}\n")
                f.write(f"- **Unsupported Claims:** {eval_summary.get('issues', {}).get('unsupported_claims_count', 0)}\n")
                f.write(f"- **Source Credibility:** {eval_summary.get('source_quality', {}).get('credibility_score', 0.0):.1%}\n\n")

                if "red_team_report" in result and result["red_team_report"]:
                    f.write("### Detailed Red Team Report\n\n")
                    f.write(result["red_team_report"])
                    f.write("\n\n")
                f.write("---\n\n")

            # Write research report
            if "research_report" in result:
                f.write("## Research Report\n\n")
                f.write(result["research_report"])
            else:
                f.write("No research report available.\n")

        print(f"\n✅ RAGaaS analysis report saved to: {output_file}")
        return output_file

    async def run(self):
        """
        Main method to run the RAGaaS platform research.
        """
        print("="*80)
        print("RAGaaS Platform Research for C++ Copilot")
        print("="*80)
        print(f"Output directory: {self.output_path}")
        print("="*80)
        print("\nStarting research...\n")

        # Run research
        result = await self.research_ragaas_platforms()

        # Save results
        output_file = self.save_results(result)

        # Print summary
        if result.get("status") == "completed":
            print("\n" + "="*80)
            print("Research Complete")
            print("="*80)

            if "usage_stats" in result:
                stats = result["usage_stats"]
                openai_stats = stats.get("openai", {})
                tavily_stats = stats.get("tavily", {})
                print(f"Total tokens: {openai_stats.get('total_tokens', 0):,}")
                print(f"  - Input: {openai_stats.get('prompt_tokens', 0):,}")
                print(f"  - Output: {openai_stats.get('completion_tokens', 0):,}")
                print(f"Tavily calls: {tavily_stats.get('api_calls', 0):,}")

            if "red_team_evaluation" in result:
                overall_score = result["red_team_evaluation"].get("overall_score", 0.0)
                print(f"Objectivity Score: {overall_score:.1%}")

            print(f"\nReport saved to: {output_file}")
        else:
            print(f"\n❌ Research failed: {result.get('error', 'Unknown error')}")


async def main():
    """
    Main entry point for running the RAGaaS platform research.
    """
    investigator = investigate_ragaas()
    await investigator.run()


if __name__ == "__main__":
    asyncio.run(main())

