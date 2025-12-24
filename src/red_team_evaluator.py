"""
Red Team Evaluation Module

This module provides adversarial evaluation of research results to identify:
- Bias and one-sided arguments
- Missing counter-evidence
- Source quality issues
- Unsupported claims
- Objectivity metrics
"""

import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json

from langchain_core.messages import HumanMessage, SystemMessage
from deep_research.model_config import get_model
from deep_research.usage_tracker import get_tracker


def extract_sources_from_report(report: str) -> List[str]:
    """
    Extract source URLs from a research report.

    Looks for sources in the "Sources" section at the end of the report,
    typically in format: [1] Title: URL

    Args:
        report: The research report text

    Returns:
        List of source URLs found in the report
    """
    sources = []

    # Look for Sources section
    sources_match = re.search(r'###\s*Sources?\s*\n(.*?)(?=\n###|\Z)', report, re.IGNORECASE | re.DOTALL)
    if sources_match:
        sources_text = sources_match.group(1)

        # Extract URLs from citation format: [N] Title: URL
        url_pattern = r'https?://[^\s\)]+'
        urls = re.findall(url_pattern, sources_text)
        sources.extend(urls)

    # Also look for inline URLs in the report
    inline_urls = re.findall(r'https?://[^\s\)]+', report)
    sources.extend(inline_urls)

    # Remove duplicates while preserving order
    seen = set()
    unique_sources = []
    for url in sources:
        # Clean URL (remove trailing punctuation)
        clean_url = url.rstrip('.,;:!?)')
        if clean_url not in seen:
            seen.add(clean_url)
            unique_sources.append(clean_url)

    return unique_sources


@dataclass
class BiasMetrics:
    """Metrics for detecting bias in research reports."""
    one_sided_score: float = 0.0  # 0-1, higher = more one-sided
    missing_counter_evidence: List[str] = field(default_factory=list)
    confirmation_bias_indicators: List[str] = field(default_factory=list)
    source_diversity_score: float = 0.0  # 0-1, higher = more diverse sources
    quantitative_ratio: float = 0.0  # Ratio of quantitative to qualitative claims


@dataclass
class SourceQualityMetrics:
    """Metrics for assessing source quality."""
    total_sources: int = 0
    primary_sources: int = 0  # Official, authoritative sources
    secondary_sources: int = 0  # News, blogs, aggregators
    academic_sources: int = 0
    source_credibility_score: float = 0.0  # 0-1, higher = more credible
    missing_citations: List[str] = field(default_factory=list)


@dataclass
class ObjectivityScore:
    """Overall objectivity assessment."""
    overall_score: float = 0.0  # 0-1, higher = more objective
    bias_metrics: BiasMetrics = field(default_factory=BiasMetrics)
    source_quality: SourceQualityMetrics = field(default_factory=SourceQualityMetrics)
    unsupported_claims: List[str] = field(default_factory=list)
    counter_evidence_gaps: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


class RedTeamEvaluator:
    """
    Red team evaluator for assessing research objectivity and quality.

    This evaluator acts as an adversarial reviewer, identifying:
    - Potential biases and one-sided arguments
    - Missing counter-evidence or alternative perspectives
    - Source quality issues
    - Unsupported claims
    - Areas needing more quantitative data
    """

    def __init__(self):
        """Initialize the red team evaluator."""
        self.evaluator_model = get_model("google/gemini-2.5-flash")
        self.tracker = get_tracker()

    async def evaluate_report(
        self,
        report: str,
        research_query: str,
        sources: Optional[List[str]] = None
    ) -> ObjectivityScore:
        """
        Perform comprehensive red team evaluation of a research report.

        Args:
            report: The research report to evaluate
            research_query: The original research query/question
            sources: Optional list of source URLs used in the report

        Returns:
            ObjectivityScore with detailed metrics and recommendations
        """
        # Run parallel evaluations
        bias_analysis = await self._analyze_bias(report, research_query)
        source_analysis = await self._analyze_sources(report, sources or [])
        claim_verification = await self._verify_claims(report, research_query)

        # Calculate overall objectivity score
        overall_score = self._calculate_objectivity_score(
            bias_analysis, source_analysis, claim_verification
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(
            bias_analysis, source_analysis, claim_verification
        )

        return ObjectivityScore(
            overall_score=overall_score,
            bias_metrics=bias_analysis,
            source_quality=source_analysis,
            unsupported_claims=claim_verification.get("unsupported", []),
            counter_evidence_gaps=claim_verification.get("missing_counter_evidence", []),
            recommendations=recommendations
        )

    async def _analyze_bias(self, report: str, query: str) -> BiasMetrics:
        """Analyze the report for bias and one-sided arguments."""
        prompt = f"""You are a red team evaluator analyzing a research report created by gpt 5.1 and tavily api for bias and objectivity issues.

Research Query: {query}

Research Report:
{report}

Analyze this report for:
1. **One-sided arguments**: Are alternative perspectives or counter-arguments missing?
2. **Confirmation bias**: Does the report only present evidence supporting a particular conclusion?
3. **Source diversity**: Are sources from diverse perspectives, or mostly from one viewpoint?
4. **Quantitative vs qualitative**: What ratio of claims are quantitative (with numbers/data) vs qualitative (opinions/descriptions)?

Provide your analysis in JSON format:
{{
    "one_sided_score": <0.0-1.0, where 1.0 is completely one-sided>,
    "missing_counter_evidence": [<list of specific counter-arguments or alternative perspectives that should be included>],
    "confirmation_bias_indicators": [<list of specific examples where only supporting evidence is presented>],
    "source_diversity_score": <0.0-1.0, where 1.0 is highly diverse sources>,
    "quantitative_ratio": <0.0-1.0, ratio of quantitative to total claims>
}}"""

        response = self.evaluator_model.invoke([
            SystemMessage(content="You are an expert research evaluator specializing in bias detection and objectivity assessment."),
            HumanMessage(content=prompt)
        ])

        self.tracker.track_openai_response(response)

        # Parse response (simplified - in production, use structured output)
        # For now, return a structured analysis
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if json_match:
                analysis = json.loads(json_match.group())
                return BiasMetrics(
                    one_sided_score=analysis.get("one_sided_score", 0.5),
                    missing_counter_evidence=analysis.get("missing_counter_evidence", []),
                    confirmation_bias_indicators=analysis.get("confirmation_bias_indicators", []),
                    source_diversity_score=analysis.get("source_diversity_score", 0.5),
                    quantitative_ratio=analysis.get("quantitative_ratio", 0.5)
                )
        except:
            pass

        # Fallback: return default metrics
        return BiasMetrics()

    async def _analyze_sources(self, report: str, sources: List[str]) -> SourceQualityMetrics:
        """Analyze source quality and credibility."""
        sources_text = "\n".join([f"- {s}" for s in sources]) if sources else "No sources provided"

        prompt = f"""Analyze the sources used in this research report for quality and credibility.

Research Report:
{report}

Sources Used:
{sources_text}

Evaluate:
1. **Source types**: Count primary sources (official, authoritative) vs secondary (news, blogs)
2. **Source credibility**: Overall credibility score based on source types
3. **Missing citations**: Are there claims in the report that lack source citations?

Provide your analysis in JSON format:
{{
    "total_sources": <number>,
    "primary_sources": <number of official/authoritative sources>,
    "secondary_sources": <number of news/blogs/aggregators>,
    "academic_sources": <number of academic papers/journals>,
    "source_credibility_score": <0.0-1.0, higher = more credible>,
    "missing_citations": [<list of claims that should have citations but don't>]
}}"""

        response = self.evaluator_model.invoke([
            SystemMessage(content="You are an expert in source evaluation and citation analysis."),
            HumanMessage(content=prompt)
        ])

        self.tracker.track_openai_response(response)

        # Parse response
        import json
        import re

        try:
            json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if json_match:
                analysis = json.loads(json_match.group())
                # Ensure missing_citations items are strings
                missing_citations = analysis.get("missing_citations", [])
                if missing_citations:
                    missing_citations = [str(item) if not isinstance(item, str) else item for item in missing_citations]

                return SourceQualityMetrics(
                    total_sources=int(analysis.get("total_sources", len(sources))),
                    primary_sources=int(analysis.get("primary_sources", 0)),
                    secondary_sources=int(analysis.get("secondary_sources", 0)),
                    academic_sources=int(analysis.get("academic_sources", 0)),
                    source_credibility_score=float(analysis.get("source_credibility_score", 0.5)),
                    missing_citations=missing_citations
                )
        except Exception as e:
            print(f"[RED TEAM] Warning: Failed to parse source quality JSON: {e}")
            pass

        # Fallback: basic analysis
        return SourceQualityMetrics(
            total_sources=len(sources),
            source_credibility_score=0.5
        )

    async def _verify_claims(self, report: str, query: str) -> Dict[str, List[str]]:
        """Verify claims and identify unsupported assertions."""
        prompt = f"""You are a fact-checker reviewing a research report. Identify:

1. **Unsupported claims**: Assertions that lack evidence or citations
2. **Missing counter-evidence**: Areas where alternative viewpoints or contradictory evidence should be presented

Research Query: {query}

Research Report:
{report}

Provide your analysis in JSON format:
{{
    "unsupported": [<list of specific unsupported claims with their locations in the report>],
    "missing_counter_evidence": [<list of claims that need counter-evidence or alternative perspectives>]
}}"""

        response = self.evaluator_model.invoke([
            SystemMessage(content="You are an expert fact-checker and evidence evaluator."),
            HumanMessage(content=prompt)
        ])

        self.tracker.track_openai_response(response)

        # Parse response
        import json
        import re

        try:
            json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                # Ensure all items in lists are strings
                result = {
                    "unsupported": [],
                    "missing_counter_evidence": []
                }
                if "unsupported" in parsed:
                    for item in parsed["unsupported"]:
                        if isinstance(item, str):
                            result["unsupported"].append(item)
                        elif isinstance(item, dict):
                            # Extract text from dict if it's structured
                            result["unsupported"].append(str(item.get("claim", item.get("text", str(item)))))
                        else:
                            result["unsupported"].append(str(item))

                if "missing_counter_evidence" in parsed:
                    for item in parsed["missing_counter_evidence"]:
                        if isinstance(item, str):
                            result["missing_counter_evidence"].append(item)
                        elif isinstance(item, dict):
                            # Extract text from dict if it's structured
                            result["missing_counter_evidence"].append(str(item.get("gap", item.get("text", str(item)))))
                        else:
                            result["missing_counter_evidence"].append(str(item))

                return result
        except Exception as e:
            print(f"[RED TEAM] Warning: Failed to parse claim verification JSON: {e}")
            pass

        return {"unsupported": [], "missing_counter_evidence": []}

    def _calculate_objectivity_score(
        self,
        bias: BiasMetrics,
        sources: SourceQualityMetrics,
        claims: Dict[str, List[str]]
    ) -> float:
        """Calculate overall objectivity score (0-1, higher = more objective)."""
        # Weighted components
        bias_component = (1.0 - bias.one_sided_score) * 0.4  # Lower one-sided = better
        source_component = sources.source_credibility_score * 0.3
        diversity_component = bias.source_diversity_score * 0.2
        quantitative_component = bias.quantitative_ratio * 0.1

        # Penalize unsupported claims
        unsupported_penalty = min(len(claims.get("unsupported", [])) * 0.05, 0.2)

        score = (bias_component + source_component + diversity_component +
                quantitative_component) * (1.0 - unsupported_penalty)

        return max(0.0, min(1.0, score))

    def _generate_recommendations(
        self,
        bias: BiasMetrics,
        sources: SourceQualityMetrics,
        claims: Dict[str, List[str]]
    ) -> List[str]:
        """Generate actionable recommendations for improving objectivity."""
        recommendations = []

        if bias.one_sided_score > 0.6:
            recommendations.append(
                f"⚠️ High one-sided score ({bias.one_sided_score:.2f}): "
                "Consider adding alternative perspectives or counter-arguments."
            )

        if bias.source_diversity_score < 0.5:
            recommendations.append(
                "⚠️ Low source diversity: Include sources from diverse perspectives "
                "and viewpoints."
            )

        if bias.quantitative_ratio < 0.3:
            recommendations.append(
                "⚠️ Low quantitative data: Increase use of specific numbers, "
                "statistics, and measurable metrics."
            )

        if sources.source_credibility_score < 0.6:
            recommendations.append(
                "⚠️ Source quality concerns: Prioritize primary sources, "
                "official documents, and authoritative publications."
            )

        unsupported_count = len(claims.get("unsupported", []))
        if unsupported_count > 0:
            recommendations.append(
                f"⚠️ {unsupported_count} unsupported claim(s) identified: "
                "Add citations or evidence for these assertions."
            )

        missing_counter = len(claims.get("missing_counter_evidence", []))
        if missing_counter > 0:
            recommendations.append(
                f"⚠️ {missing_counter} area(s) need counter-evidence: "
                "Present alternative viewpoints or contradictory evidence."
            )

        if not recommendations:
            recommendations.append("✅ Report shows good objectivity and balanced perspective.")

        return recommendations

    def format_evaluation_report(self, score: ObjectivityScore) -> str:
        """Format the evaluation results as a readable report."""
        report_lines = [
            "# Red Team Evaluation Report",
            f"**Generated:** {datetime.now().isoformat()}",
            "",
            "## Overall Objectivity Score",
            f"**Score: {score.overall_score:.2%}** ({'Excellent' if score.overall_score > 0.8 else 'Good' if score.overall_score > 0.6 else 'Needs Improvement'})",
            "",
            "## Bias Analysis",
            f"- **One-Sided Score:** {score.bias_metrics.one_sided_score:.2%} "
              f"({'High' if score.bias_metrics.one_sided_score > 0.6 else 'Moderate' if score.bias_metrics.one_sided_score > 0.4 else 'Low'})",
            f"- **Source Diversity:** {score.bias_metrics.source_diversity_score:.2%}",
            f"- **Quantitative Ratio:** {score.bias_metrics.quantitative_ratio:.2%}",
            ""
        ]

        if score.bias_metrics.missing_counter_evidence:
            report_lines.extend([
                "### Missing Counter-Evidence",
                *[f"- {item}" for item in score.bias_metrics.missing_counter_evidence],
                ""
            ])

        if score.bias_metrics.confirmation_bias_indicators:
            report_lines.extend([
                "### Confirmation Bias Indicators",
                *[f"- {item}" for item in score.bias_metrics.confirmation_bias_indicators],
                ""
            ])

        report_lines.extend([
            "## Source Quality Analysis",
            f"- **Total Sources:** {score.source_quality.total_sources}",
            f"- **Primary Sources:** {score.source_quality.primary_sources}",
            f"- **Secondary Sources:** {score.source_quality.secondary_sources}",
            f"- **Academic Sources:** {score.source_quality.academic_sources}",
            f"- **Credibility Score:** {score.source_quality.source_credibility_score:.2%}",
            ""
        ])

        if score.source_quality.missing_citations:
            report_lines.extend([
                "### Missing Citations",
                *[f"- {item}" for item in score.source_quality.missing_citations],
                ""
            ])

        if score.unsupported_claims:
            report_lines.extend([
                "## Unsupported Claims",
                *[f"- {item}" for item in score.unsupported_claims],
                ""
            ])

        if score.counter_evidence_gaps:
            report_lines.extend([
                "## Counter-Evidence Gaps",
                *[f"- {item}" for item in score.counter_evidence_gaps],
                ""
            ])

        report_lines.extend([
            "## Recommendations",
            *[f"{rec}" for rec in score.recommendations],
            ""
        ])

        return "\n".join(report_lines)

    def get_evaluation_summary(self, score: ObjectivityScore) -> Dict[str, Any]:
        """Get a summary dictionary of the evaluation results."""
        return {
            "overall_score": score.overall_score,
            "bias_metrics": {
                "one_sided_score": score.bias_metrics.one_sided_score,
                "source_diversity_score": score.bias_metrics.source_diversity_score,
                "quantitative_ratio": score.bias_metrics.quantitative_ratio,
                "missing_counter_evidence_count": len(score.bias_metrics.missing_counter_evidence),
                "confirmation_bias_indicators_count": len(score.bias_metrics.confirmation_bias_indicators)
            },
            "source_quality": {
                "total_sources": score.source_quality.total_sources,
                "primary_sources": score.source_quality.primary_sources,
                "secondary_sources": score.source_quality.secondary_sources,
                "academic_sources": score.source_quality.academic_sources,
                "credibility_score": score.source_quality.source_credibility_score,
                "missing_citations_count": len(score.source_quality.missing_citations)
            },
            "issues": {
                "unsupported_claims_count": len(score.unsupported_claims),
                "counter_evidence_gaps_count": len(score.counter_evidence_gaps)
            },
            "recommendations_count": len(score.recommendations)
        }

    def generate_refinement_feedback(
        self,
        score: ObjectivityScore,
        report: str,
        research_query: str
    ) -> Dict[str, Any]:
        """
        Generate actionable feedback for report refinement.

        Args:
            score: ObjectivityScore from evaluation
            report: Current report text
            research_query: Original research query

        Returns:
            Dictionary with:
            - priority_issues: List of critical issues to address
            - specific_suggestions: Concrete improvement suggestions
            - missing_elements: What should be added
            - refinement_prompt: Formatted prompt for the writer
        """
        feedback = {
            "priority_issues": [],
            "specific_suggestions": [],
            "missing_elements": [],
            "refinement_prompt": ""
        }

        # High priority: Low objectivity score
        if score.overall_score < 0.6:
            feedback["priority_issues"].append({
                "severity": "critical",
                "issue": "Overall objectivity score is below acceptable threshold",
                "current_score": score.overall_score,
                "target_score": 0.75
            })
        elif score.overall_score < 0.75:
            feedback["priority_issues"].append({
                "severity": "high",
                "issue": "Overall objectivity score needs improvement",
                "current_score": score.overall_score,
                "target_score": 0.75
            })

        # High priority: Unsupported claims
        if len(score.unsupported_claims) > 0:
            feedback["priority_issues"].append({
                "severity": "high",
                "issue": f"{len(score.unsupported_claims)} unsupported claims found",
                "claims": score.unsupported_claims[:5]  # Top 5
            })

        # Medium priority: Missing counter-evidence
        if len(score.counter_evidence_gaps) > 0:
            feedback["priority_issues"].append({
                "severity": "medium",
                "issue": "Missing counter-evidence or alternative perspectives",
                "gaps": score.counter_evidence_gaps[:3]
            })

        # Missing citations
        if len(score.source_quality.missing_citations) > 0:
            feedback["priority_issues"].append({
                "severity": "medium",
                "issue": f"{len(score.source_quality.missing_citations)} claims missing citations",
                "citations_needed": score.source_quality.missing_citations[:5]
            })

        # Generate specific suggestions
        if score.bias_metrics.one_sided_score > 0.6:
            feedback["specific_suggestions"].append(
                "Add alternative viewpoints and counter-arguments to balance the report"
            )
            if score.bias_metrics.missing_counter_evidence:
                feedback["missing_elements"].extend(
                    score.bias_metrics.missing_counter_evidence[:3]
                )

        if score.bias_metrics.quantitative_ratio < 0.3:
            feedback["specific_suggestions"].append(
                "Include more quantitative data: specific numbers, statistics, dates, and metrics"
            )
            feedback["missing_elements"].append("Quantitative metrics and statistics")

        if score.source_quality.source_credibility_score < 0.6:
            feedback["specific_suggestions"].append(
                "Prioritize primary sources and authoritative publications over secondary sources"
            )
            feedback["missing_elements"].append("High-credibility primary sources")

        if score.bias_metrics.source_diversity_score < 0.5:
            feedback["specific_suggestions"].append(
                "Include sources from diverse perspectives and viewpoints"
            )
            feedback["missing_elements"].append("Diverse source perspectives")

        # Build refinement prompt
        feedback["refinement_prompt"] = self._build_refinement_prompt(
            score, feedback, report, research_query
        )

        return feedback

    def _build_refinement_prompt(
        self,
        score: ObjectivityScore,
        feedback: Dict[str, Any],
        report: str,
        research_query: str
    ) -> str:
        """Build a detailed prompt for report refinement."""
        prompt = f"""You are refining a research report based on red team evaluation feedback.

Original Research Query: {research_query}

Current Report Objectivity Score: {score.overall_score:.1%} (Target: ≥75%)

CRITICAL ISSUES TO ADDRESS:
"""
        for issue in feedback["priority_issues"]:
            prompt += f"- [{issue['severity'].upper()}] {issue['issue']}\n"
            if "claims" in issue and issue["claims"]:
                # Ensure claims are strings before joining
                claims_list = issue['claims'][:3]
                claims_str = ', '.join(str(c) if not isinstance(c, str) else c for c in claims_list)
                prompt += f"  Specific claims needing support: {claims_str}\n"
            if "gaps" in issue and issue["gaps"]:
                # Ensure gaps are strings before joining
                gaps_list = issue['gaps'][:2]
                gaps_str = ', '.join(str(g) if not isinstance(g, str) else g for g in gaps_list)
                prompt += f"  Missing perspectives: {gaps_str}\n"

        prompt += f"\nSPECIFIC IMPROVEMENTS NEEDED:\n"
        for suggestion in feedback["specific_suggestions"]:
            prompt += f"- {suggestion}\n"

        if feedback["missing_elements"]:
            prompt += f"\nMISSING ELEMENTS TO ADD:\n"
            for element in set(feedback["missing_elements"]):  # Remove duplicates
                prompt += f"- {element}\n"

        prompt += f"\nRED TEAM RECOMMENDATIONS:\n"
        for rec in score.recommendations:
            prompt += f"- {rec}\n"

        prompt += f"""
CURRENT REPORT:
{report}

TASK:
Refine the above report by addressing all the issues identified by the red team evaluation.
Maintain all accurate information while:
1. Adding missing citations for unsupported claims
2. Including counter-evidence and alternative perspectives where identified
3. Increasing quantitative data and specific metrics
4. Improving source diversity and credibility
5. Ensuring balanced, objective presentation

IMPORTANT:
- Preserve all valid content and accurate information
- Do not remove existing good content
- Focus on adding missing elements and improving objectivity
- Maintain the report structure and organization

Return the refined report that addresses these concerns while preserving all valid content.
"""
        return prompt

