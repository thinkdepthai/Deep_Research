
"""
Full Multi-Agent Research System

This module integrates all components of the research system:
- User clarification and scoping
- Research brief generation
- Multi-agent research coordination
- Final report generation

The system orchestrates the complete research workflow from initial user
input through final report delivery.
"""
import os

from typing_extensions import Literal
from dotenv import load_dotenv
from pathlib import Path

from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END

from deep_research.utils import get_today_str
from deep_research.prompts import final_report_generation_with_helpfulness_insightfulness_hit_citation_prompt
from deep_research.state_scope import AgentState, AgentInputState
from deep_research.research_agent_scope import clarify_with_user, write_research_brief, write_draft_report
from deep_research.multi_agent_supervisor import supervisor_agent
from deep_research.usage_tracker import get_tracker
from deep_research.red_team_evaluator import RedTeamEvaluator, extract_sources_from_report
from deep_research.model_config import get_model


# ===== Config =====
load_dotenv()

writer_model = get_model(max_tokens=32000)

# ===== FINAL REPORT GENERATION =====

async def final_report_generation(state: AgentState):
    """
    Generate initial final report.

    This is the "blue team" initial report generation.
    Red team evaluation and refinement will happen in subsequent nodes.
    """
    print("\n" + "="*80)
    print("[BLUE TEAM] Generating initial final report...")
    print("="*80)

    notes = state.get("notes", [])

    findings = "\n".join(notes)

    final_report_prompt = final_report_generation_with_helpfulness_insightfulness_hit_citation_prompt.format(
        research_brief=state.get("research_brief", ""),
        findings=findings,
        date=get_today_str(),
        draft_report=state.get("draft_report", ""),
        user_request=state.get("user_request", "")
    )

    final_report = await writer_model.ainvoke([HumanMessage(content=final_report_prompt)])

    # Track token usage
    tracker = get_tracker()
    tracker.track_openai_response(final_report)

    report_length = len(final_report.content)
    print(f"[BLUE TEAM] ✓ Initial report generated ({report_length:,} characters)")
    print(f"[BLUE TEAM] → Proceeding to red team evaluation...\n")

    return {
        "final_report": final_report.content,
        "messages": ["Initial final report generated"],
    }

# ===== RED TEAM EVALUATION NODE =====

async def red_team_evaluation_node(state: AgentState):
    """
    Red team evaluation node that provides feedback for refinement.

    This evaluates the current report and generates actionable feedback
    for the blue team to improve the report.
    """
    if os.getenv("ENABLE_RED_TEAM_EVAL", "false").lower() != "true":
        print("[RED TEAM] Evaluation disabled (ENABLE_RED_TEAM_EVAL=false)")
        return {}

    current_report = state.get("final_report", "")
    if not current_report:
        print("[RED TEAM] ⚠️  No report available for evaluation")
        return {}

    current_iteration = state.get("red_team_iteration_count", 0) + 1
    print("\n" + "="*80)
    print(f"[RED TEAM] Evaluation Cycle #{current_iteration}")
    print("="*80)
    print("[RED TEAM] Analyzing report for objectivity, bias, and quality issues...")

    try:
        evaluator = RedTeamEvaluator()
        sources = extract_sources_from_report(current_report)
        research_query = state.get("research_brief", state.get("user_request", ""))

        print(f"[RED TEAM] - Extracted {len(sources)} sources from report")
        print("[RED TEAM] - Running bias analysis...")

        # Evaluate the report
        red_team_evaluation = await evaluator.evaluate_report(
            current_report,
            research_query,
            sources
        )

        overall_score = red_team_evaluation.overall_score
        print(f"[RED TEAM] ✓ Evaluation complete")
        print(f"[RED TEAM] - Overall Objectivity Score: {overall_score:.1%}")

        # Generate actionable feedback
        print("[RED TEAM] - Generating actionable feedback...")
        feedback = evaluator.generate_refinement_feedback(
            red_team_evaluation,
            current_report,
            research_query
        )

        priority_count = len(feedback["priority_issues"])
        suggestions_count = len(feedback["specific_suggestions"])

        print(f"[RED TEAM] ✓ Feedback generated:")
        print(f"[RED TEAM]   - Priority issues: {priority_count}")
        print(f"[RED TEAM]   - Specific suggestions: {suggestions_count}")

        if priority_count > 0:
            print("[RED TEAM] Priority issues identified:")
            for i, issue in enumerate(feedback["priority_issues"][:3], 1):
                print(f"[RED TEAM]   {i}. [{issue['severity'].upper()}] {issue['issue']}")

        # Create new history entry
        # Note: With Annotated[list[dict], operator.add], we return just the new item(s)
        # LangGraph will automatically merge with existing list
        new_history_entry = {
            "iteration": current_iteration,
            "score": red_team_evaluation.overall_score,
            "timestamp": get_today_str(),
            "priority_issues_count": priority_count,
            "suggestions_count": suggestions_count
        }

        # Store evaluation and feedback
        result = {
            "red_team_evaluation": evaluator.get_evaluation_summary(red_team_evaluation),
            "red_team_report": evaluator.format_evaluation_report(red_team_evaluation),
            "red_team_feedback": feedback,
            "red_team_iteration_count": current_iteration,
            "red_team_feedback_history": [new_history_entry]  # LangGraph will add this to existing list using operator.add
        }

        print(f"[RED TEAM] → Proceeding to quality check...\n")
        return result

    except Exception as e:
        # Don't fail the research if red team evaluation fails
        print(f"[RED TEAM] ⚠️  Warning: Red team evaluation failed: {e}")
        return {}

# ===== REPORT REFINEMENT NODE =====

async def refine_report_with_feedback(state: AgentState):
    """
    Refine the report based on red team evaluation feedback.

    This is the "blue team" response to red team criticism.
    Takes the feedback and improves the report accordingly.
    """
    current_iteration = state.get("refinement_iterations", 0) + 1
    print("\n" + "="*80)
    print(f"[BLUE TEAM] Refinement Cycle #{current_iteration}")
    print("="*80)

    current_report = state.get("final_report", "")
    feedback = state.get("red_team_feedback", {})
    refinement_prompt = feedback.get("refinement_prompt", "")

    if not refinement_prompt:
        print("[BLUE TEAM] ⚠️  No feedback available, skipping refinement")
        return {
            "final_report": current_report,
            "refinement_iterations": current_iteration
        }

    # Show what we're addressing
    priority_issues = feedback.get("priority_issues", [])
    suggestions = feedback.get("specific_suggestions", [])

    print(f"[BLUE TEAM] Addressing {len(priority_issues)} priority issues:")
    for i, issue in enumerate(priority_issues[:3], 1):
        print(f"[BLUE TEAM]   {i}. [{issue['severity'].upper()}] {issue['issue']}")

    if suggestions:
        print(f"[BLUE TEAM] Implementing {len(suggestions)} improvements:")
        for i, suggestion in enumerate(suggestions[:3], 1):
            print(f"[BLUE TEAM]   {i}. {suggestion}")

    print("[BLUE TEAM] Refining report based on red team feedback...")

    # Use writer model to refine based on feedback
    refined_report = await writer_model.ainvoke([
        HumanMessage(content=refinement_prompt)
    ])

    # Track token usage
    tracker = get_tracker()
    tracker.track_openai_response(refined_report)

    original_length = len(current_report)
    refined_length = len(refined_report.content)
    length_change = refined_length - original_length

    print(f"[BLUE TEAM] ✓ Report refined")
    print(f"[BLUE TEAM] - Original length: {original_length:,} characters")
    print(f"[BLUE TEAM] - Refined length: {refined_length:,} characters")
    print(f"[BLUE TEAM] - Change: {length_change:+,} characters")
    print(f"[BLUE TEAM] → Proceeding to red team re-evaluation...\n")

    return {
        "final_report": refined_report.content,
        "refinement_iterations": current_iteration,
        "messages": [f"Report refined based on red team feedback (iteration {current_iteration})"]
    }

# ===== ITERATION CONTROL =====

def should_refine_report(state: AgentState) -> Literal["refine_report", "finalize_report"]:
    """
    Decide whether to continue refining or finalize the report.

    Returns:
        "refine_report" if quality is below threshold and iterations remain
        "finalize_report" if quality is acceptable or max iterations reached
    """
    print("\n" + "-"*80)
    print("[DECISION] Quality Check & Iteration Control")
    print("-"*80)

    # Check if red team evaluation is enabled
    if os.getenv("ENABLE_RED_TEAM_EVAL", "false").lower() != "true":
        print("[DECISION] Red team evaluation disabled → Finalizing report")
        return "finalize_report"

    # Check if feedback loop is enabled
    if os.getenv("RED_TEAM_FEEDBACK_LOOP", "true").lower() != "true":
        print("[DECISION] Feedback loop disabled → Finalizing report")
        return "finalize_report"

    # Check if we have red team evaluation
    if "red_team_evaluation" not in state:
        print("[DECISION] No red team evaluation available → Finalizing report")
        return "finalize_report"

    eval_summary = state["red_team_evaluation"]
    overall_score = eval_summary.get("overall_score", 1.0)

    # Get thresholds from state or environment
    min_score = state.get("min_objectivity_score", float(os.getenv("MIN_OBJECTIVITY_SCORE", "0.75")))
    max_iterations = state.get("max_refinement_iterations", int(os.getenv("MAX_REFINEMENT_ITERATIONS", "3")))
    current_iterations = state.get("refinement_iterations", 0)

    print(f"[DECISION] Current objectivity score: {overall_score:.1%}")
    print(f"[DECISION] Minimum threshold: {min_score:.1%}")
    print(f"[DECISION] Current refinement iterations: {current_iterations}/{max_iterations}")

    # Finalize if quality is acceptable
    if overall_score >= min_score:
        print(f"[DECISION] ✓ Quality threshold met ({overall_score:.1%} ≥ {min_score:.1%})")
        print("[DECISION] → Finalizing report\n")
        return "finalize_report"

    # Finalize if we've reached max iterations
    if current_iterations >= max_iterations:
        print(f"[DECISION] ⚠️  Max iterations reached ({current_iterations}/{max_iterations})")
        print("[DECISION] → Finalizing report (quality may be below threshold)\n")
        return "finalize_report"

    # Otherwise, continue refining
    print(f"[DECISION] ⚠️  Quality below threshold ({overall_score:.1%} < {min_score:.1%})")
    print(f"[DECISION] → Continuing refinement (iteration {current_iterations + 1}/{max_iterations})\n")
    return "refine_report"

# ===== GRAPH CONSTRUCTION =====
# Build the overall workflow
deep_researcher_builder = StateGraph(AgentState, input_schema=AgentInputState)

# Add workflow nodes
deep_researcher_builder.add_node("clarify_with_user", clarify_with_user)
deep_researcher_builder.add_node("write_research_brief", write_research_brief)
deep_researcher_builder.add_node("write_draft_report", write_draft_report)
deep_researcher_builder.add_node("supervisor_subgraph", supervisor_agent)
deep_researcher_builder.add_node("final_report_generation", final_report_generation)
deep_researcher_builder.add_node("red_team_evaluation", red_team_evaluation_node)
deep_researcher_builder.add_node("refine_report", refine_report_with_feedback)

# Add workflow edges
deep_researcher_builder.add_edge(START, "clarify_with_user")
deep_researcher_builder.add_edge("write_research_brief", "write_draft_report")
deep_researcher_builder.add_edge("write_draft_report", "supervisor_subgraph")
deep_researcher_builder.add_edge("supervisor_subgraph", "final_report_generation")
deep_researcher_builder.add_edge("final_report_generation", "red_team_evaluation")

# Add conditional edge for feedback loop
deep_researcher_builder.add_conditional_edges(
    "red_team_evaluation",
    should_refine_report,
    {
        "refine_report": "refine_report",
        "finalize_report": END
    }
)

# Loop back from refinement to evaluation
deep_researcher_builder.add_edge("refine_report", "red_team_evaluation")

# Compile the full workflow
agent = deep_researcher_builder.compile()
