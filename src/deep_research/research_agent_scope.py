
"""User Clarification and Research Brief Generation.

This module implements the scoping phase of the research workflow, where we:
1. Assess if the user's request needs clarification
2. Generate a detailed research brief from the conversation

The workflow uses structured output to make deterministic decisions about
whether sufficient context exists to proceed with research.
"""

import os
from datetime import datetime
from typing_extensions import Literal

from langchain_core.messages import AIMessage, HumanMessage, get_buffer_string
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command

from deep_research import logging as dr_logging
from deep_research.llm_factory import get_chat_model
from deep_research.prompts import transform_messages_into_research_topic_human_msg_prompt, draft_report_generation_prompt, clarify_with_user_instructions
from deep_research.state_scope import AgentState, ResearchQuestion, AgentInputState, DraftReport, ClarifyWithUser

logger = dr_logging.get_logger(__name__)

# ===== UTILITY FUNCTIONS =====






def get_today_str() -> str:
    """Get current date in a human-readable format."""
    return datetime.now().strftime("%a %b %-d, %Y")

# ===== CONFIGURATION =====

DEFAULT_INTERACTIVE = os.getenv("DEEP_RESEARCH_INTERACTIVE", "false").lower() == "true"

# Initialize model
logger.debug("Initializing scope models: scope_primary and scope_creative")
model = get_chat_model("scope_primary")
creative_model = get_chat_model("scope_creative")

# ===== WORKFLOW NODES =====

def clarify_with_user(state: AgentState) -> Command[Literal["write_research_brief"]]:
    logger.debug("clarify_with_user called with %d messages", len(state.get("messages", [])))
    # Headless mode: skip interactive clarification and proceed.
    if not DEFAULT_INTERACTIVE:
        return Command(goto="write_research_brief")

    # Interactive path: run structured clarification and possibly ask a question.
    structured_output_model = model.with_structured_output(ClarifyWithUser)
    response = structured_output_model.invoke([
        HumanMessage(content=clarify_with_user_instructions.format(
            messages=get_buffer_string(messages=state.get("messages", [])), 
            date=get_today_str()
        ))
    ])

    if response.need_clarification:
        return Command(
            goto=END, 
            update={"messages": [AIMessage(content=response.question)]}
        )

    return Command(goto="write_research_brief")

def write_research_brief(state: AgentState) -> Command[Literal["write_draft_report"]]:
    """
    Transform the conversation history into a comprehensive research brief.

    Uses structured output to ensure the brief follows the required format
    and contains all necessary details for effective research.
    """
    logger.debug(
        "write_research_brief invoked with %d messages", len(state.get("messages", []))
    )
    # Set up structured output model
    structured_output_model = model.with_structured_output(ResearchQuestion)

    # Generate research brief from conversation history
    prompt = transform_messages_into_research_topic_human_msg_prompt.format(
        messages=get_buffer_string(state.get("messages", [])),
        date=get_today_str()
    )
    logger.debug("write_research_brief invoking structured_output_model with prompt_length=%d", len(prompt))
    response = structured_output_model.invoke([HumanMessage(content=prompt)])
    logger.debug("write_research_brief produced research_brief length=%d", len(response.research_brief))

    # Update state with generated research brief and pass it to the supervisor
    return Command(
            goto="write_draft_report", 
            update={"research_brief": response.research_brief}
        )

def write_draft_report(state: AgentState) -> Command[Literal["__end__"]]:
    """
    Final report generation node.

    Synthesizes all research findings into a comprehensive final report
    """
    logger.debug(
        "write_draft_report invoked with research_brief present=%s",
        bool(state.get("research_brief")),
    )
    # Set up structured output model
    structured_output_model = creative_model.with_structured_output(DraftReport)
    research_brief = state.get("research_brief", "")
    draft_report_prompt = draft_report_generation_prompt.format(
        research_brief=research_brief,
        date=get_today_str()
    )

    response = structured_output_model.invoke([HumanMessage(content=draft_report_prompt)])
    logger.debug("write_draft_report produced draft_report length=%d", len(response.draft_report))

    return {
        "research_brief": research_brief,
        "draft_report": response.draft_report, 
        "supervisor_messages": ["Here is the draft report: " + response.draft_report, research_brief]
    }

# ===== GRAPH CONSTRUCTION =====

# Build the scoping workflow
deep_researcher_builder = StateGraph(AgentState, input_schema=AgentInputState)

# Add workflow nodes
deep_researcher_builder.add_node("clarify_with_user", clarify_with_user)
deep_researcher_builder.add_node("write_research_brief", write_research_brief)
deep_researcher_builder.add_node("write_draft_report", write_draft_report)

# Add workflow edges
deep_researcher_builder.add_edge(START, "clarify_with_user")
deep_researcher_builder.add_edge("write_research_brief", "write_draft_report")
deep_researcher_builder.add_edge("write_draft_report", END)

# Compile the workflow
scope_research = deep_researcher_builder.compile()
