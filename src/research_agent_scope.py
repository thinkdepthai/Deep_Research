
"""User Clarification and Research Brief Generation.

This module implements the scoping phase of the research workflow, where we:
1. Assess if the user's request needs clarification
2. Generate a detailed research brief from the conversation

The workflow uses structured output to make deterministic decisions about
whether sufficient context exists to proceed with research.
"""

from datetime import datetime
from typing_extensions import Literal

from deep_research.model_config import get_model
from langchain_core.messages import HumanMessage, get_buffer_string
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command

from deep_research.prompts import transform_messages_into_research_topic_human_msg_prompt, draft_report_generation_prompt, clarify_with_user_instructions
from deep_research.state_scope import AgentState, ResearchQuestion, AgentInputState, DraftReport
from deep_research.usage_tracker import get_tracker

# ===== UTILITY FUNCTIONS =====

def get_today_str() -> str:
    """Get current date in a human-readable format."""
    return datetime.now().strftime("%d/%m/%Y, %H:%M:%S")

# ===== CONFIGURATION =====

# Initialize model
model = get_model()
creative_model = get_model()

# ===== WORKFLOW NODES =====

def clarify_with_user(state: AgentState) -> Command[Literal["write_research_brief"]]:
    #uncomment if you want to enable this module
    """
    Determine if the user's request contains sufficient information to proceed with research.

    Uses structured output to make deterministic decisions and avoid hallucination.
    Routes to either research brief generation or ends with a clarification question.
    """

    """
    # Set up structured output model
    structured_output_model = model.with_structured_output(ClarifyWithUser)

    # Invoke the model with clarification instructions
    response = structured_output_model.invoke([
        HumanMessage(content=clarify_with_user_instructions.format(
            messages=get_buffer_string(messages=state["messages"]),
            date=get_today_str()
        ))
    ])

    # Route based on clarification need
    if response.need_clarification:
        return Command(
            goto=END,
            update={"messages": [AIMessage(content=response.question)]}
        )
    else:
    """
    return Command(
        goto="write_research_brief"
    )

def write_research_brief(state: AgentState) -> Command[Literal["write_draft_report"]]:
    """
    Transform the conversation history into a comprehensive research brief.

    Uses structured output to ensure the brief follows the required format
    and contains all necessary details for effective research.
    """
    # Set up structured output model
    structured_output_model = model.with_structured_output(ResearchQuestion)

    try:
        # Generate research brief from conversation history
        response = structured_output_model.invoke([
            HumanMessage(content=transform_messages_into_research_topic_human_msg_prompt.format(
                messages=get_buffer_string(state.get("messages", [])),
                date=get_today_str()
            ))
        ])

        # Track token usage
        tracker = get_tracker()
        tracker.track_openai_response(response)

        # Update state with generated research brief and pass it to the supervisor
        return Command(
                goto="write_draft_report",
                update={"research_brief": response.research_brief}
            )
    except Exception as e:
        error_msg = str(e)
        # Check for common API errors
        if "429" in error_msg or "quota" in error_msg.lower() or "insufficient_quota" in error_msg.lower():
            raise RuntimeError(
                "OpenAI API Quota Exceeded: Your OpenAI API key has exceeded its quota or billing limit. "
                "Please check your OpenAI account billing and usage at https://platform.openai.com/usage. "
                "You may need to add payment information or upgrade your plan."
            ) from e
        elif "401" in error_msg or "invalid" in error_msg.lower() and "api key" in error_msg.lower():
            raise RuntimeError(
                "OpenAI API Key Error: Your API key is invalid or expired. "
                "Please check your OPENAI_API_KEY in your .env file or environment variables."
            ) from e
        else:
            raise RuntimeError(
                f"Error generating research brief: {error_msg}. "
                "Please check your API configuration and try again."
            ) from e

def write_draft_report(state: AgentState) -> Command[Literal["__end__"]]:
    """
    Final report generation node.

    Synthesizes all research findings into a comprehensive final report
    """
    # Set up structured output model
    structured_output_model = creative_model.with_structured_output(DraftReport)
    research_brief = state.get("research_brief", "")
    draft_report_prompt = draft_report_generation_prompt.format(
        research_brief=research_brief,
        date=get_today_str()
    )

    try:
        response = structured_output_model.invoke([HumanMessage(content=draft_report_prompt)])

        # Track token usage
        tracker = get_tracker()
        tracker.track_openai_response(response)

        return {
            "research_brief": research_brief,
            "draft_report": response.draft_report,
            "supervisor_messages": ["Here is the draft report: " + response.draft_report, research_brief]
        }
    except Exception as e:
        error_msg = str(e)
        # Check for common API errors
        if "429" in error_msg or "quota" in error_msg.lower() or "insufficient_quota" in error_msg.lower():
            raise RuntimeError(
                "OpenAI API Quota Exceeded: Your OpenAI API key has exceeded its quota or billing limit. "
                "Please check your OpenAI account billing and usage at https://platform.openai.com/usage. "
                "You may need to add payment information or upgrade your plan."
            ) from e
        elif "401" in error_msg or "invalid" in error_msg.lower() and "api key" in error_msg.lower():
            raise RuntimeError(
                "OpenAI API Key Error: Your API key is invalid or expired. "
                "Please check your OPENAI_API_KEY in your .env file or environment variables."
            ) from e
        else:
            raise RuntimeError(
                f"Error generating draft report: {error_msg}. "
                "Please check your API configuration and try again."
            ) from e

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
