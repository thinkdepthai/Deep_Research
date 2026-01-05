
"""Research Agent Implementation.

This module implements a research agent that can perform iterative web searches
and synthesis to answer complex research questions.
"""

from typing_extensions import Literal

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, filter_messages
from deep_research.llm_factory import get_chat_model
from deep_research.state_research import ResearcherState, ResearcherOutputState
from deep_research.utils import _tavily_search_tool, _think_tool, get_today_str
from deep_research.prompts import research_agent_prompt, compress_research_system_prompt, compress_research_human_message
from deep_research import logging as dr_logging

logger = dr_logging.get_logger(__name__)

# ===== CONFIGURATION =====

# Set up tools and model binding
# Note: tools are intentionally ordered; keep consistent with prompts
# to avoid regressions.
tools = [_tavily_search_tool, _think_tool]
tools_by_name = {tool.name: tool for tool in tools}

# Initialize models
model = get_chat_model("researcher_main")
model_with_tools = model.bind_tools(tools)
summarization_model = get_chat_model("researcher_summarizer")
compress_model = get_chat_model("researcher_compressor")

# ===== AGENT NODES =====

def llm_call(state: ResearcherState):
    """Analyze current state and decide on next actions."""
    msg_count = len(state.get("researcher_messages", []))
    logger.debug("llm_call invoked with %d messages", msg_count)

    response = model_with_tools.invoke(
        [SystemMessage(content=research_agent_prompt)] + state["researcher_messages"]
    )

    logger.info(
        "llm_call produced response tool_calls=%s num_tool_calls=%d",
        bool(response.tool_calls),
        len(response.tool_calls or []),
    )
    return {
        "researcher_messages": [response]
    }

def tool_node(state: ResearcherState):
    """Execute all tool calls from the previous LLM response."""
    tool_calls = state["researcher_messages"][-1].tool_calls
    logger.info("tool_node executing %d tool calls", len(tool_calls or []))

    # Execute all tool calls
    observations = []
    for tool_call in tool_calls:
        tool = tools_by_name[tool_call["name"]]
        logger.debug("Invoking tool %s with args=%s", tool_call["name"], tool_call["args"])
        observations.append(tool.invoke(tool_call["args"]))

    # Create tool message outputs
    tool_outputs = [
        ToolMessage(
            content=observation,
            name=tool_call["name"],
            tool_call_id=tool_call["id"]
        ) for observation, tool_call in zip(observations, tool_calls)
    ]

    return {"researcher_messages": tool_outputs}

def compress_research(state: ResearcherState) -> dict:
    """Compress research findings into a concise summary."""

    system_message = compress_research_system_prompt.format(date=get_today_str())
    messages = [SystemMessage(content=system_message)] + state.get("researcher_messages", []) + [HumanMessage(content=compress_research_human_message)]
    logger.info("compress_research invoked with %d messages", len(messages))
    response = compress_model.invoke(messages)

    # Extract raw notes from tool and AI messages
    raw_notes = [
        str(m.content) for m in filter_messages(
            state["researcher_messages"], 
            include_types=["tool", "ai"]
        )
    ]

    logger.debug("compress_research produced raw_notes_count=%d", len(raw_notes))
    return {
        "compressed_research": str(response.content),
        "raw_notes": ["\n".join(raw_notes)]
    }

# ===== ROUTING LOGIC =====

def should_continue(state: ResearcherState) -> Literal["tool_node", "compress_research"]:
    """Determine whether to continue research or provide final answer."""
    messages = state["researcher_messages"]
    last_message = messages[-1]

    decision = "tool_node" if last_message.tool_calls else "compress_research"
    logger.info("should_continue decision=%s (has_tool_calls=%s)", decision, bool(last_message.tool_calls))
    return decision

# ===== GRAPH CONSTRUCTION =====

# Build the agent workflow
agent_builder = StateGraph(ResearcherState, output_schema=ResearcherOutputState)

# Add nodes to the graph
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)
agent_builder.add_node("compress_research", compress_research)

# Add edges to connect nodes
agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    {
        "tool_node": "tool_node", # Continue research loop
        "compress_research": "compress_research", # Provide final answer
    },
)
agent_builder.add_edge("tool_node", "llm_call") # Loop back for more research
agent_builder.add_edge("compress_research", END)

# Compile the agent
researcher_agent = agent_builder.compile()
