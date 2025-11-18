
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

"""
Fast Testing Mode - Bypasses clarification, thinking, and draft generation
"""

from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END

from deep_research.utils import get_today_str
from deep_research.prompts import final_report_generation_with_helpfulness_insightfulness_hit_citation_prompt
from deep_research.state_scope import AgentState, AgentInputState
from deep_research.research_agent_scope import clarify_with_user, write_research_brief, write_draft_report
from deep_research.multi_agent_supervisor import supervisor_agent

# ===== TESTING CONFIG =====
FAST_TEST_MODE = False  # Set to False for full pipeline

from langchain.chat_models import init_chat_model
writer_model = init_chat_model(model="openai:gpt-5", max_tokens=40000)

# ===== BYPASS NODES FOR TESTING =====

async def fast_clarify_with_user(state: AgentState):
    """Skip user clarification - just route to next step"""
    # Still need to return Command to route properly
    from langgraph.types import Command
    return Command(goto="write_research_brief")

async def fast_write_research_brief(state: AgentState):
    """Skip research brief generation - use user request directly as brief"""
    from langgraph.types import Command
    
    # Get user request from messages
    messages = state.get("messages", [])
    user_request = messages[-1].content if messages else "Research topic"
    
    # Return Command with research_brief that supervisor will use
    return Command(
        goto="write_draft_report",
        update={"research_brief": user_request}
    )

async def fast_write_draft_report(state: AgentState):
    """Skip draft report - pass empty draft and research brief to supervisor"""
    research_brief = state.get("research_brief", "")
    
    # CRITICAL: supervisor_messages needs the research_brief so it knows what to search
    return {
        "draft_report": "",
        "supervisor_messages": [HumanMessage(content=research_brief)]  # This goes to supervisor
    }

async def fast_final_report_generation(state: AgentState):
    """Fast final report - just return notes without LLM synthesis"""
    notes = state.get("notes", [])
    findings = "\n".join(notes)
    
    return {
        "final_report": findings,
        "messages": [HumanMessage(content="Here is the final report: " + findings)],
    }

# ===== FULL NODES (unchanged) =====

async def final_report_generation(state: AgentState):
    """Full final report generation"""
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

    return {
        "final_report": final_report.content, 
        "messages": ["Here is the final report: " + final_report.content],
    }

# ===== GRAPH CONSTRUCTION =====

deep_researcher_builder = StateGraph(AgentState, input_schema=AgentInputState)

# Choose nodes based on mode
if FAST_TEST_MODE:
    print("🚀 FAST TEST MODE - Bypassing clarification, brief, draft, and final report generation")
    deep_researcher_builder.add_node("clarify_with_user", fast_clarify_with_user)
    deep_researcher_builder.add_node("write_research_brief", fast_write_research_brief)
    deep_researcher_builder.add_node("write_draft_report", fast_write_draft_report)
    deep_researcher_builder.add_node("final_report_generation", fast_final_report_generation)
else:
    print("📚 FULL MODE - Running complete research pipeline")
    deep_researcher_builder.add_node("clarify_with_user", clarify_with_user)
    deep_researcher_builder.add_node("write_research_brief", write_research_brief)
    deep_researcher_builder.add_node("write_draft_report", write_draft_report)
    deep_researcher_builder.add_node("final_report_generation", final_report_generation)

# Supervisor always runs (this is what you're testing)
deep_researcher_builder.add_node("supervisor_subgraph", supervisor_agent)

# Add workflow edges
deep_researcher_builder.add_edge(START, "clarify_with_user")
deep_researcher_builder.add_edge("write_research_brief", "write_draft_report")
deep_researcher_builder.add_edge("write_draft_report", "supervisor_subgraph")
deep_researcher_builder.add_edge("supervisor_subgraph", "final_report_generation")
deep_researcher_builder.add_edge("final_report_generation", END)

# Compile
agent = deep_researcher_builder.compile()
