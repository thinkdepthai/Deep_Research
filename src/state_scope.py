
"""State Definitions and Pydantic Schemas for Research Scoping.

This defines the state objects and structured schemas used for
the research agent scoping workflow, including researcher state management and output schemas.
"""

import operator
from typing_extensions import Optional, Annotated, List, Sequence

from langchain_core.messages import BaseMessage
from langgraph.graph import MessagesState
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

# ===== STATE DEFINITIONS =====

class AgentInputState(MessagesState):
    """Input state for the full agent - only contains messages from user input."""
    pass

class AgentState(MessagesState):
    """
    Main state for the full multi-agent research system.

    Extends MessagesState with additional fields for research coordination.
    Note: Some fields are duplicated across different state classes for proper
    state management between subgraphs and the main workflow.
    """

    # Research brief generated from user conversation history
    research_brief: Optional[str]
    # Messages exchanged with the supervisor agent for coordination
    supervisor_messages: Annotated[Sequence[BaseMessage], add_messages]
    # Raw unprocessed research notes collected during the research phase
    raw_notes: Annotated[list[str], operator.add] = []
    # Processed and structured notes ready for report generation
    notes: Annotated[list[str], operator.add] = []
    # Draft research report
    draft_report: str
    # Final formatted research report
    final_report: str
    # Red team evaluation summary (dict with metrics)
    red_team_evaluation: Optional[dict]
    # Red team evaluation detailed report (markdown string)
    red_team_report: Optional[str]
    # Blue-Red feedback loop fields
    red_team_iteration_count: int = 0
    red_team_feedback: Optional[dict] = None  # Current feedback for refinement
    red_team_feedback_history: Annotated[list[dict], operator.add] = []  # Track feedback across iterations
    refinement_iterations: int = 0
    max_refinement_iterations: int = 3  # Default max iterations
    min_objectivity_score: float = 0.75  # Default quality threshold

# ===== STRUCTURED OUTPUT SCHEMAS =====

class ClarifyWithUser(BaseModel):
    """Schema for user clarification decision and questions."""

    need_clarification: bool = Field(
        description="Whether the user needs to be asked a clarifying question.",
    )
    question: str = Field(
        description="A question to ask the user to clarify the report scope",
    )
    verification: str = Field(
        description="Verify message that we will start research after the user has provided the necessary information.",
    )

class ResearchQuestion(BaseModel):
    """Schema for structured research brief generation."""

    research_brief: str = Field(
        description="A research question that will be used to guide the research.",
    )

class DraftReport(BaseModel):
    """Schema for structured draft report generation."""

    draft_report: str = Field(
        description="A draft report that will be used to guide the research.",
    )
