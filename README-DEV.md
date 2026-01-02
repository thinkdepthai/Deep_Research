# Developer Notes

## Interactive vs. Headless Mode
- **Integration / CI runs**: Leave `DEEP_RESEARCH_INTERACTIVE` unset or set to `false` (default headless behavior).
- **Manual interactive sessions**: Set `DEEP_RESEARCH_INTERACTIVE=true` to enable the interactive clarification step.

## Agentic workflow (overview)
- Scope: optional clarify → generate research brief → draft report → supervisor orchestrates parallel researchers → notes aggregated → final report.
- Parallelism: supervisor can launch up to 3 researcher agents; each researcher loops search/think until ready to compress findings.
- Tools: `_tavily_search_tool`, `_think_tool`, `_refine_draft_report_tool`, `ConductResearch`, `ResearchComplete`.

```
User request
    |
    v
clarify_with_user (skipped when DEEP_RESEARCH_INTERACTIVE=false)
    v
write_research_brief  --> research_brief
    v
write_draft_report    --> draft_report
    v
+--------------------------------------+
| supervisor_subgraph (multi_agent_supervisor)
|  - decides topics, iterations <= 15
|  - issues ConductResearch / think / refine
+--------------------------------------+
        | (tool call: ConductResearch, <=3 parallel)
        v
  +-------------------------------+
  | researcher_agent              |
  |  llm_call -> tool_node loop   |
  |  tools: tavily_search, think  |
  |  exit: compress_research      |
  +-------------------------------+
        ^ compressed_research, raw_notes (ToolMessage)
        |
   refine_draft_report tool (optional)
        |
notes + draft_report
        v
final_report_generation --> final_report + user message
```

## Key components
- `research_agent_scope.py`: clarify request (optional), create `research_brief`, draft initial report.
- `multi_agent_supervisor.py` + `state_multi_agent_supervisor.py`: supervise, fan out `ConductResearch`, gather notes, optional draft refinement.
- `research_agent.py` + `state_research.py`: per-topic researcher loop (LLM + tools → compress findings).
- `research_agent_full.py`: wires scoping → supervisor subgraph → final report generation.
- `utils.py`: tool implementations (`tavily_search`, `think`, `refine_draft_report`).

## Data flow notes
- Supervisor aggregates `compressed_research` from researchers via tool messages, can refine draft, then hands `notes` and `draft_report` into final report generation.
- Iteration guards: supervisor max iterations (`max_researcher_iterations=15`), bounded parallel researchers (`max_concurrent_researchers=3`).
