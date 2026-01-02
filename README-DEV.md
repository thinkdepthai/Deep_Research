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

## How to add a custom search provider
The search layer is pluggable via `search_factory`.

1) Pick a backend name and config block
- In your config (per stage):
  ```yaml
  search:
    backend: customsearch
    customsearch:
      api_key: "..."
      base_url: "https://example.com"
      max_results: 5          # optional, overrides defaults
      topic: general          # optional
      include_raw_content: true
      timeout_seconds: 30     # optional
  ```

2) Provide a provider module
Implement a module that exposes either:
- `PROVIDER` object implementing the `SearchProvider` protocol (`build_client`, `search`, `defaults`), or
- a `register_search_provider(register_fn)` function that calls `register_fn(name, provider_obj)`.

The dynamic loader will try, in order:
- `deep_research.providers.<backend>`
- `deep_research_search_<backend>`
- `<backend>` (any importable module)

3) Normalize results
Your provider’s `search` should return a dict compatible with downstream processing (a list of results under `results`, each having at least `url`, `title`, `content`, and optionally `raw_content`).

4) Swap out the stub (override)
If you need to replace the built-in stubbed `customsearch` provider with a real one at runtime, use `override_provider`:

```python
from deep_research import search_factory

class RealProvider:
    def build_client(self, cfg):
        ...
    def search(self, client, query, *, max_results, include_raw_content, topic, timeout_seconds):
        ...
    def defaults(self, cfg):
        ...

search_factory.override_provider("customsearch", RealProvider())
```

5) Test
Add/extend unit tests to cover provider registration and defaults. You can monkeypatch `search_factory.register_provider` in tests to inject fakes.

6) Fallback/compatibility
If the backend is missing or not importable, `search_factory` raises `SearchConfigError`. Ensure your module is on `PYTHONPATH` in the environment that runs the code/tests.

## Data flow notes
- Supervisor aggregates `compressed_research` from researchers via tool messages, can refine draft, then hands `notes` and `draft_report` into final report generation.
- Iteration guards: supervisor max iterations (`max_researcher_iterations=15`), bounded parallel researchers (`max_concurrent_researchers=3`).
