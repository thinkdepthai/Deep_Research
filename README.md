# ThinkDepth.ai Deep Research

A powerful multi-agent research system that uses Self-Balancing Agentic AI to conduct deep, comprehensive research on complex topics. The system addresses the challenge of balancing multiple factors for long-horizon and complex tasks by explicitly reasoning about self-balancing rules that guide the interaction of different requirements at different stages.

## Key Features

- **Self-Balancing Agentic AI**: Explicitly guides the interaction between information gap closing and generation gap closing at different stages
- **Multi-Stage Research Process**:
  - Information collection stage focuses on closing information gaps via web search
  - Final report generation stage optimizes for closing generation gaps
- **Red Team Evaluation**: Built-in adversarial evaluation system to assess objectivity, bias, and source quality
- **Blue-Red Feedback Loop**: Iterative refinement process where red team feedback improves report quality
- **Token Usage Tracking**: Comprehensive tracking of API usage (OpenAI/OpenRouter tokens and Tavily calls)
- **Multiple Research Modes**:
  - General deep research queries
  - Library competitor analysis
  - RAGaaS platform analysis

## Architecture

The system uses a supervisor-agent architecture with:

- **Research Agents**: Individual agents that conduct focused research on specific topics
- **Multi-Agent Supervisor**: Coordinates multiple research agents using a diffusion algorithm
- **LangGraph**: Orchestrates the multi-stage, multi-agent workflow
- **LangChain**: Integrates LLMs and tools (Tavily for web search)
- **Red Team Evaluator**: Assesses report quality, objectivity, and bias

## Installation

### Prerequisites

- Python 3.11 or higher
- pip or uv package manager

### Quick Start

1. **Clone the repository**:

   ```bash
   git clone <repository-url>
   cd Deep_Research
   ```

2. **Install dependencies**:

   **Using pip (recommended)**:

   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

   **Using uv**:

   ```bash
   pip install uv
   uv sync
   ```

3. **Set up API keys**:

   Create a `.env` file in the project root:

   ```bash
   cp .env.example .env
   ```

   Edit `.env` and add your API keys:

   ```
   OPENAI_API_KEY=your_openai_api_key_here
   TAVILY_API_KEY=your_tavily_api_key_here
   ```

   **Alternative**: Set environment variables:

   ```bash
   # Linux/macOS
   export OPENAI_API_KEY='Your OpenAI API Key'
   export TAVILY_API_KEY='Your Tavily API Key'

   # Windows PowerShell
   $env:OPENAI_API_KEY='Your OpenAI API Key'
   $env:TAVILY_API_KEY='Your Tavily API Key'
   ```

4. **Verify installation**:
   ```bash
   python -c "import deep_research; print('Installation successful!')"
   ```

For detailed installation instructions, see [INSTALLATION.md](INSTALLATION.md).

## Usage

### Command-Line Interface

Run general research queries:

```bash
python -m tasks.main "Your research question here"
```

Example:

```bash
python -m tasks.main "What are the latest developments in quantum computing?"
```

### Library Competitor Analysis

Research competitors for Boost C++ libraries:

```bash
python -m tasks.Investigate_competitors
```

This will:

- Read libraries from `boost_library.json`
- Research top 5-6 competitors for each library
- Generate quantitative comparison reports
- Save results to `report_by_library/`

### RAGaaS Platform Analysis

Research RAGaaS platforms for C++ Copilot use case:

```bash
python -m tasks.investigate_ragaas
```

This will:

- Research top 6-8 RAGaaS platforms
- Analyze features, capabilities, and pricing
- Generate comprehensive comparison report
- Save results to `report_by_ragaas/`

### Jupyter Notebook

For interactive exploration:

```bash
jupyter notebook thinkdepthai_deepresearch.ipynb
```

## Configuration

### Environment Variables

The following environment variables can be configured in `.env`:

- `OPENAI_API_KEY`: Your OpenAI API key (required)
- `TAVILY_API_KEY`: Your Tavily API key (required)
- `USE_OPENROUTER`: Set to `true` to use OpenRouter instead of OpenAI (optional)
- `OPENROUTER_API_KEY`: Your OpenRouter API key (optional)
- `ENABLE_RED_TEAM_EVAL`: Enable red team evaluation (default: `false`)
- `RED_TEAM_FEEDBACK_LOOP`: Enable iterative refinement (default: `true`)
- `MIN_OBJECTIVITY_SCORE`: Minimum score threshold (default: `0.75`)
- `MAX_REFINEMENT_ITERATIONS`: Max refinement cycles (default: `3`)

### Red Team Evaluation

The red team evaluation system can be enabled to assess report quality:

- **Objectivity Score**: Overall quality assessment (0-100%)
- **Bias Analysis**: One-sided score, source diversity, quantitative ratio
- **Source Quality**: Credibility score, missing citations
- **Issue Detection**: Unsupported claims, counter-evidence gaps

Enable it by setting `ENABLE_RED_TEAM_EVAL=true` in your `.env` file.

## Project Structure

```
Deep_Research/
├── src/                          # Core source code
│   ├── research_agent.py         # Individual research agent
│   ├── research_agent_full.py    # Full research workflow
│   ├── multi_agent_supervisor.py # Supervisor coordination
│   ├── red_team_evaluator.py    # Red team evaluation
│   ├── usage_tracker.py         # Token usage tracking
│   ├── prompts.py               # Prompt templates
│   ├── utils.py                 # Utility functions
│   └── model_config.py          # LLM configuration
├── tasks/                        # Task-specific scripts
│   ├── main.py                  # CLI entry point
│   ├── Investigate_competitors.py # Library competitor research
│   └── investigate_ragaas.py    # RAGaaS platform research
├── report_by_library/           # Library competitor reports
├── report_by_ragaas/            # RAGaaS analysis reports
├── rules/                        # Cursor rules for AI assistance
├── requirements.txt             # Python dependencies
├── pyproject.toml               # Project configuration
└── README.md                    # This file
```

## Experiments

ThinkDepth.ai deep research was ranked **#1** and established a new state-of-the-art result on [DeepResearch Bench](https://huggingface.co/spaces/Ayanami0730/DeepResearch-Leaderboard/discussions/4/files) on Nov 17th, 2025.

- Outperformed Google Gemini 2.5 pro deep research by **2.78%**
- Outperformed OpenAI deep research by **6.04%**
- Outperformed Anthropic Claude deep research by **7.45%**

See the [DeepResearch Bench Leaderboard](https://huggingface.co/spaces/muset-ai/DeepResearch-Bench-Leaderboard) for current rankings.

## Technical Details

### Self-Balancing Agentic AI

The system implements Self-Balancing Test-Time Diffusion Deep Research algorithm:

1. **Information Collection Stage**:

   - Focuses on closing information gaps
   - Makes external web search tool calls
   - Refines draft reports incrementally

2. **Final Report Generation Stage**:
   - Fully optimizes for closing generation gaps
   - Produces comprehensive, well-structured reports
   - Includes citations and source references

### Multi-Agent Coordination

- Uses diffusion algorithm for parallel research
- Supervisor coordinates multiple research agents
- Balances information gathering and report generation
- Supports iterative refinement based on feedback

### Red Team Evaluation

The red team evaluation system provides:

- **Bias Detection**: Identifies one-sided arguments and missing counter-evidence
- **Source Quality Assessment**: Evaluates credibility and citation quality
- **Claim Verification**: Checks for unsupported claims
- **Actionable Feedback**: Generates specific suggestions for improvement

## API Usage Tracking

The system tracks:

- **Input tokens**: Prompt tokens used
- **Output tokens**: Completion tokens generated
- **Total tokens**: Combined usage
- **Tavily calls**: Number of web search API calls

Usage statistics are included in all generated reports.

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

See [LICENSE](LICENSE) file for details.

## Contact

**Primary Contact**: [Paichun Lin's LinkedIn](https://www.linkedin.com/in/paichunjimlin) | paichul@cs.stanford.edu

## Additional Resources

- [Blog Post: Self-Balancing Agentic AI](https://paichunlin.substack.com/p/self-balancing-agentic-ai-test-time)
- [ThinkDepth.ai Website](https://thinkdepth.ai)
- [DeepResearch Bench Leaderboard](https://huggingface.co/spaces/muset-ai/DeepResearch-Bench-Leaderboard)

## Troubleshooting

### Import Errors

If you see import errors, make sure the package is installed:

```bash
pip install -e .
```

### API Key Issues

- Verify your `.env` file exists and contains valid API keys
- Check that environment variables are set correctly
- Restart your terminal/IDE after setting environment variables

### Token Usage

Monitor your API usage:

- Reports include usage statistics
- Check OpenAI/Tavily dashboards for detailed usage
- Consider using OpenRouter for potentially lower costs

For more troubleshooting help, see [INSTALLATION.md](INSTALLATION.md).
