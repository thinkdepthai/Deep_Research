### ThinkDepth.ai Deep Research
Our ThinkDepth.ai deep research 1) addresses the issue of balancing multiple factors for long horizon and complex tasks and 2) addresses the issue of balancing between model capability and structural flexibility. It solves those issues by explicitly reasoning about the self-balancing rules that guide the interaction of different requirements at different stages. For example, those self-balancing rules allow ThinkDepth.ai deep research to explicitly guide the interaction between information gap closing and generation gap closing at different stages.


In the information collection stage, it focuses on closing the information gap by making external web search tool calls while doing a bit of generation gap closing by refining the draft report. Once the information gap is fully closed, it transitions to the final report generation stage. In the final report generation stage, it then fully optimizes for closing the generation gap. This explicit multi-stage self-balancing rules reasoning leads to the development of Self-Balancing Test-Time Diffusion Deep Research algorithm and more effective context engineering. We call this paradigm Self-Balancing Agentic AI. 

Check out our <a href="https://paichunlin.substack.com/p/self-balancing-agentic-ai-test-time">blog post</a> for more technical details.

Primary Contact: <a href="https://www.linkedin.com/in/paichunjimlin">Paichun Lin's LinkedIn</a> | paichul@cs.stanford.edu

### Setup
Please follow the instructions to run the demo:
1. pip install uv
2. in ~/.zshrc or ~/.bashrc, enter

export OPENAI_API_KEY='Your OpenAI API Key'

export TAVILY_API_KEY='Your Tavily API Key'

3. uv sync
4. uv run jupyter notebook thinkdepthai_deepresearch.ipynb

### Experiments
<a href="https://thinkdepth.ai">ThinkDepth.ai</a> deep research is ranked #1 and established a new state-of-art result on <a href="https://huggingface.co/spaces/Ayanami0730/DeepResearch-Leaderboard/discussions/4/files">DeepResearch  Bench</a> on Oct 29th, 2025.
* It outperformed Google Gemini 2.5 pro deep research by 2.78%.
* It outperformed OpenAI deep research by 6.04%.
* It outperformed Anthropic Claude deep research by  7.45%.

<img width="890" height="453" alt="DeepResearch Bench Results" src="https://github.com/user-attachments/assets/313fd4cc-bb05-4792-880f-d66e9c59082a" />
