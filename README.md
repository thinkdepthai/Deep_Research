### ThinkDepth.ai Deep Research
Our ThinkDepth.ai deep research 1) addresses the issue of balancing multiple factors for long horizon and complex tasks and 2) addresses the issue of balancing between model capability and structural flexibility. It solves those issues by explicitly reasoning about the self-balancing rules that guide the interaction of different requirements at different stages. For example, those self-balancing rules allow ThinkDepth.ai deep research to explicitly guide the interaction between information gap closing and generation gap closing at different stages.


In the information collection stage, it focuses on closing the information gap by making external web search tool calls while doing a bit of generation gap closing by refining the draft report. Once the information gap is fully closed, it transitions to the final report generation stage. In the final report generation stage, it then fully optimizes for closing the generation gap. This explicit multi-stage self-balancing rules reasoning leads to the development of Self-Balancing Test-Time Diffusion Deep Research algorithm and more effective context engineering. We call this paradigm Self-Balancing Agentic AI. 

Check out our <a href="https://paichunlin.substack.com/p/self-balancing-agentic-ai-test-time">blog post</a> for more technical details.

Primary Contact: <a href="https://www.linkedin.com/in/paichunjimlin">Paichun Lin's LinkedIn</a> | paichul@cs.stanford.edu

### Setup
Please follow the instructions to run the demo:
1. Install uv
```
pip install uv
```
2. in ~/.zshrc or ~/.bashrc, enter your API keys info:
```
export OPENAI_API_KEY='Your OpenAI API Key'

export TAVILY_API_KEY='Your Tavily API Key'
```
3. Install all packages
```
uv sync
```
4. Run the demo in the notebook
```
uv run jupyter notebook thinkdepthai_deepresearch.ipynb
```

### Experiments
<a href="https://thinkdepth.ai">ThinkDepth.ai</a> deep research was ranked #1 and established a new state-of-art result on <a href="https://huggingface.co/spaces/Ayanami0730/DeepResearch-Leaderboard/discussions/4/files">DeepResearch  Bench</a> on Nov 17th, 2025. 
It is #1 open source deep research agent on DeepResearch Bench since Nov 22nd, 2025. 
* It outperformed Google Gemini 2.5 pro deep research by 2.78%.
* It outperformed OpenAI deep research by 6.04%.
* It outperformed Anthropic Claude deep research by  7.45%.

<img width="899" height="463" alt="benchmark" src="https://github.com/user-attachments/assets/1ddd8bd0-1d04-467e-a00d-394e9dc967f8" />

### DeepResearch Bench Leaderboard Screenshot

<img width="1178" height="751" alt="huggingface_leaderboard" src="https://github.com/user-attachments/assets/2d88256a-5e77-46f8-bd51-fe083bfcc780" />

<a href="https://huggingface.co/spaces/muset-ai/DeepResearch-Bench-Leaderboard"> DeepResearch Bench Leaderboard </a> 

### Example Generated Report
For the task "Write a paper to discuss the influence of AI interaction on interpersonal relations, considering AI's potential to fundamentally change how and why individuals relate to each other.", a snapshot of the generated report is shared below:


<img width="1005" height="645" alt="report" src="https://github.com/user-attachments/assets/7fccc245-a83b-4b95-9abe-f1d56fef607d" />
