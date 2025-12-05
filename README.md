## GAIA Benchmark Agent (`gaia_benchmark_agent`)

Minimal Gradio app + LangGraph agent for answering GAIA benchmark questions

Based on the template provided in Hugging Face's Agents Course.
As I expanded the app to make development iteration easier, I found a locally hosted app to be handy for experimentation and sufficient for checking answers.

[agents-course](https://huggingface.co/agents-course)
[Final_Assignment_Template](https://huggingface.co/spaces/agents-course/Final_Assignment_Template)

## Local development (uv)

### 1) Set up the environment

From the project root:

```bash
uv venv .venv
# Optional: activate; uv run will use the venv automatically
source .venv/bin/activate  # macOS/Linux

# Install dependencies from pyproject.toml / uv.lock
uv sync
```

### 2) Run the Gradio app

```bash
uv run python app.py
```

Then open the printed local URL in your browser (by default `http://127.0.0.1:7860`).

## Optional: LangGraph Studio

To explore the LangGraph-based agent in Studio:

```bash
cd studio
uv run langgraph dev
```
