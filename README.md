---
title: Template Final Assignment
emoji: 🕵🏻‍♂️
colorFrom: indigo
colorTo: indigo
sdk: gradio
sdk_version: 5.25.2
app_file: app.py
pinned: false
hf_oauth: true
# optional, default duration is 8 hours/480 minutes. Max duration is 30 days/43200 minutes.
hf_oauth_expiration_minutes: 480
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

## Local Development (uv)

### 1) Set up the environment with uv

```bash
cd /Users/micahshanks/Dev/hf_agents_course/final_project/Final_Assignment_Template
uv venv .venv
# Optional: activate; uv run will use the venv automatically
source .venv/bin/activate  # macOS/Linux

# Install dependencies from requirements.txt
uv pip install -r requirements.txt
```

### 2) Set your Hugging Face token

Export one of the following environment variables before starting the app:

```bash
export HUGGINGFACE_HUB_TOKEN=YOUR_TOKEN   # preferred
# or
export HUGGING_FACE_HUB_TOKEN=YOUR_TOKEN
```

### 3) Run the Gradio app

```bash
uv run python app.py
```

You should see output similar to:

```
* Running on local URL:  http://127.0.0.1:7860
* To create a public link, set `share=True` in `launch()`.
```

Open the printed URL in your browser.

## LangGraph Studio (optional)

To explore the graph used in Studio:

```bash
cd /Users/micahshanks/Dev/hf_agents_course/final_project/Final_Assignment_Template/studio
uv run langgraph dev
```

If you see an error about `langgraph.json`, ensure the file exists in the `studio` directory.