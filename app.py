import os
import gradio as gr
import requests
import pandas as pd
import typing as t
import json

from langchain_core.messages import HumanMessage
from graph import graph as agent_graph

# --- Constants ---
DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"
QUESTIONS_CACHE_PATH = os.path.join(os.path.dirname(__file__), "questions_cache.json")

# Global LangGraph execution settings (used for all batch runs)
DEFAULT_RECURSION_LIMIT = 25
DEFAULT_MAX_CONCURRENCY = 5

def load_cached_questions() -> list:
    try:
        if os.path.exists(QUESTIONS_CACHE_PATH):
            with open(QUESTIONS_CACHE_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, list) else []
    except Exception:
        pass
    return []

def save_cached_questions(questions_data: list) -> None:
    try:
        with open(QUESTIONS_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(questions_data, f, ensure_ascii=False, indent=2)
    except Exception:
        # Non-fatal if caching fails
        pass

def fetch_questions(profile: gr.OAuthProfile | None, refresh: bool = True, questions_state: t.Optional[list] = None):
    if not profile:
        return "Please Login to Hugging Face with the button.", gr.update(value=pd.DataFrame()), (questions_state or []), gr.update(value=True)
    username = f"{profile.username}"
    api_url = os.getenv("API_URL", DEFAULT_API_URL)
    questions_url = f"{api_url}/questions"
    try:
        if not refresh:
            # Prefer in-memory state, then disk cache
            questions_data = questions_state or load_cached_questions()
        else:
            questions_data = []

        if not questions_data:
            resp = requests.get(questions_url, timeout=15)
            resp.raise_for_status()
            questions_data = resp.json() or []
            # Enrich with file_url if API provided only file_name
            for q in questions_data:
                fname = (q.get("file_name") or "").strip()
                if fname and not q.get("file_url"):
                    q["file_url"] = f"{questions_url.rsplit('/', 1)[0]}/files/{fname}" if "/questions" in questions_url else f"{api_url}/files/{fname}"
            # Save fresh fetch to disk cache
            if questions_data:
                save_cached_questions(questions_data)
        if not questions_data:
            return "Fetched questions list is empty.", pd.DataFrame(), (questions_state or []), gr.update(value=False)
        rows = []
        for q in questions_data:
            row = {
                "Selected": False,  # Default: nothing selected until user chooses
                "Task ID": q.get("task_id"),
                "Question": q.get("question"),
                # Always include columns so they appear in the table
                "File Name": q.get("file_name") or "",
                "File URL": q.get("file_url") or "",
            }
            rows.append(row)
        qdf = pd.DataFrame(rows)
        # Ensure boolean dtype for proper checkbox rendering
        if "Selected" in qdf.columns:
            qdf["Selected"] = qdf["Selected"].astype(bool)
        # Enforce presence of attachment columns
        for col in ["File Name", "File URL"]:
            if col not in qdf.columns:
                qdf[col] = ""
        # By default, no rows are selected and the 'select all' checkbox is unchecked.
        return f"Fetched {len(questions_data)} questions for '{username}'.", gr.update(value=qdf), questions_data, gr.update(value=False)
    except Exception as e:
        # Attempt to fall back to disk cache on network error
        cached = load_cached_questions()
        if cached:
            rows = []
            for q in cached:
                row = {
                    "Selected": False,  # Default: nothing selected until user chooses
                    "Task ID": q.get("task_id"),
                    "Question": q.get("question"),
                    "File Name": q.get("file_name") or "",
                    "File URL": (q.get("file_url") or (f"{questions_url.rsplit('/', 1)[0]}/files/{q.get('file_name')}" if q.get('file_name') else "")),
                }
                rows.append(row)
            qdf = pd.DataFrame(rows)
            if "Selected" in qdf.columns:
                qdf["Selected"] = qdf["Selected"].astype(bool)
            for col in ["File Name", "File URL"]:
                if col not in qdf.columns:
                    qdf[col] = ""
            # By default, no rows are selected and the 'select all' checkbox is unchecked.
            return (
                f"Network error, loaded {len(cached)} cached questions.",
                gr.update(value=qdf),
                cached,
                gr.update(value=False),
            )
        return f"Error fetching questions: {e}", gr.update(value=pd.DataFrame()), (questions_state or []), gr.update()

def run_agent(selection_df: t.Any = None):
    """Run agent on selected questions IN PARALLEL using LangGraph's batch()."""
    if selection_df is None:
        return "No questions table provided. Click 'Fetch Questions' first.", pd.DataFrame(), []
    try:
        df = selection_df if isinstance(selection_df, pd.DataFrame) else pd.DataFrame(selection_df)
    except Exception:
        return "Invalid questions table provided.", pd.DataFrame(), []
    if df.empty:
        return "Questions table is empty. Click 'Fetch Questions' first.", pd.DataFrame(), []
    if "Selected" in df.columns:
        df = df[df["Selected"] == True]
    if df.empty:
        return "No questions selected.", pd.DataFrame(), []
    
    # Prepare batch inputs for parallel execution
    batch_inputs = []
    task_metadata = []
    
    for _, row in df.iterrows():
        task_id = row.get("Task ID")
        question_text = row.get("Question")
        file_name = row.get("File Name") or ""
        
        if not task_id or question_text is None:
            continue
        
        user_content = question_text

        state_input = {"messages": [HumanMessage(content=user_content)]}
        if file_name:
            state_input["attachment_file_name"] = file_name
        
        batch_inputs.append(state_input)
        task_metadata.append({"task_id": task_id, "question": question_text})
    
    if not batch_inputs:
        return "No valid questions to process.", pd.DataFrame(), []
    
    # 🎯 Use LangGraph's native batch() method for parallel execution
    try:
        results = agent_graph.batch(
            batch_inputs,
            config={
                "recursion_limit": DEFAULT_RECURSION_LIMIT,
                "max_concurrency": DEFAULT_MAX_CONCURRENCY,
            },
        )
    except Exception as e:
        return f"Error running batch: {e}", pd.DataFrame(), []
    
    # Build outputs
    results_log = []
    answers_payload = []
    
    for metadata, result in zip(task_metadata, results):
        try:
            answer = result["messages"][-1].content
        except Exception as e:
            answer = f"AGENT ERROR: {e}"
        
        results_log.append({
            "Task ID": metadata["task_id"],
            "Question": metadata["question"],
            "Submitted Answer": answer
        })
        answers_payload.append({
            "task_id": metadata["task_id"],
            "submitted_answer": answer
        })
    
    return (
        f"Ran agent on {len(answers_payload)} questions in parallel. Review answers below.",
        pd.DataFrame(results_log),
        answers_payload
    )

def select_all_toggle(select_all: bool, current_df: t.Any = None):
    try:
        df = current_df if isinstance(current_df, pd.DataFrame) else pd.DataFrame(current_df)
    except Exception:
        return gr.update()
    if df.empty or "Selected" not in df.columns:
        return gr.update()
    df["Selected"] = bool(select_all)
    # Ensure dtype remains boolean for checkbox rendering
    df["Selected"] = df["Selected"].astype(bool)
    return gr.update(value=df)

def submit_answers(profile: gr.OAuthProfile | None, questions_state: t.Optional[list] = None, answers_state: t.Optional[list] = None):
    """Submit pre-generated answers from answers_state."""
    if not profile:
        return "Please Login to Hugging Face with the button."
    username = f"{profile.username}".strip()
    
    # Use pre-generated answers if available
    if answers_state:
        answers_payload = answers_state
    else:
        return "No answers to submit. Click 'Run Agent' first to generate answers."
    
    if not answers_payload:
        return "No answers to submit. Click 'Run Agent' first."

    # Submit the answers
    space_id = os.getenv("SPACE_ID")
    agent_code = f"https://huggingface.co/spaces/{space_id}/tree/main" if space_id else None
    submission_data = {"username": username, "agent_code": agent_code, "answers": answers_payload}
    api_url = DEFAULT_API_URL
    submit_url = f"{api_url}/submit"
    
    try:
        response = requests.post(submit_url, json=submission_data, timeout=60)
        response.raise_for_status()
        r = response.json()
        return (
            "Submission Successful!\n"
            f"User: {r.get('username')}\n"
            f"Overall Score: {r.get('score', 'N/A')}% "
            f"({r.get('correct_count', '?')}/{r.get('total_attempted', '?')} correct)\n"
            f"Message: {r.get('message', 'No message received.')}"
        )
    except Exception as e:
        return f"Submission Failed: {e}"

def run_and_submit_all(profile: gr.OAuthProfile | None):
    """Fetch all questions, run agent in parallel using batch(), then submit."""
    if not profile:
        return "Please Login to Hugging Face with the button.", None
    
    username = f"{profile.username}"
    api_url = DEFAULT_API_URL
    questions_url = f"{api_url}/questions"
    submit_url = f"{api_url}/submit"
    
    # Fetch questions
    print(f"Fetching questions from: {questions_url}")
    try:
        response = requests.get(questions_url, timeout=15)
        response.raise_for_status()
        questions_data = response.json()
        if not questions_data:
            return "Fetched questions list is empty.", None
        print(f"Fetched {len(questions_data)} questions.")
    except Exception as e:
        return f"Error fetching questions: {e}", None
    
    # Prepare batch inputs for parallel execution
    batch_inputs = []
    task_metadata = []
    
    for item in questions_data:
        task_id = item.get("task_id")
        question_text = item.get("question")
        file_name = item.get("file_name") or ""
        
        if not task_id or question_text is None:
            continue
        
        user_content = question_text

        state_input = {"messages": [HumanMessage(content=user_content)]}
        if file_name:
            state_input["attachment_file_name"] = file_name
        
        batch_inputs.append(state_input)
        task_metadata.append({"task_id": task_id, "question": question_text})
    
    # 🎯 Run all questions in parallel using LangGraph's batch()
    print(f"Running agent on {len(batch_inputs)} questions in parallel...")
    try:
        results = agent_graph.batch(
            batch_inputs,
            config={
                "recursion_limit": DEFAULT_RECURSION_LIMIT,
                "max_concurrency": DEFAULT_MAX_CONCURRENCY,
            },
        )
    except Exception as e:
        return f"Error running batch: {e}", None
    
    # Build submission payload
    results_log = []
    answers_payload = []
    
    for metadata, result in zip(task_metadata, results):
        try:
            answer = result["messages"][-1].content
        except Exception as e:
            answer = f"AGENT ERROR: {e}"
        
        results_log.append({
            "Task ID": metadata["task_id"],
            "Question": metadata["question"],
            "Submitted Answer": answer
        })
        answers_payload.append({
            "task_id": metadata["task_id"],
            "submitted_answer": answer
        })
    
    if not answers_payload:
        return "No answers produced.", pd.DataFrame(results_log)
    
    # Submit
    space_id = os.getenv("SPACE_ID")
    agent_code = f"https://huggingface.co/spaces/{space_id}/tree/main" if space_id else None
    submission_data = {"username": username.strip(), "agent_code": agent_code, "answers": answers_payload}
    
    print(f"Submitting {len(answers_payload)} answers...")
    try:
        response = requests.post(submit_url, json=submission_data, timeout=120)
        response.raise_for_status()
        result_data = response.json()
        final_status = (
            f"Submission Successful!\n"
            f"User: {result_data.get('username')}\n"
            f"Overall Score: {result_data.get('score', 'N/A')}% "
            f"({result_data.get('correct_count', '?')}/{result_data.get('total_attempted', '?')} correct)\n"
            f"Message: {result_data.get('message', 'No message received.')}"
        )
        return final_status, pd.DataFrame(results_log)
    except Exception as e:
        return f"Submission Failed: {e}", pd.DataFrame(results_log)


# --- Build Gradio Interface using Blocks ---
with gr.Blocks() as demo:
    gr.Markdown("# GAIA Benchmark Agent Evaluation")
    gr.Markdown(
        """
        **How to Use This App:**

        1.  **Login:** Click the login button and authenticate with your Hugging Face account.
        2.  **Fetch Questions:** Click "Fetch Questions" to load questions from the GAIA benchmark API.
        3.  **Select Questions:** Toggle individual questions or use "Select all questions" checkbox.
        4.  **Run Agent:** Click "Run Agent" to process selected questions in parallel (up to 5 concurrent).
        5.  **Review Answers:** Check the results in the "Agent Answers" table before submitting.
        6.  **Submit:** Click "Submit Answers for Scoring" to send your cached answers to the evaluation API.

        """
    )

    login_button = gr.LoginButton()

    # Shared state
    questions_state = gr.State([])
    answers_state = gr.State([])

    # Buttons & controls
    fetch_button = gr.Button("Fetch Questions")
    run_button = gr.Button("Run Agent (no submission)")
    submit_button = gr.Button("Submit Answers for Scoring")
    refresh_cb = gr.Checkbox(label="Refresh questions from API", value=False)
    # Default: no questions selected and 'select all' unchecked until user opts in
    select_all_cb = gr.Checkbox(label="Select all questions", value=False)

    status_output = gr.Textbox(label="Status", lines=5, interactive=False)
    questions_table = gr.Dataframe(
        label="Fetched Questions (toggle 'Selected' to choose)",
        headers=["Selected", "Task ID", "Question"],
        datatype=["bool", "number", "str"],
        interactive=True,
    )
    answers_table = gr.DataFrame(label="Agent Answers", wrap=True)

    # Wire up actions
    # Note: Profile is automatically injected by LoginButton when function accepts it as first parameter
    fetch_button.click(
        fn=fetch_questions,
        inputs=[refresh_cb, questions_state],
        outputs=[status_output, questions_table, questions_state, select_all_cb],
    )

    run_button.click(
        fn=run_agent,
        inputs=[questions_table],
        outputs=[status_output, answers_table, answers_state],
    )

    select_all_cb.change(
        fn=select_all_toggle,
        inputs=[select_all_cb, questions_table],
        outputs=[questions_table],
    )

    submit_button.click(
        fn=submit_answers,
        inputs=[questions_state, answers_state],
        outputs=[status_output],
    )

if __name__ == "__main__":
    print("\n" + "-"*30 + " App Starting " + "-"*30)
    # Check for SPACE_HOST and SPACE_ID at startup for information
    space_host_startup = os.getenv("SPACE_HOST")
    space_id_startup = os.getenv("SPACE_ID") # Get SPACE_ID at startup

    if space_host_startup:
        print(f"✅ SPACE_HOST found: {space_host_startup}")
        print(f"   Runtime URL should be: https://{space_host_startup}.hf.space")
    else:
        print("ℹ️  SPACE_HOST environment variable not found (running locally?).")

    if space_id_startup: # Print repo URLs if SPACE_ID is found
        print(f"✅ SPACE_ID found: {space_id_startup}")
        print(f"   Repo URL: https://huggingface.co/spaces/{space_id_startup}")
        print(f"   Repo Tree URL: https://huggingface.co/spaces/{space_id_startup}/tree/main")
    else:
        print("ℹ️  SPACE_ID environment variable not found (running locally?). Repo URL cannot be determined.")

    print("-"*(60 + len(" App Starting ")) + "\n")

    print("Launching Gradio Interface for Basic Agent Evaluation...")
    demo.launch(debug=True, share=False)