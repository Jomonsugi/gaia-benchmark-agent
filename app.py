import os
import gradio as gr
import requests
import pandas as pd
import typing as t
import json

from langchain_core.messages import HumanMessage
from graph import graph as agent_graph

# (Keep Constants as is)
# --- Constants ---
DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"
QUESTIONS_CACHE_PATH = os.path.join(os.path.dirname(__file__), "questions_cache.json")

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
                "Selected": True,
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
        return f"Fetched {len(questions_data)} questions for '{username}'.", gr.update(value=qdf), questions_data, gr.update(value=True)
    except Exception as e:
        # Attempt to fall back to disk cache on network error
        cached = load_cached_questions()
        if cached:
            rows = []
            for q in cached:
                row = {
                    "Selected": True,
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
            return (
                f"Network error, loaded {len(cached)} cached questions.",
                gr.update(value=qdf),
                cached,
                gr.update(value=True),
            )
        return f"Error fetching questions: {e}", gr.update(value=pd.DataFrame()), (questions_state or []), gr.update()

def run_agent(selection_df: t.Any = None):
    # Expect a pandas DataFrame with columns: Selected (bool), Task ID, Question
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
    results_log = []
    answers_payload = []
    for _, row in df.iterrows():
        task_id = row.get("Task ID")
        question_text = row.get("Question")
        file_url = row.get("File URL") or ""
        if not task_id or question_text is None:
            continue
        try:
            # Include attachment info so the graph can use URL-aware tools
            user_content = question_text
            if file_url:
                user_content += f"\n\nATTACHMENT_URL: {file_url}"

            submitted_answer = agent_graph.invoke(
                {"messages": [HumanMessage(content=user_content)]},
                config={"recursion_limit": 8}
            )["messages"][-1].content
        except Exception as e:
            submitted_answer = f"AGENT ERROR: {e}"
        answers_payload.append({"task_id": task_id, "submitted_answer": submitted_answer})
        results_log.append({"Task ID": task_id, "Question": question_text, "Submitted Answer": submitted_answer})
    if not answers_payload:
        return "No answers produced.", pd.DataFrame(results_log), []
    return f"Ran agent on {len(answers_payload)} questions. Review answers below.", pd.DataFrame(results_log), answers_payload

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

def submit_answers(profile: gr.OAuthProfile | None, questions_state: t.Optional[list] = None):
    if not profile:
        return "Please Login to Hugging Face with the button."
    username = f"{profile.username}".strip()
    questions = questions_state or []
    if not questions:
        return "No questions cached. Click 'Fetch Questions' first."
    
    answers_payload = []
    for item in questions:
        task_id = item.get("task_id")
        question_text = item.get("question")
        file_name = item.get("file_name") or ""
        if not task_id or question_text is None:
            continue
        try:
            # Include attachment info so the agent can download GAIA files
            user_content = question_text
            if file_name:
                user_content += f"\n\nAttached file: {file_name}"

            submitted_answer = agent_graph.invoke(
                {"messages": [HumanMessage(content=user_content)]},
                config={"recursion_limit": 8}
            )["messages"][-1].content
        except Exception as e:
            submitted_answer = f"AGENT ERROR: {e}"
        answers_payload.append({"task_id": task_id, "submitted_answer": submitted_answer})

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

def run_and_submit_all( profile: gr.OAuthProfile | None):
    """
    Fetches all questions, runs the agent graph on them, submits all answers,
    and displays the results.
    """
    # --- Determine HF Space Runtime URL and Repo URL ---
    space_id = os.getenv("SPACE_ID") # Get the SPACE_ID for sending link to the code

    if profile:
        username = f"{profile.username}"
        print(f"User logged in: {username}")
    else:
        print("User not logged in.")
        return "Please Login to Hugging Face with the button.", None

    api_url = DEFAULT_API_URL
    questions_url = f"{api_url}/questions"
    submit_url = f"{api_url}/submit"

    # In the case of an app running as a hugging Face space, this link points toward your codebase
    agent_code = f"https://huggingface.co/spaces/{space_id}/tree/main" if space_id else None
    if agent_code:
        print(agent_code)

    # 2. Fetch Questions
    print(f"Fetching questions from: {questions_url}")
    try:
        response = requests.get(questions_url, timeout=15)
        response.raise_for_status()
        questions_data = response.json()
        if not questions_data:
             print("Fetched questions list is empty.")
             return "Fetched questions list is empty or invalid format.", None
        print(f"Fetched {len(questions_data)} questions.")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching questions: {e}")
        return f"Error fetching questions: {e}", None
    except requests.exceptions.JSONDecodeError as e:
         print(f"Error decoding JSON response from questions endpoint: {e}")
         print(f"Response text: {response.text[:500]}")
         return f"Error decoding server response for questions: {e}", None
    except Exception as e:
        print(f"An unexpected error occurred fetching questions: {e}")
        return f"An unexpected error occurred fetching questions: {e}", None

    # 3. Run your Agent
    results_log = []
    answers_payload = []
    print(f"Running agent on {len(questions_data)} questions...")
    for item in questions_data:
        task_id = item.get("task_id")
        question_text = item.get("question")
        file_name = item.get("file_name") or ""
        if not task_id or question_text is None:
            print(f"Skipping item with missing task_id or question: {item}")
            continue
        try:
            # Include attachment info so the agent can download GAIA files
            user_content = question_text
            if file_name:
                user_content += f"\n\nAttached file: {file_name}"

            submitted_answer = agent_graph.invoke(
                {"messages": [HumanMessage(content=user_content)]},
                config={"recursion_limit": 8}
            )["messages"][-1].content
            
            answers_payload.append({"task_id": task_id, "submitted_answer": submitted_answer})
            results_log.append({"Task ID": task_id, "Question": question_text, "Submitted Answer": submitted_answer})
        except Exception as e:
             print(f"Error running agent on task {task_id}: {e}")
             results_log.append({"Task ID": task_id, "Question": question_text, "Submitted Answer": f"AGENT ERROR: {e}"})

    if not answers_payload:
        print("Agent did not produce any answers to submit.")
        return "Agent did not produce any answers to submit.", pd.DataFrame(results_log)

    # 4. Prepare Submission 
    submission_data = {"username": username.strip(), "agent_code": agent_code, "answers": answers_payload}
    status_update = f"Agent finished. Submitting {len(answers_payload)} answers for user '{username}'..."
    print(status_update)

    # 5. Submit
    print(f"Submitting {len(answers_payload)} answers to: {submit_url}")
    try:
        response = requests.post(submit_url, json=submission_data, timeout=60)
        response.raise_for_status()
        result_data = response.json()
        final_status = (
            f"Submission Successful!\n"
            f"User: {result_data.get('username')}\n"
            f"Overall Score: {result_data.get('score', 'N/A')}% "
            f"({result_data.get('correct_count', '?')}/{result_data.get('total_attempted', '?')} correct)\n"
            f"Message: {result_data.get('message', 'No message received.')}"
        )
        print("Submission successful.")
        results_df = pd.DataFrame(results_log)
        return final_status, results_df
    except requests.exceptions.HTTPError as e:
        error_detail = f"Server responded with status {e.response.status_code}."
        try:
            error_json = e.response.json()
            error_detail += f" Detail: {error_json.get('detail', e.response.text)}"
        except requests.exceptions.JSONDecodeError:
            error_detail += f" Response: {e.response.text[:500]}"
        status_message = f"Submission Failed: {error_detail}"
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df
    except requests.exceptions.Timeout:
        status_message = "Submission Failed: The request timed out."
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df
    except requests.exceptions.RequestException as e:
        status_message = f"Submission Failed: Network error - {e}"
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df
    except Exception as e:
        status_message = f"An unexpected error occurred during submission: {e}"
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df


# --- Build Gradio Interface using Blocks ---
with gr.Blocks() as demo:
    gr.Markdown("# Basic Agent Evaluation Runner")
    gr.Markdown(
        """
        **Instructions:**

        1.  Please clone this space, then modify the code to define your agent's logic, the tools, the necessary packages, etc ...
        2.  Log in to your Hugging Face account using the button below. This uses your HF username for submission.
        3.  Use the buttons below:
            - Fetch Questions: pulls questions and displays them.
            - Run Agent (no submission): runs locally on cached questions and shows answers.
            - Submit Answers for Scoring: submits the last-run answers to the scoring API.

        ---
        **Disclaimers:**
        Once clicking on the "submit button, it can take quite some time ( this is the time for the agent to go through all the questions).
        This space provides a basic setup and is intentionally sub-optimal to encourage you to develop your own, more robust solution. For instance for the delay process of the submit button, a solution could be to cache the answers and submit in a seperate action or even to answer the questions in async.
        """
    )

    gr.LoginButton()

    # Shared state
    questions_state = gr.State([])
    answers_state = gr.State([])

    # Buttons & controls
    fetch_button = gr.Button("Fetch Questions")
    run_button = gr.Button("Run Agent (no submission)")
    submit_button = gr.Button("Submit Answers for Scoring")
    refresh_cb = gr.Checkbox(label="Refresh questions from API", value=False)
    select_all_cb = gr.Checkbox(label="Select all questions", value=True)

    status_output = gr.Textbox(label="Status", lines=5, interactive=False)
    questions_table = gr.Dataframe(
        label="Fetched Questions (toggle 'Selected' to choose)",
        headers=["Selected", "Task ID", "Question"],
        datatype=["bool", "number", "str"],
        interactive=True,
    )
    answers_table = gr.DataFrame(label="Agent Answers", wrap=True)

    # Wire up actions
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
        inputs=[questions_state],
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