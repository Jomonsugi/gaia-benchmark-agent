import os
import re
from typing import TypedDict, Annotated

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage, AIMessage
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

from tools import (
    wikipedia_search_tool,
    tavily_web_search_tool,
    youtube_transcript_tool,
    web_fetch_page_tool,
    interactive_web_browse_tool,
    youtube_video_visual_qa_tool,
    read_text_file_tool,
    read_excel_file_tool,
    transcribe_audio_tool,
    chess_best_move_from_image_tool,
    vision_qa_tool,
    download_file_attachment,
)

hf_token = os.getenv("HUGGING_FACE_HUB_TOKEN")


class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    attachment_file_name: str | None
    attachment_local_path: str | None


def load_llm():
    llm = HuggingFaceEndpoint(
        # repo_id="Qwen/Qwen2.5-Coder-32B-Instruct",
        # repo_id="Qwen/Qwen3-VL-235B-A22B-Instruct",
        repo_id="Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8",
        # repo_id="deepseek-ai/DeepSeek-R1-0528",
        huggingfacehub_api_token=hf_token,
    )
    return ChatHuggingFace(llm=llm, verbose=True)


def create_graph():
    chat = load_llm()
    tools = [
        wikipedia_search_tool,
        tavily_web_search_tool,
        youtube_transcript_tool,
        web_fetch_page_tool,
        interactive_web_browse_tool,
        youtube_video_visual_qa_tool,
        read_text_file_tool,
        read_excel_file_tool,
        transcribe_audio_tool,
        chess_best_move_from_image_tool,
        vision_qa_tool,
    ]
    chat_with_tools = chat.bind_tools(tools)

    sys_msg = SystemMessage(content="""
You are a general AI assistant. I will ask you a question. Report your thoughts, and
finish your answer with the following template: FINAL ANSWER: [YOUR FINAL ANSWER].
YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma separated
list of numbers and/or strings.
If you are asked for a number, don’t use comma to write your number neither use units such as $ or
percent sign unless specified otherwise.
If you are asked for a string, don’t use articles, neither abbreviations (e.g. for cities), and write the
digits in plain text unless specified otherwise.
If you are asked for a comma separated list, apply the above rules depending of whether the element
to be put in the list is a number or a string.
""")

    def prepare_attachment(state: AgentState):
        file_name = state.get("attachment_file_name")
        if not file_name:
            return {}

        file_path = download_file_attachment(file_name)

        if not isinstance(file_path, str) or file_path.startswith("ERROR"):
            message = (
                f"Attachment download failed for '{file_name}'. "
                f"Details: {file_path}"
            )
            return {
                "attachment_local_path": None,
                "messages": [SystemMessage(content=message)],
            }

        info_message = SystemMessage(
            content=(
                f"Attachment '{file_name}' has been downloaded to: {file_path}.\n"
                "Use the appropriate file-processing tool (read_text_file, ExcelReader, "
                "transcribe_audio, chess_best_move_from_image) with this path."
            )
        )

        return {
            "attachment_local_path": file_path,
            "messages": [info_message],
        }

    def assistant(state: AgentState):
        msgs = state.get("messages", [])
        if msgs and isinstance(msgs[0], str):
            msgs = [HumanMessage(content=msgs[0])]
        out = chat_with_tools.invoke([sys_msg] + msgs)
        return {"messages": [out]}

    def extract_final_answer(content: str) -> str:
        text = content if isinstance(content, str) else str(content)
        marker = "FINAL ANSWER:"
        idx = text.upper().rfind(marker)
        if idx != -1:
            return text[idx + len(marker):].strip()
        return text.strip()

    def finalize(state: AgentState):
        messages = state.get("messages", [])
        if not messages:
            return {}
        last = messages[-1]
        if isinstance(last, AIMessage):
            final_text = extract_final_answer(last.content)
        else:
            final_text = extract_final_answer(getattr(last, "content", last))

        final_message = AIMessage(content=final_text)
        return {"messages": messages[:-1] + [final_message]}

    builder = StateGraph(AgentState)
    builder.add_node("prepare_attachment", prepare_attachment)
    builder.add_node("assistant", assistant)
    builder.add_node("finalize", finalize)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "prepare_attachment")
    builder.add_edge("prepare_attachment", "assistant")
    builder.add_conditional_edges("assistant", tools_condition, {"tools": "tools", END: "finalize"})
    builder.add_edge("tools", "assistant")
    builder.add_edge("finalize", END)
    return builder.compile()


# Compiled graph to be imported by app and Studio
graph = create_graph()