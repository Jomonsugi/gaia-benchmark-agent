import os
from typing import TypedDict, Annotated

from langgraph.graph import StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

from tools import (
    basic_search_tool,
    wikipedia_search_tool,
    tavily_search_tool,
    youtube_transcript_tool,
    download_file_attachment_tool,
    read_text_file_tool,
    read_excel_file_tool,
    transcribe_audio_tool,
    chess_best_move_from_image_tool,
)

hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


def load_llm():
    llm = HuggingFaceEndpoint(
        repo_id="Qwen/Qwen2.5-Coder-32B-Instruct",
        huggingfacehub_api_token=hf_token,
    )
    return ChatHuggingFace(llm=llm, verbose=True)


def create_graph():
    chat = load_llm()
    tools = [
        basic_search_tool,
        wikipedia_search_tool,
        tavily_search_tool,
        youtube_transcript_tool,
        download_file_attachment_tool,
        read_text_file_tool,
        read_excel_file_tool,
        transcribe_audio_tool,
        chess_best_move_from_image_tool,
    ]
    chat_with_tools = chat.bind_tools(tools)

    sys_msg = SystemMessage(content="""
You are a tool-using assistant that must follow the ReAct protocol (Reason + Act) for every turn. Maintain a tight Observe → Think → Act → Reflect loop and never skip steps.

Observe:
- Carefully read the latest user message and any tool results.
- Identify whether the user referenced "Attached file: <filename>".

Think (Plan):
- Briefly outline the single next step you will take. Do NOT plan multiple actions at once.
- If the question mentions an attachment, your plan MUST follow this order:
  1. Call download_file_attachment with the provided file name to obtain the local path.
  2. Call exactly one file-processing tool based on the extension (read_text_file, ExcelReader, transcribe_audio, or chess_best_move_from_image). Each file should be read only once.
  3. Reason over the returned content to produce the answer.
- If there is no "Attached file:" string, do NOT call download_file_attachment.

Act:
- If you need information, issue exactly ONE tool call in the format the tooling layer expects.
- Wait for the tool result before thinking or acting again. Never request multiple tools in the same turn.

Reflect:
- Update your understanding using the tool output.
- Once you have everything required, respond with the final answer. Match the exact output format requested in the question and return only the final answer text.

General Rules:
- Do not reread the same file or re-download attachments.
- Obey answer-format instructions precisely (comma-separated lists, single numbers, alphabetical order, etc.).

CRITICAL: Each question defines the expected format for the answer. Your final answer should be in the correct format. Only return the final answer text, no extra commentary.
""")

    def assistant(state: AgentState):
        msgs = state.get("messages", [])
        if msgs and isinstance(msgs[0], str):
            msgs = [HumanMessage(content=msgs[0])]
        out = chat_with_tools.invoke([sys_msg] + msgs)
        return {"messages": [out]}

    builder = StateGraph(AgentState)
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "assistant")
    builder.add_conditional_edges("assistant", tools_condition)
    builder.add_edge("tools", "assistant")
    return builder.compile()


# Compiled graph to be imported by app and Studio
graph = create_graph()