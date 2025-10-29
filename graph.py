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
    download_gaia_attachment_tool,
    read_text_file_tool,
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
        download_gaia_attachment_tool,
        read_text_file_tool,
    ]
    chat_with_tools = chat.bind_tools(tools)

    sys_msg = SystemMessage(content="""
You are a helpful assistant tasked with answering questions. The questions are complex and require the use of tools to answer.

When a question mentions an attached file (look for "Attached file: <filename>"):
1. First use download_gaia_attachment with the file name to get the local path
2. Then use read_text_file with that path to read the file contents (for .py, .txt, .json, .csv, .md files)
3. Use the file contents to answer the question

After using the applicable tool(s), return only the final answer text (no extra commentary).
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