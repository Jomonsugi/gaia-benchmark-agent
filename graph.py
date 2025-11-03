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
    read_excel_file_tool,
    transcribe_audio_tool,
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
        read_excel_file_tool,
        transcribe_audio_tool,
    ]
    chat_with_tools = chat.bind_tools(tools)

    sys_msg = SystemMessage(content="""
You are a helpful assistant tasked with answering questions. The questions are complex and require the use of various tools to answer the questions.

When a question mentions an attached file (look for "Attached file: <filename>"), follow these steps in order:
1. First, you MUST use the download_gaia_attachment with the file name to get the local path
2. Then use a tool to read the file contents based on the file type ONCE - the full file contents will be returned:
   - For text files (.py, .txt, .md, .json, .csv): use read_text_file
   - For Excel files (.xlsx, .xls): use ExcelReader
   - For audio files (.mp3, .wav, .m4a): use transcribe_audio
3. Use the file contents from step 2 to answer the question - DO NOT re-read the same file

IMPORTANT: Only read each file ONCE. The tool output contains all the data you need.

The assistant should only use one tool at a time. The reponse from the tool will indicate the next tool to use, or if the question is answered, the assistant should return the final answer.
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