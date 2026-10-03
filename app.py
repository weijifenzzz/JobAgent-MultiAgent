"""Streamlit 入口：连接配置、页面、会话与对话执行服务。"""

import traceback
from contextlib import closing

import pymysql
import streamlit as st
import streamlit_analytics2 as streamlit_analytics
from langchain_community.chat_message_histories import StreamlitChatMessageHistory

from services.conversation import stream_conversation
from persistence import open_database
from runtime_context import LOCAL_USER_ID
from settings import initialize_environment
from ui.chat import StreamingReply, configure_page, render_header, render_history, render_input
from ui.session import (
    STOPPED, finish_request, get_pending_request,
    initialize_session, start_request, stop_request,
)
from ui.sidebar import read_streamlit_secret, render_sidebar
from ui.conversations import select_conversation, restore_conversation, render_recovery
from ui.memory import render_memory_panel
from ui.resume import render_resume_panel
from services.memory import memory_intent


def execute_pending_request(graph, model_config, message_history, database):
    """页面负责加载提示、会话写回和错误展示；执行服务只运行图。"""
    user_input = get_pending_request(st.session_state)
    if user_input is None:
        return
    if st.session_state["stop_execution"]:
        stop_request(st.session_state)
        st.rerun()

    try:
        preview = StreamingReply()
        result = None
        thread_id = st.session_state["thread_id"]
        with database.threads.lock(thread_id), st.spinner("🤔 AI 正在思考中..."):
            database.threads.touch(thread_id, user_input)
            with closing(stream_conversation(
                graph,
                user_input,
                message_history.messages,
                model_config,
                thread_id=thread_id,
                resume=st.session_state["resume_execution"],
            )) as events:
                for event in events:
                    if st.session_state["stop_execution"]:
                        break
                    preview.update(event)
                    if event.type == "done":
                        result = event.result
            database.threads.touch(thread_id)
        if st.session_state["stop_execution"]:
            finish_request(st.session_state, STOPPED)
        else:
            if result is None:
                raise RuntimeError("对话流未正常完成，请重试。")
            message_history.clear()
            message_history.add_messages(result.messages)
            finish_request(
                st.session_state, result.reply, result.agent_sequence, result.agent_results
            )
    except Exception as exc:
        traceback.print_exc()
        st.error(f"执行错误: {exc}")
        reply = STOPPED if st.session_state["stop_execution"] else ":( 抱歉，发生了一些错误。请重试。"
        finish_request(st.session_state, reply)
        st.session_state["last_execution_error"] = "本轮执行失败，已保存的进度仍在数据库中。请检查终端错误并使用恢复按钮，或新建会话。"
    st.rerun()


def main():
    configure_page()
    initialize_environment(read_streamlit_secret)
    initialize_session(st.session_state)
    # 先预留顶部区域，数据库连接完成后再填充，避免会话列表藏在配置下方。
    navigation = st.sidebar.container()
    resume_area = st.sidebar.container()
    sidebar = render_sidebar()
    render_header()
    try:
        with open_database() as database:
            run_page(database, sidebar, navigation, resume_area)
    except (pymysql.MySQLError, ValueError):
        st.error("MySQL 会话存储不可用。请检查 .env 中的 MYSQL_* 配置、MySQL 服务，并运行 python -m scripts.init_mysql 初始化。")


def run_page(database, sidebar, navigation, resume_area):
    # 环境配置加载后再导入图模块，保持工具读取配置的顺序。
    from agents import define_graph

    graph = define_graph(checkpointer=database.checkpointer, store=database.store)
    history = StreamlitChatMessageHistory()
    thread_id = select_conversation(database.threads, navigation)
    snapshot = restore_conversation(graph, thread_id, history)
    has_resume = render_resume_panel(graph, database.threads, thread_id, snapshot,
                                     container=resume_area)
    render_memory_panel(database.store, LOCAL_USER_ID,
                        disabled=st.session_state["is_processing"], container=navigation)
    if error := st.session_state.pop("last_execution_error", None):
        st.error(error)
    render_recovery(snapshot)
    conversation_container = st.container()
    input_container = st.container()

    # 与原页面一样，仅统计聊天区域，不统计侧边栏密钥输入。
    streamlit_analytics.start_tracking()
    try:
        with input_container:
            action = render_input(st.session_state["is_processing"], st.session_state["input_key"],
                                  blocked=bool(snapshot.next))
            if action:
                kind, text = action
                if kind == "stop":
                    stop_request(st.session_state)
                    st.rerun()
                if not has_resume and memory_intent(text) is None:
                    st.error("❌ 请先为当前会话绑定可用简历，然后再提交查询。")
                elif not sidebar.model_config["OPENAI_API_KEY"] or not sidebar.model_config["OPENAI_BASE_URL"]:
                    st.error("❌ 请先配置模型 API Key 和 Base URL。")
                else:
                    start_request(st.session_state, text)
                    st.rerun()

        with conversation_container:
            render_history(
                st.session_state["user_query_history"],
                st.session_state["response_history"],
                st.session_state["agent_sequence_history"],
                st.session_state["agent_results_history"],
                st.session_state["pending_index"],
            )
            execute_pending_request(graph, sidebar.model_config, history, database)
    finally:
        streamlit_analytics.stop_tracking()


if __name__ == "__main__":
    main()
