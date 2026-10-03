"""数据库会话选择和页面历史重建。"""

import streamlit as st

from services.conversation import conversation_turns, thread_config


def new_conversation(repository):
    thread_id = repository.create()
    st.query_params["thread"] = thread_id
    st.session_state.pop("conversation_search", None)
    _activate_conversation(thread_id)
    st.rerun()


def _activate_conversation(thread_id):
    """切换时清理上一会话的临时执行标记；历史随后从 checkpoint 重建。"""
    st.session_state["thread_id"] = thread_id
    st.query_params["thread"] = thread_id
    st.session_state["pending_index"] = None
    st.session_state["is_processing"] = False
    st.session_state["stop_execution"] = False
    st.session_state["resume_execution"] = False
    st.session_state.pop("last_execution_error", None)
    st.session_state["input_key"] += 1


def select_conversation(repository, navigation=None):
    rows = repository.list_threads()
    if not rows:
        new_conversation(repository)
    ids = [row["thread_id"] for row in rows]
    current = st.session_state.get("thread_id")
    selected = st.query_params.get("thread") or current
    busy = st.session_state["is_processing"]
    if busy and current in ids:
        selected = current
    if selected not in ids:
        selected = ids[0]
    if current != selected:
        _activate_conversation(selected)
    selected_row = next(row for row in rows if row["thread_id"] == selected)

    with navigation if navigation is not None else st.sidebar:
        st.subheader("💬 会话列表")
        new, refresh = st.columns([3, 1])
        with new:
            if st.button("➕ 新会话", key="new_conversation", use_container_width=True, disabled=busy):
                new_conversation(repository)
        with refresh:
            if st.button("刷新", key="refresh_conversations", use_container_width=True, disabled=busy):
                st.rerun()
        search = st.text_input("搜索会话", key="conversation_search", placeholder="按标题查找", disabled=busy)
        filtered = [row for row in rows if search.strip().casefold() in row["title"].casefold()]
        st.caption(f"共 {len(rows)} 个会话 · 按最近活动排序")
        if not filtered:
            st.info("没有匹配的会话，请换个关键词。")
        else:
            with st.container(height=320):
                for row in filtered:
                    title = " ".join(row["title"].split()) or "新会话"
                    label = title[:40] + ("…" if len(title) > 40 else "")
                    if st.button(label, key=f"conversation_{row['thread_id']}",
                                 type="primary" if row["thread_id"] == selected else "secondary",
                                 help=title, use_container_width=True, disabled=busy):
                        if row["thread_id"] != selected:
                            _activate_conversation(row["thread_id"])
                            st.rerun()
                    if updated := row.get("updated_at"):
                        st.caption(f"最近活动：{updated:%m-%d %H:%M}")
        st.caption(f"当前会话：{selected_row['title']}")
        with st.expander("会话信息"):
            st.text(f"ID：{selected}")
            st.caption("会话已保存到本地 MySQL。当前共用会话列表，暂未区分用户。")
        st.divider()
    st.query_params["thread"] = selected
    return selected


def restore_conversation(graph, thread_id, history):
    """页面列表只是数据库历史的投影，不能作为持久化模式下的模型输入来源。"""
    snapshot = graph.get_state(thread_config(thread_id))
    if st.session_state["pending_index"] is None:
        messages = list(snapshot.values.get("messages", []))
        history.clear()
        history.add_messages(messages)
        turns = conversation_turns(messages)
        st.session_state["user_query_history"] = [t["question"] for t in turns]
        st.session_state["response_history"] = [t["reply"] or "该轮任务尚未完成。" for t in turns]
        st.session_state["agent_sequence_history"] = [t["sequence"] for t in turns]
        st.session_state["agent_results_history"] = [t["results"] for t in turns]
    return snapshot


def render_recovery(snapshot):
    """恢复是 graph.stream(None)，不是再发送一遍原问题。"""
    if not snapshot.next or st.session_state["is_processing"]:
        return
    st.warning("本会话有未完成任务。可恢复执行，或新建会话；失败节点可能重新调用模型或工具。")
    if st.button("▶ 恢复未完成任务", key="resume_thread"):
        if not st.session_state["user_query_history"]:
            st.error("未找到对应的用户问题，请新建会话。")
            return
        st.session_state["pending_index"] = len(st.session_state["user_query_history"]) - 1
        st.session_state["is_processing"] = True
        st.session_state["stop_execution"] = False
        st.session_state["resume_execution"] = True
        st.rerun()
