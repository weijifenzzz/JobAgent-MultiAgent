"""长期记忆面板：人工录入不调用模型，与聊天命令共用 Store 服务。"""

import streamlit as st
from pydantic import ValidationError

from services.memory import KINDS, MemoryEntry, delete_memory, key_for_label, list_memories, save_memory


def render_memory_panel(store, user_id, *, disabled=False, container=None):
    records = list_memories(store, user_id)
    with container if container is not None else st.sidebar:
        with st.expander(f"🧠 长期记忆（{len(records)}）", expanded=False):
            st.caption("偏好和事实在所有会话中共用，重启后仍保留。当前使用本地身份 local_user。")
            st.caption("聊天中可说：请记住：我的目标城市是上海；查看我的记忆；请忘记我的目标城市。")
            if notice := st.session_state.pop("memory_notice", None):
                st.success(notice)
            if not records:
                st.info("还没有长期记忆，可以在下面添加。")
            kind = st.selectbox("记忆类别", list(KINDS), format_func=KINDS.get,
                                key="memory_kind", disabled=disabled)
            entries = {r["key"]: r for r in records if r["kind"] == kind}
            selected = st.selectbox("选择记忆", ["__new__", *entries], key=f"memory_selection_{kind}",
                                    format_func=lambda key: "＋ 添加一条记忆" if key == "__new__" else entries[key]["label"],
                                    disabled=disabled)
            current = entries.get(selected, {})
            with st.form(f"memory_editor_{kind}_{selected}"):
                label = st.text_input("名称", value=current.get("label", ""), max_chars=60,
                                      placeholder="例如：目标城市、技能", disabled=disabled)
                content = st.text_area("内容", value=current.get("content", ""), max_chars=1000,
                                       placeholder="例如：希望在上海工作", disabled=disabled)
                save = st.form_submit_button("保存记忆", disabled=disabled)
            if save:
                try:
                    entry = MemoryEntry(kind=kind, key=selected if current else key_for_label(label),
                                        label=label, content=content)
                except ValidationError:
                    st.error("请输入有效的名称和内容。")
                else:
                    save_memory(store, user_id, entry, source="manual")
                    st.session_state["memory_notice"] = "记忆已保存，其他会话也能使用。"
                    st.rerun()
            if current:
                if st.button("删除这条记忆", key=f"delete_memory_{kind}_{selected}", disabled=disabled):
                    delete_memory(store, user_id, kind, selected)
                    st.session_state.pop(f"memory_selection_{kind}", None)
                    st.session_state["memory_notice"] = "已从长期记忆删除；原聊天和检查点仍保留。"
                    st.rerun()
