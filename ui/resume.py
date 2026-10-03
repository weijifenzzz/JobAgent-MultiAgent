"""当前会话的简历面板；会话选定后才读取和绑定，不再覆盖全局 PDF。"""

import streamlit as st

from services.resumes import bind_resume, read_resume_bytes, save_resume
from settings import PROJECT_ROOT


def _save_and_bind(graph, repository, thread_id, content, filename, expected_resume_id):
    with repository.lock(thread_id):
        resume_id = save_resume(content)
        changed = bind_resume(graph, thread_id, resume_id, filename,
                              expected_resume_id=expected_resume_id)
        repository.touch(thread_id)
    return changed


def render_resume_panel(graph, repository, thread_id, snapshot, *, container=None):
    resume_id = snapshot.values.get("resume_id")
    filename = snapshot.values.get("resume_filename", "简历.pdf")
    available = False
    with container if container is not None else st.sidebar:
        st.subheader("📄 当前会话的简历")
        if notice := st.session_state.pop("resume_notice", None):
            st.success(notice)
        if resume_id:
            try:
                read_resume_bytes(resume_id)
                available = True
                st.success(f"已绑定：{filename}")
            except (OSError, ValueError) as exc:
                st.error(str(exc))
            st.caption(f"简历 ID：{resume_id[:12]}…")
            if snapshot.values.get("resume_context_start", 0):
                st.caption("已隔离更换简历前的模型上下文；旧聊天仍可查看，请重新分析当前简历。")
        else:
            st.info("本会话尚未绑定简历。上传 PDF 或选择演示简历。")
            if snapshot.values.get("messages"):
                st.caption("旧会话没有简历 ID，无法确认当时使用哪份文件，请手动重新绑定。")
        if snapshot.next:
            st.caption("有未完成任务时不能更换简历。请先恢复执行，或新建会话。")
            if resume_id and not available:
                st.caption("若原文件丢失，可重新上传完全相同的 PDF 修复；不同内容仍不能绑定。")
        disabled = st.session_state["is_processing"] or (
            bool(snapshot.next) and (not resume_id or available)
        )
        st.caption("更换后保留旧聊天，但后续模型只使用更换后的对话和当前简历；长期记忆继续共用。")
        # 每段会话、每次绑定有独立上传控件，避免另一个会话遗留的上传值自动覆盖绑定。
        widget_id = f"{thread_id}_{resume_id or 'unbound'}"
        with st.form(f"resume_form_{widget_id}"):
            uploaded = st.file_uploader("上传简历（PDF格式）", type="pdf",
                                        key=f"resume_upload_{widget_id}", disabled=disabled)
            submitted = st.form_submit_button("绑定到当前会话", disabled=disabled)
        content = name = None
        if submitted:
            if uploaded is None:
                st.warning("请先选择 PDF 文件。")
            else:
                content, name = uploaded.getvalue(), uploaded.name
        if not resume_id:
            demo = PROJECT_ROOT / "dummy_resume.pdf"
            if demo.is_file() and st.button("使用演示简历", key=f"demo_resume_{thread_id}", disabled=disabled):
                content, name = demo.read_bytes(), demo.name
        if content is not None:
            try:
                changed = _save_and_bind(graph, repository, thread_id, content, name, resume_id)
            except (OSError, ValueError, RuntimeError) as exc:
                st.error(f"简历绑定失败：{exc}")
            else:
                st.session_state["resume_notice"] = (
                    "已绑定到当前会话。更换前的聊天保留，后续请重新分析当前简历。"
                    if changed else "该文件与当前简历相同，绑定和模型上下文保持不变。"
                )
                st.rerun()
    return available
