"""内容寻址的简历文件和会话绑定；不依赖 Streamlit，不让模型选择路径。"""

import hashlib
import os
import re
import tempfile
from pathlib import Path

from settings import TEMP_DIR
from services.conversation import thread_config

RESUME_DIR = TEMP_DIR / "resumes"


def resume_path(resume_id: str) -> Path:
    if not isinstance(resume_id, str) or not re.fullmatch(r"[0-9a-f]{64}", resume_id):
        raise ValueError("简历 ID 无效，请重新上传并绑定简历。")
    root = RESUME_DIR.resolve()
    path = (root / f"{resume_id}.pdf").resolve()
    if path.parent != root:
        raise ValueError("简历文件路径无效。")
    return path


def read_resume_bytes(resume_id: str) -> bytes:
    path = resume_path(resume_id)
    if not path.is_file():
        raise FileNotFoundError("当前会话绑定的简历文件已丢失，请重新上传原文件或绑定另一份简历。")
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != resume_id:
        raise ValueError("简历文件内容与 ID 不一致，请重新上传原文件修复。")
    return content


def save_resume(content: bytes) -> str:
    """相同 PDF 字节得到相同 ID；临时写入后原子替换，避免读到半个文件。"""
    import pymupdf

    if not content:
        raise ValueError("简历文件为空。")
    try:
        with pymupdf.open(stream=content, filetype="pdf") as document:
            if not document.is_pdf or document.needs_pass or document.page_count == 0:
                raise ValueError("请上传可直接打开、非加密且包含页面的 PDF。")
    except (RuntimeError, ValueError) as exc:
        raise ValueError("无法读取该 PDF，请确认文件有效且未加密。") from exc
    resume_id = hashlib.sha256(content).hexdigest()
    path = resume_path(resume_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.read_bytes() == content:
        return resume_id
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as file:
            temporary = Path(file.name)
            file.write(content)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return resume_id


def bind_resume(graph, thread_id, resume_id, filename, *, expected_resume_id=None):
    """调用方持有会话锁；绑定立即进入 checkpoint，不必等用户发消息。"""
    read_resume_bytes(resume_id)
    config = thread_config(thread_id)
    snapshot = graph.get_state(config)
    current = snapshot.values.get("resume_id")
    if current != expected_resume_id:
        raise ValueError("简历绑定已在另一个页面改变，请刷新后重试。")
    if current == resume_id:
        return False
    if snapshot.next:
        raise ValueError("会话有未完成任务，不能更换简历。请先恢复任务，或新建会话。")
    filename = Path(filename.replace("\\", "/")).name[:200] or "resume.pdf"
    graph.update_state(config, {
        "resume_id": resume_id,
        "resume_filename": filename,
        # 保留完整聊天用于页面展示；新简历不再使用切换前的模型上下文。
        "resume_context_start": len(snapshot.values.get("messages", [])),
        "next_step": "", "needs_followup": "", "task_completed": True,
    }, as_node="ResumeBinding")
    return True


def model_messages(state):
    """旧 checkpoint 没有边界字段时保持原历史；只在明确换绑后隔离旧内容。"""
    messages = state.get("messages", [])
    start = state.get("resume_context_start", 0)
    return list(messages[start:])
