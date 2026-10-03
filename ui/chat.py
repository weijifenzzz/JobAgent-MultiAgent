"""聊天页面组件；只绘制界面、返回用户动作，不调用 Agent。"""

import streamlit as st
from streamlit_chat import message

QUICK_QUESTIONS = [
    ("📝", "总结我的简历"),
    ("🔍", "搜索 GenAI 相关岗位"),
    ("💼", "分析我的简历并推荐合适岗位"),
    ("✉️", "生成求职信"),
    ("🌐", "识别 GenAI 相关的科技行业最新趋势"),
    ("🔍", "查找新兴技术及其对岗位机会的影响"),
    ("📈", "根据我的简历生成职业路径可视化"),
    ("🏢", "阿里的 GenAI 相关岗位"),
]


def configure_page():
    st.set_page_config(
        page_title="JobAgent-MultiAgent 职业助手", page_icon="👨‍💼",
        layout="wide", initial_sidebar_state="expanded",
    )
    st.markdown("""
    <style>
        /* 主标题样式 */
        h1 {
            color: #1f77b4;
            font-weight: 600;
        }

        /* 侧边栏样式 */
        [data-testid="stSidebar"] {
            background-color: #f8f9fa;
        }

        /* 按钮样式优化 */
        .stButton>button {
            border-radius: 8px;
            font-weight: 500;
        }

        /* 输入框样式 */
        .stTextInput>div>div>input,
        .stTextArea>div>div>textarea {
            border-radius: 8px;
            font-size: 14px;
        }

        /* 文件上传器样式 */
        [data-testid="stFileUploader"] {
            border-radius: 8px;
            border: 2px dashed #ccc;
            padding: 10px;
        }

        /* 成功/警告/错误消息样式 */
        .stSuccess, .stWarning, .stError, .stInfo {
            border-radius: 8px;
            padding: 10px;
            font-size: 14px;
        }

        /* Pills 样式优化 */
        .stPills {
            margin-top: 10px;
        }

        /* 聊天消息容器 - 缩小字体 */
        [data-testid="stChatMessageContent"] {
            border-radius: 10px;
            font-size: 14px;
        }

        /* 聊天消息内容 */
        .stChatMessage {
            font-size: 14px;
            background-color: #f0f2f6;
            border-radius: 12px;
        }

        /* Markdown 内容字体 */
        .stMarkdown {
            font-size: 14px;
        }

        /* 减小标题字体 */
        h5 {
            font-size: 16px;
        }
    </style>
    """, unsafe_allow_html=True)


def render_header():
    st.title("JobAgent-MultiAgent 职业助手 👨‍💼")
    st.markdown("""
    <div style='background-color: #f0f2f6; padding: 15px; border-radius: 10px; margin-bottom: 20px;'>
        <p style='margin: 0; color: #31333F;'>
            🚀 基于 LangGraph 的多智能体求职助手系统<br>
            💡 支持简历分析、岗位搜索、求职信生成、公司调研等功能<br>
            📧 <a href='http://mail.qq.com/cgi-bin/qm_share?t=qm_mailme&email=HC4lKS4sKyovLSpcbW0yf3Nx' target='_blank'>联系作者</a>
        </p>
    </div>
    """, unsafe_allow_html=True)


def render_clear_button(label="🗑️ 清除聊天"):
    _, controls = st.columns([6, 1])
    with controls:
        return st.button(label, use_container_width=True)


def render_input(is_processing, input_key, *, blocked=False):
    """返回 (动作, 内容)，让入口决定如何处理；不会自行调用 rerun。"""
    st.markdown("##### 💡 快捷问题")
    columns = st.columns(4)
    for index, (icon, question) in enumerate(QUICK_QUESTIONS):
        with columns[index % 4]:
            if st.button(
                f"{icon} {question}", key=f"quick_{index}",
                use_container_width=True, disabled=is_processing or blocked,
            ):
                return "submit", question

    if is_processing:
        st.chat_input(
            "💬 处理中，请稍候...", key=f"chat_input_disabled_{input_key}", disabled=True
        )
        _, controls = st.columns([5, 1])
        with controls:
            if st.button("⏹️ 停止", use_container_width=True, key="stop_btn_inline"):
                return "stop", ""
    elif blocked:
        st.chat_input("请先恢复未完成任务，或新建会话。", key=f"chat_blocked_{input_key}", disabled=True)
    else:
        query = st.chat_input(
            "😀请输入您的问题（按 Enter 发送，Shift+Enter 换行）",
            key=f"chat_input_{input_key}",
        )
        if query:
            return "submit", query
    return None


class AssistantBubble:
    """流式预览和历史回答共用一个聊天框结构，状态、阶段结果和正文都在框内。"""

    def __init__(self):
        self.container = st.chat_message("assistant", avatar="🤖")
        with self.container:
            self.status = st.empty()
            self.stages = st.empty()
            self.body = st.empty()

    def show_sequence(self, sequence):
        names = [name.strip().removeprefix("🤖").strip().removesuffix(" Agent") for name in sequence]
        if names:
            self.status.caption("执行顺序：" + " → ".join(names))
        else:
            self.status.empty()

    def show_stages(self, results):
        if not results:
            self.stages.empty()
            return
        with self.stages.container():
            for result in results:
                with st.expander(f"{result['agent']} 的阶段结果"):
                    st.markdown(result["text"])

    def finish(self, reply, sequence, results):
        self.show_sequence(sequence)
        self.show_stages(results[:-1])
        self.body.markdown(reply)


class StreamingReply(AssistantBubble):
    """一轮对话只创建一个助手框，所有片段在它的正文占位符中更新。"""

    def __init__(self):
        super().__init__()
        self.buffers = {}
        self.message_ids = {}
        self.sequence = []
        self.results = []
        self.body.markdown("正在思考…")

    def update(self, event):
        if event.type == "done":
            if event.result is not None:
                self.finish(event.result.reply, event.result.agent_sequence, event.result.agent_results)
            else:
                self.body.markdown(event.text)
            return
        agent = event.agent
        if agent not in self.buffers:
            self.buffers[agent] = ""
            self.sequence.append(agent)
            self.show_sequence(self.sequence)
            # 切换 Agent 时，把已经完成的结果留在同一聊天框的折叠区域。
            self.show_stages(self.results)
            self.body.markdown("正在处理…")

        if event.type == "text_delta":
            if self.message_ids.get(agent) != event.message_id:
                self.buffers[agent] = ""
                self.message_ids[agent] = event.message_id
            self.buffers[agent] += event.text
            self.body.markdown(self.buffers[agent] + " ▌")
        elif event.type == "text_reset":
            self.buffers[agent] = ""
            self.message_ids[agent] = event.message_id
            self.body.markdown("正在调用工具…")
        elif event.type == "agent_done":
            self.buffers[agent] = event.text
            self.results.append({"agent": agent, "text": event.text})
            self.body.markdown(event.text or "此步骤已完成。")


def render_history(questions, responses, sequences, agent_results=None, pending_index=None):
    for index, (question, response) in enumerate(zip(questions, responses)):
        message(question, is_user=True, key=f"{index}_user", avatar_style="adventurer")
        if index == pending_index:
            continue  # 本轮答案由 StreamingReply 显示，不再画一个重复的“思考中”气泡。
        results = agent_results[index] if agent_results else []
        AssistantBubble().finish(response, sequences[index], results)
