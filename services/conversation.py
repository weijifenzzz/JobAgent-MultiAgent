"""对话执行服务：输入数据，运行图，返回结果；不读写页面状态。"""

from dataclasses import dataclass, field
from typing import Iterator, Literal, Sequence

from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage
from langgraph.config import get_stream_writer
from runtime_context import AgentObserver, RunContext, LOCAL_USER_ID

NO_REPLY = "本轮未生成助手回复，请重新提问。"


@dataclass
class RecordingObserver:
    """无界面调用时也可以记录角色顺序，供 CLI 或测试使用。"""
    agent_sequence: list[str] = field(default_factory=list)

    def write_agent_name(self, name: str):
        self.agent_sequence.append(name)

    def clear_agent_sequence(self):
        self.agent_sequence.clear()

    def get_agent_sequence(self):
        return list(self.agent_sequence)


@dataclass
class ConversationResult:
    reply: str
    messages: list[BaseMessage]
    agent_sequence: list[str]
    agent_results: list[dict[str, str]] = field(default_factory=list)


@dataclass
class ConversationEvent:
    """服务层事件；result 仅用于本地写回历史，前端只接收 public_data()。"""
    type: Literal["agent_start", "text_delta", "text_reset", "agent_done", "done"]
    agent: str = ""
    text: str = ""
    message_id: str = ""
    result: ConversationResult | None = None

    def public_data(self) -> dict:
        """未来 API 可将此字典编码为 SSE；不输出图状态、密钥或回调对象。"""
        return {
            "type": self.type,
            "agent": self.agent,
            "text": self.text,
            "message_id": self.message_id,
        }


VISIBLE_AGENTS = {
    "ResumeAnalyzer", "JobSearcher", "CoverLetterGenerator", "WebResearcher", "ChatBot",
    "MemoryManager",
}


def _text_content(content) -> str:
    """只提取正文，忽略结构化内容中的推理、工具调用等非文本块。"""
    if isinstance(content, str):
        return content
    return "".join(
        block if isinstance(block, str) else block.get("text", "")
        for block in content
        if isinstance(block, str) or (isinstance(block, dict) and block.get("type") == "text")
    )


class _StreamObserver(RecordingObserver):
    def write_agent_name(self, name: str):
        super().write_agent_name(name)
        agent = next((role for role in VISIBLE_AGENTS if role in name), "")
        if agent:
            # 由图运行线程发出事件；这里不访问任何 Streamlit 组件。
            get_stream_writer()({"event": "agent_start", "agent": agent})


def thread_config(thread_id: str) -> dict:
    return {"recursion_limit": 15, "configurable": {"thread_id": thread_id}}


def conversation_turns(messages):
    """从外层消息重建页面每轮问答；中间 Agent 输出仍作为阶段结果保留。"""
    turns = []
    for message in messages:
        if isinstance(message, HumanMessage):
            turns.append({"question": _text_content(message.content), "reply": "",
                          "sequence": [], "results": []})
        elif isinstance(message, AIMessage) and turns:
            turn = turns[-1]
            text = _text_content(message.content)
            turn["reply"] = text
            if message.name:
                turn["sequence"].append(message.name)
                turn["results"].append({"agent": message.name, "text": text})
    return turns


def _prepare_run(graph, user_input, history, thread_id, resume):
    """持久化模式以数据库为准；新一轮重置路由标志，恢复则不覆盖旧状态。"""
    config = thread_config(thread_id) if thread_id else {"recursion_limit": 15}
    if thread_id:
        snapshot = graph.get_state(config)
        messages = list(snapshot.values.get("messages", []))
        if resume:
            if not snapshot.next:
                raise ValueError("该会话没有待恢复的任务。")
            boundary = next((i + 1 for i in range(len(messages) - 1, -1, -1)
                             if isinstance(messages[i], HumanMessage)), len(messages))
            return None, config, messages, boundary
        if snapshot.next:
            raise ValueError("该会话有未完成任务，请先恢复执行，或新建会话。")
    else:
        if resume:
            raise ValueError("恢复执行必须提供 thread_id 和 checkpointer。")
        messages = list(history)
    messages.append(HumanMessage(content=user_input))
    inputs = {"messages": messages, "user_input": user_input}
    if thread_id:
        inputs.update(next_step="", needs_followup="", task_completed=False)
    return inputs, config, messages, len(messages)


def stream_conversation(
    graph,
    user_input: str,
    history: Sequence[BaseMessage],
    model_config: dict,
    *,
    thread_id: str | None = None,
    resume: bool = False,
    user_id: str = LOCAL_USER_ID,
) -> Iterator[ConversationEvent]:
    """实时产出事件；完成后通过 done.result 返回权威历史，异常交给调用端。"""
    observer = _StreamObserver()
    inputs, config, input_messages, input_count = _prepare_run(
        graph, user_input, history, thread_id, resume
    )
    final_messages = input_messages
    agent_results = []
    started = set()
    completed = set()
    tool_messages = set()
    if resume:
        for message in input_messages[input_count:]:
            if isinstance(message, AIMessage) and message.name in VISIBLE_AGENTS:
                text = _text_content(message.content)
                agent_results.append({"agent": message.name, "text": text})
                yield ConversationEvent("agent_done", message.name, text)
    stream = graph.stream(
        inputs,
        config,
        context=RunContext(model_config=dict(model_config), observer=observer, user_id=user_id),
        stream_mode=["messages", "updates", "custom"],
        subgraphs=True,
        version="v2",
        **({"durability": "sync"} if thread_id else {}),
    )
    try:
        for part in stream:
            mode, namespace, data = part["type"], part["ns"], part["data"]
            if mode == "custom":
                if data.get("event") == "agent_start":
                    agent = data.get("agent")
                    if agent in VISIBLE_AGENTS and agent not in started:
                        started.add(agent)
                        yield ConversationEvent("agent_start", agent)

            elif mode == "messages":
                chunk, metadata = data
                # 子图里的节点通常叫 model，角色名需要从外层命名空间识别。
                agent = namespace[0].split(":", 1)[0] if namespace else metadata.get("langgraph_node")
                if agent == "MemoryManager":
                    continue  # 只显示实际写入后的确认，隐藏提取模型的内部 JSON。
                if agent not in VISIBLE_AGENTS or agent in completed:
                    continue  # 不展示 Supervisor 路由文本和已经完成的节点消息。
                if not isinstance(chunk, AIMessageChunk) or metadata.get("langgraph_node") == "tools":
                    continue
                if agent not in started:
                    started.add(agent)
                    yield ConversationEvent("agent_start", agent)
                message_id = chunk.id or f"{agent}:current"
                key = (agent, message_id)
                if chunk.tool_call_chunks or chunk.tool_calls:
                    if key not in tool_messages:
                        tool_messages.add(key)
                        # 工具调用可能在正文片段之后才出现：撤回本次调用的预览。
                        yield ConversationEvent("text_reset", agent, message_id=message_id)
                    continue
                if key in tool_messages:
                    continue
                text = _text_content(chunk.content)
                if text:
                    yield ConversationEvent("text_delta", agent, text, message_id)

            elif mode == "updates" and not namespace:
                # 只用外层节点的结果写回历史，忽略内层 Agent 的临时工具消息。
                for agent, update in data.items():
                    if not isinstance(update, dict) or "messages" not in update:
                        continue
                    final_messages = list(update["messages"])
                    if agent not in VISIBLE_AGENTS:
                        continue
                    response = next((
                        msg for msg in reversed(final_messages[input_count:])
                        if isinstance(msg, AIMessage) and msg.name == agent
                    ), None)
                    if response is None:
                        continue
                    text = _text_content(response.content)
                    completed.add(agent)
                    agent_results.append({"agent": agent, "text": text})
                    # 替换预览，保留节点附加的标题/说明；没有 token 的模型也能显示结果。
                    yield ConversationEvent("agent_done", agent, text)
    finally:
        stream.close()

    if thread_id:
        snapshot = graph.get_state(config)
        if snapshot.next:
            raise RuntimeError("任务尚未完成，请使用会话恢复入口。")
        final_messages = list(snapshot.values.get("messages", []))
    response = next((
        msg for msg in reversed(final_messages[input_count:]) if isinstance(msg, AIMessage)
    ), None)
    reply = _text_content(response.content) if response is not None else ""
    sequence = observer.get_agent_sequence()
    if thread_id:
        turns = conversation_turns(final_messages)
        if turns:
            sequence, agent_results = turns[-1]["sequence"], turns[-1]["results"]
    result = ConversationResult(reply or NO_REPLY, final_messages, sequence, agent_results)
    yield ConversationEvent("done", text=result.reply, result=result)


def run_conversation(
    graph,
    user_input: str,
    history: Sequence[BaseMessage],
    model_config: dict,
    observer: AgentObserver | None = None,
    *,
    thread_id: str | None = None,
    resume: bool = False,
    user_id: str = LOCAL_USER_ID,
) -> ConversationResult:
    """异常交给调用端处理；本函数不展示错误、不操作 session_state。"""
    observer = observer if observer is not None else RecordingObserver()
    observer.clear_agent_sequence()
    inputs, config, _, input_message_count = _prepare_run(graph, user_input, history, thread_id, resume)
    output = graph.invoke(
        inputs,
        config,
        context=RunContext(model_config=dict(model_config), observer=observer, user_id=user_id),
        **({"durability": "sync"} if thread_id else {}),
    )

    # 当前外层图返回完整历史；只提取输入边界之后的新 AI 消息。
    if thread_id:
        snapshot = graph.get_state(config)
        if snapshot.next:
            raise RuntimeError("任务尚未完成，请使用会话恢复入口。")
        output = snapshot.values
    messages = output.get("messages", [])
    new_messages = messages[input_message_count:]
    response = next(
        (msg for msg in reversed(new_messages) if isinstance(msg, AIMessage)),
        None,
    )
    reply = response.content if response is not None and response.content else NO_REPLY
    if thread_id:
        turns = conversation_turns(messages)
        if turns:
            return ConversationResult(reply, messages, turns[-1]["sequence"], turns[-1]["results"])
    return ConversationResult(reply, messages, list(observer.get_agent_sequence()))
