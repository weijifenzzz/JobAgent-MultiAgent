"""不访问模型 API：验证真实嵌套 Agent 的 invoke 也能向外层传出 token。"""

import json
import os
import unittest
from threading import Event
from typing import Any
from unittest.mock import MagicMock, Mock, patch

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langchain_core.outputs import ChatGenerationChunk
from langchain_core.tools import tool

from services.conversation import ConversationEvent, NO_REPLY, stream_conversation


class ScriptedModel(BaseChatModel):
    scripts: list[list[AIMessageChunk]]
    calls: int = 0
    completed_calls: int = 0
    release: Any = None

    @property
    def _llm_type(self):
        return "offline-stream-test"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, *args, **kwargs):
        raise AssertionError("Nested invoke should select streaming through graph callbacks")

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        script = self.scripts[self.calls]
        self.calls += 1
        for index, chunk in enumerate(script):
            yield ChatGenerationChunk(message=chunk)
            if self.release is not None and index == 0:
                if not self.release.wait(timeout=5):
                    raise AssertionError("No token reached the consumer before model completion")
        self.completed_calls += 1


class StreamingTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {"LANGCHAIN_TRACING_V2": "false"})
        self.env.start()
        self.addCleanup(self.env.stop)
        with patch("dotenv.load_dotenv"):
            import agents
        self.agents = agents

    def test_chatbot_tokens_arrive_before_invoke_returns_and_hide_supervisor(self):
        release = Event()
        model = ScriptedModel(scripts=[[
            AIMessageChunk(content="不客气，"), AIMessageChunk(content="祝你求职顺利！"),
        ]], release=release)
        models = [FakeListChatModel(responses=['{"steps": ["ChatBot"]}']), model]
        with patch.object(self.agents, "init_chat_model", side_effect=models):
            events = []
            try:
                for event in stream_conversation(self.agents.define_graph(), "谢谢", [], {}):
                    events.append(event)
                    if event.type == "text_delta" and not release.is_set():
                        self.assertEqual(model.completed_calls, 0)
                        release.set()
            finally:
                release.set()
        self.assertEqual("".join(e.text for e in events if e.type == "text_delta"), "不客气，祝你求职顺利！")
        self.assertTrue(all(e.agent != "Supervisor" for e in events))
        self.assertEqual(events[-1].result.reply, "不客气，祝你求职顺利！")
        self.assertEqual(len(events[-1].result.messages), 2)

    def test_real_nested_agent_tool_loop_streams_without_tool_arguments(self):
        @tool
        def resume_extractor() -> str:
            """Read an offline test resume."""
            return "Python developer"

        model = ScriptedModel(scripts=[
            [AIMessageChunk(content="", tool_call_chunks=[{
                "name": "resume_extractor", "args": "{}", "id": "call-test", "index": 0,
            }])],
            [AIMessageChunk(content="你的优势是"), AIMessageChunk(content=" Python。")],
        ])
        with patch.object(self.agents, "init_chat_model", side_effect=[
            FakeListChatModel(responses=['{"steps": ["ResumeAnalyzer"]}']), model,
        ]), patch.object(self.agents, "ResumeExtractorTool", return_value=resume_extractor):
            events = list(stream_conversation(self.agents.define_graph(), "分析简历", [], {}))
        self.assertEqual(model.calls, 2)
        self.assertEqual("".join(e.text for e in events if e.type == "text_delta"), "你的优势是 Python。")
        self.assertTrue(any(e.type == "text_reset" for e in events))
        self.assertTrue(all(e.agent == "ResumeAnalyzer" for e in events if e.type == "text_delta"))
        self.assertEqual(events[-1].result.reply, "你的优势是 Python。")
        self.assertEqual(len(events[-1].result.messages), 2)  # 不把内层工具消息重复写回外层历史。

    def test_multi_agent_results_keep_node_added_title(self):
        first = ScriptedModel(scripts=[[AIMessageChunk(content="简历分析结果")]])
        second = ScriptedModel(scripts=[[AIMessageChunk(content="岗位推荐结果")]])
        with patch.object(self.agents, "init_chat_model", side_effect=[
            FakeListChatModel(responses=['{"steps": ["ResumeAnalyzer", "JobSearcher"]}']), first, second,
        ]):
            events = list(stream_conversation(self.agents.define_graph(), "分析简历并推荐岗位", [], {}))
        result = events[-1].result
        self.assertEqual([r["agent"] for r in result.agent_results], ["ResumeAnalyzer", "JobSearcher"])
        self.assertIn("基于简历分析的个性化岗位推荐", result.reply)
        self.assertIn("岗位推荐结果", result.reply)
        self.assertEqual(len(result.messages), 3)

    def test_checkpointed_agents_continue_without_duplicate_messages_or_stale_routing(self):
        from langgraph.checkpoint.memory import InMemorySaver
        from services.conversation import thread_config
        graph = self.agents.define_graph(checkpointer=InMemorySaver())
        models = [
            FakeListChatModel(responses=['{"steps": ["ResumeAnalyzer", "JobSearcher"]}']),
            ScriptedModel(scripts=[[AIMessageChunk(content="简历分析结果")]]),
            ScriptedModel(scripts=[[AIMessageChunk(content="岗位推荐结果")]]),
            FakeListChatModel(responses=['{"steps": ["ChatBot"]}']),
            ScriptedModel(scripts=[[AIMessageChunk(content="不客气")]]),
        ]
        with patch.object(self.agents, "init_chat_model", side_effect=models):
            first = list(stream_conversation(graph, "分析简历并推荐岗位", [], {}, thread_id="persisted"))
            second = list(stream_conversation(graph, "谢谢", [HumanMessage(content="stale cache")], {}, thread_id="persisted"))
        self.assertTrue(any(e.type == "text_delta" for e in first))
        self.assertEqual([r["agent"] for r in first[-1].result.agent_results], ["ResumeAnalyzer", "JobSearcher"])
        self.assertEqual(len(second[-1].result.messages), 5)
        self.assertEqual(second[-1].result.agent_sequence, ["ChatBot"])
        self.assertEqual(second[-1].result.reply, "不客气")
        state = graph.get_state(thread_config("persisted")).values
        self.assertEqual(state["needs_followup"], "")
        self.assertNotIn("stale cache", [m.content for m in state["messages"]])

    def test_fallback_no_tokens_and_public_payload_excludes_state(self):
        def chunks(state, config, **kwargs):
            state["messages"].append(AIMessage(content="complete answer", name="ChatBot"))
            yield {"type": "updates", "ns": (), "data": {"ChatBot": state}}

        graph = Mock(stream=chunks)
        events = list(stream_conversation(graph, "hi", [], {"api_key": "never-export"}))
        self.assertEqual([e.type for e in events], ["agent_done", "done"])
        self.assertEqual(events[-1].result.reply, "complete answer")
        payload = json.dumps([e.public_data() for e in events])
        self.assertNotIn("never-export", payload)
        self.assertNotIn("callback", payload)

    def test_no_new_answer_and_midstream_error(self):
        graph = Mock(stream=lambda *args, **kwargs: (yield from ()))
        events = list(stream_conversation(graph, "new", [AIMessage(content="old")], {}))
        self.assertEqual(events[-1].result.reply, NO_REPLY)
        history = [HumanMessage(content="previous")]

        def broken(*args, **kwargs):
            yield {"type": "messages", "ns": (), "data": (
                AIMessageChunk(content="partial", id="id"), {"langgraph_node": "ChatBot"}
            )}
            raise RuntimeError("offline failure")

        with self.assertRaisesRegex(RuntimeError, "offline failure"):
            list(stream_conversation(Mock(stream=broken), "new", history, {}))
        self.assertEqual(len(history), 1)

    def test_closing_consumer_closes_underlying_graph_stream(self):
        closed = []

        def chunks(*args, **kwargs):
            try:
                yield {"type": "custom", "ns": (), "data": {"event": "agent_start", "agent": "ChatBot"}}
                raise AssertionError("Should not consume more after close")
            finally:
                closed.append(True)

        events = stream_conversation(Mock(stream=chunks), "new", [], {})
        self.assertEqual(next(events).type, "agent_start")
        events.close()
        self.assertEqual(closed, [True])

    def test_openai_compatible_invoke_uses_streaming_http_without_live_network(self):
        import httpx
        from langchain_openai import ChatOpenAI
        from langgraph.graph import StateGraph, START, END
        requests = []

        def respond(request):
            requests.append(json.loads(request.content))
            chunks = []
            for text in ["hello", " world"]:
                chunks.append("data: " + json.dumps({
                    "id": "test-http", "object": "chat.completion.chunk", "created": 0,
                    "model": "test-model", "choices": [{"index": 0, "delta": {
                        "role": "assistant", "content": text,
                    }, "finish_reason": None}],
                }) + "\n\n")
            chunks.append("data: [DONE]\n\n")
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, text="".join(chunks))

        with httpx.Client(transport=httpx.MockTransport(respond)) as client:
            from llms import get_llm
            # 测试实际工厂，防止重新引入 streaming=False 导致消息流被禁用。
            with patch("llms.ChatOpenAI", side_effect=lambda **kw: ChatOpenAI(http_client=client, **kw)):
                model = get_llm(provider="openai", model="test-model", api_key="offline-key",
                                base_url="https://example.invalid/v1")
            self.assertNotIn("streaming", model.model_fields_set)

            def chatbot(state):
                output = model.invoke(state["messages"])
                return {"messages": state["messages"] + [AIMessage(content=output.content, name="ChatBot")]}

            graph = StateGraph(self.agents.AgentState).add_node("ChatBot", chatbot)
            graph.add_edge(START, "ChatBot")
            graph.add_edge("ChatBot", END)
            events = list(stream_conversation(graph.compile(), "hi", [], {}))
        self.assertTrue(requests[0]["stream"])
        self.assertEqual("".join(e.text for e in events if e.type == "text_delta"), "hello world")

    def test_preview_resets_between_model_calls_and_reconciles_node_result(self):
        from ui.chat import StreamingReply
        with patch("ui.chat.st") as st:
            st.container.return_value = MagicMock()
            preview = StreamingReply()
            preview.update(ConversationEvent("text_delta", "ResumeAnalyzer", "before tool", "first"))
            preview.update(ConversationEvent("text_reset", "ResumeAnalyzer", message_id="first"))
            self.assertEqual(preview.buffers["ResumeAnalyzer"], "")
            preview.update(ConversationEvent("text_delta", "ResumeAnalyzer", "actual ", "second"))
            preview.update(ConversationEvent("text_delta", "ResumeAnalyzer", "answer", "second"))
            self.assertEqual(preview.buffers["ResumeAnalyzer"], "actual answer")
            preview.update(ConversationEvent("agent_done", "ResumeAnalyzer", "Title: actual answer"))
            self.assertEqual(st.empty.return_value.markdown.call_args.args[0], "Title: actual answer")


if __name__ == "__main__":
    unittest.main()
