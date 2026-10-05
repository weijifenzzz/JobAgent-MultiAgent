"""验证知识库工具和真实嵌套 Agent 循环，不访问模型服务或 Docker。"""

import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessageChunk
from pydantic import ValidationError

from services.conversation import stream_conversation
from test_streaming import ScriptedModel


class ResearchModel(ScriptedModel):
    bound_names: list[str] = []
    tool_contents: list[str] = []

    def bind_tools(self, tools, **kwargs):
        self.bound_names = [tool.name for tool in tools]
        return self

    def _stream(self, messages, **kwargs):
        self.tool_contents = [m.content for m in messages if m.type == "tool"]
        yield from super()._stream(messages, **kwargs)


class KnowledgeAgentTests(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, {"LANGCHAIN_TRACING_V2": "false", "LANGSMITH_TRACING": "false"})
        env.start()
        self.addCleanup(env.stop)
        with patch("dotenv.load_dotenv"):
            import agents
            import tools
        self.agents, self.tools = agents, tools
        self.factory = patch("services.knowledge.KnowledgeIndex")
        self.index_class = self.factory.start()
        self.addCleanup(self.factory.stop)
        self.index = self.index_class.return_value
        self.url = "https://github.com/example/tutorial/blob/version/memory.md"
        self.index.search.return_value = [SimpleNamespace(score=0.8, payload={
            "title": "记忆与检索", "section": "短期与长期记忆", "text": "短期保存当前上下文，长期跨会话复用。",
            "source_url": self.url, "source_revision": "version", "unused_private_field": "hidden",
        })]

    def test_tool_returns_sources_and_closes_client(self):
        result = self.tools.search_knowledge_base.invoke({"query": "  记忆区别  "})
        self.index.search.assert_called_once_with("记忆区别", limit=3)
        self.index.close.assert_called_once()
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["results"][0]["source_url"], self.url)
        self.assertNotIn("hidden", json.dumps(result))

    def test_empty_and_error_do_not_claim_success_or_leak_error(self):
        self.index.search.return_value = []
        self.assertEqual(self.tools.search_knowledge_base.invoke({"query": "记忆"})["status"], "empty")
        self.index.search.side_effect = RuntimeError("private-api-key-in-error")
        result = self.tools.search_knowledge_base.invoke({"query": "记忆"})
        self.assertEqual(result["status"], "unavailable")
        self.assertEqual(result["results"], [])
        self.assertNotIn("private-api-key", json.dumps(result))
        self.assertEqual(self.index.close.call_count, 2)

    def test_invalid_input_does_not_open_clients(self):
        self.assertEqual(self.tools.search_knowledge_base.invoke({"query": "   "})["status"], "invalid_query")
        for args in [{"query": "问题", "limit": 6}, {"query": "问题", "limit": 0}, {"query": "x" * 2001}]:
            with self.assertRaises(ValidationError):
                self.tools.search_knowledge_base.invoke(args)
        self.index_class.assert_not_called()

    def test_real_graph_tool_loop_streams_and_keeps_only_final_answer(self):
        reply = f"短期用于当前上下文，长期可跨会话复用。[记忆与检索]({self.url})"
        model = ResearchModel(scripts=[
            [AIMessageChunk(content="", tool_call_chunks=[{
                "name": "search_knowledge_base", "args": '{"query":"短期记忆和长期记忆"}',
                "id": "kb-call", "index": 0,
            }])],
            [AIMessageChunk(content=reply)],
        ])
        with patch.object(self.agents, "init_chat_model", side_effect=[
            FakeListChatModel(responses=["WebResearcher"]), model,
        ]):
            events = list(stream_conversation(self.agents.define_graph(), "解释Agent记忆", [], {}))
        self.assertEqual(set(model.bound_names), {"search_knowledge_base", "google_search", "scrape_website"})
        self.assertIn(self.url, model.tool_contents[0])
        self.assertEqual(events[-1].result.reply, reply)
        self.assertEqual(len(events[-1].result.messages), 2)
        self.assertEqual(events[-1].result.messages[-1].name, "WebResearcher")
        self.assertEqual("".join(e.text for e in events if e.type == "text_delta"), reply)
        self.index.search.assert_called_once()

    def test_invalid_supervisor_output_uses_knowledge_fallback(self):
        for query in ["什么是RAG？", "分析 Agent 工作原理", "Agent 岗位面试如何回答记忆问题？"]:
            with self.subTest(query=query), patch.object(self.agents, "init_chat_model", return_value=FakeListChatModel(responses=["invalid-route"])):
                runtime = SimpleNamespace(context=SimpleNamespace(model_config={}), store=None)
                result = self.agents.supervisor_node({"user_input": query, "messages": []}, runtime)
            self.assertEqual(result["next_step"], "WebResearcher")
        for query in ["搜索上海 Agent 招聘岗位", "分析我的简历中的 Agent 项目", "帮我写求职信", "谢谢"]:
            self.assertFalse(self.agents._is_knowledge_question(query))


if __name__ == "__main__":
    unittest.main()
