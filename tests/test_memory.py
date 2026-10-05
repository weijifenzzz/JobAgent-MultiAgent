"""长期记忆回归：用真实 Store/图和模拟模型验证，不访问外部 API。"""

import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from runtime_context import LOCAL_USER_ID
from services.conversation import run_conversation, stream_conversation, thread_config
from services.memory import (
    MemoryEntry, delete_memory, handle_memory_request, list_memories,
    memory_context, memory_intent, memory_namespace, save_memory,
)


def entry(content="上海", **kwargs):
    return MemoryEntry(**{"kind": "preferences", "key": "preferred_city",
                          "label": "目标城市", "content": content, **kwargs})


def extractor(*entries):
    return FakeListChatModel(responses=[json.dumps(
        {"entries": [item.model_dump() for item in entries]}, ensure_ascii=False
    )])


class MemoryTests(unittest.TestCase):
    def setUp(self):
        self.store = InMemoryStore()
        env = patch.dict(os.environ, {"LANGCHAIN_TRACING_V2": "false"})
        env.start()
        self.addCleanup(env.stop)
        with patch("dotenv.load_dotenv"):
            import agents
        self.agents = agents

    def test_upsert_is_per_user_and_kind_not_per_thread(self):
        save_memory(self.store, "alice", entry(), thread_id="thread-a")
        save_memory(self.store, "alice", entry("杭州"), thread_id="thread-b")
        save_memory(self.store, "bob", entry("北京"))
        save_memory(self.store, "alice", entry("曾在上海工作", kind="facts"))
        self.assertEqual(len(list_memories(self.store, "alice")), 2)
        record = self.store.get(memory_namespace("alice", "preferences"), "preferred_city")
        self.assertEqual(record.value["content"], "杭州")
        self.assertEqual(record.value["source_thread_id"], "thread-b")
        self.assertTrue(delete_memory(self.store, "alice", "preferences", "preferred_city"))
        self.assertFalse(delete_memory(self.store, "alice", "preferences", "preferred_city"))
        self.assertEqual(list_memories(self.store, "bob")[0]["content"], "北京")
        self.assertEqual(list_memories(self.store, "alice")[0]["kind"], "facts")

    def test_normal_chat_and_listing_never_call_extraction_model(self):
        factory = Mock(side_effect=AssertionError("must not call model"))
        self.assertIsNone(handle_memory_request(self.store, "alice", "推荐上海的职位", factory))
        self.assertIsNone(memory_intent("不要记住我的信息"))
        self.assertIn("没有已保存", handle_memory_request(self.store, "alice", "查看我的记忆", factory))
        save_memory(self.store, "alice", entry())
        self.assertIn("上海", handle_memory_request(self.store, "alice", "查看我的记忆", factory))
        factory.assert_not_called()

    def test_commands_save_replace_and_delete_only_valid_existing_keys(self):
        for city in ("上海", "杭州"):
            reply = handle_memory_request(self.store, "alice", f"请记住：目标城市是{city}",
                                          lambda: extractor(entry(city)))
            self.assertIn("已保存", reply)
        self.assertEqual(len(list_memories(self.store, "alice")), 1)
        reply = handle_memory_request(self.store, "alice", "请忘记目标城市", lambda: extractor(
            entry(), entry(key="unknown", label="不存在")
        ))
        self.assertIn("没有执行删除", reply)
        self.assertEqual(len(list_memories(self.store, "alice")), 1)
        reply = handle_memory_request(self.store, "alice", "请忘记目标城市", lambda: extractor(entry("杭州")))
        self.assertIn("已删除", reply)
        self.assertEqual(list_memories(self.store, "alice"), [])

    def test_invalid_extraction_is_not_written_or_reported_as_saved(self):
        for response in ("not json", '{"entries":[{"key":"incomplete"}]}'):
            reply = handle_memory_request(self.store, "alice", "请记住：上海", lambda: FakeListChatModel(responses=[response]))
            self.assertIn("没有修改", reply)
        self.assertEqual(list_memories(self.store, "alice"), [])

    def test_context_handles_pagination_and_literal_json_braces(self):
        for i in range(105):
            save_memory(self.store, "alice", entry(f'值{i} {{"example": 1}}', key=f"item_{i}"))
        self.assertEqual(len(list_memories(self.store, "alice")), 105)
        context = memory_context(SimpleNamespace(store=self.store, context=SimpleNamespace(user_id="alice")))
        self.assertIn("总计105条", context)
        from chains import get_finish_chain
        chain = get_finish_chain(FakeListChatModel(responses=["ok"]), memory_context=context)
        self.assertEqual(chain.invoke({"messages": []}).content, "ok")

    def test_real_graph_stream_hides_extraction_and_new_thread_reads_store(self):
        graph = self.agents.define_graph(checkpointer=InMemorySaver(), store=self.store)
        with patch.object(self.agents, "init_chat_model", return_value=extractor(entry())):
            events = list(stream_conversation(graph, "请记住：目标城市是上海", [], {}, thread_id="a"))
        self.assertIn("已保存", events[-1].result.reply)
        self.assertFalse(any(e.type == "text_delta" for e in events))
        self.assertEqual(events[-1].result.agent_sequence, ["MemoryManager"])
        from chains import get_finish_chain
        captured = []

        def finish(llm, memory_context=""):
            captured.append(memory_context)
            return get_finish_chain(llm, memory_context=memory_context)

        with patch.object(self.agents, "init_chat_model", side_effect=[
            FakeListChatModel(responses=['{"steps": ["ChatBot"]}']), FakeListChatModel(responses=["你偏好上海。"])
        ]), patch.object(self.agents, "get_finish_chain", side_effect=finish):
            result = run_conversation(graph, "我的偏好是什么？", [], {}, thread_id="b")
        self.assertIn("上海", captured[0])
        self.assertEqual(result.reply, "你偏好上海。")
        self.assertEqual(len(result.messages), 2)  # 不复制会话 a 的聊天，也不把记忆追加进 State。
        self.assertEqual(result.messages[0].content, "我的偏好是什么？")
        state = graph.get_state(thread_config("b")).values
        self.assertNotIn("config", state)
        self.assertNotIn("callback", state)
        self.assertNotIn("user_id", state)
        self.assertEqual(list_memories(self.store, LOCAL_USER_ID)[0]["content"], "上海")

    def test_specialist_receives_memory_in_system_prompt(self):
        save_memory(self.store, LOCAL_USER_ID, entry())
        fake_agent = Mock()
        from langchain_core.messages import AIMessage
        fake_agent.invoke.return_value = {"messages": [AIMessage(content="分析完成")]}
        with patch.object(self.agents, "init_chat_model", return_value=FakeListChatModel(responses=['{"steps": ["ResumeAnalyzer"]}'])), \
                patch.object(self.agents, "create_agent", return_value=fake_agent) as create:
            run_conversation(self.agents.define_graph(store=self.store), "分析简历", [], {})
        self.assertIn("上海", create.call_args.kwargs["system_prompt"])


if __name__ == "__main__":
    unittest.main()
