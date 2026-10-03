"""真实 Streamlit 页面回归，模型调用及配置、简历路径均隔离。"""

import os
import tempfile
import unittest
from contextlib import ExitStack, nullcontext
from copy import deepcopy
from types import SimpleNamespace
from uuid import uuid4
from pathlib import Path
from unittest.mock import Mock, patch

from langchain_core.messages import AIMessage, AIMessageChunk
from streamlit.testing.v1 import AppTest
from langgraph.store.memory import InMemoryStore
from resume_fixtures import pdf_bytes


class AppTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch("dotenv.load_dotenv"))
        self.stack.enter_context(patch.dict(os.environ, {"LANGCHAIN_TRACING_V2": "false"}))
        import agents
        import ui.sidebar as sidebar

        directory = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.directory = directory
        (directory / "dummy_resume.pdf").write_bytes(pdf_bytes())
        self.stack.enter_context(patch("ui.resume.PROJECT_ROOT", directory))
        self.stack.enter_context(patch("services.resumes.RESUME_DIR", directory / "resumes"))
        self.stack.enter_context(patch.object(sidebar, "load_initial_config", return_value={
            "model_name": "test-model", "api_key": "test-key", "base_url": "https://example.invalid/v1",
            "temperature": 0.3, "serper_key": "", "firecrawl_key": "",
        }))
        self.save = self.stack.enter_context(patch.object(sidebar, "save_persistent_config", return_value=True))
        self.stack.enter_context(patch("settings.initialize_environment"))
        self.stack.enter_context(patch("streamlit_analytics2.start_tracking"))
        self.stack.enter_context(patch("streamlit_analytics2.stop_tracking"))
        self.graph = Mock()
        self.states = {}
        self.pending = {}
        self.graph.get_state.side_effect = lambda config: SimpleNamespace(
            values=deepcopy(self.states.get(config["configurable"]["thread_id"], {})),
            next=self.pending.get(config["configurable"]["thread_id"], ())
        )
        self.graph.stream.side_effect = self.respond
        def update_state(config, values, **kwargs):
            thread_id = config["configurable"]["thread_id"]
            self.states.setdefault(thread_id, {}).update(deepcopy(values))
            self.pending[thread_id] = ()
        self.graph.update_state.side_effect = update_state
        self.rows = [{"thread_id": str(uuid4()), "title": "新会话"}]
        repository = Mock()
        repository.list_threads.side_effect = lambda: list(self.rows)
        def create():
            thread_id = str(uuid4())
            self.rows.insert(0, {"thread_id": thread_id, "title": "新会话"})
            return thread_id
        repository.create.side_effect = create
        repository.lock.side_effect = lambda thread_id: nullcontext()
        self.store = InMemoryStore()
        self.stack.enter_context(patch("persistence.open_database", side_effect=lambda: nullcontext(
            SimpleNamespace(checkpointer=Mock(), threads=repository, store=self.store)
        )))
        self.stack.enter_context(patch.object(agents, "define_graph", return_value=self.graph))
        self.app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=20)
        self.app.run()
        self.assertFalse(self.app.exception)
        self.click_label("使用演示简历")
        self.assertFalse(self.app.exception)

    def respond(self, state, config, **kwargs):
        thread_id = config["configurable"]["thread_id"]
        if state is None:
            state = deepcopy(self.states[thread_id])
        else:
            state = {**deepcopy(self.states.get(thread_id, {})), **state}
        kwargs["context"].observer.agent_sequence.append("ChatBot")
        yield {"type": "custom", "ns": (), "data": {"event": "agent_start", "agent": "ChatBot"}}
        for text in ["test ", "answer"]:
            yield {"type": "messages", "ns": (), "data": (
                AIMessageChunk(content=text, id="test-answer"), {"langgraph_node": "ChatBot"}
            )}
        state["messages"].append(AIMessage(content="test answer", name="ChatBot"))
        self.states[config["configurable"]["thread_id"]] = deepcopy(state)
        self.pending[thread_id] = ()
        yield {"type": "updates", "ns": (), "data": {"ChatBot": state}}

    def click_label(self, label):
        next(button for button in self.app.button if button.label == label).click().run()

    def test_chat_rerun_and_clear(self):
        self.app.chat_input[0].set_value("hello").run()
        self.assertFalse(self.app.exception)
        self.assertEqual(self.app.session_state["response_history"][-1], "test answer")
        self.assertEqual(self.app.session_state["agent_sequence_history"][-1], ["ChatBot"])
        self.graph.stream.assert_called_once()
        self.app.run()
        self.graph.stream.assert_called_once()
        self.app.chat_input[0].set_value("follow-up").run()
        self.assertEqual(len(self.app.session_state["langchain_messages"]), 4)
        previous_thread = self.app.session_state["thread_id"]
        self.click_label("➕ 新会话")
        self.assertFalse(self.app.exception)
        self.assertEqual(self.app.session_state["langchain_messages"], [])
        self.assertEqual(self.app.session_state["response_history"], [])
        self.app.button(key=f"conversation_{previous_thread}").click().run()
        self.assertEqual(len(self.app.session_state["langchain_messages"]), 4)
        self.assertEqual(self.app.session_state["response_history"][-1], "test answer")

    def test_memory_panel_save_switch_edit_delete_without_model(self):
        from services.memory import list_memories
        from runtime_context import LOCAL_USER_ID

        next(x for x in self.app.text_input if x.label == "名称").set_value("目标城市")
        next(x for x in self.app.text_area if x.label == "内容").set_value("上海")
        self.click_label("保存记忆")
        self.assertFalse(self.app.exception)
        self.assertEqual(list_memories(self.store, LOCAL_USER_ID)[0]["content"], "上海")
        self.click_label("➕ 新会话")
        self.app.selectbox(key="memory_selection_preferences").select("preferred_city").run()
        self.assertEqual(next(x for x in self.app.text_area if x.label == "内容").value, "上海")
        next(x for x in self.app.text_area if x.label == "内容").set_value("杭州")
        self.click_label("保存记忆")
        self.assertEqual(list_memories(self.store, LOCAL_USER_ID)[0]["content"], "杭州")
        self.click_label("删除这条记忆")
        self.assertFalse(self.app.exception)
        self.assertEqual(list_memories(self.store, LOCAL_USER_ID), [])
        self.graph.stream.assert_not_called()

    def test_resume_upload_binding_survives_switch_and_new_page(self):
        first = self.app.session_state["thread_id"]
        first_id = self.states[first]["resume_id"]
        self.click_label("➕ 新会话")
        second = self.app.session_state["thread_id"]
        self.assertNotIn("resume_id", self.states.get(second, {}))
        uploaded = SimpleNamespace(name="new.pdf", getvalue=lambda: pdf_bytes("Second candidate Java developer"))
        with patch("ui.resume.st.file_uploader", return_value=uploaded):
            self.click_label("绑定到当前会话")
        self.assertFalse(self.app.exception)
        second_id = self.states[second]["resume_id"]
        self.assertNotEqual(second_id, first_id)
        self.app.button(key=f"conversation_{first}").click().run()
        self.assertEqual(self.states[first]["resume_id"], first_id)
        self.assertTrue(any("dummy_resume.pdf" in x.value for x in self.app.success))
        self.app.button(key=f"conversation_{second}").click().run()
        self.assertTrue(any("new.pdf" in x.value for x in self.app.success))
        fresh = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=20)
        fresh.query_params["thread"] = second
        fresh.run()
        self.assertFalse(fresh.exception)
        self.assertTrue(any("new.pdf" in x.value for x in fresh.success))
        self.assertEqual(self.states[second]["resume_id"], second_id)
        self.graph.stream.assert_not_called()

    def test_resume_missing_does_not_fall_back_to_demo_and_legacy_requires_binding(self):
        from services.resumes import resume_path
        thread_id = self.app.session_state["thread_id"]
        resume_id = self.states[thread_id]["resume_id"]
        resume_path(resume_id).unlink()
        self.app.run()
        self.assertTrue(any("丢失" in x.value for x in self.app.error))
        self.assertFalse(any(x.label == "使用演示简历" for x in self.app.button))
        self.assertEqual(self.states[thread_id]["resume_id"], resume_id)
        self.states[thread_id] = {"messages": [AIMessage(content="legacy answer", name="ChatBot")]}
        self.app.run()
        self.assertTrue(any("旧会话没有简历 ID" in x.value for x in self.app.caption))
        self.assertNotIn("resume_id", self.states[thread_id])
        self.graph.stream.assert_not_called()

    def test_search_filters_list_without_switching_current_conversation(self):
        current = self.app.session_state["thread_id"]
        self.rows[0]["title"] = "分析简历"
        other = str(uuid4())
        self.rows.insert(0, {"thread_id": other, "title": "Python 岗位"})
        self.app.run()
        self.assertEqual(self.app.session_state["thread_id"], current)
        self.app.text_input(key="conversation_search").set_value("python").run()
        keys = [button.key for button in self.app.button]
        self.assertIn(f"conversation_{other}", keys)
        self.assertNotIn(f"conversation_{current}", keys)
        self.assertEqual(self.app.session_state["thread_id"], current)
        self.app.button(key=f"conversation_{other}").click().run()
        self.assertEqual(self.app.session_state["thread_id"], other)
        self.assertEqual(self.app.query_params["thread"], [other])
        self.app.text_input(key="conversation_search").set_value("没有这种标题").run()
        self.assertTrue(any("没有匹配的会话" in item.value for item in self.app.info))
        self.assertEqual(self.app.session_state["thread_id"], other)
        self.app.button(key="new_conversation").click().run()
        self.assertEqual(self.app.text_input(key="conversation_search").value, "")
        self.assertNotEqual(self.app.session_state["thread_id"], other)
        self.assertFalse(self.app.exception)

    def test_duplicate_titles_switch_by_id_and_navigation_is_first(self):
        original = self.app.session_state["thread_id"]
        other = str(uuid4())
        self.rows.append({"thread_id": other, "title": self.rows[0]["title"]})
        self.app.run()
        self.app.button(key=f"conversation_{other}").click().run()
        self.assertEqual(self.app.session_state["thread_id"], other)
        self.assertNotEqual(original, other)
        self.assertEqual(self.app.sidebar.button[0].key, "new_conversation")
        self.assertEqual(self.app.sidebar.text_input[0].key, "conversation_search")
        self.graph.stream.assert_not_called()

    def test_external_thread_url_change_and_empty_database(self):
        other = str(uuid4())
        self.rows.append({"thread_id": other, "title": "从链接打开"})
        self.app.query_params["thread"] = other
        self.app.run()
        self.assertEqual(self.app.session_state["thread_id"], other)
        self.rows.clear()
        self.app.run()
        self.assertFalse(self.app.exception)
        self.assertEqual(len(self.rows), 1)
        self.assertEqual(self.app.session_state["thread_id"], self.rows[0]["thread_id"])

    def test_fresh_page_restores_saved_thread_without_reinvoking_model(self):
        self.app.chat_input[0].set_value("remember me").run()
        thread_id = self.app.session_state["thread_id"]
        fresh = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=20)
        fresh.query_params["thread"] = thread_id
        fresh.run()
        self.assertFalse(fresh.exception)
        self.assertEqual(fresh.session_state["response_history"], ["test answer"])
        self.assertEqual(len(fresh.session_state["langchain_messages"]), 2)
        self.graph.stream.assert_called_once()

    def test_quick_question_and_save_settings(self):
        self.app.button(key="quick_0").click().run()
        self.assertFalse(self.app.exception)
        self.assertEqual(self.graph.stream.call_args.args[0]["user_input"], "总结我的简历")
        self.app.text_input(key="model_name_input").set_value("changed-model").run()
        self.click_label("💾 保存配置")
        self.assertEqual(self.save.call_args.args[0]["model_name"], "changed-model")
        self.assertFalse(self.app.exception)

    def test_failed_request_can_be_retried(self):
        self.graph.stream.side_effect = RuntimeError("offline test failure")
        with patch("traceback.print_exc"):
            self.app.chat_input[0].set_value("fail").run()
        self.assertFalse(self.app.exception)
        self.assertFalse(self.app.session_state["is_processing"])
        self.assertEqual(self.app.session_state["langchain_messages"], [])
        self.graph.stream.side_effect = self.respond
        self.app.chat_input[0].set_value("retry").run()
        self.assertEqual(self.app.session_state["response_history"][-1], "test answer")

    def test_resume_button_does_not_resubmit_user_message(self):
        def fail(state, config, **kwargs):
            thread_id = config["configurable"]["thread_id"]
            self.states[thread_id] = deepcopy(state)
            self.pending[thread_id] = ("ChatBot",)
            raise RuntimeError("offline failure after checkpoint")
        self.graph.stream.side_effect = fail
        with patch("traceback.print_exc"):
            self.app.chat_input[0].set_value("recover this question").run()
        self.assertTrue(self.app.chat_input[0].disabled)
        self.assertFalse(any(button.label == "⏹️ 停止" for button in self.app.button))
        self.graph.stream.side_effect = self.respond
        self.click_label("▶ 恢复未完成任务")
        self.assertFalse(self.app.exception)
        self.assertIsNone(self.graph.stream.call_args.args[0])
        self.assertEqual(len(self.app.session_state["langchain_messages"]), 2)
        self.assertEqual(self.app.session_state["response_history"], ["test answer"])


if __name__ == "__main__":
    unittest.main()
