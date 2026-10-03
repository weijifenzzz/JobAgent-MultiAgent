"""离线回归：配置、会话、服务，以及真实 LangGraph 的结束路由。"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from langchain_core.messages import AIMessage, HumanMessage

import settings
from services.conversation import NO_REPLY, run_conversation
from ui.session import (
    THINKING, STOPPED, clear_conversation, finish_request,
    get_pending_request, initialize_session, start_request, stop_request,
)


class ConversationTests(unittest.TestCase):
    def test_mutating_graph_and_last_reply(self):
        history = [HumanMessage(content="previous"), AIMessage(content="old")]

        def invoke(state, config, *, context, **kwargs):
            self.assertEqual(config["recursion_limit"], 15)
            context.observer.write_agent_name("ResumeAnalyzer")
            state["messages"].append(AIMessage(content="analysis"))
            context.observer.write_agent_name("JobSearcher")
            state["messages"].append(AIMessage(content="jobs"))
            return state

        result = run_conversation(Mock(invoke=invoke), "new", history, {})
        self.assertEqual(result.reply, "jobs")
        self.assertEqual(len(history), 2)
        self.assertEqual(len(result.messages), 5)
        self.assertEqual(result.agent_sequence, ["ResumeAnalyzer", "JobSearcher"])

    def test_no_new_ai_never_echoes_user_or_old_answer(self):
        graph = Mock(invoke=lambda state, config, **kwargs: state)
        result = run_conversation(graph, "thanks", [AIMessage(content="old")], {})
        self.assertEqual(result.reply, NO_REPLY)

    def test_error_leaves_caller_history_unchanged(self):
        history = []
        graph = Mock()
        graph.invoke.side_effect = RuntimeError("offline test failure")
        with self.assertRaises(RuntimeError):
            run_conversation(graph, "hello", history, {})
        self.assertEqual(history, [])

    def test_real_finish_route_still_calls_chatbot(self):
        with patch("dotenv.load_dotenv"), patch.dict(os.environ, {"LANGCHAIN_TRACING_V2": "false"}):
            import agents
            with patch.object(agents, "init_chat_model"), \
                 patch.object(agents, "get_supervisor_chain") as supervisor, \
                 patch.object(agents, "get_finish_chain") as finish:
                supervisor.return_value.invoke.return_value = AIMessage(content="Finish")
                finish.return_value.invoke.return_value = AIMessage(content="不客气，祝你求职顺利！")
                result = run_conversation(agents.define_graph(), "暂时没有其他问题，谢谢。", [], {})
                self.assertEqual(result.reply, "不客气，祝你求职顺利！")
                finish.return_value.invoke.assert_called_once()
                self.assertTrue(any("ChatBot" in name for name in result.agent_sequence))


class SessionTests(unittest.TestCase):
    def setUp(self):
        self.state = {}
        initialize_session(self.state)

    def test_request_is_executed_once_even_if_reply_matches_placeholder(self):
        start_request(self.state, "hello")
        start_request(self.state, "duplicate")
        self.assertEqual(get_pending_request(self.state), "hello")
        finish_request(self.state, THINKING, ["ChatBot"])
        self.assertIsNone(get_pending_request(self.state))
        self.assertEqual(len(self.state["user_query_history"]), 2)
        self.assertEqual(self.state["agent_sequence_history"][-1], ["ChatBot"])

    def test_stop_and_clear(self):
        start_request(self.state, "hello")
        stop_request(self.state)
        self.assertEqual(self.state["response_history"][-1], STOPPED)
        self.assertFalse(self.state["is_processing"])
        history = Mock()
        clear_conversation(self.state, history)
        history.clear.assert_called_once()
        self.assertEqual(self.state["response_history"], [])
        self.assertEqual(self.state["agent_sequence_history"], [])
        start_request(self.state, "again")
        self.assertEqual(get_pending_request(self.state), "again")


class SettingsTests(unittest.TestCase):
    def test_config_roundtrip_and_priority(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(settings, "CONFIG_FILE", Path(directory) / "temp" / "config.json"), \
             patch.dict(os.environ, {"MODEL_NAME": "environment-model"}, clear=True):
            read_secret = lambda name, default: "secret-model" if name == "MODEL_NAME" else default
            self.assertEqual(settings.load_initial_config(read_secret)["model_name"], "environment-model")
            self.assertTrue(settings.save_persistent_config({"model_name": "saved-model", "temperature": 0.0}))
            config = settings.load_initial_config(read_secret)
            self.assertEqual(config["model_name"], "saved-model")
            self.assertEqual(config["temperature"], 0.0)


if __name__ == "__main__":
    unittest.main()
