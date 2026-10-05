"""离线验证运行时依赖可用，但不进入图状态、流式更新和检查点。"""

from schemas import RouteSchema

import os
import unittest
from unittest.mock import Mock, patch

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from runtime_context import RunContext
from services.conversation import RecordingObserver, run_conversation, stream_conversation


class RuntimeContextTests(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, {"LANGCHAIN_TRACING_V2": "false", "LANGSMITH_TRACING": "false"})
        env.start()
        self.addCleanup(env.stop)
        with patch("dotenv.load_dotenv"):
            import agents
        self.agents = agents

    def test_both_service_entrypoints_separate_dependencies_from_inputs(self):
        model_config = {"model": "offline-model", "OPENAI_API_KEY": "runtime-only-test-key"}

        def check(state, config, context):
            self.assertEqual(set(state), {"messages", "user_input"})
            self.assertEqual(config, {"recursion_limit": 15})
            self.assertEqual(context.model_config, model_config)
            self.assertIsNot(context.model_config, model_config)
            self.assertNotIn("runtime-only-test-key", repr(context))
            context.observer.write_agent_name("ChatBot")
            state["messages"].append(AIMessage(content="answer", name="ChatBot"))

        def invoke(state, config, *, context, **kwargs):
            check(state, config, context)
            return state

        # 流式 observer 会调用 LangGraph writer，隔离它但保留角色记录。
        def stream(state, config, *, context, **kwargs):
            with patch("services.conversation.get_stream_writer", return_value=lambda data: None):
                check(state, config, context)
            yield {"type": "updates", "ns": (), "data": {"ChatBot": state}}

        result = run_conversation(Mock(invoke=invoke), "hello", [], model_config)
        events = list(stream_conversation(Mock(stream=stream), "hello", [], model_config))
        self.assertEqual(result.agent_sequence, ["ChatBot"])
        self.assertEqual(events[-1].result.agent_sequence, ["ChatBot"])
        self.assertEqual(events[-1].result.reply, "answer")

    def test_real_graph_checkpoints_and_updates_exclude_runtime_dependencies(self):
        # 单元测试用内存 checkpointer 隔离数据库。
        saver = InMemorySaver()
        graph = self.agents.define_graph(checkpointer=saver)
        config = {"configurable": {"thread_id": "isolated-runtime-test"}}
        observer = RecordingObserver()
        model_config = {"OPENAI_API_KEY": "runtime-only-test-key"}
        context = RunContext(model_config=model_config, observer=observer)
        with patch.object(self.agents, "init_chat_model") as init_model, \
             patch.object(self.agents, "get_supervisor_chain") as supervisor, \
             patch.object(self.agents, "get_finish_chain") as finish:
            supervisor.return_value.invoke.return_value = RouteSchema(steps=["ChatBot"])
            finish.return_value.invoke.return_value = AIMessage(content="goodbye")
            updates = list(graph.stream(
                {"messages": [HumanMessage(content="thanks")], "user_input": "thanks"},
                config, context=context, stream_mode="updates",
            ))
            self.assertEqual(init_model.call_count, 2)
            for call in init_model.call_args_list:
                self.assertIs(call.args[0], model_config)

        self.assertTrue(any("ChatBot" in name for name in observer.get_agent_sequence()))
        self.assertEqual(graph.get_state(config).values["messages"][-1].content, "goodbye")
        for update in updates:
            for state in update.values():
                self.assertNotIn("config", state)
                self.assertNotIn("callback", state)
            self.assertNotIn("runtime-only-test-key", repr(update))
        checkpoints = list(saver.list(config))
        self.assertGreater(len(checkpoints), 1)
        for checkpoint in checkpoints:
            channels = checkpoint.checkpoint["channel_values"]
            self.assertNotIn("config", channels)
            self.assertNotIn("callback", channels)
            self.assertNotIn("runtime-only-test-key", repr(checkpoint))
            self.assertNotIn("RecordingObserver", repr(checkpoint))

    def test_reusing_graph_gets_each_calls_config_and_observer(self):
        graph = self.agents.define_graph()
        first, second = RecordingObserver(), RecordingObserver()
        with patch.object(self.agents, "init_chat_model") as init_model, \
             patch.object(self.agents, "get_supervisor_chain") as supervisor, \
             patch.object(self.agents, "get_finish_chain") as finish:
            supervisor.return_value.invoke.return_value = RouteSchema(steps=["ChatBot"])
            finish.return_value.invoke.return_value = AIMessage(content="goodbye")
            run_conversation(graph, "thanks", [], {"model": "first"}, first)
            run_conversation(graph, "thanks", [], {"model": "second"}, second)
        self.assertEqual([c.args[0]["model"] for c in init_model.call_args_list],
                         ["first", "first", "second", "second"])
        self.assertEqual(len(first.agent_sequence), 1)
        self.assertEqual(len(second.agent_sequence), 1)


if __name__ == "__main__":
    unittest.main()
