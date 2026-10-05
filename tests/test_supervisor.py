"""验证模型任务规划、输出校验、有限重试以及两步协作。"""

import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import ValidationError

from chains import get_supervisor_chain
from schemas import RouteSchema
from services.conversation import run_conversation


class CountingModel(FakeListChatModel):
    calls: int = 0

    def _call(self, *args, **kwargs):
        self.calls += 1
        return super()._call(*args, **kwargs)


class SupervisorTests(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, {"LANGCHAIN_TRACING_V2": "false", "LANGSMITH_TRACING": "false"})
        env.start()
        self.addCleanup(env.stop)
        with patch("dotenv.load_dotenv"):
            import agents
        self.agents = agents
        self.runtime = SimpleNamespace(context=SimpleNamespace(model_config={}), store=None)

    def test_schema_rejects_unknown_empty_duplicate_or_unsupported_order(self):
        for steps in [[], ["Finish"], ["unknown"], ["ChatBot", "ChatBot"],
                      ["JobSearcher", "ResumeAnalyzer"], ["WebResearcher", "JobSearcher"],
                      ["ResumeAnalyzer", "JobSearcher", "CoverLetterGenerator"]]:
            with self.subTest(steps=steps), self.assertRaises(ValidationError):
                RouteSchema(steps=steps)
        with self.assertRaises(ValidationError):
            RouteSchema(steps=["ChatBot"], reason="extra")

    def test_invalid_json_then_valid_plan_retries_once(self):
        model = CountingModel(responses=["WebResearcher", '{"steps":["WebResearcher"]}'])
        plan = get_supervisor_chain(model).invoke({"messages": [HumanMessage(content="解释 RAG")]})
        self.assertEqual(plan.steps, ["WebResearcher"])
        self.assertEqual(model.calls, 2)

    def test_invalid_step_order_also_retries(self):
        model = CountingModel(responses=[
            '{"steps":["JobSearcher","ResumeAnalyzer"]}',
            '{"steps":["ResumeAnalyzer","JobSearcher"]}',
        ])
        self.assertEqual(get_supervisor_chain(model).invoke({"messages": []}).steps,
                         ["ResumeAnalyzer", "JobSearcher"])
        self.assertEqual(model.calls, 2)

    def test_persistent_invalid_output_fails_without_keyword_fallback(self):
        model = CountingModel(responses=["ResumeAnalyzer"])
        state = {"user_input": "分析我的简历并找工作", "messages": []}
        with patch.object(self.agents, "init_chat_model", return_value=model):
            with self.assertRaisesRegex(RuntimeError, "连续两次"):
                self.agents.supervisor_node(state, self.runtime)
        self.assertEqual(model.calls, 2)
        self.assertNotIn("next_step", state)
        self.assertNotIn("needs_followup", state)

    def test_all_new_requests_use_model_even_when_old_keywords_match(self):
        cases = [
            ("解释简历和求职信的区别", ["WebResearcher"]),
            ("简历分析和岗位匹配中 RAG 起什么作用", ["WebResearcher"]),
            ("分析我的简历，再推荐合适岗位", ["ResumeAnalyzer", "JobSearcher"]),
            ("分析简历并写求职信", ["ResumeAnalyzer", "CoverLetterGenerator"]),
            ("暂时没有其他问题，谢谢", ["ChatBot"]),
        ]
        for query, steps in cases:
            model = CountingModel(responses=[json.dumps({"steps": steps})])
            with self.subTest(query=query), patch.object(self.agents, "init_chat_model", return_value=model):
                state = self.agents.supervisor_node({"user_input": query, "messages": []}, self.runtime)
            self.assertEqual(model.calls, 1)
            self.assertEqual(state["next_step"], steps[0])
            self.assertEqual(state["needs_followup"], steps[1] if len(steps) == 2 else "")

    def test_followup_does_not_replan_and_rejects_invalid_checkpoint(self):
        with patch.object(self.agents, "init_chat_model") as init:
            state = self.agents.supervisor_node({"needs_followup": "JobSearcher"}, self.runtime)
            self.assertEqual(state["next_step"], "JobSearcher")
            self.assertEqual(state["needs_followup"], "")
            init.assert_not_called()
            with self.assertRaises(ValidationError):
                self.agents.supervisor_node({"needs_followup": "unknown"}, self.runtime)

    def test_real_two_step_graph_plans_once_and_passes_analysis(self):
        for target in ["JobSearcher", "CoverLetterGenerator"]:
            planner = FakeListChatModel(responses=[json.dumps({"steps": ["ResumeAnalyzer", target]})])
            analyzer, worker = Mock(), Mock()
            analyzer.invoke.return_value = {"messages": [AIMessage(content="Python 开发经验")]}
            worker.invoke.return_value = {"messages": [AIMessage(content="任务完成")]}
            with self.subTest(target=target), patch.object(self.agents, "init_chat_model", side_effect=[planner, Mock(), Mock()]) as init, \
                 patch.object(self.agents, "create_agent", side_effect=[analyzer, worker]):
                result = run_conversation(self.agents.define_graph(), "请处理我的求职任务", [], {})
            self.assertEqual(init.call_count, 3)
            self.assertEqual([m.name for m in result.messages if isinstance(m, AIMessage)], ["ResumeAnalyzer", target])
            self.assertIn("Python 开发经验", worker.invoke.call_args.args[0]["messages"][-1].content)


if __name__ == "__main__":
    unittest.main()
