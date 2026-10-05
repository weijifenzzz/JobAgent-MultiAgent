"""真实 PDF、真实内存 checkpoint 和嵌套 Agent 的简历隔离回归。"""

from schemas import RouteSchema

import os
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import Mock, patch

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from resume_fixtures import pdf_bytes
from services.conversation import run_conversation, stream_conversation, thread_config
from services.resumes import bind_resume, model_messages, read_resume_bytes, resume_path, save_resume


class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch("dotenv.load_dotenv"))
        self.stack.enter_context(patch.dict(os.environ, {"LANGCHAIN_TRACING_V2": "false", "LANGSMITH_TRACING": "false"}))
        import agents
        from tools import ResumeExtractorTool
        self.agents, self.Tool = agents, ResumeExtractorTool
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.stack.enter_context(patch("services.resumes.RESUME_DIR", self.root / "resumes"))
        self.a = pdf_bytes("Candidate ALPHA Python developer")
        self.b = pdf_bytes("Candidate BRAVO Java developer")
        self.a_id, self.b_id = save_resume(self.a), save_resume(self.b)
        self.saver = InMemorySaver()
        self.graph = agents.define_graph(checkpointer=self.saver)

    def test_content_addressed_files_reuse_and_validate(self):
        self.assertEqual(save_resume(self.a), self.a_id)
        self.assertNotEqual(self.a_id, self.b_id)
        self.assertEqual(len(list((self.root / "resumes").glob("*.pdf"))), 2)
        with self.assertRaises(ValueError):
            resume_path("../../resume")
        with self.assertRaises(ValueError):
            save_resume(b"not a PDF")
        resume_path(self.a_id).write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "不一致"):
            read_resume_bytes(self.a_id)
        self.assertEqual(save_resume(self.a), self.a_id)  # 原文件可修复损坏副本。
        self.assertEqual(read_resume_bytes(self.a_id), self.a)

    def test_tool_reads_injected_id_and_has_no_model_path_argument(self):
        tool = self.Tool(resume_id=self.a_id)
        self.assertEqual(tool.get_input_schema().model_json_schema()["properties"], {})
        self.assertIn("ALPHA", tool.invoke({}))
        self.assertIn("BRAVO", self.Tool(resume_id=self.b_id).invoke({}))
        self.assertIn("尚未绑定", self.Tool().invoke({}))
        resume_path(self.a_id).unlink()
        self.assertIn("丢失", tool.invoke({}))
        blank = save_resume(pdf_bytes(""))
        self.assertIn("OCR", self.Tool(resume_id=blank).invoke({}))

    def test_binding_immediately_persists_and_survives_recompile(self):
        bind_resume(self.graph, "a", self.a_id, "resume.pdf")
        bind_resume(self.graph, "b", self.b_id, "resume.pdf")
        graph = self.agents.define_graph(checkpointer=self.saver)
        for thread, resume_id in (("a", self.a_id), ("b", self.b_id)):
            snapshot = graph.get_state(thread_config(thread))
            self.assertEqual(snapshot.values["resume_id"], resume_id)
            self.assertFalse(snapshot.next)
            self.assertNotIn("messages", snapshot.values)  # 没发问题也已经保存绑定。
        self.assertFalse(bind_resume(graph, "a", self.a_id, "other.pdf", expected_resume_id=self.a_id))
        with self.assertRaisesRegex(ValueError, "另一个页面"):
            bind_resume(graph, "a", self.b_id, "new.pdf")

    def test_change_keeps_history_but_excludes_old_analysis_from_all_model_input(self):
        bind_resume(self.graph, "a", self.a_id, "old.pdf")
        old = [HumanMessage(content="old resume request"), AIMessage(content="OLD_ANALYSIS", name="ResumeAnalyzer")]
        self.graph.update_state(thread_config("a"), {"messages": old}, as_node="ResumeBinding")
        bind_resume(self.graph, "a", self.b_id, "new.pdf", expected_resume_id=self.a_id)
        snapshot = self.graph.get_state(thread_config("a"))
        self.assertEqual(len(snapshot.values["messages"]), 2)
        self.assertEqual(model_messages(snapshot.values), [])
        agent = Mock()
        agent.invoke.return_value = {"messages": [AIMessage(content="NEW_RESULT")]}
        route = Mock()
        route.invoke.return_value = RouteSchema(steps=["JobSearcher"])
        with patch.object(self.agents, "init_chat_model", return_value=FakeListChatModel(responses=["unused"])), \
             patch.object(self.agents, "get_supervisor_chain", return_value=route), \
             patch.object(self.agents, "create_agent", return_value=agent):
            result = run_conversation(self.graph, "找工作", [], {}, thread_id="a")
        self.assertEqual(len(result.messages), 4)
        self.assertEqual([m.content for m in route.invoke.call_args.args[0]["messages"]], ["找工作"])
        self.assertEqual([m.content for m in agent.invoke.call_args.args[0]["messages"]], ["找工作"])
        self.assertNotIn("基于简历分析", result.reply)
        self.assertFalse(bind_resume(self.graph, "a", self.b_id, "again.pdf", expected_resume_id=self.b_id))
        self.assertEqual(self.graph.get_state(thread_config("a")).values["resume_context_start"], 2)

    def test_failed_task_blocks_rebinding_then_recovers_original_resume(self):
        bind_resume(self.graph, "a", self.a_id, "a.pdf")
        with patch.object(self.agents, "supervisor_node"):
            # 图已编译；替换它调用的 chain，制造可恢复的节点错误。
            with patch.object(self.agents, "init_chat_model", return_value=FakeListChatModel(responses=['{"steps": ["ResumeAnalyzer"]}'])), \
                 patch.object(self.agents, "create_agent", side_effect=RuntimeError("offline failure")):
                with self.assertRaisesRegex(RuntimeError, "offline failure"):
                    run_conversation(self.graph, "分析简历", [], {}, thread_id="a")
        self.assertEqual(self.graph.get_state(thread_config("a")).next, ("ResumeAnalyzer",))
        with self.assertRaisesRegex(ValueError, "未完成"):
            bind_resume(self.graph, "a", self.b_id, "b.pdf", expected_resume_id=self.a_id)
        self.assertEqual(self.graph.get_state(thread_config("a")).values["resume_id"], self.a_id)
        resume_path(self.a_id).unlink()
        self.assertEqual(save_resume(self.a), self.a_id)
        self.assertFalse(bind_resume(self.graph, "a", self.a_id, "a.pdf", expected_resume_id=self.a_id))
        self.assertEqual(self.graph.get_state(thread_config("a")).next, ("ResumeAnalyzer",))
        agent = Mock()
        agent.invoke.return_value = {"messages": [AIMessage(content="Recovered")]}
        with patch.object(self.agents, "init_chat_model", return_value=FakeListChatModel(responses=["unused"])), \
             patch.object(self.agents, "create_agent", return_value=agent) as create:
            result = run_conversation(self.graph, "", [], {}, thread_id="a", resume=True)
        self.assertIn("ALPHA", create.call_args.args[1][0].invoke({}))
        self.assertEqual(len(result.messages), 2)

    def test_real_nested_agent_calls_bound_tool(self):
        from test_streaming import ScriptedModel
        bind_resume(self.graph, "a", self.a_id, "a.pdf")
        model = ScriptedModel(scripts=[
            [AIMessageChunk(content="", tool_call_chunks=[{
                "name": "resume_extractor", "args": "{}", "id": "resume-call", "index": 0,
            }])], [AIMessageChunk(content="Analysis complete")],
        ])
        # 捕获原工具真实执行结果，不替换它的文件读取。
        from tools import ResumeExtractorTool
        actual_run = ResumeExtractorTool._run
        extracted = []

        def run(tool):
            text = actual_run(tool)
            extracted.append(text)
            return text

        with patch.object(ResumeExtractorTool, "_run", run), \
             patch.object(self.agents, "init_chat_model", side_effect=[
                 FakeListChatModel(responses=['{"steps": ["ResumeAnalyzer"]}']), model,
             ]):
            events = list(stream_conversation(self.graph, "分析简历", [], {}, thread_id="a"))
        self.assertIn("ALPHA", extracted[0])
        self.assertNotIn("BRAVO", extracted[0])
        self.assertEqual(events[-1].result.reply, "Analysis complete")


if __name__ == "__main__":
    unittest.main()
