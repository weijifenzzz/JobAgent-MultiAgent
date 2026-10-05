"""真实 MySQL + 独立 Python 进程验证；模型被替换，不消耗 API 额度。"""

from schemas import RouteSchema

import argparse
import os
import subprocess
import sys
from unittest.mock import Mock, patch

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import ChatGeneration, ChatResult

from persistence import open_database
from services.conversation import stream_conversation, thread_config
from settings import PROJECT_ROOT

FACT = "我的目标岗位是 Python 后端工程师。"
TEST_SECRET = "synthetic-runtime-key-not-for-storage"


class OfflineAgentModel(BaseChatModel):
    """执行真实 create_agent 子图，只有模型响应是本地合成的。"""

    @property
    def _llm_type(self):
        return "mysql-integration-test"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="合成的分析与建议。"))])


def check_page_restore(thread_id):
    """启动全新页面会话，验证真实 MySQL 的记录能重建聊天区域。"""
    import tempfile
    from pathlib import Path
    from streamlit.testing.v1 import AppTest
    import ui.sidebar as sidebar
    with tempfile.TemporaryDirectory() as directory, \
         patch("ui.resume.PROJECT_ROOT", Path(directory)), \
         patch("services.resumes.RESUME_DIR", Path(directory) / "resumes"), \
         patch.object(sidebar, "load_initial_config", return_value={
             "model_name": "offline", "api_key": "offline", "base_url": "https://example.invalid",
             "temperature": 0.3, "serper_key": "", "firecrawl_key": "",
         }), patch("settings.initialize_environment"), \
         patch("streamlit_analytics2.start_tracking"), patch("streamlit_analytics2.stop_tracking"):
        (Path(directory) / "dummy_resume.pdf").write_bytes(b"test resume")
        page = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=30)
        page.query_params["thread"] = thread_id
        page.run()
        assert not page.exception
        assert page.session_state["thread_id"] == thread_id
        assert len(page.session_state["langchain_messages"]) == 4
        assert "Python" in page.session_state["response_history"][-1]
    print("PASS: fresh Streamlit page restored actual MySQL history")


def worker(phase, thread_id):
    from agents import define_graph
    import agents

    def answer(inputs):
        humans = [m.content for m in inputs["messages"] if isinstance(m, HumanMessage)]
        if phase == "fail":
            raise RuntimeError("simulated-process-failure")
        if FACT not in humans:
            raise AssertionError("历史事实未从 checkpoint 传入模型。")
        return AIMessage(content="你的目标岗位是 Python 后端工程师。")

    supervisor = Mock()
    supervisor.invoke.return_value = RouteSchema(steps=["ChatBot"])
    if phase == "recover":
        supervisor.invoke.side_effect = AssertionError("恢复不应重新执行已完成的 Supervisor")
    finish = Mock(invoke=answer)
    with open_database() as db, patch.object(agents, "init_chat_model", return_value=OfflineAgentModel()), \
         patch.object(agents, "get_supervisor_chain", return_value=supervisor), \
         patch.object(agents, "get_finish_chain", return_value=finish):
        graph = define_graph(checkpointer=db.checkpointer)
        with db.threads.lock(thread_id):
            try:
                events = list(stream_conversation(
                    graph, ("分析简历并推荐岗位" if phase == "collaborate" else
                            "我的目标岗位是什么？" if phase == "continue" else FACT),
                    [HumanMessage(content="这个本地缓存应被忽略")],
                    {"OPENAI_API_KEY": TEST_SECRET}, thread_id=thread_id, resume=phase == "recover",
                ))
            except RuntimeError as exc:
                if phase != "fail" or str(exc) != "simulated-process-failure":
                    raise
                assert graph.get_state(thread_config(thread_id)).next == ("ChatBot",)
                print("PASS: saved failed task; next=ChatBot")
                return
            assert phase != "fail", "预期的模拟故障没有发生"
            result = events[-1].result
            assert len(result.messages) == (3 if phase == "collaborate" else 4 if phase == "continue" else 2)
            if phase == "collaborate":
                assert result.agent_sequence == ["ResumeAnalyzer", "JobSearcher"]
            else:
                assert FACT in [m.content for m in result.messages]
                assert "Python" in result.reply
            assert not graph.get_state(thread_config(thread_id)).next
            for checkpoint in db.checkpointer.list(thread_config(thread_id)):
                serialized = repr(checkpoint)
                assert TEST_SECRET not in serialized
                assert "_StreamObserver" not in serialized
            db.threads.touch(thread_id)
            print(f"PASS: {phase}; messages={len(result.messages)}; runtime excluded")


def main():
    load_dotenv(PROJECT_ROOT / ".env")
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    os.environ["LANGSMITH_TRACING"] = "false"
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["write", "continue", "fail", "recover", "collaborate", "page"])
    parser.add_argument("--thread")
    args = parser.parse_args()
    if args.phase:
        if args.phase == "page":
            check_page_restore(args.thread)
        else:
            worker(args.phase, args.thread)
        return
    with open_database() as db:
        conversation = db.threads.create("验证：跨进程续聊")
        failed = db.threads.create("验证：故障后恢复")
        collaboration = db.threads.create("验证：多 Agent 检查点")
    for phase, thread_id in [("write", conversation), ("continue", conversation),
                             ("fail", failed), ("recover", failed),
                             ("collaborate", collaboration), ("page", conversation)]:
        subprocess.run([sys.executable, "-m", "scripts.verify_mysql_restart",
                        "--phase", phase, "--thread", thread_id], cwd=PROJECT_ROOT, check=True)
    with open_database() as first, open_database() as second:
        with first.threads.lock(conversation):
            try:
                with second.threads.lock(conversation):
                    raise AssertionError("Concurrent execution lock not enforced")
            except RuntimeError:
                print("PASS: concurrent execution blocked")
    print(f"PASS: restart verification complete; conversation={conversation}; recovery={failed}")
    print(f"Three synthetic verification conversations retained; collaboration={collaboration}.")


if __name__ == "__main__":
    main()
