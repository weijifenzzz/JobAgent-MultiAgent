"""真实 MySQL 跨进程验证简历绑定；合成 PDF、离线模型，结束清理测试会话。"""

from schemas import RouteSchema

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest.mock import Mock, patch

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage

from persistence import open_database
from services.conversation import run_conversation, thread_config
from services.resumes import bind_resume, model_messages, save_resume
from settings import PROJECT_ROOT


def worker(phase, root, threads):
    import agents
    from tools import ResumeExtractorTool

    files = [(root / name).read_bytes() for name in ("a.pdf", "b.pdf")]
    ids = [hashlib.sha256(content).hexdigest() for content in files]
    with patch("services.resumes.RESUME_DIR", root / "resumes"), open_database() as db:
        graph = agents.define_graph(checkpointer=db.checkpointer, store=db.store)
        if phase == "write":
            for thread, content in zip(threads, files):
                with db.threads.lock(thread):
                    resume_id = save_resume(content)
                    bind_resume(graph, thread, resume_id, "resume.pdf")
                    db.threads.touch(thread)
            print("PASS: two independent bindings saved before any message")
            return

        expected = ids if phase == "read" else [ids[1], ids[1]]
        for thread, resume_id in zip(threads, expected):
            snapshot = graph.get_state(thread_config(thread))
            assert snapshot.values["resume_id"] == resume_id
            assert not snapshot.next
            text = ResumeExtractorTool(resume_id=resume_id).invoke({})
            assert ("ALPHA" if resume_id == ids[0] else "BRAVO") in text

        if phase == "read":
            # 绑定信息由图合并保留，服务输入没有显式重复传 resume_id。
            route = Mock()
            route.invoke.return_value = RouteSchema(steps=["ChatBot"])
            finish = Mock()
            finish.invoke.return_value = AIMessage(content="Saved reply for ALPHA")
            with db.threads.lock(threads[0]), \
                 patch.object(agents, "init_chat_model", return_value=Mock()), \
                 patch.object(agents, "get_supervisor_chain", return_value=route), \
                 patch.object(agents, "get_finish_chain", return_value=finish):
                result = run_conversation(graph, "Hello ALPHA", [], {}, thread_id=threads[0])
                assert len(result.messages) == 2
                bind_resume(graph, threads[0], ids[1], "resume.pdf", expected_resume_id=ids[0])
            print("PASS: new process restored both PDFs, ran offline conversation, then changed A binding")
        else:
            state = graph.get_state(thread_config(threads[0])).values
            assert len(state["messages"]) == 2
            assert state["resume_context_start"] == 2
            assert model_messages(state) == []
            assert isinstance(state["messages"][0], HumanMessage)
            with db.threads.connection.cursor() as cursor:
                cursor.execute(
                    "SELECT checkpoint FROM checkpoints WHERE thread_id = %s "
                    "AND checkpoint_ns = '' ORDER BY checkpoint_id DESC LIMIT 1",
                    (threads[0],),
                )
                checkpoint = json.loads(cursor.fetchone()["checkpoint"])
            values = checkpoint["channel_values"]
            assert values["resume_id"] == ids[1]
            assert values["resume_filename"] == "resume.pdf"
            assert values["resume_context_start"] == 2
            print("PASS: third process restored new binding and context boundary from MySQL checkpoint JSON")
        for thread in threads:
            for checkpoint in db.checkpointer.list(thread_config(thread)):
                assert str(root) not in repr(checkpoint)


def main():
    load_dotenv(PROJECT_ROOT / ".env")
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    os.environ["LANGSMITH_TRACING"] = "false"
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["write", "read", "restored"])
    parser.add_argument("--root", type=Path)
    parser.add_argument("--threads", nargs=2)
    args = parser.parse_args()
    if args.phase:
        worker(args.phase, args.root, args.threads)
        return

    import pymupdf

    threads = []
    try:
        with tempfile.TemporaryDirectory(prefix="jobagent-resume-check-") as directory:
            root = Path(directory)
            for name, text in (("a.pdf", "Candidate ALPHA Python"), ("b.pdf", "Candidate BRAVO Java")):
                with pymupdf.open() as document:
                    document.new_page().insert_text((72, 72), text)
                    document.save(root / name)
            with open_database() as db:
                for name in ("A", "B"):
                    threads.append(db.threads.create(f"临时验证简历绑定 {name}"))
            for phase in ("write", "read", "restored"):
                result = subprocess.run(
                    [sys.executable, "-m", "scripts.verify_resume_binding", "--phase", phase,
                     "--root", str(root), "--threads", *threads],
                    cwd=PROJECT_ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace",
                    env={**os.environ, "PYTHONIOENCODING": "utf-8"},
                )
                # 不直接输出依赖库异常，避免连接配置进入终端。
                if result.returncode:
                    raise RuntimeError(f"Verification phase failed: {phase}")
                for line in result.stdout.splitlines():
                    if line.startswith("PASS:"):
                        print(line)
    finally:
        if threads:
            with open_database() as db:
                for thread in threads:
                    db.checkpointer.delete_thread(thread)
                    with db.threads.connection.cursor() as cursor:
                        cursor.execute("DELETE FROM app_threads WHERE thread_id = %s", (thread,))
            print("PASS: removed only the temporary verification threads")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Verification failed ({type(exc).__name__}); check local MySQL configuration.")
        sys.exit(1)
