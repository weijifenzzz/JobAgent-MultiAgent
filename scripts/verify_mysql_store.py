"""真实 MySQL 跨进程验证 Store；使用独立测试身份和模拟模型，结束后清理测试数据。"""

import argparse
import json
import os
import subprocess
import sys
from unittest.mock import patch
from uuid import uuid4

from dotenv import load_dotenv
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from persistence import open_database
from services.conversation import stream_conversation, thread_config
from services.memory import delete_memory, list_memories, memory_namespace
from settings import PROJECT_ROOT


def worker(phase, user_id, thread_id):
    import agents
    from chains import get_finish_chain

    with open_database() as db:
        graph = agents.define_graph(checkpointer=db.checkpointer, store=db.store)
        current = list_memories(db.store, user_id)
        if phase in ("write", "update", "delete"):
            city = "上海" if phase == "write" else "杭州"
            entries = [{"kind": "preferences", "key": "preferred_city", "label": "目标城市", "content": city}]
            if phase == "write":
                assert not current
                entries.append({"kind": "facts", "key": "skills", "label": "技能", "content": "熟悉 Python"})
                text = "请记住：目标城市是上海，我熟悉 Python。"
            elif phase == "update":
                assert len(current) == 2
                text = "请记住：目标城市改为杭州。"
            else:
                assert any(r["content"] == "杭州" for r in current)
                text = "请忘记我的目标城市。"
            model = FakeListChatModel(responses=[json.dumps({"entries": entries}, ensure_ascii=False)])
            with patch.object(agents, "init_chat_model", return_value=model):
                events = list(stream_conversation(graph, text, [], {}, thread_id=thread_id, user_id=user_id))
            assert not any(e.type == "text_delta" for e in events), "Extraction JSON leaked"
            assert ("已删除" if phase == "delete" else "已保存") in events[-1].result.reply
            record = db.store.get(memory_namespace(user_id, "preferences"), "preferred_city")
            if phase == "delete":
                assert record is None and len(list_memories(db.store, user_id)) == 1
            else:
                assert record.value["content"] == city
                assert record.value["source_thread_id"] == thread_id
                assert len(list_memories(db.store, user_id)) == 2
        else:
            expected_city = "上海" if phase == "read" else "杭州"
            assert len(current) == (1 if phase == "read_deleted" else 2)
            assert list_memories(db.store, user_id + "_other") == []
            captured = []

            def finish(llm, memory_context=""):
                captured.append(memory_context)
                assert "熟悉 Python" in memory_context
                if phase == "read_deleted":
                    assert "preferred_city" not in memory_context
                else:
                    assert expected_city in memory_context
                return get_finish_chain(llm, memory_context=memory_context)

            with patch.object(agents, "init_chat_model", side_effect=[
                FakeListChatModel(responses=["ChatBot"]), FakeListChatModel(responses=["已参考长期记忆。"])
            ]), patch.object(agents, "get_finish_chain", side_effect=finish):
                events = list(stream_conversation(graph, "请结合我的偏好给些建议。", [], {},
                                                  thread_id=thread_id, user_id=user_id))
            assert len(captured) == 1
            assert len(events[-1].result.messages) == 2, "Other thread's history must not be copied"
        assert not graph.get_state(thread_config(thread_id)).next
    print(f"PASS: {phase}")


def main():
    load_dotenv(PROJECT_ROOT / ".env")
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    os.environ["LANGSMITH_TRACING"] = "false"
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["write", "read", "update", "read_updated", "delete", "read_deleted"])
    parser.add_argument("--user")
    parser.add_argument("--thread")
    args = parser.parse_args()
    if args.phase:
        worker(args.phase, args.user, args.thread)
        return

    user_id = "store_test_" + uuid4().hex
    threads = []
    try:
        for phase in ("write", "read", "update", "read_updated", "delete", "read_deleted"):
            with open_database() as db:
                thread_id = db.threads.create(f"临时 Store 验证：{phase}")
                threads.append(thread_id)
            result = subprocess.run(
                [sys.executable, "-m", "scripts.verify_mysql_store", "--phase", phase,
                 "--user", user_id, "--thread", thread_id], cwd=PROJECT_ROOT,
                capture_output=True, text=True, encoding="utf-8", errors="replace",
                env={**os.environ, "PYTHONIOENCODING": "utf-8"},
            )
            if result.returncode:
                # 避免第三方连接异常或 traceback 意外展示 .env 中的连接凭据。
                raise RuntimeError(f"Store verification failed at phase {phase}; exit={result.returncode}")
            print(f"PASS: independent process {phase}")
    finally:
        with open_database() as db:
            for record in list_memories(db.store, user_id):
                delete_memory(db.store, user_id, record["kind"], record["key"])
            for thread_id in threads:
                db.checkpointer.delete_thread(thread_id)
                with db.threads.connection.cursor() as cursor:
                    cursor.execute("DELETE FROM app_threads WHERE thread_id = %s", (thread_id,))
    print("PASS: cross-process save/read/update/delete, separate threads and user namespaces")
    print("Temporary test memories and conversations removed; local_user data untouched.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Store verification failed: {type(exc).__name__}", file=sys.stderr)
        sys.exit(1)
