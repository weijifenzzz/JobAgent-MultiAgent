"""独立验证向量检索，不调用聊天模型：python -m scripts.search_knowledge \"问题\"。"""

import argparse

from services.knowledge import KnowledgeIndex
from settings import initialize_environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument("--limit", type=int, default=3, choices=range(1, 21))
    args = parser.parse_args()
    index = None
    try:
        initialize_environment()
        index = KnowledgeIndex()
        results = index.search(args.query, args.limit)
        for rank, result in enumerate(results, 1):
            payload = result.payload
            print(f"\n{rank}. {payload['title']} | {payload['section']} | score={result.score:.4f}")
            print(payload["source_url"])
            print(payload["text"][:600])
        if not results:
            print("集合中没有可检索的片段，请先导入。")
    except Exception as exc:
        detail = str(exc) if isinstance(exc, ValueError) else "请检查服务状态、权限及 .env 配置。"
        print(f"检索失败（{type(exc).__name__}）：{detail}")
        raise SystemExit(1) from None
    finally:
        if index:
            index.close()


if __name__ == "__main__":
    main()
