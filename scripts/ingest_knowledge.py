"""预览或导入知识库：python -m scripts.ingest_knowledge [--dry-run]。"""

import argparse
from collections import Counter

from services.knowledge import KnowledgeIndex, load_chunks
from settings import initialize_environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="仅本地切分，不调用 API、不修改 Qdrant")
    args = parser.parse_args()
    index = None
    try:
        initialize_environment()
        chunks = load_chunks()
        for name, count in Counter(c["file"] for c in chunks).items():
            print(f"{name}: {count} 个片段")
        print(f"共 {len(chunks)} 个片段。")
        if args.dry_run:
            return
        index = KnowledgeIndex()
        print(f"模型：{index.model}；维度：{index.dimensions}；集合：{index.collection}")
        print("仅将选定的公开教程文本发送给向量模型服务。")
        print(index.ingest(chunks))
    except Exception as exc:
        # 不输出第三方错误体，避免其中包含请求头、凭证或完整配置。
        detail = str(exc) if isinstance(exc, (ValueError, FileNotFoundError)) else "请检查网络、依赖、服务权限及 .env 配置；已写入批次可重跑恢复。"
        print(f"导入失败（{type(exc).__name__}）：{detail}")
        raise SystemExit(1) from None
    finally:
        if index:
            index.close()


if __name__ == "__main__":
    main()
