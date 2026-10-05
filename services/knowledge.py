"""知识库文档切分、向量调用与 Qdrant 读写；不依赖页面或聊天模型配置。"""

import hashlib
import json
import math
import uuid
from pathlib import Path

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from openai import OpenAI
from qdrant_client import QdrantClient, models

from settings import PROJECT_ROOT, get_setting

KNOWLEDGE_DIR = PROJECT_ROOT / "data" / "knowledge" / "hello-agents"
PIPELINE = "hello-agents-markdown-v1-1200-180"
OWNER = "jobagent-knowledge-v1"


def load_chunks(directory=KNOWLEDGE_DIR):
    directory = Path(directory).resolve()
    manifest = json.loads((directory / "sources.json").read_text(encoding="utf-8"))
    headers = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3"), ("####", "h4")],
        strip_headers=False,
    )
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200, chunk_overlap=180,
        separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""],
    )
    chunks = []
    for source in manifest["documents"]:
        path = (directory / source["file"]).resolve()
        if path.parent != directory or path.suffix != ".md":
            raise ValueError("来源清单包含非法文件路径。")
        content = path.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        if digest != source["sha256"]:
            raise ValueError(f"{path.name} 与 sources.json 的校验值不符，请重新下载。")
        sections = headers.split_text(content.decode("utf-8"))
        doc_id = hashlib.sha256((manifest["repository"] + source["source_path"]).encode()).hexdigest()
        index = 0
        for section in sections:
            heading = " / ".join(section.metadata.values()) or source["title"]
            for text in splitter.split_text(section.page_content):
                chunks.append({
                    "owner": OWNER, "pipeline": PIPELINE, "doc_id": doc_id,
                    "title": source["title"], "section": heading,
                    "text": text, "chunk_index": index, "file": source["file"],
                    "source_url": source["source_url"], "source_revision": manifest["revision"],
                    "source_sha256": digest, "license": manifest["license"],
                    "author": manifest["author"],
                })
                index += 1
        if index == 0:
            raise ValueError(f"{path.name} 没有可导入内容。")
    if not chunks:
        raise ValueError("来源清单没有正文。")
    return chunks


def embedding_text(chunk):
    return f"{chunk['title']}\n{chunk['section']}\n\n{chunk['text']}"


def point_id(chunk, model, dimensions):
    identity = json.dumps([PIPELINE, chunk["doc_id"], chunk["source_revision"],
                           chunk["source_sha256"], chunk["chunk_index"],
                           embedding_text(chunk), model, dimensions], ensure_ascii=False)
    return str(uuid.uuid5(uuid.NAMESPACE_URL, identity))


class KnowledgeIndex:
    def __init__(self):
        self.model = get_setting("EMBEDDING_MODEL", "text-embedding-v4").strip()
        self.dimensions = int(get_setting("EMBEDDING_DIMENSIONS", "1024"))
        self.collection = get_setting("QDRANT_COLLECTION", "jobagent_knowledge_v1").strip()
        key = get_setting("EMBEDDING_API_KEY").strip()
        base_url = get_setting("EMBEDDING_BASE_URL").strip()
        if not key or not base_url or not self.model or not self.collection or self.dimensions <= 0:
            raise ValueError("请检查 .env 中的 EMBEDDING_* 和 QDRANT_COLLECTION 配置。")
        self.embedding = OpenAI(api_key=key, base_url=base_url, timeout=60, max_retries=2)
        self.qdrant = QdrantClient(
            url=get_setting("QDRANT_URL", "http://localhost:6333"),
            api_key=get_setting("QDRANT_API_KEY") or None, timeout=60,
        )

    def close(self):
        self.embedding.close()
        self.qdrant.close()

    def existing_points(self, create=False):
        if not self.qdrant.collection_exists(self.collection):
            if not create:
                raise ValueError("知识库集合不存在，请先导入。")
            self.qdrant.create_collection(
                self.collection,
                vectors_config=models.VectorParams(size=self.dimensions, distance=models.Distance.COSINE),
            )
        info = self.qdrant.get_collection(self.collection)
        vectors = info.config.params.vectors
        if not isinstance(vectors, models.VectorParams) or vectors.size != self.dimensions or vectors.distance != models.Distance.COSINE:
            raise ValueError("集合的向量维度或距离配置不匹配，请使用新的 QDRANT_COLLECTION。")
        points = {}
        offset = None
        while True:
            records, offset = self.qdrant.scroll(
                self.collection, limit=256, offset=offset, with_vectors=False,
                with_payload=["owner", "embedding_model", "embedding_dimensions", "doc_id"],
            )
            for record in records:
                payload = record.payload or {}
                if (payload.get("owner") != OWNER or payload.get("embedding_model") != self.model
                        or payload.get("embedding_dimensions") != self.dimensions):
                    raise ValueError("集合含其他来源或其他向量模型的数据，请改用新的 QDRANT_COLLECTION。")
                points[str(record.id)] = payload
            if offset is None:
                break
        return points

    def embed(self, texts):
        response = self.embedding.embeddings.create(
            model=self.model, input=texts, dimensions=self.dimensions, encoding_format="float",
        )
        items = sorted(response.data, key=lambda item: item.index)
        if [item.index for item in items] != list(range(len(texts))):
            raise ValueError("向量服务返回的条数或索引不正确。")
        vectors = [item.embedding for item in items]
        if any(len(v) != self.dimensions or not all(math.isfinite(x) for x in v) or not any(v) for v in vectors):
            raise ValueError("向量服务返回了无效向量或维度与配置不一致。")
        return vectors

    def ingest(self, chunks, report=print):
        existing = self.existing_points(create=True)
        desired = {point_id(c, self.model, self.dimensions): c for c in chunks}
        pending = [(pid, c) for pid, c in desired.items() if pid not in existing]
        skipped = len(desired) - len(pending)
        report(f"共 {len(desired)} 个片段，已有 {skipped} 个，需要向量化 {len(pending)} 个。")
        # v4 每次最多 10 条；成功写入的批次可在重跑时直接跳过。
        for start in range(0, len(pending), 10):
            batch = pending[start:start + 10]
            vectors = self.embed([embedding_text(c) for _, c in batch])
            points = [models.PointStruct(
                id=pid, vector=vector,
                payload={**chunk, "embedding_model": self.model, "embedding_dimensions": self.dimensions},
            ) for (pid, chunk), vector in zip(batch, vectors)]
            self.qdrant.upsert(self.collection, points=points, wait=True)
            report(f"已写入 {min(start + 10, len(pending))}/{len(pending)} 个新片段。")
        # 全部新片段写完才移除本次文档的旧版本，不清空集合，也不删除其他文档。
        doc_ids = {c["doc_id"] for c in chunks}
        stale = [pid for pid, payload in existing.items()
                 if payload["doc_id"] in doc_ids and pid not in desired]
        if stale:
            self.qdrant.delete(self.collection, points_selector=models.PointIdsList(points=stale), wait=True)
        return {"chunks": len(desired), "written": len(pending), "skipped": skipped, "removed_old": len(stale)}

    def search(self, query, limit=5):
        if not query.strip():
            raise ValueError("查询内容不能为空。")
        self.existing_points()  # 第一版小型知识库：校验模型，防止不同向量空间混用。
        vector = self.embed([query])[0]
        return self.qdrant.query_points(
            self.collection, query=vector, limit=limit, with_payload=True, with_vectors=False,
        ).points
