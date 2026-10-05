"""知识库导入的去重、断点恢复和模型隔离；不调用真实向量服务。"""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from qdrant_client import QdrantClient

from services.knowledge import KnowledgeIndex, OWNER, PIPELINE, load_chunks


def sample(index=0, revision="v1"):
    return {
        "owner": OWNER, "pipeline": PIPELINE, "doc_id": "test-document",
        "title": "测试资料", "section": "第一节", "text": f"测试内容 {index}",
        "chunk_index": index, "source_revision": revision, "source_sha256": revision,
    }


class KnowledgeTests(unittest.TestCase):
    def setUp(self):
        self.index = KnowledgeIndex.__new__(KnowledgeIndex)
        self.index.model = "test-embedding"
        self.index.dimensions = 3
        self.index.collection = "knowledge-test"
        self.index.qdrant = QdrantClient(":memory:")
        self.index.embed = Mock(side_effect=lambda texts: [[1.0, 0.0, 0.0] for _ in texts])

    def tearDown(self):
        self.index.qdrant.close()

    def ingest(self, chunks):
        return self.index.ingest(chunks, report=lambda _: None)

    def test_repeat_skips_embedding_and_points(self):
        chunks = [sample(i) for i in range(13)]
        self.assertEqual(self.ingest(chunks)["written"], 13)
        self.assertEqual([len(c.args[0]) for c in self.index.embed.call_args_list], [10, 3])
        self.index.embed.reset_mock()
        result = self.ingest(chunks)
        self.assertEqual(result["skipped"], 13)
        self.index.embed.assert_not_called()
        self.assertEqual(len(self.index.existing_points()), 13)

    def test_failed_batch_can_resume(self):
        chunks = [sample(i) for i in range(13)]
        self.index.embed.side_effect = [[[1.0, 0.0, 0.0]] * 10, RuntimeError("network")]
        with self.assertRaises(RuntimeError):
            self.ingest(chunks)
        self.assertEqual(len(self.index.existing_points()), 10)
        self.index.embed.side_effect = lambda texts: [[1.0, 0.0, 0.0] for _ in texts]
        result = self.ingest(chunks)
        self.assertEqual((result["written"], result["skipped"]), (3, 10))

    def test_revision_replaces_old_chunks_only_after_success(self):
        self.ingest([sample(i) for i in range(3)])
        self.index.embed.side_effect = RuntimeError("network")
        with self.assertRaises(RuntimeError):
            self.ingest([sample(0, "v2")])
        self.assertEqual(len(self.index.existing_points()), 3)
        self.index.embed.side_effect = lambda texts: [[1.0, 0.0, 0.0] for _ in texts]
        result = self.ingest([sample(0, "v2")])
        self.assertEqual(result["removed_old"], 3)
        self.assertEqual(len(self.index.existing_points()), 1)

    def test_same_dimension_different_model_is_rejected(self):
        self.ingest([sample()])
        self.index.model = "other-model"
        self.index.embed.reset_mock()
        with self.assertRaisesRegex(ValueError, "其他向量模型"):
            self.ingest([sample()])
        self.index.embed.assert_not_called()

    def test_different_dimensions_rejected(self):
        self.ingest([sample()])
        self.index.dimensions = 4
        with self.assertRaisesRegex(ValueError, "维度"):
            self.index.existing_points()

    def test_manifest_selects_only_body_and_checks_integrity(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            content = "# 文档\n\n## 记忆\n\n" + "短期记忆和长期记忆。" * 200
            raw = content.encode()
            (directory / "lesson.md").write_bytes(raw)
            (directory / "LICENSE.txt").write_text("not knowledge", encoding="utf-8")
            (directory / "unlisted.md").write_text("not knowledge", encoding="utf-8")
            manifest = {"repository": "https://example.org/repo", "revision": "v1",
                        "license": "test", "author": "test", "documents": [{
                            "file": "lesson.md", "title": "教程", "source_path": "lesson.md",
                            "source_url": "https://example.org/lesson", "sha256": hashlib.sha256(raw).hexdigest(),
                        }]}
            (directory / "sources.json").write_text(json.dumps(manifest), encoding="utf-8")
            chunks = load_chunks(directory)
            self.assertGreater(len(chunks), 1)
            self.assertTrue(all(c["file"] == "lesson.md" and len(c["text"]) <= 1200 for c in chunks))
            self.assertIn("记忆", chunks[-1]["section"])
            (directory / "lesson.md").write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "校验"):
                load_chunks(directory)


if __name__ == "__main__":
    unittest.main()
