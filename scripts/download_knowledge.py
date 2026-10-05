"""下载固定版本的 Hello-Agents 中文资料：python -m scripts.download_knowledge。"""

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

import httpx

from settings import PROJECT_ROOT

REPO = "datawhalechina/hello-agents"
DESTINATION = PROJECT_ROOT / "data" / "knowledge" / "hello-agents"
CHAPTERS = (4, 6, 8, 9, 12)


def download(revision=None):
    manifest_path = DESTINATION / "sources.json"
    # 再次下载默认沿用已经选定的版本，更新必须明确传入 --revision。
    if revision is None and manifest_path.exists():
        revision = json.loads(manifest_path.read_text(encoding="utf-8"))["revision"]
    with httpx.Client(timeout=60, follow_redirects=True) as client:
        response = client.get(f"https://api.github.com/repos/{REPO}/commits/{revision or 'main'}")
        response.raise_for_status()
        revision = response.json()["sha"]
        response = client.get(f"https://api.github.com/repos/{REPO}/git/trees/{revision}?recursive=1")
        response.raise_for_status()
        tree = response.json()
        if tree.get("truncated"):
            raise ValueError("GitHub 目录返回不完整，停止下载。")
        paths = [item["path"] for item in tree["tree"] if item["type"] == "blob"]
        selected = []
        for chapter in CHAPTERS:
            matches = [p for p in paths if p.startswith(f"docs/chapter{chapter}/")
                       and p.endswith(".md") and re.search(r"[\u4e00-\u9fff]", Path(p).name)]
            if len(matches) != 1:
                raise ValueError(f"第 {chapter} 章未找到唯一的中文正文。")
            selected.append(matches[0])
        answers = "Extra-Chapter/Extra01-参考答案.md"
        if answers not in paths:
            raise ValueError("未找到面试参考答案。")
        selected.append(answers)
        # 全部下载成功后再保存，避免网络失败造成半份来源清单。
        files = {}
        sources = []
        for path in selected + ["LICENSE.txt"]:
            response = client.get(f"https://raw.githubusercontent.com/{REPO}/{revision}/{quote(path)}")
            response.raise_for_status()
            content = response.content
            content.decode("utf-8")
            name = Path(path).name
            files[name] = content
            if path != "LICENSE.txt":
                sources.append({
                    "file": name, "title": Path(path).stem,
                    "source_path": path,
                    "source_url": f"https://github.com/{REPO}/blob/{revision}/{quote(path)}",
                    "sha256": hashlib.sha256(content).hexdigest(),
                })
        DESTINATION.mkdir(parents=True, exist_ok=True)
        for name, content in files.items():
            (DESTINATION / name).write_bytes(content)
        manifest = {
            "repository": f"https://github.com/{REPO}", "revision": revision,
            "author": "Datawhale / Hello-Agents contributors",
            "license": "CC-BY-NC-SA-4.0",
            "downloaded_at": datetime.now(timezone.utc).isoformat(),
            "documents": sources,
        }
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"已保存 {len(sources)} 份正文、许可证和 sources.json。版本：{revision}")
        print(f"目录：{DESTINATION}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", help="指定 Git commit；首次默认 main，之后默认沿用本地版本")
    args = parser.parse_args()
    try:
        download(args.revision)
    except (httpx.HTTPError, OSError, ValueError, KeyError) as exc:
        print(f"下载失败（{type(exc).__name__}）。请检查网络、GitHub 访问权限和 sources.json。")
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
