# 本地知识库导入

已实现资料下载、向量化入库、命令行检索，并通过 `search_knowledge_base` 工具接入现有 WebResearcher。

资料选用 Datawhale 的 Hello-Agents：第 4、6、8、9、12 章中文正文和 Extra01 面试参考答案。正文来自 https://github.com/datawhalechina/hello-agents ，许可为 CC BY-NC-SA 4.0，原许可证与作者信息随下载保存。

本次选用版本为 `4b014ad47e2658af24b59f21e7bdb3f89a66205e`，按当前规则得到 500 个片段。要复现这批资料，可以执行 `python -m scripts.download_knowledge --revision 4b014ad47e2658af24b59f21e7bdb3f89a66205e`。其他版本的片段数量可能不同。

## 准备配置

在项目根目录的 `.env` 中填写以下变量。聊天模型仍使用现有页面配置，两者互不覆盖。

```dotenv
EMBEDDING_API_KEY=填写百炼密钥
EMBEDDING_BASE_URL=填写百炼的OpenAI兼容BaseURL
EMBEDDING_MODEL=text-embedding-v4
EMBEDDING_DIMENSIONS=1024
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=jobagent_knowledge_v1
```

如果 Qdrant 启用了认证，再配置 `QDRANT_API_KEY`。本机 Python 通过映射端口访问 Docker 内的 Qdrant。

## 执行命令

在 Anaconda Prompt 执行：

```bat
conda activate jobagent
cd /d "E:\agent develop\projects\JobAgent-MultiAgent"
python -m pip install -r requirements.txt
python -m scripts.download_knowledge
python -m scripts.ingest_knowledge --dry-run
python -m scripts.ingest_knowledge
python -m scripts.search_knowledge "Agent 的短期记忆和长期记忆有什么区别？"
```

下载脚本保存 6 份 Markdown、`LICENSE.txt` 和 `sources.json` 到 `data/knowledge/hello-agents/`。第一次下载时解析 main 对应的 commit，所有文件均取自该 commit；再次执行会使用来源清单中的版本。下载依赖 GitHub 网络连通，脚本不会关闭证书验证。

`--dry-run` 只读取本地正文并显示切分数量，不访问向量服务或 Qdrant。实际导入会将这些公开教程的文本发送到配置的向量服务，产生相应 API 用量；不会读取简历和会话。

导入按 Markdown 标题划分章节，再按最多 1200 个字符、最多 180 个字符重叠切分。每批最多 10 个片段。章标题与小节标题一起参与向量化，许可证和来源清单不作为正文导入。

## 查看结果

打开 http://localhost:6333/dashboard ，找到配置的 collection，查看 points 数量和 payload。

每个 point 表示一个正文片段：

| 字段 | 含义 |
| --- | --- |
| vector | 向量模型生成的数值数组，用于相似度检索 |
| text | 对应的正文片段 |
| title / section | 文档标题和小节标题 |
| source_url / source_revision | 固定版本的原文链接和 Git commit |
| file / source_sha256 | 本地文件名和原文件校验值 |
| doc_id / chunk_index | 文档标识和片段序号 |
| embedding_model / embedding_dimensions | 使用的向量模型和维度 |
| author / license | 资料作者与许可证 |
| owner / pipeline | 本项目导入标记和切分规则版本 |

命令行检索会把问题向量化，返回最相近的片段、来源和相似度分数。分数不是回答正确率。这个命令本身不会生成完整答案；聊天中的 WebResearcher 会将工具返回的片段交给聊天模型组织回答。

## 在聊天中使用

启动页面后可以提问：“请根据知识库解释 Agent 的短期记忆和长期记忆有什么区别，并附上来源。”Supervisor 根据当前意图返回经过校验的 JSON 任务安排，将知识问题交给 WebResearcher；该 Agent 同时拥有知识库、网页搜索和网页抓取工具。

知识类问题通过提示词优先选择本地知识库，时效性问题使用网络工具。检索失败或资料不足时说明情况；明确要求仅依据知识库时，不应用网络或常识补全。这里采用模型自主工具调用，并非每轮都由代码强制检索；来源引用也由提示词约束，尚未增加自动引用核验。

知识库工具每次最多返回 5 个片段，只创建临时客户端，不修改图 State、MySQL 表或已有 Qdrant 记录。工具异常不会将接口异常原文和配置传给模型。前端沿用原来的 WebResearcher 流式展示，无需重新导入已有资料。

页面仍沿用原有的简历绑定校验：如果要求先绑定简历，可绑定演示简历再测试；知识库工具本身不会读取简历。长期记忆与知识库是两套独立功能。

## 重复执行与更新

同一份文档、版本、切分规则和向量模型生成稳定的 point ID。重复导入会检查 Qdrant 中已有的 ID，跳过已成功写入的片段，不再次消耗这些片段的向量额度。网络中断后可以执行相同命令继续。

更新上游内容时运行 `python -m scripts.download_knowledge --revision main`，预览切分后再导入。新片段全部写入成功后，程序才删除本次文档的旧片段。更新中断时可能暂时包含新旧版本，重新执行导入完成更新即可。这不是事务性版本切换，第一版请不要同时运行两个导入进程。

本地正文和来源清单的哈希必须一致；不要直接改下载文件后导入。更换模型或维度时使用新的 `QDRANT_COLLECTION` 并重新生成向量，脚本会拒绝混用模型或维度。删除本地文件不会自动删除 Qdrant 中的数据。

`data/knowledge/` 是下载产物，已加入 `.gitignore`。提交脚本与说明即可，不必将第三方教程复制进自己的仓库。下载内容继续遵守原许可。
