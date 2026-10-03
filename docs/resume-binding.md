# 独立保存简历 + 会话绑定 resume_id

## 使用方式

1. 选择或新建会话，在“当前会话的简历”中选择 PDF，点击“绑定到当前会话”。
2. 绑定成功后再发送“分析我的简历”。新会话不会自动继承上一段会话的文件。
3. 切换会话或重启应用后，按 thread_id 从 MySQL checkpoint 恢复绑定。
4. 需要演示数据时明确点击“使用演示简历”。不会自动回退到演示文件或旧的 `temp/resume.pdf`。

旧 checkpoint 没有 resume_id，程序无法确定旧聊天当时使用哪个文件，必须手动重新绑定。旧文件不会被自动删除，可以选择原文件重新上传。旧会话若还有待恢复任务，先恢复或新建会话。

## 文件与状态流转

```text
上传 PDF 字节
  → 校验 PDF 可打开、未加密、含页面
  → SHA-256 得到 resume_id
  → 保存 temp/resumes/<resume_id>.pdf
  → graph.update_state 保存当前会话绑定
  → Agent 从 State 取 resume_id，创建带绑定的简历工具
  → 模型调用工具 {}，工具按 ID 定位 PDF
  → load_resume 提取文字
```

哈希按完整文件字节计算；字节完全相同复用文件，文本相同但 PDF 元数据不同可能产生不同 ID。原始文件名只用于展示，同名文件不会互相覆盖。

例如 A 会话绑定 `hash_a`，B 会话绑定 `hash_b`。之后把 A 改绑到 `hash_c`，B 仍使用 `hash_b`，原 `hash_a.pdf` 也保留。文件存储目录由项目绝对路径确定，不受启动时工作目录影响。

## 修改的代码与上下文

### services/resumes.py：独立文件服务

- `save_resume()` 校验并保存 PDF，使用同目录临时文件和 `os.replace()`，避免正常并发读到未写完整的文件。
- `resume_path()` 只接受 64 位小写十六进制 ID，并检查解析后的文件路径仍在存储目录。
- `read_resume_bytes()` 检查文件存在、实际内容哈希匹配。绑定丢失或文件损坏时明确报错，不悄悄换文件。
- `bind_resume()` 在调用方持有会话锁时写 checkpoint；比较页面看到的旧 ID，防止陈旧页面覆盖另一个页面刚更新的绑定。
- `model_messages()` 提供当前简历对应的模型消息范围，页面仍可读取完整 messages。

重新上传相同字节不会重置上下文；文件丢失/损坏时，相同原文件可以修复存储副本。有未完成任务时禁止换成另一份简历，允许修复同一份文件后继续执行。

### agents.py：把绑定持久化并传到内部 Agent

AgentState 新增：

```python
resume_id: str             # 文件身份
resume_filename: str       # 页面显示用的原始文件名
resume_context_start: int  # 当前简历上下文从 messages 的哪个下标开始
```

图增加管理节点 `ResumeBinding`，它只连到 END。绑定时使用：

```python
graph.update_state(config, updates, as_node="ResumeBinding")
```

这会创建新的 checkpoint，逻辑上把此次更新归属于 ResumeBinding，不会调用该节点函数，也不会执行模型。它的后继是 END，所以绑定后不会遗留待执行的 Supervisor 任务。即使还没提问，绑定也已经持久化。

ResumeAnalyzer 和 CoverLetterGenerator 构造工具时显式传入：

```python
ResumeExtractorTool(resume_id=state.get("resume_id"))
```

这是外层 State 到内部 create_agent 工具的连接。内层 Agent 不必把 resume_id 当消息或让模型生成路径。

Supervisor、简历分析、求职信、岗位搜索、网页研究、ChatBot 的模型输入都经过 `model_messages()`。岗位搜索和求职信寻找旧 ResumeAnalyzer 结果时，也只在该范围查找。Supervisor 不再把截取后的消息覆盖回完整 State.messages。

### tools.py 和 data_loader.py：按绑定读取，保留错误语义

ResumeExtractorTool 的绑定 ID 保存在 Pydantic PrivateAttr 中，模型看到的参数 schema 为空。模型只能调用 `resume_extractor({})`；ID 和文件路径由程序决定。

工具校验对应文件后将路径交给 `load_resume()`。加载器对缺失、空文件、解析错误和未提取到文字的扫描件抛出异常；工具转成 ToolException，由框架的 `handle_tool_error=True` 返回明确的工具错误。错误文本不再当作正常简历内容返回。没有新增 OCR 能力。

原来未使用的 `save_resume_pdf()` 删除，文件保存统一由独立服务负责。

### app.py、ui/resume.py、ui/sidebar.py、ui/session.py：先选会话，再处理简历

app.main 预留侧边栏简历区域；run_page 先选择 thread_id、恢复 snapshot，再绘制该会话的简历面板。不能在选定会话前上传并覆盖一个全局文件。

ui/resume.py 使用表单明确提交，上传组件 key 包含 thread_id 和当前 resume_id。切换会话不会把上一段会话的上传值自动绑定过来。保存与 checkpoint 更新共用现有 MySQL 会话锁；处理请求期间禁用更换。

ui/sidebar.py 只返回模型配置；移除旧的固定路径上传及自动演示回退。ui/session.py 移除全局 resume_saved / resume_filename 缓存，绑定以 checkpoint 为准。app.py 的发起请求校验读取当前会话的文件可用状态。

services/conversation.py 本次无需修改。现有服务每轮读取 checkpoint，再提交新消息和路由字段；LangGraph 保留没有出现在此次输入里的 resume_id 等状态字段。

## 更换简历如何避免旧结果混入

假设旧 messages 有 4 条。绑定不同简历时：

```python
resume_context_start = 4
```

下一轮把新问题追加到 messages 下标 4，模型只读取 `messages[4:]`。旧四条仍在数据库和聊天界面中，但不会给模型，也不会被当成当前简历分析。

这是保守的范围隔离：更换前的一般聊天同样不再提供给模型，需要的任务条件请重新说明。相同文件重复上传不改变边界。LangGraph Store 中用户明确保存的跨会话偏好/事实仍继续注入，不会因为更换简历而被删除；过时的长期事实需在记忆面板更新。

## MySQL 数据有什么变化

没有新增数据库表、SQL 列或依赖，无需执行迁移脚本。

- app_threads 仍保存会话目录，绑定操作更新最近活动时间。
- 绑定和换绑产生新的 checkpoint。
- 当前 MySQL 适配器把 resume_id、resume_filename、resume_context_start 这样的字符串/整数保存在 checkpoints.checkpoint JSON 的 channel_values 中。
- messages 等复杂值由 checkpoint_blobs 保存，checkpoint_writes 继续承担执行写入/恢复职责。
- Store 表不保存本次简历绑定；一份 PDF 属于文件层，一段会话绑定哪个 PDF 属于 checkpoint State。

原 PDF 字节不会写入 checkpoint；分析后产生的聊天文本仍正常保存。可通过以下只读查询查看绑定历史（不同 checkpoint 会重复显示）：

```sql
SELECT thread_id, checkpoint_id,
       JSON_UNQUOTE(JSON_EXTRACT(checkpoint, '$.channel_values.resume_id')) AS resume_id,
       JSON_UNQUOTE(JSON_EXTRACT(checkpoint, '$.channel_values.resume_filename')) AS resume_filename,
       JSON_EXTRACT(checkpoint, '$.channel_values.resume_context_start') AS context_start
FROM checkpoints
WHERE checkpoint_ns = ''
  AND JSON_EXTRACT(checkpoint, '$.channel_values.resume_id') IS NOT NULL
ORDER BY thread_id, checkpoint_id DESC;
```

数据库和 PDF 文件是两个存储位置，未使用分布式事务：先保存文件，再更新绑定；中途失败最多留下未绑定文件。当前不自动清理旧 PDF，因为其他会话/历史 checkpoint 可能仍引用它们。

虽然目录叫 temp，`temp/resumes/` 是当前方案的持久文件目录，不应随意删除；备份或迁移要同时保留 MySQL 数据和该目录。现有 `.gitignore` 的 temp/ 规则覆盖所有上传文件。

## 重启与验证

Anaconda Prompt 中停止原服务（Ctrl+C），执行：

```bat
conda activate jobagent
cd /d "E:\agent develop\projects\JobAgent-MultiAgent"
python -m streamlit run app.py
```

手工验证：A 会话上传 A.pdf 并分析；新建 B 会话上传 B.pdf 并分析；切回 A 查看绑定；重启服务后再次切换；最后在 A 改绑新简历并重新分析。

自动化验证：

```bat
python -m unittest discover -s tests -v
python -m scripts.verify_resume_binding
```

单元/页面测试覆盖真实 PDF、去重与内容验证、工具参数不可改路径、嵌套 Agent 调用工具、上下文隔离、故障恢复、上传和会话切换、重新打开页面。模型响应使用离线替身。

MySQL 脚本在独立进程中写入、读取、续聊、换绑并再次恢复，核对数据库 JSON 中的绑定；使用合成 PDF，结束后清理本次临时会话与文件，不调用模型 API，也不改已有个人会话。
