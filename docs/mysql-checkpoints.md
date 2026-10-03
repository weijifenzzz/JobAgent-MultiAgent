# MySQL 会话持久化

应用使用社区维护的 `langgraph-checkpoint-mysql[pymysql,aiomysql,asyncmy]==3.0.0`，checkpointer 实现为 `PyMySQLSaver`；跨会话记忆使用同包的 `PyMySQLStore`。
适配器项目：https://github.com/tjni/langgraph-checkpoint-mysql

本机验证版本：MySQL 8.0.34、Python 3.11、LangGraph 1.2.12、langgraph-checkpoint 4.2.0。
数据库名默认 `jobagent`，使用 `utf8mb4_0900_ai_ci`。适配器的 `JSON_TABLE` 查询使用 MySQL 8 的默认 utf8mb4 排序规则；混用 `utf8mb4_unicode_ci` 会导致 1267 错误。

## 初始化与启动

在项目根目录 `.env` 中配置（真实密码不要提交）：

```dotenv
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_USER=你的用户名
MYSQL_PASSWORD=你的密码
MYSQL_DATABASE=jobagent
```

在 Anaconda Prompt 执行：

```bat
conda activate jobagent
cd /d "E:\agent develop\projects\JobAgent-MultiAgent"
python -m pip install -r requirements.txt
python -m scripts.init_mysql
python -m streamlit run app.py
```

本次本机接入已经安装依赖并初始化数据库；日常只需要最后的启动命令。
初始化需要建库/建表权限；正常运行不执行 DDL，也不会修改其他数据库。
MySQL 不可用时页面显示错误，不会悄悄改用内存保存。

## 会话如何恢复

1. 每个会话分配 UUID，作为 LangGraph 的 `configurable.thread_id`。
2. `app_threads` 提供侧边栏顶部的可滚动会话列表，按最近活动排序；可按标题搜索、点击切换，当前会话高亮。页面 URL 的 `?thread=...` 保存当前会话标识。
3. 新一轮执行前，服务层通过 `graph.get_state(config)` 从 MySQL 读取消息，不信任页面里的旧历史。
4. 当前 State 的消息字段仍采用“完整列表替换”：读出旧消息后添加一条用户消息再提交，同时重置 `next_step`、`needs_followup`、`task_completed`。本次没有引入消息 reducer。
5. `durability="sync"` 让检查点在下一步执行前保存。它不表示逐 token 保存；最后尚未完成的模型输出可能需要重新生成。
6. 页面刷新或服务重启后，按相同 thread_id 恢复 State，并将 HumanMessage 和各个 Agent 的 AIMessage 重建为问答与阶段结果。
7. 如果 `snapshot.next` 非空，页面提供“恢复未完成任务”，使用 `graph.stream(None, config, context=...)`；不重复插入原问题。当前按钮处理执行故障/停止后的待执行节点，尚未实现 `interrupt()` 的人工审批表单。

`StreamlitChatMessageHistory` 现在只是页面缓存，MySQL checkpoint 是持久化模式下的历史来源。
“新会话”创建新的 thread_id，旧记录保留供再次选择；不再把清空页面误认为删除数据库历史。
同一个会话执行时通过 MySQL `GET_LOCK` 排他保护，避免两个页面同时写入完整历史相互覆盖。
恢复失败节点可能再次调用模型和工具；数据库 checkpoint 不保证外部工具副作用只执行一次。

模型 API Key、观察器和数据库连接不在 AgentState 中。每次调用/恢复重新提供 `RunContext`；数据库配置由 `persistence.py` 从环境读取。
本地 `.env` 和网页配置文件的保存方式未改变。跨会话 Store 见 [长期记忆说明](long-term-memory.md)。没有实现多用户鉴权；当前会话列表用于本地单用户应用。
本次接入前只存在于旧 Streamlit 进程内的会话没有自动导入 MySQL；数据库列表从本次接入后的会话开始。

## 会话与检查点的五张表

以下五张表继续负责会话与检查点。接入长期记忆后另有 `store`、`store_migrations` 两张表，因此 `jobagent` 当前共七张项目表。

| 表 | 数据含义 | 常用字段 |
|---|---|---|
| `app_threads` | 应用自己的会话目录，一行代表一段会话 | `thread_id` 唯一标识；`title` 首次提问摘要或验证标题；`created_at`、`updated_at` 时间 |
| `checkpoint_migrations` | 适配器已执行的建表/升级版本，一行是一个迁移编号 | `v`；不是聊天次数 |
| `checkpoints` | 图在不同执行步骤的状态快照索引与调度信息，一次问答通常产生多行 | `thread_id`、`checkpoint_id`、`parent_checkpoint_id`、`checkpoint_ns`、`checkpoint`、`metadata` |
| `checkpoint_blobs` | 按 channel/version 保存实际状态值，例如消息列表 | `channel`（如 messages）、`version`、`type`（序列化格式）、`blob`（二进制数据） |
| `checkpoint_writes` | 节点/任务针对某个检查点的写入，用于故障恢复与已完成结果复用 | `task_id`、`task_path`、`idx`、`channel`、`blob` |

`checkpoint_ns` 为空字符串表示外层图；带节点命名空间的值通常属于内层 Agent 子图。
`checkpoint_ns_hash` 是适配器生成的命名空间索引字段。
`checkpoint` JSON 中的 channel_versions 等信息与 blobs 中的数据配合恢复完整 State；metadata 中可见执行来源、步骤等信息。
`blob` 在 DataGrip 显示为二进制是正常的，应通过 saver/`graph.get_state()` 反序列化，不要直接当 UTF-8 文本或 JSON 编辑。

会话简历新增 `resume_id`、`resume_filename`、`resume_context_start` 三个 State 字段，不增加表或 SQL 列。当前适配器将字符串、整数等简单值直接保存在 `checkpoints.checkpoint` JSON 的 `channel_values` 中；消息列表等复杂值进入 `checkpoint_blobs`。PDF 原文件独立保存，详见 [会话简历绑定](resume-binding.md)。

因此，这套表与“一行一条聊天消息”的 `tb_ai_chat_record` 不同：它还记录执行进度、子图和待恢复的任务。需要独立的业务聊天记录表时，可以另建投影，不能用它替代 checkpoint 调度信息。

可在 DataGrip 的 `jobagent` 数据源运行：

```sql
SELECT thread_id, title, created_at, updated_at
FROM app_threads ORDER BY updated_at DESC;

SELECT thread_id, checkpoint_ns, COUNT(*) AS snapshot_count
FROM checkpoints GROUP BY thread_id, checkpoint_ns;

SELECT channel, type, COUNT(*) AS version_count
FROM checkpoint_blobs GROUP BY channel, type;
```

## 验证方法与结果

离线回归（模拟模型与数据库边界）：

```bat
python -m unittest discover -s tests -v
```

真实 MySQL 跨进程验证：

```bat
python -m scripts.verify_mysql_restart
```

后者会创建标记为“验证：…”的合成会话并保留，便于查看数据库；每个阶段启动新的 Python 进程，不调用真实模型 API。

- 第一个进程写入目标岗位，退出后第二个进程不传旧历史仍能读取事实并续聊，消息总数由 2 变为 4。
- 模拟 ChatBot 节点故障，下一进程恢复该节点；测试禁止重跑已完成的 Supervisor。
- 运行真实 ResumeAnalyzer → Supervisor → JobSearcher 图及 create_agent 子图，验证多 Agent 检查点。
- 全新 Streamlit 页面从真实数据库恢复两轮历史。
- 检查合成密钥、观察器未进入检查点；验证同一会话并发锁。

本次这些检查均已通过。用户自己的模型仍可按以下步骤做端到端体验验证：

1. 点击“新会话”，发送“请记住：我的目标岗位是 Python 后端工程师”，等待完成。
2. 记录侧边栏 thread_id 或收藏当前 URL。
3. 终端 Ctrl+C 停止 Streamlit，再重新运行 `python -m streamlit run app.py`。
4. 打开原 URL，或从侧边栏选择原会话；旧问答应显示。
5. 提问“我刚才说的目标岗位是什么？”；此时模型输入历史来自 MySQL。
