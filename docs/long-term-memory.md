# LangGraph Store 跨会话长期记忆

## 先理解两个不同的标识

`thread_id` 表示“哪一段会话”，用于 checkpoint。`user_id` 表示“谁的长期记忆”，用于 Store 的 namespace。目前没有登录系统，页面和服务统一使用 `local_user`。未来接入登录时，后端应将经过认证的用户 ID 传给 `RunContext`，并为会话目录增加所属用户和访问控制；只增加 namespace 不等于实现登录鉴权。

会话 A 和会话 B 的消息列表各自独立，但都可以读取：

```python
("users", "local_user", "preferences")  # 求职偏好
("users", "local_user", "facts")        # 用户自述事实，例如技能和经历
```

本次采用官方定义的 `BaseStore` 接口和社区 MySQL 实现 `PyMySQLStore`。不是 `InMemoryStore`，也没有自行用 SQL 替代 Store API。

- LangGraph Store 文档：https://docs.langchain.com/oss/python/langgraph/stores
- MySQL 适配器：https://github.com/tjni/langgraph-checkpoint-mysql

## 实际使用

1. 侧边栏展开“长期记忆”，选择“偏好”，名称填“目标城市”、内容填“上海”，点击保存。
2. 新建另一会话，发送“查看我的记忆”，可看到刚才保存的城市。
3. 发送“请记住：我的目标城市改为杭州，我熟悉 Python。”。这会调用当前配置的模型提取条目，然后实际写入 Store，写成功后才返回确认。
4. 在面板选择已有条目检查提取内容，必要时直接修正。自然语言提取不保证语义百分之百正确。
5. 发送“请忘记我的目标城市。”，或选择该条目点击删除。删除后其他会话不再从 Store 读到城市，但原聊天和 checkpoint 不会被擦除。
6. 重启应用后，记忆仍然存在。

普通问答不会自动存储偏好。请使用明确前缀 `请记住：`、`更新记忆：`、`请忘记`、`删除记忆：`，或使用面板。记忆请求处理后直接结束本轮；如果还要搜索岗位等业务任务，请另发一条消息。

面板直接使用 Store，不调用模型。通过聊天保存/删除通常增加一次提取模型调用；“查看我的记忆”由代码读取并生成列表，不调用模型。页面的聊天入口仍要求模型配置完整。

## 保存链路与代码上下文

```text
页面输入 → services/conversation.py 构造本轮输入和 RunContext
  → MemoryManager（agents.py 的 memory_node）
      ├─ 普通问题 → Supervisor → 对应业务 Agent → END
      └─ 明确记忆命令 → services/memory.py
            查看：store.search → 列表回复
            保存：模型提取 → Pydantic 校验 → store.put → 成功回复
            删除：模型匹配条目 → 校验已存在 → store.delete → 成功回复
          → END
```

### 1. persistence.py：提供数据库支持的 Store

`DatabaseSession` 在 checkpointer、threads 之外新增 `store`。`initialize_database()` 执行 `PyMySQLStore(connection).setup()`，由适配器创建它自己的表。

`open_database()` 给 saver 和 Store 各开一个连接。它们内部各有自己的锁，不能依赖一个锁保护另一个组件；LangGraph 可能在后台保存 checkpoint，所以两者不能共用同一个 PyMySQL 连接。页面结束或 rerun 时连接由上下文管理器关闭。

当前适配器版本的 Store 包入口会导入同步、异步驱动，因此 requirements 同时安装 `pymysql,aiomysql,asyncmy` extras；应用自身仍使用同步 `PyMySQLStore`。

### 2. runtime_context.py / services/conversation.py：传入用户身份

`RunContext` 新增 `user_id`，默认是 `LOCAL_USER_ID = "local_user"`。两个对话服务入口都接受可选的 `user_id` 参数，并传给 `context=RunContext(...)`。它不由模型生成，也没有混进 `AgentState` 的业务状态。

`thread_id` 仍放在 `configurable` 中。两个值即使不同也不冲突：同一个 user_id 可以有多个 thread_id。

### 3. agents.py：给图挂 Store，再增加入口节点

```python
graph = workflow.compile(checkpointer=checkpointer, store=store)
```

Store 由图注入 `runtime.store`，不需要把数据库连接放进 State。只在传入 Store 时添加 `MemoryManager` 入口，未配置 Store 的离线调用继续走原流程。

`memory_node` 先检查本轮 `state["user_input"]`。普通问题只返回 `{"next_step": "Supervisor"}`；明确命令调用 `handle_memory_request()`，再把真实处理结果作为 `AIMessage(name="MemoryManager")` 加入消息。条件边根据 `next_step` 进入 Supervisor 或结束。

`config["configurable"]["thread_id"]` 仅作为记忆来源写入 value，不参与 namespace。

### 4. services/memory.py：记忆业务逻辑

- `memory_intent()`：判断明确的保存/删除/查看前缀，不为普通聊天额外调用模型。
- `extract_entries()`：把当前请求和已有记忆交给模型，要求输出 `kind/key/label/content`。只分析本次输入，不自动提取演示简历、工具搜索结果、模型建议。
- `MemoryEntry` / `MemoryPlan`：验证 JSON 格式、类型、键格式和长度；格式错误不写入。它们不能证明模型提取的内容真实或准确。
- `memory_namespace()`：由程序根据 user_id 和类别决定命名空间。
- `save_memory()`：调用 `store.put(namespace, key, value)`。同一 namespace/key 表示同一个条目，重复保存会替换该条目的整个 value，不会追加第二条。不同属性应使用不同 key。
- `delete_memory()`：通过 `store.get()` 确认存在后调用 `store.delete()`。
- `list_memories()`：按两类 namespace 分页 `store.search()`，再次检查完整 namespace，供面板和模型使用。

常用 key 包括 `preferred_city`、`target_role`、`skills`、`experience`。面板新增时根据名称选择常用 key，其他名称生成稳定哈希；编辑已有条目保留 key。自然语言提取要求复用同一属性的已有 key，以实现更新。

一次保存多个条目时逐条写入，不是覆盖用户全部记忆。Store 写入和 checkpoint 写入也不是同一个数据库事务：极端情况下记忆已写入而节点保存失败，恢复会再次执行。相同 namespace/key 的 put 是覆盖操作，避免同一个 key 重复创建，但不能保证模型重试时总是产生完全相同的条目。可在面板检查实际数据。

## 读取链路：为什么新会话能用到

```text
新会话的某个 Agent 开始执行
  → memory_context(runtime)
  → 使用 runtime.context.user_id 搜索 runtime.store
  → 生成长期记忆参考文本
  → 与原系统提示词、本会话 messages 一起发给模型
  → 模型根据当前问题与记忆回答
```

`agents.py` 中的 Supervisor、ResumeAnalyzer、JobSearcher、CoverLetterGenerator、WebResearcher、ChatBot 都接入 `memory_context(runtime)`。读取不依赖旧会话的 messages，因此换 thread_id 后仍有效。

`chains.py` 的两个 chain 构造函数新增可选参数 `memory_context`，用 `SystemMessage(content=...)` 插入提示词。这里使用消息对象，避免记忆 JSON 中的 `{}` 被 `ChatPromptTemplate` 当成占位符。

记忆作为参考数据，提示模型以本轮用户要求优先，不执行记录中夹带的指令，也不要把历史对话中的旧值说成当前已保存值。应用不会把读取的整份记忆反复追加进外层 `State.messages`；原聊天文字和模型引用记忆产生的回答仍会按正常流程保存在 checkpoint。

当前最多注入前 50 条记忆，并提示总条数；面板/查看命令读取全部条目。这适合少量偏好和个人事实，未配置 embedding、向量检索或基于问题的相关性排序，不是文档 RAG。

## 页面与流式处理

`app.py` 在 `define_graph()` 中同时传入 saver 和 store；会话恢复后调用 `ui/memory.py` 的 `render_memory_panel()`。切换会话只改变 thread_id，面板的 user_id 保持不变。

`ui/memory.py` 仅处理表单和展示；保存、删除复用 `services/memory.py`，不调用模型。操作完成后 rerun，再从 MySQL 读取最新记录。

`services/conversation.py` 将 `MemoryManager` 加入可显示角色，但过滤它的 `messages` token 事件。提取模型产生的 JSON 属于内部处理结果，不显示为回答；只有实际操作后的确认，通过节点 `updates` 事件显示在原有聊天框中。

## 数据库改动

现有五张表的结构未更改。`jobagent` 新增以下两张表，均由 Store 的 `setup()` 维护。

### store：一行是一条长期记忆

| 字段 | 含义 |
|---|---|
| `prefix` | namespace 的数据库文本形式，例如 `users.local_user.preferences` |
| `key` | 该 namespace 下的条目名称，例如 `preferred_city` |
| `value` | JSON 数据，包含 label、content、source、source_thread_id |
| `created_at` | 首次创建时间 |
| `updated_at` | 最近一次保存时间 |

联合主键是 `(prefix, key)`，另有 prefix 索引。示例：

```json
{
  "label": "目标城市",
  "content": "上海",
  "source": "chat",
  "source_thread_id": "保存这条记忆的会话 UUID"
}
```

`source=manual` 表示面板保存，`source=chat` 表示聊天命令保存。面板保存的 `source_thread_id` 为 null；更新时记录最近一次来源。`source_thread_id` 是追溯字段，不是外键，不决定可见范围。

### store_migrations：表结构迁移记录

这里只有版本字段 `v`。当前适配器 3.0.0 初始化后包含版本 0、1，分别对应 Store 数据表和索引的迁移。它不保存偏好、用户数量或会话数量，也不与每条记忆做关联。

### 与原表的关系

```text
app_threads ──thread_id──> checkpoint 系列表：每段会话的历史和执行位置

RunContext.user_id ──namespace──> store：多个会话共用的个人偏好/事实

store_migrations：Store 的结构版本
checkpoint_migrations：checkpointer 的结构版本
```

Store 不需要通过 JOIN app_threads 才能读取。新增会话不会复制记忆，删除一条 Store 记忆不会级联删除 checkpoint。

DataGrip 可执行：

```sql
SELECT prefix, `key`, value, created_at, updated_at
FROM jobagent.store
WHERE prefix IN ('users.local_user.preferences', 'users.local_user.facts')
ORDER BY prefix, `key`;

SELECT v FROM jobagent.store_migrations ORDER BY v;
```

初始化不会从旧聊天自动生成长期记忆。刚完成接入时 `store` 为空是正常现象，需要明确保存。

## 验证与重启

已通过 37 项离线测试，包含面板保存/编辑/删除、跨会话读取、提取 JSON 隐藏、普通聊天不写入、不同用户和类别隔离、非法提取结果不写入。

```bat
python -m unittest discover -s tests -v
python -m scripts.verify_mysql_store
```

第二条使用真实 MySQL：六个独立 Python 进程依次保存、读取、更新、读取更新结果、删除、读取删除结果，每阶段使用不同会话。模型用合成响应，验证流程和持久化，不验证真实模型的提取准确率。最后清理此次生成的测试身份、测试会话和 checkpoint，不修改 `local_user` 数据。

本机已安装新增依赖、执行初始化。停止原 Streamlit 后，在 Anaconda Prompt 重启：

```bat
conda activate jobagent
cd /d "E:\agent develop\projects\JobAgent-MultiAgent"
python -m streamlit run app.py
```

在其他环境部署时，先执行 `python -m pip install -r requirements.txt` 和 `python -m scripts.init_mysql`。
