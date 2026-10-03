# JobAgent-MultiAgent

一个用 LangGraph 和 Streamlit 写的求职助手，支持简历分析、岗位搜索、求职信撰写和网页调研。基于 [connwang7/JobAgent-MultiAgent](https://github.com/connwang7/JobAgent-MultiAgent) 修改，原项目参考了 [JobPilot-Multi-agent](https://github.com/chenchong911/JobPilot-Multi-agent)。

这份代码增加了流式显示、MySQL 会话持久化、会话切换、长期记忆和按会话绑定简历，并拆分了页面与对话执行代码。目前按本地单用户使用，没有登录和用户隔离。

![原项目 Agent 分工示意图](multiagent.png)

上图沿用原项目的 Agent 分工示意，未包含新增的记忆管理和持久化部分；具体路由以当前代码为准。

## 功能

- 简历分析：读取 PDF 的文字，由模型给出分析和修改建议。每段会话单独绑定简历，上传新文件不会覆盖其他会话的文件。
- 岗位搜索：通过 Serper 搜索岗位相关网页，再整理标题、摘要和链接。结果来自搜索页面，不是招聘平台的完整岗位数据库，也不保证岗位仍在招聘。
- 求职信：根据简历和用户提供的岗位要求生成正文。模型调用保存工具时，会在本地 temp/ 下生成 Word 文件；页面暂未提供专门的下载按钮。
- 网页调研：搜索公司或行业信息，按需通过 Firecrawl 读取网页内容，再由模型整理回答。
- 多步任务：支持先分析简历，再搜索岗位或写求职信。任务由 Supervisor 路由，部分复合任务使用关键词规则。
- 会话与记忆：支持流式回复、历史会话列表和切换。会话状态通过 MySQL checkpointer 保存；偏好和事实通过 LangGraph Store 保存，可在侧边栏管理，或用“请记住：……”等明确指令操作。普通聊天不会自动写入长期记忆。

## 运行环境

本地使用 Python 3.11、MySQL 8.0.34。已验证环境中的主要依赖为 LangChain 1.4.2、LangGraph 1.2.12、Streamlit 1.64.0，MySQL 适配器为 langgraph-checkpoint-mysql 3.0.0。

需要一个支持工具调用的 OpenAI 兼容模型接口。岗位搜索需要 Serper API Key，网页抓取需要 Firecrawl API Key；没有配置对应服务时，该工具不可用。

## 本地启动

以下命令在 Windows 的 Anaconda Prompt 中执行。

```bat
git clone https://github.com/weijifenzzz/JobAgent-MultiAgent.git
cd JobAgent-MultiAgent
conda create -n jobagent python=3.11 -y
conda activate jobagent
python -m pip install -r requirements.txt
copy .env.example .env
```

已有环境和 .env 时跳过对应步骤，不要覆盖原来的配置。

编辑项目根目录的 .env，填写模型和本地 MySQL 配置：

```dotenv
OPENAI_API_KEY=你的模型密钥
OPENAI_BASE_URL=服务商提供的OpenAI兼容接口地址
MODEL_NAME=支持工具调用的模型名称

MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_USER=你的数据库用户名
MYSQL_PASSWORD=你的数据库密码
MYSQL_DATABASE=jobagent

SERPER_API_KEY=
FIRECRAWL_API_KEY=
```

需要搜索或抓取网页时，分别填入最后两项。建议首次启动前配好；修改 .env 后重启应用。

确保 MySQL 已启动，再执行：

```bat
python -m scripts.init_mysql
python -m streamlit run app.py
```

初始化脚本会创建 jobagent 数据库及会话、checkpoint、Store 所需表，账号需要建库和建表权限。初始化完成后，日常启动只需运行 `python -m streamlit run app.py`。打开终端显示的 Local URL，默认是 http://localhost:8501。

页面也可以填写和保存模型配置。已在页面保存的配置优先于 .env；如果修改 .env 后页面仍显示旧配置，可以在侧边栏重新填写并保存。

## 使用

1. 新建或选择一个会话。
2. 在“当前会话的简历”中上传 PDF，点击“绑定到当前会话”，也可以选择仓库自带的演示简历。当前普通聊天入口也要求先绑定简历。
3. 填写模型配置，点击“测试连接”，再发送问题。

可以先测试“总结我的简历”，再测试“分析我的简历并推荐上海的 Python 岗位”。写求职信时尽量提供目标公司和具体岗位要求；网页调研需要配置搜索、抓取服务。

重启后可以从会话列表继续原对话。有未完成任务时，页面会提供恢复入口；恢复过程中可能重新调用失败节点里的模型和工具。

长期记忆可以直接在侧边栏添加、修改和删除，也可以单独发送“请记住：我的目标城市是上海”或“查看我的记忆”。当前所有会话共用一份本地用户记忆。

更换简历后，旧聊天仍保留，但后续模型只使用更换后的对话；需要重新分析新简历。长期记忆不随简历更换而清空。旧会话如果没有记录简历 ID，需要手动重新绑定。

## 文件和数据

```text
app.py               Streamlit 入口
agents.py            图、路由和各 Agent 节点
chains.py / prompts.py
                     模型链和提示词
tools.py / utils.py  搜索、网页抓取、简历提取和文档保存工具
data_loader.py       PDF 文字提取、Word 写入
llms.py              模型初始化
settings.py          配置加载与保存
persistence.py       MySQL checkpointer、Store 和会话目录
runtime_context.py   模型配置等运行时依赖
services/            对话执行、简历绑定、长期记忆
ui/                  页面组件和页面状态
scripts/             数据库初始化、集成验证
tests/               自动化测试
docs/                实现说明
```

上传的简历保存在 `temp/resumes/<resume_id>.pdf`，会话在 MySQL 中记录对应的 ID。`temp/app_config.json` 保存页面配置，可能包含密钥。这些本地文件不提交到 Git。

`temp/resumes/` 虽然位于 temp 目录，但会被历史会话长期引用，不要当作普通缓存清理。备份时需要同时保留数据库和简历目录。

## 当前限制

- PDF 使用文字提取，未接入 OCR；纯扫描件需要先识别成带文字的 PDF。
- 岗位信息主要来自搜索摘要，公司、地点等字段可能不完整，需要核对原链接。
- 求职信保存工具返回的是本地路径，不是可直接访问的网页下载地址；同名公司的文件可能被覆盖。
- 工具调用由模型决定，提示词要求不等于程序校验。工具调用能力、外部服务额度和网络状态都会影响结果。
- 没有接入 RAG、用户登录或普通聊天的自动记忆保存，也没有独立的前后端 API。

## 测试

在项目根目录执行离线测试：

```bat
python -m unittest discover -s tests -v
```

测试使用模拟模型和临时文件，覆盖会话、流式事件、简历绑定、长期记忆和页面交互，不消耗模型 API 额度。

配置 MySQL 并初始化后，可以验证跨进程恢复：

```bat
python -m scripts.verify_resume_binding
python -m scripts.verify_mysql_store
```

这两个脚本使用合成数据，完成后清理自己的测试记录。离线测试和数据库验证不代表模型、Serper、Firecrawl 已完成联网验证；使用自己的服务配置后，还需要在页面实际测试。
