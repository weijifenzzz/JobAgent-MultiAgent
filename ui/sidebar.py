"""侧边栏的模型参数和外部服务配置；会话简历由 ui.resume 单独管理。"""

import os
from dataclasses import dataclass

import streamlit as st

from settings import load_initial_config, save_persistent_config


@dataclass
class SidebarResult:
    model_config: dict


def read_streamlit_secret(name, default=""):
    try:
        return st.secrets.get(name, default)
    except Exception:
        return default


def _initialize_saved_settings():
    for name, value in load_initial_config(read_streamlit_secret).items():
        key = f"saved_{name}"
        if key not in st.session_state:
            st.session_state[key] = value


def _render_model_settings():
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🤖 大模型配置")
    with st.sidebar.expander("🔧 模型参数设置", expanded=True):
        # 常用模型选择
        model_preset = st.selectbox(
            "选择预设模型",
            ["自定义", "qwen-plus", "qwen-max", "qwen-turbo", "gpt-4", "gpt-3.5-turbo", "deepseek-chat"],
            help="选择常用模型或自定义输入"
        )

        # 根据预设自动填充
        if model_preset != "自定义":
            default_model = model_preset
            # 自动填充对应的 Base URL
            if model_preset.startswith("qwen"):
                default_base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
            elif model_preset.startswith("gpt"):
                default_base_url = "https://api.openai.com/v1"
            elif model_preset == "deepseek-chat":
                default_base_url = "https://api.deepseek.com/v1"
            else:
                default_base_url = st.session_state["saved_base_url"]
        else:
            default_model = st.session_state["saved_model_name"]
            default_base_url = st.session_state["saved_base_url"]

        model_name = st.text_input(
            "模型名称",
            value=default_model,
            help="输入模型名称，如：qwen-plus, gpt-4, deepseek-chat 等",
            key="model_name_input"
        )

        api_key = st.text_input(
            "API Key",
            value=st.session_state["saved_api_key"],
            type="password",
            help="输入您的 API 密钥",
            key="api_key_input"
        )

        base_url = st.text_input(
            "API Base URL",
            value=default_base_url,
            help="输入 API 端点地址",
            key="base_url_input"
        )

        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state["saved_temperature"],
            step=0.1,
            help="控制输出的随机性，值越低越确定",
            key="temperature_input"
        )

        # 保存配置按钮
        if st.button("💾 保存配置", use_container_width=True):
            st.session_state["saved_model_name"] = model_name
            st.session_state["saved_api_key"] = api_key
            st.session_state["saved_base_url"] = base_url
            st.session_state["saved_temperature"] = temperature

            # 保存到持久化存储
            config_to_save = {
                "model_name": model_name,
                "api_key": api_key,
                "base_url": base_url,
                "temperature": temperature,
                "serper_key": st.session_state.get("saved_serper_key", ""),
                "firecrawl_key": st.session_state.get("saved_firecrawl_key", "")
            }
            if save_persistent_config(config_to_save):
                st.success("✅ 配置已保存并持久化")
            else:
                st.error("❌ 配置保存失败")

        # 显示当前配置提示
        if model_preset != "自定义":
            st.info(f"💡 已选择预设模型：{model_preset}")
            if model_preset.startswith("qwen"):
                st.caption("✓ 通义千问模型，推荐用于中文任务")
            elif model_preset.startswith("gpt"):
                st.caption("✓ OpenAI GPT 模型，需要 OpenAI API Key")
            elif model_preset == "deepseek-chat":
                st.caption("✓ DeepSeek 模型，需要 DeepSeek API Key")

        # 测试连接按钮
        if st.button("🔗 测试连接", use_container_width=True):
            if not api_key or not base_url:
                st.error("❌ 请先填写 API Key 和 Base URL")
            else:
                with st.spinner("正在测试连接..."):
                    try:
                        from llms import get_llm
                        test_llm = get_llm(
                            provider="openai",
                            model=model_name,
                            api_key=api_key,
                            base_url=base_url,
                            temperature=0.1
                        )
                        # 发送一个简单的测试消息
                        response = test_llm.invoke("Hi")
                        st.success(f"✅ 连接成功！模型响应正常")
                        with st.expander("查看测试响应"):
                            st.text(response.content[:200] + "..." if len(response.content) > 200 else response.content)
                    except Exception as e:
                        st.error(f"❌ 连接失败：{str(e)}")
                        if "404" in str(e):
                            st.warning("⚠️ 模型不存在或无权访问，请检查模型名称")
                        elif "401" in str(e):
                            st.warning("⚠️ API Key 无效，请检查密钥")
                        elif "timeout" in str(e).lower():
                            st.warning("⚠️ 连接超时，请检查网络或 Base URL")

    # 保存模型配置
    st.session_state["model_name"] = model_name

    # 构建统一的 settings
    settings = {
        "model": model_name,
        "model_provider": "openai",  # 统一使用 OpenAI 兼容接口
        "temperature": temperature,
        "OPENAI_API_KEY": api_key,
        "OPENAI_BASE_URL": base_url,
    }
    return settings


def _render_service_settings():
    # ==================== 外部服务配置====================
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔌 外部服务配置")

    with st.sidebar.expander("🔍 搜索与抓取服务"):
        serper_api_key = st.text_input(
            "Serper API Key",
            value=st.session_state["saved_serper_key"],
            type="password",
            help="用于网络搜索功能",
            key="serper_key_input"
        )

        firecrawl_api_key = st.text_input(
            "FireCrawl API Key",
            value=st.session_state["saved_firecrawl_key"],
            type="password",
            help="用于网页内容抓取",
            key="firecrawl_key_input"
        )

        # 保存外部服务配置
        if st.button("💾 保存服务配置", use_container_width=True, key="save_services"):
            st.session_state["saved_serper_key"] = serper_api_key
            st.session_state["saved_firecrawl_key"] = firecrawl_api_key

            # 保存到持久化存储
            config_to_save = {
                "model_name": st.session_state.get("saved_model_name", ""),
                "api_key": st.session_state.get("saved_api_key", ""),
                "base_url": st.session_state.get("saved_base_url", ""),
                "temperature": st.session_state.get("saved_temperature", 0.3),
                "serper_key": serper_api_key,
                "firecrawl_key": firecrawl_api_key
            }
            if save_persistent_config(config_to_save):
                st.success("✅ 服务配置已保存并持久化")
            else:
                st.error("❌ 服务配置保存失败")

        # 更新环境变量
        if serper_api_key:
            os.environ["SERPER_API_KEY"] = serper_api_key
        if firecrawl_api_key:
            os.environ["FIRECRAWL_API_KEY"] = firecrawl_api_key
    return serper_api_key, firecrawl_api_key


def _render_status(model_config, serper_api_key, firecrawl_api_key):
    model_name = model_config["model"]
    api_key = model_config["OPENAI_API_KEY"]
    base_url = model_config["OPENAI_BASE_URL"]
    temperature = model_config["temperature"]
    # ==================== 功能状态显示====================
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🛠️ 功能状态")

    # 显示当前使用的模型
    if api_key and base_url:
        st.sidebar.success(f"✅ 当前模型：{model_name}")
        with st.sidebar.expander("查看模型详情"):
            st.markdown(f"""
            - **模型名称**: `{model_name}`
            - **API 端点**: `{base_url}`
            - **Temperature**: `{temperature}`
            """)
    else:
        st.sidebar.error("❌ 请配置模型 API Key 和 Base URL")

    # 搜索功能状态
    if serper_api_key:
        st.sidebar.success("✅ Serper 搜索功能已启用")
    else:
        st.sidebar.warning("⚠️ Serper 未配置，搜索功能受限")

    # 网页抓取状态
    if firecrawl_api_key:
        st.sidebar.success("✅ FireCrawl 网页抓取已启用")
    else:
        st.sidebar.warning("⚠️ FireCrawl 未配置，网页抓取受限")

    # 工具调用支持 - 改为更灵活的判断
    # 通义千问系列、GPT系列、DeepSeek等主流模型都支持工具调用
    if any(keyword in model_name.lower() for keyword in ["qwen", "gpt", "deepseek", "glm", "claude"]):
        st.sidebar.success("✅ 支持高级工具调用功能")
    else:
        st.sidebar.info("ℹ️ 基础模型，部分高级功能可能受限")


def render_sidebar():
    st.sidebar.title("⚙️ 系统配置")
    _initialize_saved_settings()
    model_config = _render_model_settings()
    serper_key, firecrawl_key = _render_service_settings()
    _render_status(model_config, serper_key, firecrawl_key)
    return SidebarResult(model_config)
