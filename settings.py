"""配置读写。此模块不依赖 Streamlit，也不在导入时修改环境。"""

import json
import os
from pathlib import Path
from typing import Callable

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
TEMP_DIR = PROJECT_ROOT / "temp"
CONFIG_FILE = TEMP_DIR / "app_config.json"
SecretReader = Callable[[str, str], str]


def get_setting(name: str, default: str = "", secret_reader: SecretReader | None = None):
    """环境变量优先；页面可以额外提供 Streamlit secrets 读取器。"""
    value = os.getenv(name)
    if value is not None:
        return value
    return secret_reader(name, default) if secret_reader else default


def initialize_environment(secret_reader: SecretReader | None = None):
    load_dotenv(PROJECT_ROOT / ".env")
    keys = (
        "LINKEDIN_EMAIL", "LINKEDIN_PASS", "LANGCHAIN_API_KEY",
        "LANGCHAIN_PROJECT", "SERPER_API_KEY", "FIRECRAWL_API_KEY",
        "DASHSCOPE_API_KEY", "OPENAI_API_KEY", "OPENAI_BASE_URL",
        "DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL", "QWEN_API_KEY", "QWEN_BASE_URL",
    )
    for key in keys:
        os.environ[key] = get_setting(key, secret_reader=secret_reader)
    os.environ["LANGCHAIN_TRACING_V2"] = get_setting(
        "LANGCHAIN_TRACING_V2", "false", secret_reader
    )
    os.environ["LINKEDIN_SEARCH"] = get_setting(
        "LINKEDIN_JOB_SEARCH", secret_reader=secret_reader
    )


def load_persistent_config():
    if CONFIG_FILE.exists():
        try:
            with CONFIG_FILE.open(encoding="utf-8") as file:
                return json.load(file)
        except (OSError, ValueError) as exc:
            print(f"加载配置文件失败: {exc}")
    return {}


def save_persistent_config(config):
    try:
        CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with CONFIG_FILE.open("w", encoding="utf-8") as file:
            json.dump(config, file, ensure_ascii=False, indent=2)
        return True
    except (OSError, TypeError, ValueError) as exc:
        print(f"保存配置文件失败: {exc}")
        return False


def load_initial_config(secret_reader: SecretReader | None = None):
    """保持原来的优先级：已保存配置 > 环境变量/secrets > 默认值。"""
    saved = load_persistent_config()

    def read(name):
        return get_setting(name, secret_reader=secret_reader)

    return {
        "model_name": saved.get("model_name") or read("MODEL_NAME") or "qwen-plus",
        "api_key": saved.get("api_key") or read("OPENAI_API_KEY") or read("DASHSCOPE_API_KEY"),
        "base_url": saved.get("base_url") or read("OPENAI_BASE_URL") or "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "temperature": saved.get("temperature", 0.3),
        "serper_key": saved.get("serper_key") or read("SERPER_API_KEY"),
        "firecrawl_key": saved.get("firecrawl_key") or read("FIRECRAWL_API_KEY"),
    }
