import inspect
from typing import Any, Callable, TypeVar

from streamlit.delta_generator import DeltaGenerator
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx
from langchain_community.callbacks.streamlit.streamlit_callback_handler import (
    StreamlitCallbackHandler,
)
from langchain_core.agents import AgentAction


class CustomStreamlitCallbackHandler(StreamlitCallbackHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.agent_sequence = []  # 记录agent执行顺序

    def write_agent_name(self, name: str):
        self._parent_container.write(name)
        # 记录agent执行顺序
        self.agent_sequence.append(name)

    def get_agent_sequence(self):
        return self.agent_sequence

    def clear_agent_sequence(self):
        self.agent_sequence = []

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        # 显示agent正在执行的动作
        tool_name = action.tool
        self._parent_container.write(f"🔧 正在执行: {tool_name}")
        return super().on_agent_action(action, **kwargs)


def initialize_callback_handler(main_container: DeltaGenerator):
    """创建页面回调，并为回调方法绑定 Streamlit 执行上下文。"""
    V = TypeVar("V")

    def wrap_function(func: Callable[..., V]) -> Callable[..., V]:
        context = get_script_run_ctx()

        def wrapped(*args, **kwargs) -> V:
            add_script_run_ctx(ctx=context)
            return func(*args, **kwargs)

        return wrapped

    streamlit_callback_instance = CustomStreamlitCallbackHandler(
        parent_container=main_container
    )

    for method_name, method in inspect.getmembers(
        streamlit_callback_instance, predicate=inspect.ismethod
    ):
        setattr(streamlit_callback_instance, method_name, wrap_function(method))

    return streamlit_callback_instance
