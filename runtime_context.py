"""每次执行图时传入的依赖，与可保存的业务 State 分开。"""

from dataclasses import dataclass, field
from typing import Protocol

LOCAL_USER_ID = "local_user"


class AgentObserver(Protocol):
    """记录 Agent 执行过程；实现可以发送流式事件或更新界面。"""

    def write_agent_name(self, name: str) -> None: ...
    def clear_agent_sequence(self) -> None: ...
    def get_agent_sequence(self) -> list[str]: ...


@dataclass(frozen=True)
class RunContext:
    """仅供本次运行使用，不放入 State；后续恢复执行时也需重新提供。"""

    # repr=False 避免打印上下文对象时直接带出模型密钥或回调内部数据。
    model_config: dict = field(repr=False)
    observer: AgentObserver = field(repr=False)
    # 当前本地单用户；未来由登录后的后端身份赋值，不从模型输出中取用户 ID。
    user_id: str = LOCAL_USER_ID
