"""集中管理页面会话；状态由入口传入，不在导入时初始化。"""

from copy import deepcopy

THINKING = "🤔 正在思考中..."
STOPPED = "⚠️ 执行已被用户停止"


def initialize_session(state):
    defaults = {
        "user_query_history": ["你好! 👋"],
        "response_history": ["你好！我是 JobAgent-MultiAgent 职业助手，请问需要什么帮助？"],
        "agent_sequence_history": [],
        "agent_results_history": [],
        "stop_execution": False,
        "is_processing": False,
        "input_key": 0,
        "pending_index": None,
        "resume_execution": False,
    }
    for key, value in defaults.items():
        if key not in state:
            state[key] = deepcopy(value)
    # 兼容刷新前已有的聊天记录；显示文字不再作为待执行任务的判断依据。
    while len(state["agent_sequence_history"]) < len(state["response_history"]):
        state["agent_sequence_history"].append([])
    while len(state["agent_results_history"]) < len(state["response_history"]):
        state["agent_results_history"].append([])


def start_request(state, user_input):
    if state["is_processing"]:
        return
    state["resume_execution"] = False
    state["user_query_history"].append(user_input)
    state["response_history"].append(THINKING)
    state["agent_sequence_history"].append([])
    state["agent_results_history"].append([])
    state["pending_index"] = len(state["response_history"]) - 1
    state["is_processing"] = True
    state["stop_execution"] = False
    state["input_key"] += 1


def get_pending_request(state):
    index = state["pending_index"]
    if index is None:
        return None
    return state["user_query_history"][index]


def finish_request(state, reply, agent_sequence=None, agent_results=None):
    index = state["pending_index"]
    if index is not None:
        state["response_history"][index] = reply
        state["agent_sequence_history"][index] = list(agent_sequence or [])
        state["agent_results_history"][index] = list(agent_results or [])
    state["pending_index"] = None
    state["is_processing"] = False
    state["resume_execution"] = False


def stop_request(state):
    state["stop_execution"] = True
    finish_request(state, STOPPED)


def clear_conversation(state, message_history):
    state["user_query_history"] = []
    state["response_history"] = []
    state["agent_sequence_history"] = []
    state["agent_results_history"] = []
    state["pending_index"] = None
    state["stop_execution"] = False
    state["is_processing"] = False
    state["input_key"] += 1
    message_history.clear()
