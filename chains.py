import json
from pydantic import ValidationError
from langchain_core.runnables import RunnableLambda
from schemas import RouteSchema
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage
from members import get_team_members_details
from prompts import get_supervisor_prompt_template, get_finish_step_prompt


def get_supervisor_chain(llm: BaseChatModel, memory_context=""):
    """模型输出 JSON 任务安排；格式或步骤不合法时最多再生成一次。"""
    members = "\n".join(
        f"{member['name']}: {member['description']}"
        for member in get_team_members_details()
    )
    instructions = get_supervisor_prompt_template().format(
        members=members,
        format_instructions=json.dumps(RouteSchema.model_json_schema(), ensure_ascii=False),
    )
    prompt = ChatPromptTemplate.from_messages([
        # 使用消息对象，JSON 和长期记忆里的花括号不参与模板替换。
        SystemMessage(content=instructions),
        *([SystemMessage(content=memory_context)] if memory_context else []),
        MessagesPlaceholder(variable_name="messages"),
    ])
    parse = RunnableLambda(lambda message: RouteSchema.model_validate_json(message.content))
    return (prompt | llm | parse).with_retry(
        retry_if_exception_type=(ValidationError,),
        stop_after_attempt=2,
        wait_exponential_jitter=False,
    )


def get_finish_chain(llm: BaseChatModel, memory_context=""):
    """
    完成对话的链
    """
    system_prompt = get_finish_step_prompt()
    prompt = ChatPromptTemplate.from_messages([
        *([SystemMessage(content=memory_context)] if memory_context else []),
        MessagesPlaceholder(variable_name="messages"),
        ("system", system_prompt),
    ])
    return prompt | llm
