from typing import TypedDict
from langchain.agents import create_agent
from llms import get_llm
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig
import os

from langgraph.graph import StateGraph, END
from langgraph.runtime import Runtime
from runtime_context import RunContext
from services.memory import handle_memory_request, memory_context, memory_intent
from services.resumes import model_messages
from dotenv import load_dotenv
from chains import get_finish_chain, get_supervisor_chain
from pydantic import ValidationError
from schemas import RouteSchema
from tools import (
    job_search,
    ResumeExtractorTool,
    generate_letter_for_specific_job,
    get_google_search_results, 
    save_cover_letter_for_specific_job,
    scrape_website,
    search_knowledge_base,
)
from prompts import (
    get_analyzer_agent_prompt_template,
    get_search_agent_prompt_template,
    get_generator_agent_prompt_template,
    researcher_agent_prompt_template,
)

load_dotenv()


class AgentState(TypedDict):
    """节点之间传递的业务数据；不包含模型密钥和运行时对象。"""

    user_input: str
    messages: list[BaseMessage]
    next_step: str
    task_completed: bool
    needs_followup: str
    resume_id: str
    resume_filename: str
    resume_context_start: int


def init_chat_model(model_config):
    return get_llm(
        provider=model_config["model_provider"],
        model=model_config["model"],
        dashscope_api_key=model_config.get("DASHSCOPE_API_KEY") or os.environ.get("DASHSCOPE_API_KEY"),
        api_key=model_config.get("DEEPSEEK_API_KEY") or os.environ.get("DEEPSEEK_API_KEY") or model_config.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY"),
        base_url=model_config.get("DEEPSEEK_BASE_URL") or os.environ.get("DEEPSEEK_BASE_URL") or model_config.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_BASE_URL"),
        temperature=model_config.get("temperature", 0.3)
    )


def memory_node(state: AgentState, runtime: Runtime[RunContext], config: RunnableConfig):
    """明确的记忆命令直接处理；普通问题继续进入原 Supervisor 流程。"""
    text = state.get("user_input", "")
    if memory_intent(text) is None:
        return {"next_step": "Supervisor"}
    runtime.context.observer.write_agent_name("MemoryManager")
    reply = handle_memory_request(
        runtime.store, runtime.context.user_id, text,
        lambda: init_chat_model(runtime.context.model_config),
        thread_id=config.get("configurable", {}).get("thread_id"),
    )
    return {
        "messages": state["messages"] + [AIMessage(content=reply, name="MemoryManager")],
        "next_step": "__memory_done__", "task_completed": True, "needs_followup": "",
    }


def supervisor_node(state: AgentState, runtime: Runtime[RunContext]):
    """新请求由模型规划；已验证的后续任务直接继续，不重新规划。"""
    followup = state.get("needs_followup")
    if followup:
        # 兼容已有检查点，只接受当前流程支持的后续节点。
        RouteSchema(steps=["ResumeAnalyzer", followup])
        state.update(next_step=followup, needs_followup="", task_completed=False)
        return state

    chat_history = model_messages(state)
    if not chat_history:
        message = HumanMessage(content=state.get("user_input", ""))
        chat_history.append(message)
        state.setdefault("messages", []).append(message)

    llm = init_chat_model(runtime.context.model_config)
    chain = get_supervisor_chain(llm, memory_context=memory_context(runtime))
    try:
        plan = chain.invoke({"messages": chat_history})
    except ValidationError:
        raise RuntimeError("Supervisor 连续两次未返回有效任务安排，请重试。") from None

    state.update(
        next_step=plan.steps[0],
        needs_followup=plan.steps[1] if len(plan.steps) == 2 else "",
        task_completed=False,
    )
    print(f"任务安排: {' -> '.join(plan.steps)}")
    return state


def resume_analyzer_node(state: AgentState, runtime: Runtime[RunContext]):
    """
    简历分析节点 - 支持协作模式
    """
    llm = init_chat_model(runtime.context.model_config)
    
    analyzer_agent = create_agent(
        llm, [ResumeExtractorTool(resume_id=state.get("resume_id")), get_google_search_results],
        system_prompt=get_analyzer_agent_prompt_template() + memory_context(runtime)
    )
    
    runtime.context.observer.write_agent_name(" ResumeAnalyzer Agent")
    
    result = analyzer_agent.invoke({"messages": model_messages(state)})
    result_content = result["messages"][-1].content
    state["messages"].append(AIMessage(content=result_content, name="ResumeAnalyzer"))
    
    # 如果有后续任务，标记为未完成
    if state.get("needs_followup"):
        state["task_completed"] = False
        print(" 简历分析完成，准备执行后续任务...")
    else:
        state["task_completed"] = True
        print(" 简历分析完成")
    
    return state

def cover_letter_generator_node(state: AgentState, runtime: Runtime[RunContext]):
    """
    求职信生成节点 - 增强协作功能
    """
    llm = init_chat_model(runtime.context.model_config)
    
    generator_agent = create_agent(
        llm, [
            generate_letter_for_specific_job,
            save_cover_letter_for_specific_job,
            ResumeExtractorTool(resume_id=state.get("resume_id")),
        ], 
        system_prompt=get_generator_agent_prompt_template() + memory_context(runtime)
    )

    runtime.context.observer.write_agent_name(" CoverLetterGenerator Agent")
    
    # 检查是否有简历分析结果，如果有则生成更好的提示
    messages_to_use = model_messages(state)
    
    # 查找 ResumeAnalyzer 的输出
    resume_analysis = None
    for msg in reversed(messages_to_use):
        if hasattr(msg, 'name') and msg.name == "ResumeAnalyzer":
            resume_analysis = msg.content
            break
    
    if resume_analysis:
        enhanced_prompt = f"""基于以下简历分析结果，生成一份专业的求职信：

**简历分析结果：**
{resume_analysis}

请根据上述简历分析，生成一份个性化的求职信，突出候选人的关键技能和优势。"""
        
        messages_to_use.append(HumanMessage(content=enhanced_prompt))
        print(" 使用简历分析结果生成求职信")
    
    result = generator_agent.invoke({"messages": messages_to_use})
    result_content = result["messages"][-1].content
    
    # 如果是协作任务的结果，添加说明
    if resume_analysis:
        final_result = f""" **基于简历分析的个性化求职信**

{result_content}

---
*此求职信基于您的简历分析结果生成，确保与您的背景和技能高度匹配*"""
    else:
        final_result = result_content
    
    state["messages"].append(AIMessage(content=final_result, name="CoverLetterGenerator"))
    state["task_completed"] = True
    print(" 求职信生成完成")
    
    return state

def job_search_node(state: AgentState, runtime: Runtime[RunContext]):
    """
    职位搜索节点 - 支持协作模式
    """
    llm = init_chat_model(runtime.context.model_config)
    
    search_agent = create_agent(
        llm, [job_search, get_google_search_results], 
        system_prompt=get_search_agent_prompt_template() + memory_context(runtime)
    )
    
    runtime.context.observer.write_agent_name(" JobSearcher Agent")
    
    # 检查是否有简历分析结果，如果有则生成更好的搜索提示
    messages_to_use = model_messages(state)
    
    # 查找 ResumeAnalyzer 的输出
    resume_analysis = None
    for msg in reversed(messages_to_use):
        if hasattr(msg, 'name') and msg.name == "ResumeAnalyzer":
            resume_analysis = msg.content
            break
    
    if resume_analysis:
        enhanced_prompt = f"""基于以下简历分析结果，搜索和推荐合适的岗位：

**简历分析结果：**
{resume_analysis}

请根据上述简历分析，搜索匹配的岗位机会，重点关注：
1. 与候选人技能匹配的职位
2. 适合候选人经验水平的岗位
3. 候选人所在行业或相关行业的机会
4. 提供具体的岗位列表和申请建议"""
        
        messages_to_use.append(HumanMessage(content=enhanced_prompt))
        print(" 使用简历分析结果搜索匹配岗位")
    
    result = search_agent.invoke({"messages": messages_to_use})
    result_content = result["messages"][-1].content
    
    # 如果是协作任务的结果，添加说明
    if resume_analysis:
        final_result = f""" **基于简历分析的个性化岗位推荐**

{result_content}

---
*此岗位推荐基于您的简历分析结果生成，确保与您的技能和经验高度匹配*"""
    else:
        final_result = result_content
    
    state["messages"].append(AIMessage(content=final_result, name="JobSearcher"))
    state["task_completed"] = True
    print(" 岗位搜索完成")
    
    return state

def web_research_node(state: AgentState, runtime: Runtime[RunContext]):
    """
    资料研究节点：根据问题选择本地知识库或网络工具。
    """
    llm = init_chat_model(runtime.context.model_config)
    
    research_agent = create_agent(
        llm, [search_knowledge_base, get_google_search_results, scrape_website],
        system_prompt=researcher_agent_prompt_template() + memory_context(runtime)
    )
    
    runtime.context.observer.write_agent_name(" WebResearcher Agent")
    
    result = research_agent.invoke({"messages": model_messages(state)})
    result_content = result["messages"][-1].content
    state["messages"].append(AIMessage(content=result_content, name="WebResearcher"))
    state["task_completed"] = True
    return state

def chatbot_node(state: AgentState, runtime: Runtime[RunContext]):
    """聊天机器人节点"""
    llm = init_chat_model(runtime.context.model_config)
    
    runtime.context.observer.write_agent_name("🤖 ChatBot Agent")
    
    finish_chain = get_finish_chain(llm, memory_context=memory_context(runtime))
    output = finish_chain.invoke({"messages": model_messages(state)})
    
    state["messages"].append(AIMessage(content=output.content, name="ChatBot"))
    state["task_completed"] = True
    return state

def define_graph(checkpointer=None, store=None):
    """
    定义支持多Agent协作的工作流图
    """
    workflow = StateGraph(AgentState, context_schema=RunContext)
    # 仅供 update_state(as_node=...) 提交文件绑定；不启动模型或留下待执行节点。
    workflow.add_node("ResumeBinding", lambda state: {})
    workflow.add_edge("ResumeBinding", END)
    
    # 添加节点
    workflow.add_node("Supervisor", supervisor_node)
    workflow.add_node("ResumeAnalyzer", resume_analyzer_node)
    workflow.add_node("JobSearcher", job_search_node)
    workflow.add_node("CoverLetterGenerator", cover_letter_generator_node)
    workflow.add_node("WebResearcher", web_research_node)
    workflow.add_node("ChatBot", chatbot_node)
    
    # 设置入口点
    if store is not None:
        workflow.add_node("MemoryManager", memory_node)
        workflow.set_entry_point("MemoryManager")
        workflow.add_conditional_edges("MemoryManager", lambda state: state["next_step"], {
            "Supervisor": "Supervisor", "__memory_done__": END,
        })
    else:
        workflow.set_entry_point("Supervisor")
    
    # Supervisor 的条件路由
    workflow.add_conditional_edges(
        "Supervisor",
        lambda x: x["next_step"],
        {
            "ResumeAnalyzer": "ResumeAnalyzer",
            "JobSearcher": "JobSearcher", 
            "CoverLetterGenerator": "CoverLetterGenerator",
            "WebResearcher": "WebResearcher",
            "ChatBot": "ChatBot",
            "Finish": "ChatBot"  # 先生成结束语，再由 ChatBot 的条件边结束
        }
    )
    
    # 🔴 关键改动：Agent 完成后的路由逻辑
    def should_continue(state):
        """决定Agent执行完成后是否继续"""
        if state.get("task_completed", True):
            return "END"
        else:
            return "CONTINUE"
    
    # Agent 执行完成后的条件路由
    for agent in ["ResumeAnalyzer", "JobSearcher", "CoverLetterGenerator", "WebResearcher", "ChatBot"]:
        workflow.add_conditional_edges(
            agent,
            should_continue,
            {
                "END": END,
                "CONTINUE": "Supervisor"  # 🔴 回到 Supervisor 继续协作
            }
        )
    
    return workflow.compile(checkpointer=checkpointer, store=store)
