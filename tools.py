# define tools
import os
import asyncio
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr
from langchain.tools import BaseTool, tool
from langchain_core.tools import ToolException
from data_loader import load_resume, write_cover_letter_to_doc
from schemas import JobSearchInput
from utils import SerperClient,FireCrawlClient
from services.resumes import read_resume_bytes, resume_path
import json

load_dotenv()


# Job search tools

@tool
def job_search(
    keywords: str,
    location_name: str = None,
    job_type: str = None,
    limit: int = 5,
    employment_type: str = None,
    listed_at=None,
    experience=None,
    distance=None,
) -> dict:  # type: ignore
    """
    Search for job postings based on specified criteria using Serper API. Returns detailed job listings.
    """
    try:
        # 构造搜索查询
        query = f"job {keywords}"
        if location_name:
            query += f" in {location_name}"
        if job_type:
            query += f" {job_type}"
        if employment_type:
            query += f" {employment_type}"
        if experience:
            query += f" {experience} experience"
            
        # 使用SerperClient进行搜索
        client = SerperClient()
        response = client.search(query, num_results=limit)
        
        # 解析搜索结果
        jobs = []
        items = response.get("items", [])
        
        for item in items:
            title = item.get("title", "")
            link = item.get("link", "")
            snippet = item.get("snippet", "")
            
            # 提取公司名称（如果可能）
            company_name = ""
            if " at " in title:
                parts = title.split(" at ")
                if len(parts) >= 2:
                    company_name = parts[-1]
            
            job_info = {
                "job_title": title,
                "company_name": company_name,
                "job_location": location_name or "Not specified",
                "job_desc_text": snippet,
                "apply_link": link,
                "time_posted": item.get("date", "Not specified"),
            }
            
            jobs.append(job_info)
            
        return jobs
    except Exception as e:
        print(f"搜索职位时出错: {e}")
        return {"error": f"搜索职位时出错: {str(e)}"}


class ResumeToolInput(BaseModel):
    """模型无需也不能提供文件路径或覆盖会话的简历 ID。"""
    model_config = ConfigDict(extra="forbid")


class ResumeExtractorTool(BaseTool):
    name: str = "resume_extractor"
    description: str = "提取已上传的简历内容进行分析。不需要输入参数。"
    
    args_schema: type[BaseModel] = ResumeToolInput
    handle_tool_error: bool = True
    _resume_id: str | None = PrivateAttr(default=None)

    def __init__(self, *, resume_id=None, **kwargs):
        super().__init__(**kwargs)
        self._resume_id = resume_id

    def _run(self) -> str:
        """提取简历内容"""
        try:
            if not self._resume_id:
                raise ValueError("本会话尚未绑定简历，请在侧边栏上传并绑定。")
            read_resume_bytes(self._resume_id)
            return load_resume(resume_path(self._resume_id))
        except (OSError, ValueError) as exc:
            raise ToolException(f"简历提取失败：{exc}") from exc
    
    async def _arun(self) -> str:
        return await asyncio.to_thread(self._run)

# Cover Letter Generation Tool
@tool
def generate_letter_for_specific_job(resume_details: str, job_details: str) -> dict:
    """
    Generate a tailored cover letter using the provided CV and job details. This function constructs the letter as plain text.
    returns: A dictionary containing the job and resume details for generating the cover letter.
    """
    return {"job_details": job_details, "resume_details": resume_details}


@tool
def save_cover_letter_for_specific_job(
    cover_letter_content: str, company_name: str
) -> str:
    """
    Returns a download link for the generated cover letter.
    Params:
    cover_letter_content: The combine information of resume and job details to tailor the cover letter.
    """
    filename = f"temp/{company_name}_cover_letter.docx"
    file = write_cover_letter_to_doc(cover_letter_content, filename)
    abs_path = os.path.abspath(file)
    return f"Here is the download link: {abs_path}"


# Web Search Tools
@tool("google_search")
def get_google_search_results(
    query: str = Field(..., description="Search query for web")
) -> str:
    """
    search the web for the given query and return the search results.
    """
    response = SerperClient().search(query)
    items = response.get("items")
    string = []
    for result in items:
        try:
            string.append(
                "\n".join(
                    [
                        f"Title: {result['title']}",
                        f"Link: {result['link']}",
                        f"Snippet: {result['snippet']}",
                        "---",
                    ]
                )
            )
        except KeyError:
            continue

    content = "\n".join(string)
    return content


@tool("scrape_website")
def scrape_website(url: str = Field(..., description="Url to be scraped")) -> str:
    """
    Scrape the content of a website and return the text.
    """
    try:
        content = FireCrawlClient().scrape(url)
    except Exception as exc:
        return f"Failed to scrape {url}"
    return content
