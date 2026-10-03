from docx import Document
from langchain_community.document_loaders import PyMuPDFLoader
import os


def load_resume(file_path):
    """
    简单的简历加载函数
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError("简历文件不存在，请重新上传。")

        file_size = os.path.getsize(file_path)
        if file_size == 0:
            raise ValueError("PDF 文件为空")

        # 使用 PyMuPDFLoader
        loader = PyMuPDFLoader(file_path)
        pages = loader.load()

        content = ""
        for page in pages:
            content += page.page_content + "\n"

        if content.strip():
            return content.strip()
        else:
            raise ValueError("PDF 未提取到文字；扫描件需要先做 OCR。")

    except (OSError, ValueError):
        raise
    except Exception as e:
        raise ValueError("PDF 解析失败，请检查文件是否损坏或加密。") from e

def write_cover_letter_to_doc(text, filename="temp/cover_letter.docx"):
    doc = Document()
    paragraphs = text.split("\n")
    for para in paragraphs:
        if para.strip():
            doc.add_paragraph(para.strip())
    doc.save(filename)
    return filename
