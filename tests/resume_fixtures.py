"""内存中生成合成 PDF；测试不读取用户简历。"""

import pymupdf


def pdf_bytes(text="Synthetic Python developer resume"):
    with pymupdf.open() as document:
        page = document.new_page()
        if text:
            page.insert_text((72, 72), text)
        return document.tobytes()
