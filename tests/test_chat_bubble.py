"""验证流式、结束事件和历史重绘始终采用同一个助手框结构。"""

import unittest

from streamlit.testing.v1 import AppTest


PAGE = '''
import streamlit as st
from services.conversation import ConversationEvent, ConversationResult
from ui.chat import StreamingReply, render_history

phase = st.session_state.get("phase", "partial")
results = [{"agent": "ResumeAnalyzer", "text": "analysis"},
           {"agent": "JobSearcher", "text": "job answer"}]
if phase == "history":
    render_history(["question"], ["job answer"],
                   [[" ResumeAnalyzer Agent", " JobSearcher Agent"]], [results])
else:
    view = StreamingReply()
    view.update(ConversationEvent("agent_start", "ResumeAnalyzer"))
    view.update(ConversationEvent("text_delta", "ResumeAnalyzer", "anal", "first"))
    if phase in ("second", "done"):
        view.update(ConversationEvent("agent_done", "ResumeAnalyzer", "analysis"))
        view.update(ConversationEvent("agent_start", "JobSearcher"))
        view.update(ConversationEvent("text_delta", "JobSearcher", "job answer", "second"))
    if phase == "done":
        view.update(ConversationEvent("agent_done", "JobSearcher", "job answer"))
        result = ConversationResult("job answer", [],
                    [" ResumeAnalyzer Agent", " JobSearcher Agent"], results)
        view.update(ConversationEvent("done", text=result.reply, result=result))
'''


class ChatBubbleTests(unittest.TestCase):
    def test_partial_text_is_already_inside_assistant_bubble(self):
        page = AppTest.from_string(PAGE).run()
        self.assertFalse(page.exception)
        self.assertEqual(len(page.chat_message), 1)
        self.assertEqual(page.chat_message[0].name, "assistant")
        self.assertIn("anal ▌", [item.value for item in page.chat_message[0].markdown])

    def test_agent_switch_and_history_do_not_add_another_answer_bubble(self):
        page = AppTest.from_string(PAGE)
        snapshots = {}
        for phase in ("second", "done", "history"):
            page.session_state["phase"] = phase
            page.run()
            self.assertFalse(page.exception)
            self.assertEqual(len(page.chat_message), 1)
            bubble = page.chat_message[0]
            self.assertEqual(bubble.name, "assistant")
            self.assertEqual(len(bubble.expander), 1)
            self.assertEqual(bubble.expander[0].label, "ResumeAnalyzer 的阶段结果")
            snapshots[phase] = ([item.value for item in bubble.markdown],
                                [item.value for item in bubble.caption])
        self.assertEqual(snapshots["done"], snapshots["history"])
        self.assertEqual(snapshots["done"][0].count("job answer"), 1)


if __name__ == "__main__":
    unittest.main()
