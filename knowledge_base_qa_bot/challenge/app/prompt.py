"""
Prompt 組裝：把 retrieved sections + user query 變成 OpenAI 的 messages list。

格式遵守 OpenAI chat completion 慣例：
    [
        {"role": "system", "content": "<grounding instruction>"},
        {"role": "user",   "content": "<context + question>"},
    ]
"""
from app.types import Section


_SYSTEM_PROMPT = (
    "You are a customer support assistant answering strictly from the provided "
    "knowledge base sections below. Rules:\n"
    "1. Only use facts that appear in the context. Do not invent details.\n"
    "2. Cite every source you used in the 'sources' field using the exact "
    "[filename#heading] tag shown next to each section.\n"
    "3. If the context does not contain the answer, reply that you cannot "
    "confirm this from the knowledge base, and leave 'sources' empty.\n"
)


def build_messages(query: str, sections: list[Section]) -> list[dict]:
    """組成 OpenAI chat completion 的 messages list。

    參數:
        query: 使用者原問題
        sections: retrieval.search 拿到的 top-K sections

    回傳:
        2 個 dict 的 list：system instruction + user content。
    """
    # 把每個 section 排成 "[citation]\nbody" 的 block、用空行分隔
    # f-string 是 Python 3.6+ 內建的字串格式化工具，可以在字串裡嵌入變數：
    #   f"[{s.citation}]\n{s.body}" 等同於 "[" + s.citation + "]\n" + s.body
    context_blocks = [
        f"[{s.citation}]\n{s.body}"
        for s in sections
    ]
    # join(list) 是把 list 裡的字串用指定分隔符合併成一個大字串：
    #   "\n\n".join(["a", "b"]) → "a\n\nb"
    context = "\n\n".join(context_blocks)

    user_content = f"Context:\n{context}\n\nQuestion: {query}"

    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
