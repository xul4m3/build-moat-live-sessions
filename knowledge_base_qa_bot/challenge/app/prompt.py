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
    "[filename#heading] tag shown on the line above each section's body.\n"
    "3. If the context does not contain the answer, reply that you cannot "
    "confirm this from the knowledge base, and leave 'sources' empty.\n"
)


def build_messages(query: str, sections: list[Section]) -> list[dict]:
    """組成 OpenAI chat completion 的 messages list。

    參數:
        query: 使用者原問題
        sections: retrieval.search 拿到的 top-K sections。**呼叫端要負責確認非空**
                  —— retrieval 在 fallback=True 時應直接回 cannot-confirm、不呼此函式。
                  若傳空 list 進來，回出去的 user content 會是 "Context:\\n\\nQuestion: ..."
                  （Context 段空白），行為未定義。

    回傳:
        2 個 dict 的 list：system instruction + user content。
        Section 的順序原樣保留 —— 排序是 retrieval 的責任、本函式不重排。
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
