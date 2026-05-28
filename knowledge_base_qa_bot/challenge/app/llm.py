"""
OpenAI 結構化輸出 client。

用 openai SDK 的 chat.completions.parse() API + pydantic BaseModel 強制輸出格式 ——
這樣 LLM 一定回 {answer: str, sources: list[str]}、不會漏 sources、不會亂格式。

API 版本：openai>=1.40（chat.completions.parse() 已從 beta graduate；不要用 .beta.）。
"""
from openai import OpenAI
from pydantic import BaseModel


class LLMResponse(BaseModel):
    """強制 LLM 回傳的 schema。

    pydantic BaseModel 是 Python 的「執行期型別檢查」工具：
    - 宣告欄位 + 型別 -> 自動產生 __init__、驗證、序列化
    - 這個 class 同時做兩件事：
      1. 告訴 OpenAI SDK 「我要 JSON Schema 對應這個結構」
      2. 接住 LLM 回傳並 deserialize 成 Python 物件
    """
    answer: str
    sources: list[str]


class LLMRefusalError(RuntimeError):
    """LLM 回 refusal（safety / policy）而沒 parsed 結果時拋出。

    refusal 場景：LLM 認為 prompt 不該回答（例如違反 OpenAI policy）。
    在我們的 grounded QA bot 場景幾乎不會遇到，但 SDK API 有這個欄位、要處理。
    """


class LLMClient:
    """OpenAI chat completion 的薄包裝。"""

    def __init__(self, api_key: str, model: str):
        """初始化。

        參數:
            api_key: OpenAI API key（從 config 來）
            model: 模型名，例如 "gpt-4o-mini"

        Dependency injection 概念：把 api_key 和 model 作為參數傳進來，
        而不是在這裡 hardcode 或自己去讀 env var。這樣 test 才能輕鬆換掉
        （傳一個 fake api_key 就好），也讓 config 改動只在一個地方。
        """
        self._client = OpenAI(api_key=api_key)
        self._model = model

    def ask(self, messages: list[dict]) -> LLMResponse:
        """呼叫 OpenAI、強制 structured output。

        參數:
            messages: prompt.build_messages() 的回傳值

        回傳:
            LLMResponse(answer, sources)。

        例外:
            LLMRefusalError: LLM 回 refusal、沒有 parsed 結果
            openai SDK 自己的 exceptions（網路失敗、認證、rate limit）由 caller 處理。
        """
        completion = self._client.chat.completions.parse(
            model=self._model,
            messages=messages,
            response_format=LLMResponse,
        )
        # 防禦：少數錯誤情況下 OpenAI 可能回空 choices；不擋下會 IndexError
        if not completion.choices:
            raise LLMRefusalError("OpenAI returned no choices")
        message = completion.choices[0].message
        # 如果 LLM refused，message.parsed 為 None、message.refusal 有原因
        if message.parsed is None:
            raise LLMRefusalError(message.refusal or "LLM returned no parsed content")
        return message.parsed
