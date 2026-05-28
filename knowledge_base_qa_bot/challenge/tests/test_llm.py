"""驗證 LLMClient：透過 monkeypatch mock OpenAI client，不打真實 API。"""
from unittest.mock import MagicMock

import pytest

from app.llm import LLMClient, LLMResponse


@pytest.fixture
def mock_openai(monkeypatch):
    """把 OpenAI() 換成 MagicMock；test 拿到 mock 物件可以 assert 呼叫紀錄。"""
    mock_client = MagicMock()
    # llm.py 裡會做 OpenAI(api_key=...)；攔截 constructor
    monkeypatch.setattr("app.llm.OpenAI", lambda **kwargs: mock_client)
    return mock_client


def test_ask_returns_parsed_response(mock_openai):
    """LLM 回的 parsed 物件要原樣傳出去。"""
    # 模擬 OpenAI 回傳結構：completion.choices[0].message.parsed
    fake_parsed = LLMResponse(answer="5-7 business days", sources=["refund.md#refund-timeline"])
    fake_message = MagicMock(parsed=fake_parsed, refusal=None)
    fake_choice = MagicMock(message=fake_message)
    fake_completion = MagicMock(choices=[fake_choice])
    mock_openai.chat.completions.parse.return_value = fake_completion

    client = LLMClient(api_key="sk-test", model="gpt-4o-mini")
    result = client.ask([{"role": "user", "content": "anything"}])

    assert result.answer == "5-7 business days"
    assert result.sources == ["refund.md#refund-timeline"]


def test_ask_passes_messages_and_model_to_openai(mock_openai):
    """messages 跟 model 要原封不動傳給 OpenAI。"""
    fake_completion = MagicMock(
        choices=[MagicMock(message=MagicMock(parsed=LLMResponse(answer="x", sources=[]), refusal=None))]
    )
    mock_openai.chat.completions.parse.return_value = fake_completion

    client = LLMClient(api_key="sk-test", model="gpt-4o")
    messages = [{"role": "user", "content": "hi"}]
    client.ask(messages)

    # 檢查呼叫參數
    call_kwargs = mock_openai.chat.completions.parse.call_args.kwargs
    assert call_kwargs["model"] == "gpt-4o"
    assert call_kwargs["messages"] == messages
    assert call_kwargs["response_format"] is LLMResponse


def test_ask_raises_on_refusal(mock_openai):
    """LLM 如果回 refusal（safety / policy）-> 拋 LLMRefusalError、不要靜默吃掉。"""
    from app.llm import LLMRefusalError

    fake_message = MagicMock(parsed=None, refusal="I cannot help with that")
    fake_completion = MagicMock(choices=[MagicMock(message=fake_message)])
    mock_openai.chat.completions.parse.return_value = fake_completion

    client = LLMClient(api_key="sk-test", model="gpt-4o-mini")
    with pytest.raises(LLMRefusalError, match="cannot help"):
        client.ask([{"role": "user", "content": "anything"}])


def test_ask_raises_on_empty_choices(mock_openai):
    """OpenAI 回空 choices list -> 拋 LLMRefusalError、避免 IndexError。"""
    from app.llm import LLMRefusalError

    mock_openai.chat.completions.parse.return_value = MagicMock(choices=[])

    client = LLMClient(api_key="sk-test", model="gpt-4o-mini")
    with pytest.raises(LLMRefusalError, match="no choices"):
        client.ask([{"role": "user", "content": "anything"}])
