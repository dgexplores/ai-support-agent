import httpx
from src.agent import gemini


def test_gemini_uses_documented_endpoint_and_function_call_contract(monkeypatch):
    monkeypatch.setattr(gemini, 'GEMINI_API_KEY', 'fictional-key')
    monkeypatch.setattr(gemini, 'CHAT_MODEL', 'gemini-2.5-flash')
    def post(url, **kw):
        assert url == 'https://generativelanguage.googleapis.com/v1beta/openai/chat/completions'
        assert kw['timeout'] == 25.0 and kw['json']['max_tokens'] == 1024
        assert kw['json']['tool_choice'] == 'auto'
        assert kw['json']['reasoning_effort'] == 'none'
        return httpx.Response(200, request=httpx.Request('POST', url), json={
            'id': 'fixture', 'created': 1, 'object': 'chat.completion', 'model': 'gemini-fixture',
            'choices': [{'index': 0, 'finish_reason': 'tool_calls', 'message': {'role': 'assistant',
                'content': None, 'tool_calls': [{'id': 'call-1', 'type': 'function',
                    'function': {'name': 'lookup_order', 'arguments': '{"order_id":"ORD-1007"}'}}]}}],
        })
    monkeypatch.setattr(httpx, 'post', post)
    result = gemini.complete([{'role': 'user', 'content': 'fictional order'}], [{'type': 'function'}])
    assert result.choices[0].message.tool_calls[0].function.name == 'lookup_order'
