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


def test_tool_loop_preserves_provider_signature(monkeypatch):
    from groq.types.chat import ChatCompletion
    from src.agent import agent
    monkeypatch.setattr(agent.support_agent.retriever, 'retrieve', lambda q: [])
    monkeypatch.setattr(agent.support_agent.retriever, 'detect_source_conflicts', lambda p: [])
    calls = []
    def reply(messages, tools=None):
        if not calls:
            calls.append(True)
            data = {'role': 'assistant', 'tool_calls': [{'id': 'call-1', 'type': 'function',
                'function': {'name': 'lookup_order', 'arguments': '{"order_id":"ORD-1007"}'},
                'extra_content': {'google': {'thought_signature': 'fictional-signature'}}}]}
        else:
            assistant = next(m for m in messages if m.get('tool_calls'))
            assert assistant['tool_calls'][0]['extra_content']['google']['thought_signature'] == 'fictional-signature'
            data = {'role': 'assistant', 'content': 'Your fictional order has shipped via UPS.'}
        return ChatCompletion.model_validate({'id': 'fixture', 'created': 1, 'object': 'chat.completion',
            'model': 'gemini-fixture', 'choices': [{'index': 0, 'finish_reason': 'stop', 'message': data}]})
    monkeypatch.setattr(agent, '_groq_call_with_retry', reply)
    result = agent.support_agent.handle_message('Look up fictional order ORD-1007')
    assert result['tool_calls'] == ['lookup_order']
    assert not result['provider_failed'] and 'UPS' in result['response']
