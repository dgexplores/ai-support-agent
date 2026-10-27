"""Explicit Gemini REST transport; reuse only the compatible response DTO."""
import httpx
from groq.types.chat import ChatCompletion
from src.config import GEMINI_API_KEY, CHAT_MODEL

ENDPOINT = 'https://generativelanguage.googleapis.com/v1beta/openai/chat/completions'


def complete(messages, tools=None):
    if not GEMINI_API_KEY:
        raise RuntimeError('GEMINI_API_KEY is not configured')
    payload = {'model': CHAT_MODEL, 'messages': messages, 'temperature': 0, 'max_tokens': 1024}
    if tools:
        payload.update(tools=tools, tool_choice='auto')
    response = httpx.post(ENDPOINT, headers={'Authorization': 'Bearer ' + GEMINI_API_KEY},
                          json=payload, timeout=25.0)
    response.raise_for_status()
    result = ChatCompletion.model_validate(response.json())
    if not result.choices or not (result.choices[0].message.content or result.choices[0].message.tool_calls):
        raise RuntimeError('Gemini returned no answer or tool call')
    return result
