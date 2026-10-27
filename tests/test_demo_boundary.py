"""Public demo access checks never call a live provider."""
import os
os.environ.setdefault('GROQ_API_KEY', 'test-key-not-a-secret')
from fastapi.testclient import TestClient
from src.web import app as web
from src.web.demo_guard import DemoGuard


def test_daily_limit_survives_guard_restart(tmp_path):
    p = tmp_path / 'budget.sqlite'
    assert DemoGuard(p, daily_limit=1).admit('a')
    assert not DemoGuard(p, daily_limit=1).admit('b')


def test_owner_boundary_and_trace_removal(tmp_path, monkeypatch):
    monkeypatch.setattr(web, 'guard', DemoGuard(tmp_path / 'budget.sqlite'))
    calls = []
    def answer(message, session_id, debug=False):
        calls.append(message)
        return {'response': 'Fictional policy answer', 'session_id': session_id, 'trace': {'private': True}}
    monkeypatch.setattr(web.support_agent, 'handle_message', answer)
    with TestClient(web.app) as a, TestClient(web.app) as b:
        assert a.post('/api/chat', json={'message': 'test'}).status_code == 403
        sid = a.post('/api/session/new').json()['session_id']
        assert b.post('/api/chat', json={'message': 'test', 'session_id': sid}).status_code == 403
        assert b.post('/api/session/clear', params={'session_id': sid}).status_code == 403
        result = a.post('/api/chat', json={'message': 'returns?', 'session_id': sid, 'debug': True})
        assert result.status_code == 200 and result.json()['trace'] is None
        assert calls == ['returns?']
        assert a.post('/api/chat', json={'message': 'x'*1501}).status_code == 422
        assert a.post('/api/chat', content='x'*8193).status_code == 413
        assert a.post('/api/session/new', headers={'Origin': 'https://evil.example'}).status_code == 403
        assert a.post('/api/session/clear', params={'session_id': sid}).status_code == 200


def test_rate_limit(tmp_path):
    g = DemoGuard(tmp_path / 'budget.sqlite')
    assert all(g.admit('a') for _ in range(4))
    assert not g.admit('a')


def test_model_outage_is_not_reported_as_success(tmp_path, monkeypatch):
    monkeypatch.setattr(web, 'guard', DemoGuard(tmp_path / 'budget.sqlite'))
    monkeypatch.setattr(web.support_agent, 'handle_message', lambda *a, **k: {'provider_failed': True})
    with TestClient(web.app) as client:
        client.post('/api/session/new')
        assert client.post('/api/chat', json={'message': 'returns?'}).status_code == 503
