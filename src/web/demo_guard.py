"""Single-worker public-demo quotas. Count attempts before calling the provider."""
import os
import sqlite3
import time
from pathlib import Path
from threading import Lock


class DemoGuard:
    def __init__(self, path=None, daily_limit=40):
        self.path = str(path or os.getenv('DEMO_BUDGET_PATH', '/tmp/aster-demo-budget.sqlite3'))
        self.daily_limit = daily_limit
        self.lock = Lock()
        self.recent = {}
        with sqlite3.connect(self.path) as db:
            db.execute('CREATE TABLE IF NOT EXISTS usage (day TEXT PRIMARY KEY, attempts INTEGER NOT NULL)')

    def admit(self, actor, *, inference=True):
        now = time.time()
        with self.lock:
            self.recent = {k: [t for t in v if t > now - 60] for k, v in self.recent.items() if v and v[-1] > now - 60}
            key = ('chat' if inference else 'session', actor)
            recent = self.recent.setdefault(key, [])
            if len(recent) >= 4:
                return False
            if inference:
                with sqlite3.connect(self.path) as db:
                    db.execute('BEGIN IMMEDIATE')
                    day = time.strftime('%Y-%m-%d', time.gmtime(now))
                    db.execute('DELETE FROM usage WHERE day < ?', (day,))
                    row = db.execute('SELECT attempts FROM usage WHERE day = ?', (day,)).fetchone()
                    if row and row[0] >= self.daily_limit:
                        return False
                    db.execute('INSERT INTO usage VALUES (?, 1) ON CONFLICT(day) DO UPDATE SET attempts = attempts + 1', (day,))
            recent.append(now)
            return True
