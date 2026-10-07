"""Bounded, cancellable JSON transport; terminate only the dedicated H3 worker."""
import atexit
from collections import deque
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
from .config import ROOT


class Worker:
    def __init__(self, mode='auto'):
        self.mode = mode
        self.events = queue.Queue()
        self.errors = deque(maxlen=24)
        self.process = subprocess.Popen([sys.executable, '-u', str(ROOT / 'pi_h3' / 'worker.py'), mode],
            cwd=str(ROOT), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding='utf-8', errors='replace', bufsize=1,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0),
            env={**os.environ, 'PYTHONNOUSERSITE': '1', 'PYTHONUTF8': '1'})
        self.threads = [threading.Thread(target=self._read, args=(self.process.stdout, True), daemon=True),
                        threading.Thread(target=self._read, args=(self.process.stderr, False), daemon=True)]
        for thread in self.threads:
            thread.start()
        atexit.register(self.close)

    def _read(self, pipe, protocol):
        try:
            for line in pipe:
                if protocol:
                    try:
                        event = json.loads(line)
                        if not isinstance(event, dict) or not isinstance(event.get('type'), str):
                            raise TypeError('protocol event must be an object with a type')
                        self.events.put(event)
                    except (ValueError, TypeError):
                        self.errors.append(line.rstrip())
                        self.events.put({'type': 'protocol_error'})
                else:
                    self.errors.append(line.rstrip())
                    print('[H3 worker]', line.rstrip())
        finally:
            if protocol:
                self.events.put({'type': 'exit'})

    def _failure_details(self):
        process = getattr(self, 'process', None)
        code = None
        if process is not None:
            try:
                code = process.poll()
                if code is None:
                    try:
                        code = process.wait(timeout=0.2)
                    except subprocess.TimeoutExpired:
                        pass
            except (AttributeError, OSError):
                pass
        details = []
        if code is not None:
            details.append(f'exit code {code}')
        error_tail = '\n'.join(getattr(self, 'errors', ()))[-2500:]
        if error_tail:
            details.append(error_tail)
        return '\n'.join(details)

    def _worker_error(self, message):
        details = self._failure_details()
        return RuntimeError(message + ((' ' + details) if details else ''))

    def receive(self, cancelled, timeout=7200):
        start = time.monotonic()
        while time.monotonic() - start < timeout:
            if cancelled():
                self.close()
                raise InterruptedError('H3 generation stopped.')
            try:
                event = self.events.get(timeout=0.15)
            except queue.Empty:
                continue
            if event['type'] == 'exit':
                raise self._worker_error('H3 worker stopped.')
            if event['type'] == 'error':
                raise RuntimeError('H3: ' + event['message'])
            if event['type'] == 'protocol_error':
                self.close()
                raise RuntimeError('H3 worker sent an invalid response. ' + '\n'.join(self.errors)[-2500:])
            if event['type'] not in ('ready', 'status', 'progress', 'result'):
                self.close()
                raise RuntimeError('H3 worker sent an unknown response: ' + str(event['type']))
            return event
        self.close()
        raise TimeoutError('H3 worker timed out; its memory has been released.')

    def wait_ready(self, cancelled):
        event = self.receive(cancelled, timeout=120)
        if event['type'] != 'ready':
            raise RuntimeError('Unexpected H3 startup response: ' + str(event))
        return event

    def generate(self, request, cancelled, update):
        if cancelled():
            self.close()
            raise InterruptedError('H3 generation stopped.')
        if self.process.poll() is not None:
            raise self._worker_error('H3 worker is not running.')
        self.process.stdin.write(json.dumps(request) + '\n')
        self.process.stdin.flush()
        while True:
            event = self.receive(cancelled)
            if event['type'] == 'result':
                return event['path']
            update(event)

    def close(self):
        process = self.process
        try:
            if process.stdin is not None and not process.stdin.closed:
                process.stdin.close()
        except (OSError, ValueError):
            pass
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=4)
        for thread in self.threads:
            if thread is not threading.current_thread():
                thread.join(timeout=1)
        for stream in (process.stdin, process.stdout, process.stderr):
            try:
                stream.close()
            except (OSError, ValueError):
                pass
        atexit.unregister(self.close)
