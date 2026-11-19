"""Test of the HTTP client the load clients use for the LangGraph side. A small server on this machine stands for
the service. Standard library only.

  1. Requests of one worker thread go over one connection that stays open: many requests, few connections.
  2. A connection that was idle for longer than the limit is replaced before it is used.
  3. When the server closes a connection between two requests, the request is sent once more and answered.
  4. A request the server answered with an error is returned as it is and never sent twice.
  5. A request whose answer breaks off half way is not sent again either: the server took it.

Usage: python3 tests/test_http_client.py
"""
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'harness'))
import lg_http  # noqa: E402

SEEN = {'connections': 0, 'requests': 0, 'close_after': False}
LOCK = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'                      # connections stay open, as with the real service

    def setup(self):
        super().setup()
        with LOCK:
            SEEN['connections'] += 1

    def do_POST(self):
        body = self.rfile.read(int(self.headers['Content-Length']))
        with LOCK:
            SEEN['requests'] += 1
        status = 500 if self.path == '/fails' else 200
        answer = b'answer to ' + body
        self.send_response(status)
        self.send_header('Content-Length', str(len(answer) + (100 if self.path == '/breaks' else 0)))
        self.send_header('X-Request-ID', body.decode())
        self.end_headers()
        self.wfile.write(answer)
        if self.path == '/breaks':
            self.close_connection = True               # the answer promised 100 bytes more than it sends
        if SEEN['close_after']:
            self.close_connection = True               # the server drops the connection without saying so

    def log_message(self, *args):
        pass


def reset():
    with LOCK:
        SEEN.update(connections=0, requests=0)
    for key in lg_http.STATS:
        lg_http.STATS[key] = 0


def main():
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()

    def ask(i, path='/v1/process/x'):
        status, response, raw = lg_http.post(port, path, b'%d' % i, 'text/plain', 30)
        return status, response.getheader('X-Request-ID'), raw

    # 1. 200 requests from 4 worker threads: at most 4 connections, every answer belongs to its request.
    reset()
    with ThreadPoolExecutor(4) as pool:
        answers = list(pool.map(ask, range(200)))
        assert all(a == (200, str(i), b'answer to %d' % i) for i, a in enumerate(answers)), 'an answer does not belong to its request'
        assert SEEN['requests'] == 200 and 1 <= SEEN['connections'] <= 4, SEEN
        assert lg_http.STATS == {'requests': 200, 'connections_opened': SEEN['connections'], 'sent_again_after_a_closed_connection': 0}, lg_http.STATS
        first = SEEN['connections']

        # 2. After an idle time above the limit every thread opens a new connection before it sends.
        old, lg_http.IDLE_S = lg_http.IDLE_S, 0.05
        time.sleep(0.2)
        list(pool.map(ask, range(4)))
        lg_http.IDLE_S = old
        assert SEEN['connections'] > first and lg_http.STATS['sent_again_after_a_closed_connection'] == 0, (SEEN, lg_http.STATS)
    print('ok  200 requests from 4 threads used %d connections; an idle connection is replaced before use' % first)

    # 3. The server closes the connection after every answer without announcing it: the next request on that
    #    connection finds it closed, is sent once more on a new one and is answered. No request is lost or doubled.
    reset()
    SEEN['close_after'] = True
    one = ThreadPoolExecutor(1)
    answers = list(one.map(ask, range(20)))
    SEEN['close_after'] = False
    assert all(a == (200, str(i), b'answer to %d' % i) for i, a in enumerate(answers)), answers[:3]
    assert SEEN['requests'] == 20, 'a request reached the server twice or not at all: %s' % SEEN
    assert lg_http.STATS['sent_again_after_a_closed_connection'] >= 1, lg_http.STATS
    print('ok  20 requests over connections the server kept closing: all answered once, %d sent again on a new connection' % (
        lg_http.STATS['sent_again_after_a_closed_connection']))

    # 4. An answer with an error status is an answer: returned, not sent again.
    #    Asked from a new worker thread, so that no connection closed in step 3 is in play.
    reset()
    status, rid, raw = ThreadPoolExecutor(1).submit(ask, 7, '/fails').result()
    assert status == 500 and SEEN['requests'] == 1 and lg_http.STATS['sent_again_after_a_closed_connection'] == 0, (status, SEEN)
    print('ok  an answer with an error status is returned and not sent again')

    # 5. The answer breaks off: the client raises and does not send the request again, also on a connection in use.
    reset()
    worker = ThreadPoolExecutor(1)
    assert worker.submit(ask, 1).result()[0] == 200                # the connection has now served a request
    try:
        worker.submit(ask, 2, '/breaks').result()
        raise SystemExit('a broken answer must raise')
    except SystemExit:
        raise
    except Exception as e:
        broken = type(e).__name__
    assert SEEN['requests'] == 2 and lg_http.STATS['sent_again_after_a_closed_connection'] == 0, (SEEN, lg_http.STATS)
    assert worker.submit(ask, 3).result() == (200, '3', b'answer to 3'), 'the next request must go over a new connection'
    print('ok  an answer that breaks off raises %s and the request is not sent again' % broken)
    server.shutdown()
    print('ALL OK')


if __name__ == '__main__':
    main()
