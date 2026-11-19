"""HTTP client of the LangGraph side of the load clients: one kept-alive connection per worker thread.

A service behind HTTP is normally called over connections that stay open. Each worker thread of the load client
keeps one connection to the service and sends its requests over it, one after the other. RocketRide's client
keeps one websocket open in the same way.

Two cases open a new connection: the thread's connection has been idle for longer than IDLE_S (a server closes an
idle connection after a few seconds), or the server closed it between two requests. In the second case the
request is sent once more on a new connection. A request the server did answer is never sent twice.

Standard library only.
"""
import http.client
import threading
import time

IDLE_S = 2.0                 # under uvicorn's default of 5 s for closing an idle connection
STALE = (http.client.RemoteDisconnected, ConnectionResetError, BrokenPipeError, http.client.CannotSendRequest)
_local = threading.local()
_lock = threading.Lock()
STATS = {'requests': 0, 'connections_opened': 0, 'sent_again_after_a_closed_connection': 0}


def _count(key):
    with _lock:
        STATS[key] += 1


def _connection(host, port, timeout, fresh=False):
    conn = getattr(_local, 'conn', None)
    key = (host, port)
    if fresh or conn is None or _local.key != key or time.monotonic() - _local.used > IDLE_S:
        if conn is not None:
            conn.close()
        conn = http.client.HTTPConnection(host, port, timeout=timeout)
        _local.conn, _local.key, _local.reused, _local.used = conn, key, False, time.monotonic()
        _count('connections_opened')
    return conn


def post(port, path, body, content_type, timeout, host='127.0.0.1'):
    """POST body to the service. Returns (status, headers, raw bytes). Raises on a transport failure."""
    _count('requests')
    for attempt in (0, 1):
        conn = _connection(host, port, timeout, fresh=attempt == 1)
        reused = _local.reused
        try:
            conn.request('POST', path, body=body, headers={'Content-Type': content_type})
            response = conn.getresponse()
        except STALE:
            conn.close()
            _local.conn = None
            # No answer has begun, and only a connection that served a request before can have been closed by the
            # server while idle: the request was not taken, so it is sent once more.
            if attempt == 0 and reused:
                _count('sent_again_after_a_closed_connection')
                continue
            raise
        except BaseException:
            # A timeout or any other failure leaves the connection in an unknown state: never use it again.
            conn.close()
            _local.conn = None
            raise
        try:
            raw = response.read()
        except BaseException:
            # The server had begun to answer, so it took the request: never sent again.
            conn.close()
            _local.conn = None
            raise
        _local.used, _local.reused = time.monotonic(), True
        if response.getheader('Connection', '').lower() == 'close':
            conn.close()
            _local.conn = None
        return response.status, response, raw
