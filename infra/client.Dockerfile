# The load client: Python with the RocketRide client SDK from PyPI. The harness itself uses the standard library
# only and is mounted into the container at run time, so this image holds no benchmark code.
FROM --platform=linux/amd64 python:3.12-slim-bookworm
ARG ROCKETRIDE_CLIENT_VERSION=1.6.0
RUN pip install --no-cache-dir "rocketride==${ROCKETRIDE_CLIENT_VERSION}" \
    && pip freeze > /client-freeze.txt
# The drivers need these parts of the SDK. The build stops here when a client version lacks one.
RUN python3 -c "import inspect; from rocketride import RocketRideClient as C; from rocketride.schema import DocFilter, Question; \
assert 'env' in inspect.signature(C.use).parameters and 'threads' in inspect.signature(C.use).parameters; \
assert all(hasattr(C, m) for m in ['validate', 'pipe', 'chat', 'terminate']); print('client SDK ok')"
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
ENTRYPOINT ["python3", "-u"]
