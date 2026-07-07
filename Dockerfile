FROM python:3.12-slim

RUN useradd -m app
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
USER app
EXPOSE 8000

# --workers 1 is MANDATORY (PHASE3_DESIGN §7): in-process asyncio jobs, the
# global concurrency semaphore, and SSE affinity all break silently under
# multi-worker. Revisit only when jobs move to an external worker.
#
# Shell-form CMD so Railway's injected $PORT is honored (§10.F).
# --proxy-headers + FORWARDED_ALLOW_IPS env (§10.A) make request.client.host
# the real client IP behind Railway's CGNAT proxy — required for rate limiting.
# --timeout-graceful-shutdown pairs with railway.json drainingSeconds (§11.R6):
# sse-starlette shutdown grace < uvicorn graceful shutdown < drainingSeconds.
CMD uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1 --proxy-headers --timeout-graceful-shutdown 20
