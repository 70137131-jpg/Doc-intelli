"""Redis-based sliding window rate limiter for FastAPI."""

import time

from fastapi import HTTPException, Request, status

from app.core.redis import redis_client


class RateLimiter:
    """Sliding window rate limiter using Redis sorted sets."""

    def __init__(self, max_requests: int = 100, window_seconds: int = 60, key_prefix: str = "rl"):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.key_prefix = key_prefix

    async def __call__(self, request: Request):
        # Use user ID if authenticated, otherwise IP
        user = getattr(request.state, "user", None)
        if user:
            identifier = f"user:{user.id}"
        else:
            identifier = f"ip:{request.client.host}" if request.client else "ip:unknown"

        key = f"{self.key_prefix}:{identifier}"
        now = time.time()
        window_start = now - self.window_seconds

        pipe = redis_client.pipeline()
        pipe.zremrangebyscore(key, 0, window_start)
        pipe.zcard(key)
        pipe.zadd(key, {str(now): now})
        pipe.expire(key, self.window_seconds + 1)

        try:
            results = await pipe.execute()
            request_count = results[1]
        except Exception:
            # Redis down — allow request (fail open)
            return

        if request_count >= self.max_requests:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded. Max {self.max_requests} requests per {self.window_seconds}s.",
                headers={"Retry-After": str(self.window_seconds)},
            )


# Pre-configured limiters
rate_limit_default = RateLimiter(max_requests=100, window_seconds=60)
rate_limit_upload = RateLimiter(max_requests=10, window_seconds=60, key_prefix="rl:upload")
rate_limit_agents = RateLimiter(max_requests=10, window_seconds=60, key_prefix="rl:agents")
