"""Async Engine for LLM Inference & Real-Time Streaming JSONL Storage.
Optimized for NVIDIA NIM (build.nvidia.com) with strict 40 RPM Sliding-Window Rate Limiting,
Persistent Connection Pooling, Self-Healing json-repair, and Resilient Timeout Handling.
"""

import asyncio
import json
import logging
import random
import re
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
import httpx

try:
    from json_repair import repair_json
except ImportError:
    repair_json = None

from sft_generator.config import settings

logger = logging.getLogger(__name__)


class StreamingJSONLWriter:
    """Thread-safe and async-safe streaming JSONL writer with deduplication and checkpoint resume."""

    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = asyncio.Lock()
        self.existing_task_ids: Set[str] = self._load_existing_ids()

    def _load_existing_ids(self) -> Set[str]:
        """Loads existing task_ids from the output file for checkpointing."""
        ids: Set[str] = set()
        if self.file_path.exists():
            try:
                with open(self.file_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                data = json.loads(line)
                                if "task_id" in data:
                                    ids.add(data["task_id"])
                            except Exception:
                                pass
            except Exception as e:
                logger.warning(f"Could not read existing file {self.file_path}: {e}")
        logger.info(f"Loaded {len(ids)} existing task checkpoints from {self.file_path.name}")
        return ids

    async def write_sample(self, sample: Dict[str, Any]) -> bool:
        """Appends a valid sample to the JSONL file immediately upon completion."""
        task_id = sample.get("task_id")
        async with self.lock:
            if task_id and task_id in self.existing_task_ids:
                return False
            
            with open(self.file_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                f.flush()
                
            if task_id:
                self.existing_task_ids.add(task_id)
            return True


class SlidingWindowRateLimiter:
    """Strict Sliding-Window Rate Limiter guaranteeing requests never exceed max_rpm in any 60-second window."""

    def __init__(self, max_rpm: int = 38):
        self.max_rpm = max_rpm
        self.window_seconds = 60.0
        self.timestamps: deque = deque()
        self.lock = asyncio.Lock()

    async def acquire(self):
        """Acquires permission to send a request, sleeping if the 60s sliding window limit is reached."""
        while True:
            async with self.lock:
                now = time.monotonic()
                # Clean up expired timestamps outside 60s window
                while self.timestamps and (now - self.timestamps[0]) >= self.window_seconds:
                    self.timestamps.popleft()

                if len(self.timestamps) < self.max_rpm:
                    self.timestamps.append(now)
                    return

                # Calculate exact wait time until oldest request in window expires
                oldest = self.timestamps[0]
                wait_time = max(0.05, self.window_seconds - (now - oldest) + 0.1)

            logger.debug(f"[40 RPM Rate Limiter] Window full ({self.max_rpm} reqs in 60s). Pacing for {wait_time:.2f}s...")
            await asyncio.sleep(wait_time)


class APIKeyPool:
    """Manages a pool of API keys with individual Sliding-Window Rate Limiters, cooldowns, and round-robin dispatch."""

    def __init__(self, keys: List[str], max_rpm: int = 38):
        self.keys = keys if keys else ["dummy-key"]
        self.index = 0
        self.cooldowns: Dict[str, float] = {k: 0.0 for k in self.keys}
        self.limiters: Dict[str, SlidingWindowRateLimiter] = {
            k: SlidingWindowRateLimiter(max_rpm=max_rpm) for k in self.keys
        }
        self.lock = asyncio.Lock()

    async def get_next_key_and_acquire(self) -> str:
        """Selects the best healthy key and acquires a slot from its 40 RPM rate limiter."""
        async with self.lock:
            now = time.time()
            selected_key = self.keys[self.index]
            self.index = (self.index + 1) % len(self.keys)

            if self.cooldowns.get(selected_key, 0.0) > now:
                for key in self.keys:
                    if self.cooldowns.get(key, 0.0) <= now:
                        selected_key = key
                        break

        limiter = self.limiters[selected_key]
        await limiter.acquire()
        return selected_key

    async def mark_rate_limited(self, key: str, cooldown_seconds: float = 20.0):
        """Marks a key as rate-limited, putting it in temporary cooldown."""
        async with self.lock:
            self.cooldowns[key] = time.time() + cooldown_seconds
            masked = key[:8] + "..." + key[-4:] if len(key) > 12 else "key"
            logger.warning(f"Key [{masked}] placed on {cooldown_seconds:.1f}s cooldown due to 429 rate-limit.")

    async def mark_exhausted(self, key: str):
        """Marks a key with 401/403 as exhausted."""
        async with self.lock:
            self.cooldowns[key] = time.time() + 3600.0  # 1 hour
            masked = key[:8] + "..." + key[-4:] if len(key) > 12 else "key"
            logger.error(f"Key [{masked}] placed on 1-hour cooldown due to 401/403 Auth/Quota error.")


class AsyncLLMEngine:
    """Async Client managing calls to NVIDIA NIM API with strict 40 RPM Sliding-Window Throttling,
    persistent connection pooling, and self-healing JSON repair.
    """

    def __init__(self, concurrency_limit: Optional[int] = None):
        self.api_base = settings.OPENAI_API_BASE.rstrip("/")
        self.model = settings.CAR_BENCH_MODEL
        self.key_pool = APIKeyPool(settings.get_api_keys(), max_rpm=settings.MAX_RPM)
        limit = concurrency_limit or settings.CONCURRENCY_LIMIT
        self.semaphore = asyncio.Semaphore(limit)
        
        # Shared persistent connection pool
        self.timeout = httpx.Timeout(
            timeout=settings.REQUEST_TIMEOUT,
            connect=25.0,
            read=settings.REQUEST_TIMEOUT,
            write=25.0,
            pool=30.0,
        )
        self.limits = httpx.Limits(max_keepalive_connections=30, max_connections=100, keepalive_expiry=60.0)
        self._client: Optional[httpx.AsyncClient] = None
        self._client_lock = asyncio.Lock()

    async def get_client(self) -> httpx.AsyncClient:
        """Returns or initializes the shared persistent HTTP connection pool."""
        if self._client is None or self._client.is_closed:
            async with self._client_lock:
                if self._client is None or self._client.is_closed:
                    self._client = httpx.AsyncClient(
                        timeout=self.timeout,
                        limits=self.limits,
                    )
        return self._client

    async def close(self):
        """Gracefully closes the connection pool."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_tokens: Optional[int] = None,
        json_mode: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """Sends chat completion with strict rate limiting, auto-token reduction, and resilient error recovery."""
        url = f"{self.api_base}/chat/completions"
        
        clamped_temp = max(0.0, min(1.0, temperature))
        clamped_top_p = max(0.0, min(1.0, top_p))
        eff_max_tokens = max_tokens or settings.MAX_TOKENS

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": clamped_temp,
            "top_p": clamped_top_p,
            "max_tokens": eff_max_tokens,
        }

        include_json_format = json_mode and ("deepseek-r1" not in self.model.lower())
        if include_json_format:
            payload["response_format"] = {"type": "json_object"}

        # DeepSeek API: Explicitly disable thinking mode if not requested to save cost/tokens
        if "deepseek" in self.api_base.lower() or "deepseek" in self.model.lower():
            if not getattr(settings, "THINKING_MODE", False):
                payload["thinking"] = {"type": "disabled"}

        max_retries = settings.MAX_RETRIES
        client = await self.get_client()

        async with self.semaphore:
            for attempt in range(1, max_retries + 1):
                api_key = await self.key_pool.get_next_key_and_acquire()

                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "User-Agent": "CarBench-SFT-Engine/1.0",
                }

                try:
                    response = await client.post(
                        url,
                        headers=headers,
                        json=payload,
                        timeout=self.timeout,
                    )
                    
                    # 1. Success (200 OK)
                    if response.status_code == 200:
                        res_json = response.json()
                        content = res_json["choices"][0]["message"]["content"]
                        return self._parse_json_content(content)

                    # 2. Rate Limit (429 Too Many Requests)
                    elif response.status_code == 429:
                        retry_after_hdr = response.headers.get("Retry-After") or response.headers.get("x-ratelimit-reset-requests")
                        if retry_after_hdr and retry_after_hdr.isdigit():
                            delay = float(retry_after_hdr) + random.uniform(1.0, 3.0)
                        else:
                            exp_backoff = min(settings.MAX_BACKOFF_DELAY, settings.RETRY_DELAY * (2 ** (attempt - 1)))
                            delay = random.uniform(0.5 * exp_backoff, exp_backoff) + random.uniform(1.0, 3.0)

                        await self.key_pool.mark_rate_limited(api_key, cooldown_seconds=delay)
                        logger.warning(
                            f"[LLM 429 Rate-Limit] Backing off {delay:.1f}s on '{self.model}' (Attempt {attempt}/{max_retries})..."
                        )
                        await asyncio.sleep(delay)

                    # 3. Authentication, Quota & Insufficient Balance Errors (401 / 402 / 403)
                    elif response.status_code in [401, 402, 403]:
                        err_msg = response.text[:200]
                        if response.status_code == 402:
                            logger.error(f"[DeepSeek/LLM 402 Insufficient Balance] Account balance is empty. Top up at platform.deepseek.com: {err_msg}")
                        else:
                            logger.error(f"[LLM {response.status_code} Auth/Quota Error] Check API Key / credentials: {err_msg}")
                        await self.key_pool.mark_exhausted(api_key)
                        if len(self.key_pool.keys) > 1:
                            continue
                        else:
                            break

                    # 4. Format / Parameter Errors (400 / 422 Unprocessable Entity)
                    elif response.status_code in [400, 422]:
                        err_text = response.text.lower()
                        if "response_format" in payload:
                            logger.info("[LLM 422/400] Model does not accept response_format. Falling back to self-healing parser...")
                            del payload["response_format"]
                            continue
                        elif "context length" in err_text or "upper bound" in err_text or "completion tokens" in err_text:
                            cur_tokens = payload.get("max_tokens", 2048)
                            if cur_tokens > 1024:
                                payload["max_tokens"] = 1536 if cur_tokens > 1536 else 1024
                                logger.info(f"[LLM 400 Context Length] Auto-reducing max_tokens to {payload['max_tokens']} and retrying...")
                                continue
                        logger.error(f"[LLM HTTP {response.status_code} Error]: {response.text[:250]}")
                        break

                    # 5. Model Not Found (404)
                    elif response.status_code == 404:
                        logger.error(
                            f"[LLM API 404] Model '{self.model}' not found at {self.api_base}. Verify model ID and API endpoint."
                        )
                        break

                    # 6. Server Overload / Gateway Errors (500, 502, 503, 504, 529)
                    elif response.status_code in [500, 502, 503, 504, 529]:
                        delay = min(30.0, (1.8 ** attempt) + random.uniform(1.0, 4.0))
                        logger.warning(
                            f"[LLM API {response.status_code}] Server/Gateway overloaded at {self.api_base}. Retrying in {delay:.1f}s (Attempt {attempt}/{max_retries})..."
                        )
                        await asyncio.sleep(delay)

                    else:
                        logger.error(f"[LLM API HTTP {response.status_code}] Unexpected status: {response.text[:200]}")
                        break

                except (httpx.TimeoutException, httpx.RequestError) as e:
                    err_type = type(e).__name__
                    err_detail = str(e) if str(e) else "Server response timeout / Socket closed"
                    delay = min(20.0, (1.4 ** attempt) + random.uniform(1.0, 2.5))
                    logger.warning(f"[{err_type}] on attempt {attempt}: {err_detail}. Retrying in {delay:.1f}s...")
                    await asyncio.sleep(delay)
                except Exception as e:
                    logger.error(f"Unexpected error in chat_completion: {e}")
                    break

        return None

    def _parse_json_content(self, content: str) -> Optional[Dict[str, Any]]:
        """Extracts and parses JSON object from model response text using
        standard json + Self-Healing json_repair fallback.
        """
        if not content:
            return None
        text = content.strip()

        # 1. Clean <think>...</think> reasoning tags
        text_no_think = re.sub(r"<think>[\s\S]*?</think>", "", text).strip()
        if text_no_think:
            text = text_no_think

        # 2. Extract outermost JSON object { ... }
        # Strip markdown codeblock if the entire response is wrapped in ```json ... ```
        if text.startswith("```"):
            lines = text.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        # Find outermost curly braces to guard against prefix/suffix conversational chatter
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start : end + 1]

        # 3. Fast path: standard json.loads with strict=False
        try:
            parsed = json.loads(text, strict=False)
            if isinstance(parsed, dict) and "conversations" in parsed:
                return parsed
            elif isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

        # 4. Clean trailing commas and retry standard json
        try:
            cleaned_commas = re.sub(r",\s*([\]}])", r"\1", text)
            parsed = json.loads(cleaned_commas, strict=False)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

        # 5. Gold Standard Self-Healing Fallback: json_repair with truncated closure attempts
        if repair_json is not None:
            # Try directly first
            for suffix in ["", "\"]}", "\"}]}", "\"}]}]}", "\n}\n]"]:
                try:
                    candidate = text + suffix
                    repaired = repair_json(candidate, return_objects=True)
                    if isinstance(repaired, dict) and "conversations" in repaired:
                        return repaired
                    elif isinstance(repaired, dict) and len(repaired) > 0:
                        return repaired
                    elif isinstance(repaired, list) and len(repaired) > 0 and isinstance(repaired[0], dict):
                        return repaired[0]
                except Exception as e:
                    logger.debug(f"json_repair suffix '{suffix}' failed: {e}")

        logger.warning(f"Failed to parse JSON from model output: {content[:150]}...")
        return None
