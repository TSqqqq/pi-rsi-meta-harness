from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass

from .config import Config


@dataclass
class Allocation:
    gpu_ids: list[int]


class GPUScheduler:
    def __init__(self, cfg: Config):
        self.gpus = [int(x) for x in (cfg.get("resources", "gpus", []) or [])]
        self.gpus_per_trial = int(cfg.get("resources", "gpus_per_trial", 1))
        self.max_parallel = int(cfg.get("resources", "max_parallel", max(1, len(self.gpus) or 1)))
        if self.gpus_per_trial < 1:
            raise ValueError("resources.gpus_per_trial must be >= 1")
        if len(set(self.gpus)) != len(self.gpus):
            raise ValueError("resources.gpus must not contain duplicates")
        if self.gpus and self.gpus_per_trial > len(self.gpus):
            raise ValueError("resources.gpus_per_trial cannot exceed the number of configured GPUs")
        if self.max_parallel < 1:
            raise ValueError("resources.max_parallel must be >= 1")
        self._available = list(self.gpus)
        self._cond = asyncio.Condition()
        self._parallel = asyncio.Semaphore(self.max_parallel)

    @asynccontextmanager
    async def acquire(self):
        await self._parallel.acquire()
        try:
            if not self.gpus:
                yield Allocation([])
                return
            async with self._cond:
                while len(self._available) < self.gpus_per_trial:
                    await self._cond.wait()
                ids = self._available[:self.gpus_per_trial]
                del self._available[:self.gpus_per_trial]
            try:
                yield Allocation(ids)
            finally:
                async with self._cond:
                    self._available.extend(ids)
                    self._available.sort()
                    self._cond.notify_all()
        finally:
            self._parallel.release()
