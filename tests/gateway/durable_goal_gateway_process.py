"""Subprocess harness for durable-goal gateway restart integration tests."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time

from gateway.kanban_watchers import (
    GatewayKanbanWatchersMixin,
    _acquire_singleton_lock,
    _release_singleton_lock,
)
from hermes_cli import kanban_db as kb


def _lock_path():
    return kb.kanban_home() / "kanban" / ".dispatcher.lock"


def _hold_old_runtime() -> int:
    handle, state = _acquire_singleton_lock(_lock_path())
    print(json.dumps({"pid": os.getpid(), "lock_state": state}), flush=True)
    if state != "held":
        return 2
    try:
        while True:
            time.sleep(1)
    finally:
        _release_singleton_lock(handle)


class _Runner(GatewayKanbanWatchersMixin):
    def __init__(self) -> None:
        self._running = True


async def _run_watcher_once() -> None:
    runner = _Runner()

    async def _stop_after_real_tick() -> None:
        # Production watcher has a five-second startup delay, then one tick.
        await asyncio.sleep(6.25)
        runner._running = False

    watcher = asyncio.create_task(runner._kanban_dispatcher_watcher())
    stopper = asyncio.create_task(_stop_after_real_tick())
    done, _pending = await asyncio.wait(
        {watcher, stopper}, return_when=asyncio.FIRST_COMPLETED
    )
    if watcher in done:
        stopper.cancel()
        try:
            await stopper
        except asyncio.CancelledError:
            pass
        return
    await watcher


def _watch_once() -> int:
    asyncio.run(_run_watcher_once())
    runtime = None
    with kb.connect() as conn:
        runtime = kb.get_durable_goal_runtime_row(conn)
    print(
        json.dumps(
            {
                "pid": os.getpid(),
                "runtime_id": runtime["runtime_id"] if runtime else None,
            }
        ),
        flush=True,
    )
    return 0


def main() -> int:
    mode = sys.argv[1]
    if mode == "old-hold":
        return _hold_old_runtime()
    if mode == "watch-once":
        return _watch_once()
    raise SystemExit(f"unknown mode: {mode}")


if __name__ == "__main__":
    raise SystemExit(main())
