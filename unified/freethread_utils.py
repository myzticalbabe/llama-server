"""freethread_utils.py — DEPRECATED re-export shim.

Canonical source: azazael/src/CORE/freethreading.py
 kept so old imports keep working; all new code imports CORE.freethreading.
Slimmed 2026-09-25 (C-P2): was a 700-line fork with divergent behavior
(e.g. hardcoded free_threaded=True); now a pure re-export.
"""
import os
import sys
from pathlib import Path

_ft_path = str(Path(__file__).resolve().parent.parent.parent / 'azazael' / 'src')
if _ft_path not in sys.path:
    sys.path.insert(0, _ft_path)

from CORE.freethreading import (  # noqa: F401,E402
    BenchmarkResult,
    ConcurrentTaskRunner,
    DeviceParallelConfig,
    FreeThreadPool,
    OpenVINOAsyncRunner,
    ThreadSafeCounter,
    ThreadSafeDict,
    ThreadSafeList,
    benchmark_parallel,
    detect_devices,
    get_thread_info,
    is_free_threaded,
    print_benchmark_table,
    print_thread_info,
    set_current_thread_affinity,
    set_thread_affinity,
)

__all__ = [
    "BenchmarkResult", "ConcurrentTaskRunner", "DeviceParallelConfig",
    "FreeThreadPool", "OpenVINOAsyncRunner", "ThreadSafeCounter",
    "ThreadSafeDict", "ThreadSafeList", "benchmark_parallel",
    "detect_devices", "get_thread_info", "is_free_threaded",
    "print_benchmark_table", "print_thread_info",
    "set_current_thread_affinity", "set_thread_affinity",
]
