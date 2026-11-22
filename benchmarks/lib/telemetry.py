"""Process CPU, memory, disk I/O, and SQLite size telemetry sampler."""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Any

from lib.host import db_group_bytes


class ProcTelemetrySampler:
    """Samples server CPU %, RSS/HWM RAM, kernel disk I/O, and SQLite disk sizes periodically."""

    def __init__(self, pid: int, db_path: Path, interval_s: float = 0.25) -> None:
        self.pid = pid
        self.db_path = db_path
        self.interval_s = interval_s
        self.clk_tck = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100
        self._has_proc = Path(f"/proc/{self.pid}/stat").exists()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.samples: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def _read_proc_stat(self) -> float:
        raw = Path(f"/proc/{self.pid}/stat").read_text(encoding="utf-8")
        rparen = raw.rfind(")")
        fields = raw[rparen + 2 :].split()
        return (int(fields[11]) + int(fields[12])) / self.clk_tck

    def _read_proc_status(self) -> tuple[float, float, int]:
        rss_kb = 0.0
        hwm_kb = 0.0
        threads = 1
        for line in Path(f"/proc/{self.pid}/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                rss_kb = float(line.split()[1])
            elif line.startswith("VmHWM:"):
                hwm_kb = float(line.split()[1])
            elif line.startswith("Threads:"):
                threads = int(line.split()[1])
        return rss_kb / 1024.0, hwm_kb / 1024.0, threads

    def _read_proc_io(self) -> dict[str, int]:
        out: dict[str, int] = {}
        try:
            for line in Path(f"/proc/{self.pid}/io").read_text(encoding="utf-8").splitlines():
                k, v = line.split(":", 1)
                out[k.strip()] = int(v.strip())
        except OSError:
            pass
        return out

    def start(self) -> None:
        """Start background sampling thread."""
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> list[dict[str, Any]]:
        """Stop background sampling thread and return a snapshot of collected samples."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        return self.snapshot()

    def reset_samples(self) -> None:
        """Clear collected samples prior to a new benchmark phase or stage."""
        with self._lock:
            self.samples.clear()

    def snapshot(self) -> list[dict[str, Any]]:
        """Return a thread-safe copy of recorded telemetry samples."""
        with self._lock:
            return list(self.samples)

    def _run(self) -> None:
        prev_wall = time.perf_counter()
        prev_cpu = 0.0
        if self._has_proc:
            try:
                prev_cpu = self._read_proc_stat()
            except OSError:
                self._has_proc = False

        while not self._stop.wait(self.interval_s):
            now_wall = time.perf_counter()
            cpu_pct = 0.0
            rss_mb = 0.0
            hwm_mb = 0.0
            threads = 1
            io_stats: dict[str, int] = {}

            if self._has_proc:
                try:
                    now_cpu = self._read_proc_stat()
                    dt_wall = max(1e-6, now_wall - prev_wall)
                    cpu_pct = ((now_cpu - prev_cpu) / dt_wall) * 100.0
                    prev_cpu = now_cpu
                    rss_mb, hwm_mb, threads = self._read_proc_status()
                    io_stats = self._read_proc_io()
                except OSError:
                    break

            prev_wall = now_wall
            db_mb = db_group_bytes(self.db_path) / (1024.0 * 1024.0)
            sample = {
                "t": round(now_wall, 3),
                "cpu_pct_total": round(cpu_pct, 2),
                "rss_mb": round(rss_mb, 2),
                "hwm_mb": round(hwm_mb, 2),
                "threads": threads,
                "io_read_bytes": io_stats.get("read_bytes", 0),
                "io_write_bytes": io_stats.get("write_bytes", 0),
                "io_rchar": io_stats.get("rchar", 0),
                "io_wchar": io_stats.get("wchar", 0),
                "db_disk_mb": round(db_mb, 3),
            }
            with self._lock:
                self.samples.append(sample)
