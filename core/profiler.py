"""System, frame, render, and chunk performance profiler."""

import json
import time
import psutil

USE_GPU = True
try:
    import pynvml
except Exception:
    pynvml = None
    USE_GPU = False


class PerformanceProfiler:
    """Track FPS, CPU, memory, GPU, VRAM, render and compute timings."""

    def __init__(self, log_interval: float = 1.0, window: int = 120,
                 record_all: bool = False):
        self.log_interval = log_interval
        self.window = window
        self.last_log = time.perf_counter()
        self.last_frame = self.last_log

        # Full-history capture for reproducible benchmark reports. The rolling
        # `window` arrays above are for the live HUD/console readout only and
        # are trimmed, so they cannot be used to compare two runs.
        self.record_all = record_all
        self._all_frame_time = []
        self._all_render_time = []
        self._all_compute_time = []
        self._all_gpu = []
        self._all_vram = []
        self._marks = []

        self._fps = []
        self._cpu = []
        self._mem = []
        self._gpu = []  # percent
        self._vram = []  # percent
        self._render_time = []
        self._compute_time = []

        self._process = psutil.Process()
        self._gpu_handle = None
        if USE_GPU and pynvml is not None:
            try:
                pynvml.nvmlInit()
                self._gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            except Exception:
                pass

    def _sample_system(self):
        cpu = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory().percent
        gpu = None
        vram = None
        if pynvml is None or self._gpu_handle is None:
            return cpu, mem, gpu, vram
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(self._gpu_handle)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(self._gpu_handle)
            gpu = float(util.gpu)
            vram = 100.0 * int(mem_info.used) / int(mem_info.total)
        except Exception:
            pass
        return cpu, mem, gpu, vram

    def frame_start(self):
        self._frame_start = time.perf_counter()

    def frame_end(self, render_time: float = 0.0, compute_time: float = 0.0):
        now = time.perf_counter()
        frame_time = now - self._frame_start
        fps = 1.0 / frame_time if frame_time > 0.0 else 0.0

        cpu, mem, gpu, vram = self._sample_system()

        self._fps.append(fps)
        self._cpu.append(cpu)
        self._mem.append(mem)
        if gpu is not None:
            self._gpu.append(gpu)
        if vram is not None:
            self._vram.append(vram)
        self._render_time.append(render_time)
        self._compute_time.append(compute_time)

        if self.record_all:
            self._all_frame_time.append(frame_time)
            self._all_render_time.append(render_time)
            self._all_compute_time.append(compute_time)
            if gpu is not None:
                self._all_gpu.append(gpu)
            if vram is not None:
                self._all_vram.append(vram)

        self._trim()

        if now - self.last_log >= self.log_interval:
            self.last_log = now
            return self._summary()
        return None

    def _trim(self):
        for arr in (self._fps, self._cpu, self._mem, self._gpu, self._vram, self._render_time, self._compute_time):
            if len(arr) > self.window:
                arr.pop(0)

    def _stats(self, arr):
        if not arr:
            return 0.0, 0.0, 0.0
        return float(sum(arr) / len(arr)), float(min(arr)), float(max(arr))

    def _summary(self):
        fps_avg, fps_min, fps_peak = self._stats(self._fps)
        cpu_avg, cpu_min, cpu_peak = self._stats(self._cpu)
        mem_avg, mem_min, mem_peak = self._stats(self._mem)
        gpu_avg, gpu_min, gpu_peak = self._stats(self._gpu if self._gpu else [0.0])
        vram_avg, vram_min, vram_peak = self._stats(self._vram if self._vram else [0.0])
        render_avg, render_min, render_peak = self._stats(self._render_time)
        compute_avg, compute_min, compute_peak = self._stats(self._compute_time)

        proc_mem = self._process.memory_info().rss / (1024 * 1024)  # MB

        return {
            "fps": {"avg": fps_avg, "min": fps_min, "peak": fps_peak},
            "cpu": {"avg": cpu_avg, "min": cpu_min, "peak": cpu_peak},
            "mem": {"avg": mem_avg, "min": mem_min, "peak": mem_peak},
            "gpu": {"avg": gpu_avg, "min": gpu_min, "peak": gpu_peak},
            "vram": {"avg": vram_avg, "min": vram_min, "peak": vram_peak},
            "render": {"avg": render_avg, "min": render_min, "peak": render_peak},
            "compute": {"avg": compute_avg, "min": compute_min, "peak": compute_peak},
            "proc_mem_mb": proc_mem,
        }

    def format_summary(self, summary):
        lines = [
            "Performance:",
            f"  FPS      avg={summary['fps']['avg']:5.1f}  min={summary['fps']['min']:5.1f}  peak={summary['fps']['peak']:5.1f}",
            f"  CPU %    avg={summary['cpu']['avg']:5.1f}  min={summary['cpu']['min']:5.1f}  peak={summary['cpu']['peak']:5.1f}",
            f"  MEM %    avg={summary['mem']['avg']:5.1f}  min={summary['mem']['min']:5.1f}  peak={summary['mem']['peak']:5.1f}",
            f"  GPU %    avg={summary['gpu']['avg']:5.1f}  min={summary['gpu']['min']:5.1f}  peak={summary['gpu']['peak']:5.1f}",
            f"  VRAM %   avg={summary['vram']['avg']:5.1f}  min={summary['vram']['min']:5.1f}  peak={summary['vram']['peak']:5.1f}",
            f"  Render   avg={summary['render']['avg']*1000:5.2f}ms min={summary['render']['min']*1000:5.2f}ms peak={summary['render']['peak']*1000:5.2f}ms",
            f"  Compute  avg={summary['compute']['avg']*1000:5.2f}ms min={summary['compute']['min']*1000:5.2f}ms peak={summary['compute']['peak']*1000:5.2f}ms",
            f"  Proc Mem {summary['proc_mem_mb']:.1f} MB",
        ]
        return "\n".join(lines)

    def top_chunks(self, chunk_manager):
        """Return top chunks by build time and triangle count."""
        chunks = [c for c in chunk_manager.chunks.values() if getattr(c, "mesh", None) is not None]
        if not chunks:
            return [], []

        by_time = sorted(chunks, key=lambda c: getattr(c, "build_time", 0.0), reverse=True)[:3]
        by_tris = sorted(chunks, key=lambda c: getattr(c, "tri_count", 0), reverse=True)[:3]

        def fmt(c):
            key = c.key()
            tris = getattr(c, "tri_count", 0)
            verts = getattr(c, "vert_count", 0)
            bt = getattr(c, "build_time", 0.0) * 1000.0
            mem = verts * 32 + tris * 3 * 4
            return f"chunk{key} tris={tris} verts={verts} build={bt:.2f}ms mem={mem/1024:.1f}KB"

        return [fmt(c) for c in by_time], [fmt(c) for c in by_tris]

    def mark(self, label: str):
        """Record a named point in the run (e.g. a playback phase change)."""
        self._marks.append({"label": label, "frame": len(self._all_frame_time)})

    @staticmethod
    def _percentile(arr, q: float):
        if not arr:
            return 0.0
        ordered = sorted(arr)
        idx = int(round((len(ordered) - 1) * q))
        return float(ordered[idx])

    def benchmark_report(self, meta=None, warmup_frames: int = 0):
        """Aggregate the full-history capture into a comparable report.

        `warmup_frames` discards the leading frames (shader compilation, chunk
        pop-in) that would otherwise dominate the tail percentiles.
        """
        ft = self._all_frame_time[warmup_frames:]
        rt = self._all_render_time[warmup_frames:]
        ct = self._all_compute_time[warmup_frames:]
        if not ft:
            return {"error": "no frames recorded"}

        fps = [1.0 / f for f in ft if f > 0.0]
        total = sum(ft)
        return {
            "meta": meta or {},
            "frames": len(ft),
            "duration_sec": total,
            "fps": {
                "mean": len(ft) / total if total > 0 else 0.0,
                "median": self._percentile(fps, 0.5),
                # 1% low = mean of the slowest 1% of frames, the standard
                # stutter metric. Reported as FPS so higher is better.
                "low_1pct": 1.0 / self._percentile(ft, 0.99) if self._percentile(ft, 0.99) > 0 else 0.0,
                "low_0_1pct": 1.0 / self._percentile(ft, 0.999) if self._percentile(ft, 0.999) > 0 else 0.0,
            },
            "frame_ms": {
                "mean": 1000.0 * total / len(ft),
                "p50": 1000.0 * self._percentile(ft, 0.5),
                "p95": 1000.0 * self._percentile(ft, 0.95),
                "p99": 1000.0 * self._percentile(ft, 0.99),
                "max": 1000.0 * max(ft),
            },
            "render_ms": {
                "mean": 1000.0 * sum(rt) / len(rt) if rt else 0.0,
                "p95": 1000.0 * self._percentile(rt, 0.95),
                "max": 1000.0 * max(rt) if rt else 0.0,
            },
            "compute_ms": {
                "mean": 1000.0 * sum(ct) / len(ct) if ct else 0.0,
                "p95": 1000.0 * self._percentile(ct, 0.95),
                "max": 1000.0 * max(ct) if ct else 0.0,
            },
            "gpu_pct": {"mean": sum(self._all_gpu) / len(self._all_gpu) if self._all_gpu else 0.0},
            "vram_pct": {"peak": max(self._all_vram) if self._all_vram else 0.0},
            "proc_mem_mb": self._process.memory_info().rss / (1024 * 1024),
            "marks": self._marks,
        }

    def write_report(self, path: str, meta=None, warmup_frames: int = 0):
        report = self.benchmark_report(meta=meta, warmup_frames=warmup_frames)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        return report

    @staticmethod
    def format_benchmark(report):
        if "error" in report:
            return f"Benchmark: {report['error']}"
        return "\n".join([
            f"Benchmark ({report['frames']} frames, {report['duration_sec']:.1f}s)",
            f"  FPS      mean={report['fps']['mean']:7.1f}  median={report['fps']['median']:7.1f}"
            f"  1%low={report['fps']['low_1pct']:7.1f}  0.1%low={report['fps']['low_0_1pct']:7.1f}",
            f"  Frame ms mean={report['frame_ms']['mean']:6.2f}  p50={report['frame_ms']['p50']:6.2f}"
            f"  p95={report['frame_ms']['p95']:6.2f}  p99={report['frame_ms']['p99']:6.2f}"
            f"  max={report['frame_ms']['max']:7.2f}",
            f"  Render   mean={report['render_ms']['mean']:6.2f}  p95={report['render_ms']['p95']:6.2f}"
            f"  max={report['render_ms']['max']:7.2f}",
            f"  Compute  mean={report['compute_ms']['mean']:6.2f}  p95={report['compute_ms']['p95']:6.2f}"
            f"  max={report['compute_ms']['max']:7.2f}",
            f"  GPU {report['gpu_pct']['mean']:.1f}%   VRAM peak {report['vram_pct']['peak']:.1f}%"
            f"   Proc mem {report['proc_mem_mb']:.1f} MB",
        ])

    def log(self, summary, chunk_manager=None):
        out = self.format_summary(summary)
        if chunk_manager is not None:
            top_time, top_tris = self.top_chunks(chunk_manager)
            out += "\nTop chunks by build time:\n  " + "\n  ".join(top_time) if top_time else ""
            out += "\nTop chunks by triangle count:\n  " + "\n  ".join(top_tris) if top_tris else ""
        print(out)
