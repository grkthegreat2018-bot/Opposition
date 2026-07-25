"""System, frame, render, and chunk performance profiler."""

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

    def __init__(self, log_interval: float = 1.0, window: int = 120):
        self.log_interval = log_interval
        self.window = window
        self.last_log = time.perf_counter()
        self.last_frame = self.last_log

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
            mem = verts * 10 * 4 + tris * 3 * 4
            return f"chunk{key} tris={tris} verts={verts} build={bt:.2f}ms mem={mem/1024:.1f}KB"

        return [fmt(c) for c in by_time], [fmt(c) for c in by_tris]

    def log(self, summary, chunk_manager=None):
        out = self.format_summary(summary)
        if chunk_manager is not None:
            top_time, top_tris = self.top_chunks(chunk_manager)
            out += "\nTop chunks by build time:\n  " + "\n  ".join(top_time) if top_time else ""
            out += "\nTop chunks by triangle count:\n  " + "\n  ".join(top_tris) if top_tris else ""
        print(out)
